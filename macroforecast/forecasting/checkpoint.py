"""Runner-level incremental checkpointing for long pseudo-out-of-sample runs.

A multi-hour POOS run computes one (target, horizon) cell entirely in memory and
writes once at the end. If that final write fails, hours of compute are lost. The
checkpoint here persists a LEAN forecast record per origin as soon as the origin
completes, and lets a restarted run skip origins that already finished.

The guiding principle (Chan's): ``[prediction, actual, model_id]`` plus the keys
that identify the forecast cell is a sufficient statistic for every downstream
metric and test (RMSE, relative RMSE, DM, CW, MCS, forecast combination). Model
interpretation needs the fitted model and features, which is handled separately
by ``save_models`` / refit and is deliberately OUT OF SCOPE here.

``variance_prediction`` (Phase 1 density pipeline) extends that sufficient
statistic: it is a plain float already emitted by the direct policy for models
exposing ``predict_variance`` (see ``forecasting/policies/base.py``), so it is
now a FIXED lean column (like ``prediction``/``actual``) -- ``None`` when the
model does not emit it, same as before this change added it. Quantile
predictions are a ``{level: value}`` mapping, not a scalar, so they are
expanded into WIDE per-level columns rather than added to the fixed schema: the
set of levels is a per-model hyperparameter (``quantile_levels``), not known
ahead of time, so the wide columns are derived empirically per origin from
whatever the record's ``quantile_predictions`` mapping actually carries (see
``_quantile_wide_columns``). ``load_checkpoint_frame`` unions differing column
sets across origin files via ``pd.concat`` (pandas fills the gaps with NaN), so
this needs no cross-origin coordination and degrades gracefully when an old
checkpoint (written before this column existed) sits alongside new ones.

Quantile columns
----------------
A wide column is named ``qx1_`` plus the 16 hex digits of the level's IEEE-754
binary64 image (``struct.pack(">d", level).hex()``), e.g. ``0.05`` becomes
``qx1_3fa999999999999a``. The name therefore DETERMINES the level exactly: the
decode is the true inverse, so a resumed origin's mapping carries the very
``str(level)`` keys the live run produced.

The first version of this schema instead wrote ``q_<pct>`` for
``round(level * 100)``, which was neither injective nor lossless: ``0.024`` and
``0.025`` both became ``q_02`` (the second write overwriting the first),
``0.975`` and ``0.976`` both became ``q_98``, ``1/3`` came back as ``0.33``, and
every level at or above ``0.995`` became ``q_100``, which the two-digit reader
could not parse at all. Those files stay READABLE: a legacy ``q_01``..``q_99``
column still decodes, at the only value it can, its integer-percent grid point.
Both endpoints of that grid stay unread. ``q_100`` (every level at or above
``0.995``) and ``q_00`` (every level below ``0.005``) each name an endpoint that
is not a valid level -- ``1.0`` and ``0.0`` are outside the open interval a
quantile lives in -- and the true level that produced either is unrecoverable,
so reporting the endpoint would invent a quantile nobody requested. ``q_00`` is
the one the old reader did fabricate, decoding it to ``0.0``; it is now dropped
like ``q_100``. Within a row the exact family wins wholesale -- any valid
``qx1_`` value means the legacy columns beside it are ignored rather than mixed
in at a rounded level.

Format
------
``checkpoint_path`` is treated as a DIRECTORY. Each completed origin writes one
parquet file ``origin_<pos>.parquet`` containing only that origin's lean records
(scalar columns only -- no dict/struct columns; quantile predictions are
pre-expanded into wide scalar ``qx1_<hex>`` columns, see above). One-file-per-
origin makes each write atomic at the file level (a crash mid-write corrupts at
most one origin's file, which is simply ignored on resume), and resume is the
trivial act of listing the directory.

Resume identity
---------------
Listing the directory says WHICH origins finished. It says nothing about WHAT
they are, and a checkpoint directory is reached by path, not by configuration:
the same path handed to a run with a different regularisation strength, window,
panel, or seed used to hand those old parquet files straight back as this run's
forecasts (F-058). Each final ``h<h>`` directory therefore also carries a
versioned run-identity manifest, :data:`CHECKPOINT_IDENTITY_FILENAME`, written
atomically BEFORE the first origin file can be written or trusted.

The manifest is self-contained: the runner builds the identity (it is the layer
that knows what a run is) and hands this module a
:class:`CheckpointRunIdentity`, so checkpoint storage never imports policy or
model code. :func:`resolve_checkpoint_resume` is the gate, and it fails closed --
it never deletes, renames, quarantines, or adopts an existing artifact:

===========================  ==========================================  ===========================================
Final ``origin_*.parquet``   Stored / current identity                   Behaviour
===========================  ==========================================  ===========================================
none                         absent, corrupt, stale, or incomplete       replace the manifest, start fresh
present                      absent (legacy checkpoint)                  refuse
present                      malformed or wrong schema/version           refuse
present                      stored or current identity incomplete       refuse
present                      complete, digests differ                    refuse, naming what changed
present                      complete, digests match                     resume the readable origins
===========================  ==========================================  ===========================================

The version is part of that gate. It was incremented to 2 when the quantile
column grammar changed (see "Quantile columns"), because a version-1 directory's
origin files may hold rounded, collapsed levels: resuming into one would put
those rows and this run's exact-level rows in a single forecast table, which is
precisely the mixed representation ``runner._merge_checkpoint_records`` exists
to prevent. A version-1 manifest beside final origin files therefore refuses,
under the ordinary "wrong schema/version" row above; a version-1 manifest with
no origin files to describe is just a stale manifest and is replaced as before.

Refusing is not the same as invalidating. A refused directory is untouched, and
:func:`load_checkpoint_frame` / :func:`load_selection_history_frame` (and hence
``pipeline.rescore``) keep reading legacy checkpoints exactly as before. What a
legacy checkpoint cannot do is be RESUMED INTO, because nothing on disk records
the configuration that produced it. Adopting it would mean writing this run's
identity next to artifacts whose identity is unknowable, which is the false
blessing the gate exists to prevent; recovery is the user's ordinary choice of a
fresh ``checkpoint_path`` or their own removal of the old one.
"""
from __future__ import annotations

import json
import math
import os
import re
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# The lean schema: keys that identify a forecast cell plus the sufficient
# statistic (prediction, actual, model, variance). Every per-model record
# builder in ``runner.py`` produces these keys, so the lean projection is
# path-agnostic. ``variance_prediction`` is ``None`` for every policy/model that
# does not emit one (recursive, path_average, panel, combinations, and any
# direct-policy model without ``predict_variance``) -- same graceful-absence
# convention the rich (non-lean) forecast table already uses for that column.
LEAN_FORECAST_COLUMNS: tuple[str, ...] = (
    "target",
    "horizon",
    "origin",
    "origin_pos",
    "date",
    "model",
    "prediction",
    "actual",
    "forecast_policy",
    "target_transform",
    "variance_prediction",
    "vintage_id",
    "actuals_vintage_id",
)

_ORIGIN_FILE_RE = re.compile(r"^origin_(-?\d+)\.parquet$")
_ORIGIN_SELECTION_FILE_RE = re.compile(r"^origin_(-?\d+)_selection\.jsonl$")

SELECTION_HISTORY_COLUMNS: tuple[str, ...] = (
    "target",
    "arm",
    "horizon",
    "origin",
    "origin_pos",
    "date",
    "kind",
    "name",
    "value",
    "model",
    "step",
    "method",
    "score",
    "source",
)


def _origin_filename(origin_pos: Any) -> str:
    return f"origin_{int(origin_pos)}.parquet"


def _origin_selection_filename(origin_pos: Any) -> str:
    return f"origin_{int(origin_pos)}_selection.jsonl"


#: ``qx1_`` = quantile, heX encoding, grammar version 1. Fixed 20-character
#: names, disjoint from the legacy ``q_<pct>`` family by construction.
_QUANTILE_COLUMN_PREFIX = "qx1_"

_WIDE_QUANTILE_COLUMN_RE = re.compile(r"^qx1_([0-9a-f]{16})$")
#: What the first version of this schema wrote. Read-only; never written again.
_LEGACY_WIDE_QUANTILE_COLUMN_RE = re.compile(r"^q_(\d{2})$")


def _validated_quantile_level(value: Any) -> float | None:
    """The level as a plain float, or ``None`` when it cannot be a level.

    A quantile level is finite and strictly inside ``(0, 1)``. Anything else --
    a nonnumeric string, a non-finite float, ``0.0``, ``1.0``, a probability
    outside the unit interval -- is not encodable and is simply dropped by the
    callers, the same silent-skip the encoder has always applied to unusable
    entries. (A numeric string is NOT rejected here: ``float`` accepts it, and
    accepting it is deliberate, since a mapping keyed by ``str(level)`` is
    exactly what the runner contract passes around.) The old encoder reached
    ``round(inf * 100)`` for a non-finite level and raised an ``OverflowError``
    its caller did not catch; validating here closes that.
    """
    try:
        level = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        return None
    return level


def _quantile_column_name(level: float) -> str:
    """Wide checkpoint column name for a quantile level.

    ``0.05`` -> ``qx1_3fa999999999999a``: the prefix plus the 16 hex digits of
    the level's IEEE-754 binary64 image. Injective and exactly invertible, so
    distinct levels never share a column and no level is rounded on the way to
    disk (see the module docstring's "Quantile columns").
    """
    return _QUANTILE_COLUMN_PREFIX + struct.pack(">d", float(level)).hex()


def _quantile_wide_columns(record: Mapping[str, Any]) -> dict[str, float]:
    """Expand a record's ``quantile_predictions`` mapping into wide, scalar
    ``qx1_<hex>`` columns (parquet needs scalar columns; see module docstring).
    Absent, ``None``, or non-mapping values expand to no columns at all, as do
    unusable levels and unusable predictions.
    """
    value = record.get("quantile_predictions")
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for level, prediction in value.items():
        validated = _validated_quantile_level(level)
        if validated is None:
            continue
        try:
            out[_quantile_column_name(validated)] = float(prediction)
        except (TypeError, ValueError):
            continue
    return out


def _level_from_quantile_column(name: str) -> float | None:
    """Exact level a ``qx1_<hex>`` column encodes, or ``None`` if it is not one.

    Also ``None`` for a well-formed name whose payload does not decode to a
    usable level (a corrupt or hand-edited file), so a damaged column is ignored
    rather than allowed to inject a nonsense level into the mapping.
    """
    match = _WIDE_QUANTILE_COLUMN_RE.match(name)
    if match is None:
        return None
    return _validated_quantile_level(struct.unpack(">d", bytes.fromhex(match.group(1)))[0])


def _legacy_level_from_quantile_column(name: str) -> float | None:
    """Level recoverable from a legacy ``q_<pct>`` column, or ``None``.

    Only the integer-percent grid point, which is all the old encoder retained.
    ``q_100`` -- what it emitted for every level at or above ``0.995`` -- is
    deliberately NOT read: the level that produced it is unrecoverable, and
    reporting ``1.0`` would be inventing a quantile that was never requested.
    ``q_00`` is likewise unread, ``0.0`` being outside the valid range.
    """
    match = _LEGACY_WIDE_QUANTILE_COLUMN_RE.match(name)
    if match is None:
        return None
    return _validated_quantile_level(int(match.group(1)) / 100.0)


def _is_wide_quantile_column(name: str) -> bool:
    """Whether a column name belongs to either wide-quantile family."""
    return (
        _WIDE_QUANTILE_COLUMN_RE.match(name) is not None
        or _LEGACY_WIDE_QUANTILE_COLUMN_RE.match(name) is not None
    )


def _quantile_dict_from_wide(record: Mapping[str, Any]) -> dict[str, float] | None:
    """Reconstruct a ``{level_str: value}`` mapping from a lean record's wide
    quantile columns -- the inverse of :func:`_quantile_wide_columns`.

    Matches the exact string-keyed format ``forecasting/policies/direct.py``
    writes onto the rich (non-lean) forecast table (``str(level)`` for a Python
    float, e.g. ``"0.05"``/``"0.5"``/``"0.95"``), so a resumed-from-checkpoint
    origin's ``quantile_predictions`` merges back into the SAME representation
    a freshly-computed origin's does (see
    ``forecasting/runner.py::_merge_checkpoint_records``). The hex payload is an
    on-disk detail and never appears in the returned mapping.

    Precedence is per row and wholesale: if the row carries any usable
    ``qx1_<hex>`` value, ONLY that family is decoded, because a legacy column
    beside it can at best repeat one of those levels at a rounded value.
    Otherwise the legacy ``q_<pct>`` columns are decoded. Returns ``None`` (not
    ``{}``) when neither family yields anything, matching the rich table's
    ``None``-for-absent convention.
    """
    exact: dict[str, float] = {}
    legacy: dict[str, float] = {}
    for key, value in record.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        name = str(key)
        level = _level_from_quantile_column(name)
        target = exact
        if level is None:
            level = _legacy_level_from_quantile_column(name)
            target = legacy
        if level is None:
            continue
        try:
            target[str(level)] = float(value)
        except (TypeError, ValueError):
            continue
    return exact or legacy or None


def lean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project rich runner records onto the lean (scalar-only) schema.

    Missing keys in :data:`LEAN_FORECAST_COLUMNS` are filled with ``None`` so
    that fixed part of the parquet schema is stable across origins and
    execution paths. ``quantile_predictions`` (when present and non-empty) is
    additionally expanded into wide ``qx1_<hex>`` columns -- see
    :func:`_quantile_wide_columns` and the module docstring.
    """
    out: list[dict[str, Any]] = []
    for record in records:
        row = {column: record.get(column) for column in LEAN_FORECAST_COLUMNS}
        row.update(_quantile_wide_columns(record))
        out.append(row)
    return out


def completed_origin_positions(checkpoint_path: str | Path) -> set[int]:
    """Return the set of ``origin_pos`` values already persisted under the dir.

    A file that cannot be read (e.g. a partial write from a crash) is ignored, so
    its origin is recomputed rather than trusted.
    """
    directory = Path(checkpoint_path)
    if not directory.exists():
        return set()
    done: set[int] = set()
    for path in directory.glob("origin_*.parquet"):
        match = _ORIGIN_FILE_RE.match(path.name)
        if match is None:
            continue
        try:
            # A readable file is a completed origin; validate it parses.
            pd.read_parquet(path, columns=["origin_pos"])
        except Exception:
            continue
        done.add(int(match.group(1)))
    return done


def final_origin_files(checkpoint_path: str | Path) -> list[Path]:
    """Final (renamed) per-origin parquet files under the directory, sorted.

    Membership is by FILENAME, not by readability. ``completed_origin_positions``
    asks "which origins can I resume", and an unreadable file is not one of them;
    this asks "does this directory already hold user artifacts", and an unreadable
    ``origin_3.parquet`` plainly does. In-flight ``.origin_3.parquet.tmp`` writes
    are excluded by the glob, which is the point of the dot-prefixed temporary
    name in :func:`append_origin_records`.
    """
    directory = Path(checkpoint_path)
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.glob("origin_*.parquet")
        if _ORIGIN_FILE_RE.match(path.name) is not None
    )


# --------------------------------------------------------------------------- #
# Resume identity (see the module docstring's "Resume identity" section)
# --------------------------------------------------------------------------- #

#: Per-``h<h>`` manifest naming the run that owns the directory's origin files.
CHECKPOINT_IDENTITY_FILENAME = "run_identity.json"
CHECKPOINT_IDENTITY_SCHEMA = "macroforecast_checkpoint_run_identity"
CHECKPOINT_IDENTITY_VERSION = 2


@dataclass(frozen=True)
class CheckpointRunIdentity:
    """One run's resume identity, as the runner computed it.

    Self-contained by design: ``digest`` is over the canonical form of
    ``components`` and is compared as a whole, so this module never needs to know
    what a window, a model spec, or a stage policy is. ``complete`` is ``False``
    when any component could not be represented canonically -- an opaque custom
    object, a user callable -- and an incomplete identity can never establish that
    two runs agree, only that they might.
    """

    digest: str
    complete: bool
    opaque_fields: tuple[str, ...]
    components: Mapping[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        """The on-disk payload for this identity."""
        return {
            "schema": CHECKPOINT_IDENTITY_SCHEMA,
            "version": CHECKPOINT_IDENTITY_VERSION,
            "digest": self.digest,
            "digest_algorithm": "sha256",
            "complete": bool(self.complete),
            "opaque_fields": list(self.opaque_fields),
            "components": dict(self.components),
        }


def checkpoint_identity_path(checkpoint_path: str | Path) -> Path:
    """Path of the run-identity manifest for one ``h<h>`` checkpoint directory."""
    return Path(checkpoint_path) / CHECKPOINT_IDENTITY_FILENAME


def write_checkpoint_identity(
    checkpoint_path: str | Path,
    identity: CheckpointRunIdentity,
) -> Path:
    """Atomically write (or replace) the directory's run-identity manifest.

    Same tmp-file-plus-rename idiom the origin parquet writes use, so a reader
    never observes a half-written manifest and a crash leaves either the old
    manifest or the new one, never a blend of the two.
    """
    directory = Path(checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / CHECKPOINT_IDENTITY_FILENAME
    encoded = (
        json.dumps(
            identity.to_manifest(),
            sort_keys=True,
            ensure_ascii=True,
            default=_json_default,
        )
        + "\n"
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{final_path.name}.",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
        Path(tmp_name).replace(final_path)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return final_path


def read_checkpoint_identity(
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read the manifest as ``(payload, problem)``; exactly one is ``None``.

    ``problem`` is a short human phrase naming why the manifest cannot be used,
    so the caller's refusal message can say which of "absent", "unreadable", and
    "wrong schema" it hit rather than lumping them together.
    """
    path = checkpoint_identity_path(checkpoint_path)
    if not path.exists():
        return None, "no manifest is present"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"the manifest could not be read ({type(exc).__name__})"
    if not isinstance(payload, Mapping):
        return None, "the manifest is not a JSON object"
    if payload.get("schema") != CHECKPOINT_IDENTITY_SCHEMA:
        return None, (
            f"the manifest schema is {payload.get('schema')!r}, not "
            f"{CHECKPOINT_IDENTITY_SCHEMA!r}"
        )
    try:
        version = int(payload.get("version"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None, f"the manifest version {payload.get('version')!r} is not an integer"
    if version != CHECKPOINT_IDENTITY_VERSION:
        return None, (
            f"the manifest is version {version}, not "
            f"{CHECKPOINT_IDENTITY_VERSION}"
        )
    if not isinstance(payload.get("digest"), str) or not payload.get("digest"):
        return None, "the manifest carries no digest"
    if not isinstance(payload.get("complete"), bool):
        return None, "the manifest does not record whether its identity is complete"
    return dict(payload), None


def _changed_component_keys(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> list[str]:
    """Top-level identity components that differ, for an actionable message."""
    keys = sorted(set(stored) | set(current))
    changed: list[str] = []
    for key in keys:
        left = json.dumps(stored.get(key), sort_keys=True, default=str)
        right = json.dumps(current.get(key), sort_keys=True, default=str)
        if left != right:
            changed.append(key)
    return changed


def _refusal(directory: Path, n_origins: int, detail: str) -> str:
    return (
        f"checkpoint directory {str(directory)!r} already holds {n_origins} completed "
        f"origin file(s), and {detail} Resuming would return forecasts computed "
        "under a configuration this run cannot show to be the same one. Nothing was "
        "written, removed, or renamed: point checkpoint_path at a fresh directory to "
        "compute this configuration from scratch, or move/remove the existing "
        "directory yourself if its forecasts are no longer wanted. The existing "
        "files remain readable by load_checkpoint_frame() and pipeline.rescore()."
    )


def resolve_checkpoint_resume(
    checkpoint_path: str | Path,
    identity: CheckpointRunIdentity,
) -> set[int]:
    """The resume gate: which completed origins may this run reuse?

    Returns the resumable ``origin_pos`` set -- every readable origin file when
    the directory demonstrably belongs to this run, and the empty set when the
    directory holds no origin files yet (in which case the manifest is written,
    or replaced, before the caller writes anything). Raises ``ValueError`` in
    every other case; see the module docstring for the full table.

    The gate deliberately does not consult ``skip_computation``. A run that
    recomputes a completed origin still MERGES the on-disk row over its own (see
    ``runner._merge_checkpoint_records``), so a stale directory is returned to
    the caller either way.
    """
    directory = Path(checkpoint_path)
    origin_files = final_origin_files(directory)
    stored, problem = read_checkpoint_identity(directory)

    if not origin_files:
        # There is no artifact for the old manifest to describe, so replacing it
        # blesses nothing. This is also what puts the manifest on disk before the
        # first origin parquet of a fresh run.
        write_checkpoint_identity(directory, identity)
        return set()

    n_origins = len(origin_files)
    if stored is None:
        if problem == "no manifest is present":
            raise ValueError(
                _refusal(
                    directory,
                    n_origins,
                    "no manifest records which configuration produced them (a "
                    "checkpoint written before run-identity manifests existed, or "
                    "one whose manifest was removed).",
                )
            )
        raise ValueError(_refusal(directory, n_origins, f"{problem}."))

    if not stored.get("complete", False):
        raise ValueError(
            _refusal(
                directory,
                n_origins,
                "the run that wrote them recorded an INCOMPLETE identity "
                f"(opaque field(s) {list(stored.get('opaque_fields') or [])!r}), so "
                "it cannot be compared with this run's.",
            )
        )
    if not identity.complete:
        raise ValueError(
            _refusal(
                directory,
                n_origins,
                "this run's identity is INCOMPLETE: field(s) "
                f"{list(identity.opaque_fields)!r} could not be represented "
                "canonically, so equality with the stored identity cannot be "
                "established. Give those values a stable __mf_digest__ marker to "
                "make the identity comparable.",
            )
        )
    if str(stored.get("digest")) != identity.digest:
        stored_components = stored.get("components")
        changed = (
            _changed_component_keys(stored_components, identity.components)
            if isinstance(stored_components, Mapping)
            else []
        )
        detail = (
            f"this run's identity differs from theirs in {changed!r}."
            if changed
            else "this run's identity differs from theirs."
        )
        raise ValueError(_refusal(directory, n_origins, detail))

    return completed_origin_positions(directory)


def append_origin_records(
    checkpoint_path: str | Path,
    origin_pos: Any,
    records: list[dict[str, Any]],
) -> None:
    """Atomically persist one origin's lean records as ``origin_<pos>.parquet``.

    Written to a temporary file in the same directory and then renamed, so a
    reader never observes a half-written parquet file. An origin with no records
    (e.g. an empty fit/test slice) still writes a marker file so the origin is
    treated as completed and not recomputed on resume.

    Columns are the fixed :data:`LEAN_FORECAST_COLUMNS` schema plus, when any of
    these records carries quantile predictions, the wide ``qx1_<hex>`` columns
    those particular records need (sorted, for a deterministic column order).
    Every origin's file need not carry the same wide columns -- a point-only
    origin (or a run whose model does not emit quantiles) simply omits them,
    and ``load_checkpoint_frame`` unions the schemas back together across files.
    """
    directory = Path(checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    rows = lean_records(records)
    extra_columns = sorted(
        {key for row in rows for key in row if key not in LEAN_FORECAST_COLUMNS}
    )
    frame = pd.DataFrame.from_records(rows, columns=[*LEAN_FORECAST_COLUMNS, *extra_columns])
    final_path = directory / _origin_filename(origin_pos)
    tmp_path = directory / f".{_origin_filename(origin_pos)}.tmp"
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(final_path)


def append_origin_selection_records(
    checkpoint_path: str | Path,
    origin_pos: Any,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically persist one origin's optional selection-history sidecar.

    The sidecar is newline-delimited JSON next to ``origin_<pos>.parquet``. It is
    intentionally independent of the lean forecast parquet schema, so enabling
    selection history does not perturb checkpoint/rescore forecast loading.
    """
    directory = Path(checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / _origin_selection_filename(origin_pos)
    fd, tmp_name = tempfile.mkstemp(
        suffix=".tmp",
        prefix=f".{final_path.name}.",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                payload = json.dumps(
                    dict(record),
                    sort_keys=True,
                    ensure_ascii=True,
                    default=_json_default,
                )
                handle.write(payload + "\n")
        Path(tmp_name).replace(final_path)
    except OSError:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_selection_history_frame(checkpoint_path: str | Path) -> pd.DataFrame:
    """Load optional selection-history JSONL sidecars from one checkpoint dir."""
    directory = Path(checkpoint_path)
    empty = pd.DataFrame(columns=list(SELECTION_HISTORY_COLUMNS))
    if not directory.exists():
        return empty
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("origin_*_selection.jsonl")):
        match = _ORIGIN_SELECTION_FILE_RE.match(path.name)
        if match is None:
            continue
        forecast_path = directory / _origin_filename(int(match.group(1)))
        if not forecast_path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if not text:
                        continue
                    records.append(json.loads(text))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
    if not records:
        return empty
    frame = pd.DataFrame.from_records(records)
    for column in SELECTION_HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def load_checkpoint_frame(checkpoint_path: str | Path) -> pd.DataFrame:
    """Load all persisted lean records as a single frame (empty if none/missing).

    Origin files may carry different wide quantile columns, and may use either
    grammar (a point-only or pre-density-pipeline origin has none, and a
    pre-F-059 origin has legacy ``q_<pct>`` ones); ``pd.concat`` unions them,
    filling gaps with NaN, so this needs no cross-file coordination. When
    any wide quantile column is present, a ``quantile_predictions`` mapping column
    is additionally synthesized (the wide columns are kept alongside it, not
    dropped) so this frame's quantile representation matches the rich
    (non-checkpointed) forecast table's -- one ``{level_str: value}`` dict per
    row -- and every downstream consumer (``evaluate()``'s density stage,
    ``rescore()``) can use the SAME dict-based dispatch regardless of whether
    the forecasts came from a live run or a checkpoint.
    """
    directory = Path(checkpoint_path)
    empty = pd.DataFrame(columns=list(LEAN_FORECAST_COLUMNS))
    if not directory.exists():
        return empty
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.glob("origin_*.parquet")):
        if _ORIGIN_FILE_RE.match(path.name) is None:
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            continue
    if not frames:
        return empty
    result = pd.concat(frames, ignore_index=True)
    has_wide_quantiles = any(_is_wide_quantile_column(str(c)) for c in result.columns)
    if has_wide_quantiles and "quantile_predictions" not in result.columns:
        result["quantile_predictions"] = [
            _quantile_dict_from_wide(row) for row in result.to_dict(orient="records")
        ]
    return result


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)
