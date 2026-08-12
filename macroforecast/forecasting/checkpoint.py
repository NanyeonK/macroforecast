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
``pipeline.rescore``) keep reading HEALTHY legacy checkpoints exactly as before.
What changed for those readers is corruption, not age, and is described next.
What a legacy checkpoint cannot do is be RESUMED INTO, because nothing on disk
records the configuration that produced it. Adopting it would mean writing this
run's identity next to artifacts whose identity is unknowable, which is the false
blessing the gate exists to prevent; recovery is the user's ordinary choice of a
fresh ``checkpoint_path`` or their own removal of the old one.

Reading a checkpoint
--------------------
A checkpoint directory is read for two different questions, and they want
opposite answers about the same damaged file (F-060/F-061).

*Which origins may I skip?* is :func:`completed_origin_positions`, asked by the
resume gate. An unreadable origin file is not a skippable origin, so it is
ignored, and the runner recomputes that origin and overwrites the damaged file.
That is why a matching resume SELF-HEALS, and it is deliberately unchanged.

*Give me the forecasts* is :func:`load_checkpoint_frame` and
:func:`load_selection_history_frame`. No recomputation is available to a reader --
the checkpoint IS the data source -- so these fail closed with
:class:`CheckpointCorruptionError` rather than return a frame that is silently
short an origin. They fail fast on the first offending artifact in sorted
filename order rather than enumerating every corruption, they never modify the
directory, and the reader's own (pyarrow- and version-dependent) error is chained
as ``__cause__`` rather than quoted into a message users would match on. A
sidecar is parsed into a LOCAL buffer and accepted only once every nonblank line
has decoded to a JSON object, so one truncated mid-write contributes nothing
rather than its surviving prefix.

The sidecar's JSON boundary is STRICT on the way out and LENIENT on the way in
(F-062), and that asymmetry is deliberate. Writing normalizes a non-finite
numeric leaf to ``null`` and encodes with ``allow_nan=False``, because the bare
``NaN``/``Infinity`` tokens ``json.dumps`` emits by default are ECMAScript
literals that every strict RFC 8259 parser rejects. Reading still accepts those
tokens and maps them to missing, because sidecars already on disk are full of
them and refusing them would invalidate a legacy directory over a defect that
only ever lived in the writer.

That writer FAILS CLOSED, and the runner's call order is what makes that the safe
choice. Both live call sites write the sidecar BEFORE the origin's parquet, with
no ``try`` between the two calls, so a structural encoding failure propagates
before ``append_origin_records`` runs. The origin therefore never gains an
``origin_<pos>.parquet``, :func:`completed_origin_positions` never counts it, and
re-running the same configuration recomputes that origin and writes both artifacts
again. Degrading an unencodable record to force the write would do the opposite:
it lets the parquet be written, which marks the origin completed and FREEZES the
degraded sidecar, because a completed origin is skipped and never rewrites one.
Raising costs one recomputed origin; degrading costs that history permanently. So
an unencodable record cleans up its temp file, leaves any previous final sidecar
exactly as it was, and re-raises.

*How do I get the data back?* has two different answers, one per artifact, and
each message carries its own. A damaged ``origin_<pos>.parquet`` is healed by
re-running the same configuration against the same ``checkpoint_path``, because
the resume gate does not count it as completed; moving or removing it first, or
using a fresh directory, work as well. A damaged
``origin_<pos>_selection.jsonl`` is NOT healed that way. Its parquet is read
independently, so a healthy parquet keeps that origin completed however damaged
the sidecar is, the runner skips the origin, and a skipped origin writes no
sidecar. Reconstructing that history means moving or removing BOTH the sidecar
and its matching parquet and re-running with selection history enabled, or using
a fresh directory. Removing the sidecar alone is allowed, but it leaves that
origin's selection history absent rather than reconstructed.

The one caller that wants neither answer is ``runner._merge_checkpoint_records``,
which assembles the frame a finished run returns. Raising there would discard a
completed run's entire result over one file, so it alone reads through
:func:`_load_checkpoint_frame_tolerant` and warns once, naming what it excluded.
That tolerance is NOT a claim that what it excluded was irrelevant. Origins this
run computed or recomputed are already in memory and are unaffected, but an origin
a PREVIOUS run completed is supplied by that read alone, and the resume gate read
the directory before the run started, so a file intact then and damaged now is
dropped and the returned frame is short that origin. The warning states both and
names the paths; the public readers still refuse the directory outright.
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

import numpy as np
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


#: Stable ``CheckpointCorruptionError.artifact`` value for a per-origin
#: ``origin_<pos>.parquet`` forecast file.
CHECKPOINT_ORIGIN_ARTIFACT = "checkpoint origin file"

#: Stable ``CheckpointCorruptionError.artifact`` value for a per-origin
#: ``origin_<pos>_selection.jsonl`` selection-history sidecar.
CHECKPOINT_SELECTION_ARTIFACT = "selection-history sidecar"


class CheckpointCorruptionError(ValueError):
    """A checkpoint artifact could not be read, so nothing was returned.

    Raised by :func:`load_checkpoint_frame`, :func:`load_selection_history_frame`,
    and everything built on them (``pipeline.rescore``,
    ``pipeline.selection_history``, ``pipeline.selection_frequency_table``) when a
    file belonging to the checkpoint cannot be parsed. Reading is all-or-nothing
    on those paths: a reader has no way to recompute, so skipping the unreadable
    artifact would hand back a silently short result (see the module docstring's
    "Reading a checkpoint").

    It subclasses ``ValueError``, which is what these loaders' callers already
    catch, and carries the three things a handler needs:

    ``path``
        The :class:`~pathlib.Path` of the artifact that failed.
    ``artifact``
        Which kind of artifact it is: ``"checkpoint origin file"`` or
        ``"selection-history sidecar"``.
    ``line``
        The 1-based line number for a sidecar line this reader rejected, and
        ``None`` for a whole-file failure: an unreadable parquet, or a sidecar
        that could not be opened or decoded as UTF-8 at all.

    Where something underneath raised -- an unreadable parquet, an unreadable or
    non-UTF-8 sidecar, a line that is not valid JSON -- that exception is chained
    as ``__cause__``, and its text is deliberately kept out of this message
    because it is pyarrow- and version-dependent. Exactly one case has no
    ``__cause__``, by design: a sidecar line that decoded cleanly but is not a
    JSON object. Nothing failed underneath it, this reader rejected a well-formed
    value, and manufacturing a cause would misreport where the fault is.
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path,
        artifact: str,
        line: int | None = None,
    ) -> None:
        super().__init__(message)
        self.path = path
        self.artifact = artifact
        self.line = line


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
        "directory yourself if its forecasts are no longer wanted. Files that are "
        "intact remain readable by load_checkpoint_frame() and pipeline.rescore()."
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


#: Returned by :func:`_unboxed_numeric` for a value it cannot narrow, so that a
#: legitimate ``None`` is never confused with "nothing to do here".
_NOT_UNBOXED = object()


def _unboxed_numeric(value: Any) -> Any:
    """One-step unboxing for numpy scalars and single-element numpy arrays.

    Returns :data:`_NOT_UNBOXED` for anything it cannot narrow, which leaves that
    value rendered exactly as it is rendered today.

    ``.item()`` is not trustworthy on its own. ``np.longdouble.item()`` returns
    another ``np.longdouble`` wherever ``longdouble`` is wider than ``float64``, so
    it is a FIXED POINT, and following it is what made the pre-F-062 writer die with
    ``RecursionError`` on a longdouble leaf, a FINITE one included. One unboxing is
    therefore followed by an explicit narrowing rather than by re-entry.

    That narrowing covers the REAL numpy scalar types only, ``np.floating`` and
    ``np.integer``. A value still numpy-typed after one ``.item()`` and neither of
    those is reported as :data:`_NOT_UNBOXED` and keeps its pre-F-062 path unchanged
    -- ``np.clongdouble`` most of all, which is the same fixed point as
    ``np.longdouble`` and so still reaches that ``RecursionError``. Extending this
    to complex types is deliberately out of scope.
    """
    if isinstance(value, np.ndarray):
        if value.size != 1:
            # Wider arrays keep falling through to ``_json_default``'s ``str()``.
            return _NOT_UNBOXED
        value = value.item()
    if not isinstance(value, np.generic):
        return value if isinstance(value, (bool, int, float)) else _NOT_UNBOXED
    unboxed = value.item()
    if not isinstance(unboxed, np.generic):
        return unboxed
    if isinstance(unboxed, np.floating):
        # The ``longdouble`` case. ``float()`` is lossy above ``float64``, and is
        # also the only real number JSON has, so the precision goes either way.
        return float(unboxed)
    if isinstance(unboxed, np.integer):
        return int(unboxed)
    return _NOT_UNBOXED


def _json_constant_spelling(number: float) -> str:
    """The token ``json`` itself writes for a non-finite float."""
    if math.isnan(number):
        return "NaN"
    return "Infinity" if number > 0 else "-Infinity"


def _json_boundary_key(key: Any) -> Any:
    """Keep a mapping key encodable without disturbing one that already encoded.

    ``json`` renders a non-finite float KEY as a quoted string --
    ``{float("nan"): 1}`` encodes to ``{"NaN": 1}``, which is valid RFC 8259 --
    but it routes that spelling through the same ``allow_nan`` switch as a
    non-finite VALUE. Flipping the switch without handling keys would therefore
    start REFUSING a line that was standards-clean all along, so the key is
    rewritten to that exact spelling here and the switch never sees it.

    Only a non-finite :class:`float` key is rewritten -- which covers
    ``np.float64``, a genuine ``float`` subclass ``json`` already accepted as a key.
    Every other key reaches the encoder UNTOUCHED, so ``np.int64`` and
    ``np.float32`` keys still raise the ``TypeError`` they always raised: a key rule
    is also an ACCEPTANCE rule, and F-062 changes what this writer emits, not what
    it admits.
    """
    if isinstance(key, float) and not math.isfinite(key):
        return _json_constant_spelling(key)
    return key


def _json_boundary_mapping(
    mapping: Mapping[Any, Any],
    active: frozenset[int],
) -> dict[Any, Any]:
    """Normalize one mapping, REFUSING any key collision the rewrite creates.

    A plain dict comprehension over :func:`_json_boundary_key` silently drops a
    value here: ``{float("nan"): 1, "NaN": 2}`` collapses to ``{"NaN": 2}``, and two
    distinct ``float("nan")`` keys collide the same way since ``nan != nan`` keeps
    both in the source mapping. Both are refused -- writing a record that is not the
    one the runner produced is the failure this boundary exists to end, and ``json``
    rejected the mixed-key mapping outright before F-062 anyway, so overwriting
    would also be a regression.
    """
    out: dict[Any, Any] = {}
    for key, item in mapping.items():
        boundary_key = _json_boundary_key(key)
        if boundary_key in out:
            raise ValueError(
                f"a {CHECKPOINT_SELECTION_ARTIFACT} record has two mapping keys "
                f"that both encode to {boundary_key!r}, so writing it would "
                "silently drop one of their values. No sidecar was written and "
                "this origin was not completed. Re-running recomputes it."
            )
        out[boundary_key] = _json_boundary_value(item, _active=active)
    return out


def _cycle_free(container: Any, active: frozenset[int]) -> frozenset[int]:
    """Extend the current path, refusing a container that already sits on it."""
    if id(container) in active:
        raise ValueError(
            f"a {CHECKPOINT_SELECTION_ARTIFACT} record contains a circular "
            f"reference through a {type(container).__name__}; RFC 8259 JSON has "
            "no way to express one, so no sidecar was written and this origin "
            "was not completed. Re-running recomputes it."
        )
    return active | {id(container)}


def _json_boundary_value(
    value: Any,
    *,
    _active: frozenset[int] = frozenset(),
) -> Any:
    """Normalize one selection-record value for strict (``allow_nan=False``) JSON.

    Non-finite numeric leaves become ``None``/JSON ``null`` at any depth, which is
    the convention the output and reporting writers already apply to the same
    problem. This helper is deliberately LOCAL and narrow rather than a reuse of
    ``output.core._json_ready``: ``output`` imports ``forecasting``, so importing
    it back here would invert that dependency.

    It changes as little as it can: a leaf it does not recognise is returned
    UNTOUCHED, for ``_json_default`` to render as it always did. Rendering is
    therefore preserved for the flat records the runner ships and for the cases
    pinned by test -- numpy scalars, single-element arrays, the ``str()`` spelling
    of a multi-element array, the ISO spelling of a ``pd.Timestamp``. It is NOT
    preserved universally, and one divergence is deliberate: ``json`` treats only a
    real ``dict`` as an object, so a nested non-``dict`` ``Mapping`` used to fall
    through to ``_json_default``'s ``str()``, whereas the traversal below rebuilds
    it as a JSON object (and neutralizes the non-finite leaves inside it, instead of
    freezing them in a repr). Traversing mappings is the hardening; the changed
    rendering is its cost, not an accident.

    A cycle RAISES, via :func:`_cycle_free` -- JSON cannot express a back-reference,
    and the alternatives are recursing until ``RecursionError`` or quietly writing a
    record that is not the one the runner produced. ``_active`` holds the ids on the
    CURRENT PATH only, so the same object appearing twice as a sibling is encoded
    twice, which is not a cycle.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        # ``np.float64`` subclasses ``float``, so it is caught here as well.
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, pd.Timestamp):
        # The spelling ``_json_default`` already gives it, so this is not a new
        # rendering; it just avoids a needless trip through the ``default`` hook.
        return value.isoformat()
    if isinstance(value, Mapping):
        return _json_boundary_mapping(value, _cycle_free(value, _active))
    if isinstance(value, (list, tuple)):
        active = _cycle_free(value, _active)
        return [_json_boundary_value(item, _active=active) for item in value]
    unboxed = _unboxed_numeric(value)
    if unboxed is not _NOT_UNBOXED:
        return _json_boundary_value(unboxed, _active=_active)
    return value


def _encoded_selection_line(record: Mapping[str, Any]) -> str:
    """One sidecar line, encoded strictly -- or an exception and no line at all.

    There is deliberately NO "write something rather than nothing" fallback: an
    unencodable record raises and no line is produced. The module docstring carries
    the full argument for why raising is safe here and coercing would not be.
    """
    return json.dumps(
        _json_boundary_value(record),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )


def append_origin_selection_records(
    checkpoint_path: str | Path,
    origin_pos: Any,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically persist one origin's optional selection-history sidecar.

    The sidecar is newline-delimited JSON next to ``origin_<pos>.parquet``. It is
    intentionally independent of the lean forecast parquet schema, so enabling
    selection history does not perturb checkpoint/rescore forecast loading.

    Every line is RFC 8259 JSON (F-062). A non-finite numeric leaf is normalized
    to ``null`` before encoding rather than written as one of the bare
    ``NaN``/``Infinity`` tokens ``json.dumps`` emits by default, because those are
    ECMAScript literals a strict third-party parser rejects outright.
    :func:`load_selection_history_frame` still READS those tokens, so sidecars
    written before this stay loadable; see :data:`_SELECTION_JSON_DECODER`.

    Records from the shipped runner are FLAT and string-keyed, so the reachable
    non-finite leaf is one of a record's own scalar fields; the recursion through
    nested mappings, lists and tuples is hardening against caller-supplied records,
    not a claim that a nested one has been seen on disk. Note that traversing
    mappings does change one rendering: a non-``dict`` ``Mapping`` becomes a JSON
    object where it used to stringify (see :func:`_json_boundary_value`).

    This FAILS CLOSED. A record the strict encoder refuses -- mixed-type keys, a key
    collision, a cycle, an object no ``default`` hook renders -- is not coerced into
    one it accepts: the temporary file is removed, any previous final sidecar is
    left byte-for-byte as it was, and the exception propagates, so a run with
    ``selection_history=True`` raises where nothing raised before. The module
    docstring explains why that is the safe choice given the runner's call order.
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
                handle.write(_encoded_selection_line(record) + "\n")
        Path(tmp_name).replace(final_path)
    except BaseException:
        # Cleanup ONLY, and immediately re-raised. The temp file is created INSIDE
        # the checkpoint directory, and the pre-F-062 ``except OSError`` skipped
        # cleanup for every non-OSError, so such a failure left a stray dotfile
        # behind for good. That leak does not corrupt resume accounting -- the
        # name is dot-prefixed, so it matches neither the ``origin_*.parquet`` nor
        # the ``origin_*_selection.jsonl`` scan -- but nothing ever removes it
        # either, so it accumulates in a directory the user is told is ours.
        # Nothing is suppressed: a KeyboardInterrupt or a cancellation still
        # propagates untouched, and the previous final file is never replaced.
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _json_type_name(value: Any) -> str:
    """The JSON spelling of a decoded value's type, for an actionable message."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


#: The one sentence every corruption message shares. Recovery does NOT follow
#: from it: the two artifacts recover differently, so each has its own tail.
_NO_MUTATION = "The checkpoint directory was not modified, renamed, or removed."


def _origin_recovery() -> str:
    """Recovery tail for a damaged ``origin_<pos>.parquet``: a re-run heals it.

    :func:`completed_origin_positions` does not count an unreadable origin file
    as a completed origin, so re-running the same configuration against the same
    ``checkpoint_path`` recomputes that origin and overwrites the file in place.
    """
    return (
        f"{_NO_MUTATION} Recover by re-running the same configuration against the "
        "same checkpoint_path: an unreadable origin file does not count as a "
        "completed origin, so that origin is recomputed and this file is "
        "overwritten in place. Moving or removing this file first and then "
        "re-running works too, as does pointing checkpoint_path at a fresh "
        "directory to recompute from scratch."
    )


def _sidecar_recovery(path: Path) -> str:
    """Recovery tail for a damaged sidecar, which a plain re-run does NOT heal.

    The ``origin_<pos>.parquet`` beside a sidecar is read on its own, so a healthy
    parquet keeps its origin in :func:`completed_origin_positions` no matter what
    state the sidecar is in. The runner then skips that origin, and a skipped
    origin writes no sidecar, so the damaged bytes survive the re-run untouched.
    Both artifacts have to go before the origin is recomputed and the sidecar
    written again.
    """
    match = _ORIGIN_SELECTION_FILE_RE.match(path.name)
    partner = "" if match is None else f" {_origin_filename(int(match.group(1)))!r}"
    return (
        f"{_NO_MUTATION} A plain re-run does NOT heal this file while the "
        "origin_<pos>.parquet beside it still reads: that origin still counts as "
        "completed, so the run skips it, and a skipped origin writes no sidecar. "
        "To reconstruct this history, move or remove BOTH this sidecar and its "
        f"matching origin file{partner}, then re-run the same configuration with "
        "selection history enabled; pointing checkpoint_path at a fresh directory "
        "works as well. Moving only the sidecar leaves this origin's selection "
        "history ABSENT rather than reconstructed, so it is a repair only if those "
        "rows are not wanted."
    )


def _origin_corruption_message(path: Path) -> str:
    return (
        f"{CHECKPOINT_ORIGIN_ARTIFACT} {str(path)!r} could not be read, so "
        "load_checkpoint_frame() returned nothing rather than a partial frame. "
        "Skipping it would return a forecast table silently short by one origin, "
        "and every metric computed from that table would score as though the "
        f"origin had never run. {_origin_recovery()} The reader's own "
        "error is chained as this exception's __cause__."
    )


def _sidecar_unreadable_message(path: Path) -> str:
    return (
        f"{CHECKPOINT_SELECTION_ARTIFACT} {str(path)!r} could not be read, so "
        "load_selection_history_frame() returned nothing rather than a partial "
        f"frame. {_sidecar_recovery(path)} The reader's own error is chained "
        "as this exception's __cause__."
    )


def _sidecar_invalid_json_message(path: Path, line: int) -> str:
    return (
        f"{CHECKPOINT_SELECTION_ARTIFACT} {str(path)!r} is not valid JSON on line "
        f"{line}, so load_selection_history_frame() returned nothing rather than a "
        "partial frame. A sidecar is accepted whole or not at all, so the records "
        f"before line {line} were discarded too: keeping them would undercount a "
        f"selection frequency without saying so. {_sidecar_recovery(path)} The "
        "decoder's own error is chained as this exception's __cause__."
    )


def _sidecar_non_object_message(path: Path, line: int, value: Any) -> str:
    return (
        f"{CHECKPOINT_SELECTION_ARTIFACT} {str(path)!r} decoded to a JSON "
        f"{_json_type_name(value)} rather than an object on line {line}, so "
        "load_selection_history_frame() returned nothing rather than a partial "
        "frame. Every nonblank sidecar line must be a JSON object holding one "
        "selection record; a bare scalar or array carries no field names and "
        f"would become a nameless row. {_sidecar_recovery(path)}"
    )


def _legacy_json_constant(name: str) -> None:
    """Normalize a legacy bare ``NaN``/``Infinity``/``-Infinity`` to missing.

    LOAD-BEARING cross-version compatibility, not a nicety. Every sidecar written
    before F-062 holds these bare tokens, so dropping this hook would turn every
    pre-F-062 checkpoint into a :class:`CheckpointCorruptionError` -- a silent wipe
    for a defect that only ever lived in the WRITER -- and it cannot be removed
    without reconsidering :data:`CHECKPOINT_IDENTITY_VERSION`.

    ``None`` rather than the float the token spells, so a legacy bare ``NaN`` and
    a post-F-062 ``null`` reach the frame as the same missing value.
    """
    return None


#: One reusable decoder, module-level so a sidecar with many lines does not pay
#: to rebuild it per line. ``parse_constant`` fires ONLY for the three non-finite
#: tokens: ``null``/``true``/``false`` have not routed through it since Python
#: 3.1, so ordinary values are untouched by this leniency.
_SELECTION_JSON_DECODER = json.JSONDecoder(parse_constant=_legacy_json_constant)


def _read_selection_sidecar(path: Path) -> list[dict[str, Any]]:
    """Parse one sidecar into a LOCAL buffer, accepted only if every line is.

    The buffer is the whole point (F-061). Appending each decoded line straight
    into the shared result let a sidecar truncated mid-write contribute its
    surviving prefix, silently undercounting a selection frequency.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise CheckpointCorruptionError(
            _sidecar_unreadable_message(path),
            path=path,
            artifact=CHECKPOINT_SELECTION_ARTIFACT,
        ) from exc
    buffered: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            decoded = _SELECTION_JSON_DECODER.decode(text)
        except json.JSONDecodeError as exc:
            raise CheckpointCorruptionError(
                _sidecar_invalid_json_message(path, number),
                path=path,
                artifact=CHECKPOINT_SELECTION_ARTIFACT,
                line=number,
            ) from exc
        if not isinstance(decoded, Mapping):
            # Deliberately unchained: the decode SUCCEEDED, so there is no
            # underlying error to point at. This reader rejected a well-formed
            # value, and __cause__ stays None to say exactly that.
            raise CheckpointCorruptionError(
                _sidecar_non_object_message(path, number, decoded),
                path=path,
                artifact=CHECKPOINT_SELECTION_ARTIFACT,
                line=number,
            )
        buffered.append(dict(decoded))
    return buffered


def load_selection_history_frame(checkpoint_path: str | Path) -> pd.DataFrame:
    """Load optional selection-history JSONL sidecars from one checkpoint dir.

    Fails closed on a corrupt sidecar with :class:`CheckpointCorruptionError`,
    naming the file and the offending line, rather than returning the rows that
    happened to decode. Each sidecar is all-or-nothing, and the first offending
    one in sorted filename order stops the load; see the module docstring's
    "Reading a checkpoint". A sidecar with no ``origin_<pos>.parquet`` beside it
    is an orphan and is skipped BEFORE it is opened, exactly as before, so an
    orphan that is also corrupt cannot fail an otherwise healthy load. Blank
    lines are ignored, also as before.

    The recovery this artifact has is NOT the one a damaged origin parquet has,
    and the message says so: a healthy ``origin_<pos>.parquet`` keeps its origin
    completed however damaged the sidecar is, so a plain re-run skips that origin
    and rewrites no sidecar. Reconstructing the history takes moving or removing
    BOTH files and re-running with selection history enabled, or a fresh
    directory; removing the sidecar alone loses that origin's history instead.
    """
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
        # Orphan test first, and deliberately before the sidecar is opened.
        if not forecast_path.exists():
            continue
        records.extend(_read_selection_sidecar(path))
    if not records:
        return empty
    frame = pd.DataFrame.from_records(records)
    for column in SELECTION_HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def _read_origin_frames(
    directory: Path,
    *,
    strict: bool,
) -> tuple[list[pd.DataFrame], list[Path]]:
    """Read every final origin file, in sorted filename order.

    ``strict`` is the F-060 policy switch, and the only difference between the
    public loader and the runner's own merge read. When strict, the first
    unreadable file raises and nothing is returned; otherwise it is collected
    into the second element so the caller can report what it excluded.
    """
    frames: list[pd.DataFrame] = []
    skipped: list[Path] = []
    for path in sorted(directory.glob("origin_*.parquet")):
        if _ORIGIN_FILE_RE.match(path.name) is None:
            continue
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            if strict:
                raise CheckpointCorruptionError(
                    _origin_corruption_message(path),
                    path=path,
                    artifact=CHECKPOINT_ORIGIN_ARTIFACT,
                ) from exc
            skipped.append(path)
    return frames, skipped


def _assemble_checkpoint_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union the per-origin frames and restore the quantile mapping column."""
    if not frames:
        return pd.DataFrame(columns=list(LEAN_FORECAST_COLUMNS))
    result = pd.concat(frames, ignore_index=True)
    has_wide_quantiles = any(_is_wide_quantile_column(str(c)) for c in result.columns)
    if has_wide_quantiles and "quantile_predictions" not in result.columns:
        result["quantile_predictions"] = [
            _quantile_dict_from_wide(row) for row in result.to_dict(orient="records")
        ]
    return result


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

    Fails closed: an ``origin_<pos>.parquet`` that cannot be read raises
    :class:`CheckpointCorruptionError` naming that file, rather than returning
    the origins that did read. Nothing can be recomputed on this path, so a
    shortened frame would be silent partial data -- see the module docstring's
    "Reading a checkpoint". The frame is never shortened, the directory is never
    modified, and the file reported is the first offending one in sorted filename
    order. The message names the recovery this artifact actually has: re-running
    the same configuration against the same ``checkpoint_path`` recomputes the
    damaged origin and overwrites the file in place.
    """
    directory = Path(checkpoint_path)
    if not directory.exists():
        return pd.DataFrame(columns=list(LEAN_FORECAST_COLUMNS))
    frames, _ = _read_origin_frames(directory, strict=True)
    return _assemble_checkpoint_frame(frames)


def _load_checkpoint_frame_tolerant(
    checkpoint_path: str | Path,
) -> tuple[pd.DataFrame, list[Path]]:
    """``load_checkpoint_frame`` for the ONE caller that must stay completable.

    ``runner._merge_checkpoint_records`` assembles the frame a finished run
    returns. Raising there would discard a completed run's entire result over a
    single file, so this reader skips what it cannot read and reports it, and the
    runner turns that report into one warning.

    What this reader CANNOT establish is that a skipped file lies outside the run's
    origin set. Origins the run computed are already in its in-memory records, and
    origins it recomputed have already overwritten their own files, so those are
    safe. An origin a PREVIOUS run completed is contributed by this read alone, and
    the resume gate read the directory earlier: a file that was intact then and is
    damaged now is silently missing from the merged frame. A skip may therefore
    mean a stray file from a wider earlier run, or it may mean the returned frame
    is short a previously completed origin, and nothing available here tells the
    two apart. The runner's warning states both and names the paths. Public reads
    stay fail-closed through :func:`load_checkpoint_frame`.
    """
    directory = Path(checkpoint_path)
    if not directory.exists():
        return pd.DataFrame(columns=list(LEAN_FORECAST_COLUMNS)), []
    frames, skipped = _read_origin_frames(directory, strict=False)
    return _assemble_checkpoint_frame(frames), skipped


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)
