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
expanded into WIDE ``q_<pct>`` columns (e.g. ``q_05``, ``q_50``, ``q_95`` for
levels ``0.05``/``0.5``/``0.95``) rather than added to the fixed schema: the
set of levels is a per-model hyperparameter (``quantile_levels``), not known
ahead of time, so the wide columns are derived empirically per origin from
whatever the record's ``quantile_predictions`` mapping actually carries (see
``_quantile_wide_columns``). ``load_checkpoint_frame`` unions differing column
sets across origin files via ``pd.concat`` (pandas fills the gaps with NaN), so
this needs no cross-origin coordination and degrades gracefully when an old
checkpoint (written before this column existed) sits alongside new ones.

Format
------
``checkpoint_path`` is treated as a DIRECTORY. Each completed origin writes one
parquet file ``origin_<pos>.parquet`` containing only that origin's lean records
(scalar columns only -- no dict/struct columns; quantile predictions are
pre-expanded into wide scalar ``q_<pct>`` columns, see above). One-file-per-
origin makes each write atomic at the file level (a crash mid-write corrupts at
most one origin's file, which is simply ignored on resume), and resume is the
trivial act of listing the directory.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
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
)

_ORIGIN_FILE_RE = re.compile(r"^origin_(-?\d+)\.parquet$")


def _origin_filename(origin_pos: Any) -> str:
    return f"origin_{int(origin_pos)}.parquet"


def _quantile_column_name(level: float) -> str:
    """Wide checkpoint column name for a quantile level (``0.05`` -> ``q_05``)."""
    return f"q_{round(float(level) * 100):02d}"


def _quantile_wide_columns(record: Mapping[str, Any]) -> dict[str, float]:
    """Expand a record's ``quantile_predictions`` mapping into wide, scalar
    ``q_<pct>`` columns (parquet needs scalar columns; see module docstring).
    Absent, ``None``, or non-mapping values expand to no columns at all.
    """
    value = record.get("quantile_predictions")
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, float] = {}
    for level, prediction in value.items():
        try:
            out[_quantile_column_name(float(level))] = float(prediction)
        except (TypeError, ValueError):
            continue
    return out


_WIDE_QUANTILE_COLUMN_RE = re.compile(r"^q_(\d{2})$")


def _quantile_dict_from_wide(record: Mapping[str, Any]) -> dict[str, float] | None:
    """Reconstruct a ``{level_str: value}`` mapping from a lean record's wide
    ``q_<pct>`` columns -- the inverse of :func:`_quantile_wide_columns`.

    Matches the exact string-keyed format ``forecasting/policies/direct.py``
    writes onto the rich (non-lean) forecast table (``str(level)`` for a Python
    float, e.g. ``"0.05"``/``"0.5"``/``"0.95"``), so a resumed-from-checkpoint
    origin's ``quantile_predictions`` merges back into the SAME representation
    a freshly-computed origin's does (see
    ``forecasting/runner.py::_merge_checkpoint_records``). Returns ``None`` (not
    ``{}``) when the record carries no non-null ``q_<pct>`` column, matching the
    rich table's ``None``-for-absent convention.
    """
    out: dict[str, float] = {}
    for key, value in record.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        match = _WIDE_QUANTILE_COLUMN_RE.match(str(key))
        if match is None:
            continue
        level = int(match.group(1)) / 100.0
        try:
            out[str(level)] = float(value)
        except (TypeError, ValueError):
            continue
    return out or None


def lean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project rich runner records onto the lean (scalar-only) schema.

    Missing keys in :data:`LEAN_FORECAST_COLUMNS` are filled with ``None`` so
    that fixed part of the parquet schema is stable across origins and
    execution paths. ``quantile_predictions`` (when present and non-empty) is
    additionally expanded into wide ``q_<pct>`` columns -- see
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
    these records carries quantile predictions, the wide ``q_<pct>`` columns
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


def load_checkpoint_frame(checkpoint_path: str | Path) -> pd.DataFrame:
    """Load all persisted lean records as a single frame (empty if none/missing).

    Origin files may carry different wide ``q_<pct>`` quantile columns (a
    point-only or pre-density-pipeline origin has none); ``pd.concat`` unions
    them, filling gaps with NaN, so this needs no cross-file coordination. When
    any ``q_<pct>`` column is present, a ``quantile_predictions`` mapping column
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
    has_wide_quantiles = any(_WIDE_QUANTILE_COLUMN_RE.match(str(c)) for c in result.columns)
    if has_wide_quantiles and "quantile_predictions" not in result.columns:
        result["quantile_predictions"] = [
            _quantile_dict_from_wide(row) for row in result.to_dict(orient="records")
        ]
    return result
