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

Format
------
``checkpoint_path`` is treated as a DIRECTORY. Each completed origin writes one
parquet file ``origin_<pos>.parquet`` containing only that origin's lean records
(scalar columns only -- no dict/struct columns). One-file-per-origin makes each
write atomic at the file level (a crash mid-write corrupts at most one origin's
file, which is simply ignored on resume), and resume is the trivial act of
listing the directory.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# The lean schema: keys that identify a forecast cell plus the sufficient
# statistic (prediction, actual, model). Every per-model record builder in
# ``runner.py`` produces these keys, so the lean projection is path-agnostic.
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
)

_ORIGIN_FILE_RE = re.compile(r"^origin_(-?\d+)\.parquet$")


def _origin_filename(origin_pos: Any) -> str:
    return f"origin_{int(origin_pos)}.parquet"


def lean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project rich runner records onto the lean (scalar-only) schema.

    Missing keys are filled with ``None`` so the parquet schema is stable across
    origins and execution paths.
    """
    out: list[dict[str, Any]] = []
    for record in records:
        out.append({column: record.get(column) for column in LEAN_FORECAST_COLUMNS})
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
    """
    directory = Path(checkpoint_path)
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(lean_records(records), columns=LEAN_FORECAST_COLUMNS)
    final_path = directory / _origin_filename(origin_pos)
    tmp_path = directory / f".{_origin_filename(origin_pos)}.tmp"
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(final_path)


def load_checkpoint_frame(checkpoint_path: str | Path) -> pd.DataFrame:
    """Load all persisted lean records as a single frame (empty if none/missing)."""
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
    return pd.concat(frames, ignore_index=True)
