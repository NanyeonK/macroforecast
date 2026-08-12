"""Selection-history loading and aggregation for checkpointed pipeline runs.

Both public entry points read the per-origin ``origin_<pos>_selection.jsonl``
sidecars from disk on every call, including when handed a ``PipelineReport`` --
a report carries provenance, not a cached history frame. They are therefore on
the STRICT side of the checkpoint read split (see ``forecasting/checkpoint.py``):
a sidecar that cannot be read, or that holds a line which is not a JSON object,
raises ``CheckpointCorruptionError`` rather than contributing the rows that
happened to decode, because a partial sidecar undercounts a selection frequency
without saying so.
"""
from __future__ import annotations

import dataclasses as _dc
import re
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd

from macroforecast.forecasting.checkpoint import (
    SELECTION_HISTORY_COLUMNS,
    load_selection_history_frame,
)
from macroforecast.pipeline.run import _cell_checkpoint_path

_HORIZON_DIR_RE = re.compile(r"^h(-?\d+)$")


def selection_history(report_or_store: Any) -> pd.DataFrame:
    """Return tidy per-origin selection-history rows.

    ``report_or_store`` may be a ``PipelineReport`` from ``run_pipeline`` or
    ``rescore``, a checkpoint root path, or an already-loaded DataFrame. Live and
    rescored reports are resolved through their spec/checkpoint provenance so the
    returned ``target`` and ``arm`` labels use the original unsanitized names. For
    a bare checkpoint path, non-null sidecar labels are authoritative; a uniquely
    parsed directory name fills only missing labels, while an ambiguous legacy
    directory leaves missing identity unknown rather than guessing.

    Raises
    ------
    CheckpointCorruptionError
        If a sidecar belonging to the resolved checkpoint cannot be read, or holds
        a nonblank line that is not a JSON object. Each sidecar is accepted whole
        or not at all, so no partial history is returned; the exception names the
        file and the offending line. A ``ValueError`` subclass.
    """
    if isinstance(report_or_store, pd.DataFrame):
        return _normalize_history_frame(report_or_store)
    if isinstance(report_or_store, (str, PathLike)):
        return _load_history_root(Path(report_or_store))

    spec = getattr(report_or_store, "spec", None)
    provenance = getattr(report_or_store, "provenance", {}) or {}
    checkpoint_dir = provenance.get("rescored_from")
    if checkpoint_dir is None and spec is not None:
        checkpoint_dir = getattr(spec, "checkpoint_dir", None)
    if checkpoint_dir is None:
        history_provenance = provenance.get("selection_history", {})
        if isinstance(history_provenance, Mapping):
            checkpoint_dir = history_provenance.get("checkpoint_dir")
    if checkpoint_dir is None:
        root = getattr(report_or_store, "root", None)
        if root is not None:
            return _load_history_root(Path(root))
        raise ValueError(
            "selection_history requires a PipelineReport with checkpoint provenance "
            "or a checkpoint directory path"
        )
    root = Path(checkpoint_dir)
    if spec is None:
        return _load_history_root(root)
    return _load_history_spec(root, spec)


def selection_frequency_table(
    history: Any,
    *,
    by: Sequence[str] = ("arm", "horizon", "kind", "name"),
) -> pd.DataFrame:
    """Summarize selection frequencies over distinct checkpoint origins.

    Built on :func:`selection_history`, so it inherits that function's disk read
    and its refusal: a corrupt sidecar raises ``CheckpointCorruptionError``
    rather than being summarised around, which would report a frequency computed
    over fewer origins than it claims.
    """
    frame = selection_history(history)
    group_cols = [str(column) for column in by]
    missing = [column for column in group_cols if column not in frame.columns]
    if missing:
        raise ValueError(f"history is missing grouping column(s): {missing}")
    columns = [*group_cols, "n_selected", "n_origins", "frequency"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    id_cols = [
        column
        for column in ("target", "arm", "horizon", "origin_pos", "origin")
        if column in frame.columns
    ]
    if not id_cols:
        id_cols = ["__rowid__"]
        frame = frame.copy()
        frame["__rowid__"] = range(len(frame))
    denom_cols = [column for column in group_cols if column != "name"]

    selected = (
        frame.drop_duplicates(group_cols + id_cols)
        .groupby(group_cols, dropna=False)
        .size()
        .rename("n_selected")
        .reset_index()
    )
    if denom_cols:
        denominators = (
            frame.drop_duplicates(denom_cols + id_cols)
            .groupby(denom_cols, dropna=False)
            .size()
            .rename("n_origins")
            .reset_index()
        )
        result = selected.merge(denominators, on=denom_cols, how="left")
    else:
        result = selected.copy()
        result["n_origins"] = int(frame.drop_duplicates(id_cols).shape[0])
    result["frequency"] = result["n_selected"] / result["n_origins"].replace(0, pd.NA)
    return result.loc[:, columns].sort_values(group_cols).reset_index(drop=True)


def _load_history_spec(root: Path, spec: Any) -> pd.DataFrame:
    probe_spec = _dc.replace(spec, checkpoint_dir=str(root))
    frames: list[pd.DataFrame] = []
    for target in getattr(spec, "targets", ()):
        for arm in getattr(spec, "arms", ()):
            cell_dir = _cell_checkpoint_path(probe_spec, arm, target)
            if cell_dir is None:
                continue
            for horizon in getattr(spec, "horizons", ()):
                frame = load_selection_history_frame(cell_dir / f"h{int(horizon)}")
                if frame.empty:
                    continue
                frame = frame.copy()
                frame["target"] = getattr(target, "name", frame.get("target"))
                frame["arm"] = getattr(arm, "name", frame.get("arm"))
                frame["horizon"] = int(horizon)
                frames.append(frame)
    return _concat_history(frames)


def _load_history_root(root: Path) -> pd.DataFrame:
    """Load a spec-less checkpoint tree without overriding sidecar identity.

    Identity precedence is sidecar non-null value, then a uniquely parsed path,
    then ``pd.NA``. The last case is intentional for ambiguous legacy paths.
    """

    frames: list[pd.DataFrame] = []
    if not root.exists():
        return _normalize_history_frame(pd.DataFrame())
    for cell_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        target, arm = _parse_cell_dir(cell_dir.name)
        for h_dir in sorted(path for path in cell_dir.iterdir() if path.is_dir()):
            match = _HORIZON_DIR_RE.match(h_dir.name)
            if match is None:
                continue
            frame = load_selection_history_frame(h_dir)
            if frame.empty:
                continue
            frame = frame.copy()
            for column, value in (("target", target), ("arm", arm)):
                if value is None:
                    continue
                if column not in frame.columns:
                    frame[column] = value
                    continue
                present = frame[column].notna()
                frame[column] = frame[column].astype("object").where(present, value)
            frame["horizon"] = int(match.group(1))
            frames.append(frame)
    return _concat_history(frames)


def _parse_cell_dir(name: str) -> tuple[str | None, str | None]:
    """Parse an unambiguous ``<target>__<arm>`` checkpoint directory.

    Because underscores are valid inside both sanitized components, overlapping
    or repeated separators are genuinely ambiguous. Return ``(None, None)`` in
    that case so sidecar metadata can supply identity without a guessed split.
    """

    separator = name.find("__")
    if separator < 0:
        return None, name
    if name.find("__", separator + 1) >= 0:
        return None, None
    return name[:separator], name[separator + 2 :]


def _concat_history(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return _normalize_history_frame(pd.DataFrame())
    return _normalize_history_frame(pd.concat(frames, ignore_index=True))


def _normalize_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in SELECTION_HISTORY_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    ordered = [*SELECTION_HISTORY_COLUMNS, *[c for c in result.columns if c not in SELECTION_HISTORY_COLUMNS]]
    return result.loc[:, ordered]
