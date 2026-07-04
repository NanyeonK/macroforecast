"""Re-score saved forecasts from a pipeline checkpoint directory (Stage 8a).

``evaluate()`` is pure-frame (``pipeline/evaluate.py``): it needs a master forecast
frame plus the originating ``PipelineSpec``, not a fitted model. And
``forecasting/checkpoint.py::load_checkpoint_frame`` already reconstructs one
cell's lean forecast frame from its per-origin parquet files. What is missing is
the glue: a pipeline run spans MANY cells (one ``<target>__<arm>/h<h>/`` directory
per (target, arm, horizon)), so re-scoring a full run requires hand-walking that
directory tree and re-assembling a master frame before ``evaluate()`` can run.
``rescore()`` is that glue -- the "~3 lines" convenience the docstring promises.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from macroforecast.forecasting.checkpoint import load_checkpoint_frame
from macroforecast.pipeline.run import _cell_checkpoint_path


def rescore(checkpoint_dir: str | Path, spec: "Any") -> "Any":
    """Re-score a saved pipeline run from its checkpoint directory alone.

    Walks every (target, arm, horizon) cell ``spec`` describes, loads that cell's
    persisted lean forecast records from
    ``<checkpoint_dir>/<target>__<arm>/h<h>/origin_*.parquet`` (the exact layout
    ``run_pipeline(spec)`` writes when ``spec.checkpoint_dir`` is set -- see
    ``pipeline/run.py::_cell_checkpoint_path`` and ``forecasting/runner.py``'s
    per-horizon ``h<h>`` subdirectory), reassembles the master forecast frame
    (attaching ``arm``/``contender`` from ``spec`` itself -- the lean checkpoint
    schema does not carry them, only ``target``/``horizon``/``model``/etc.), and
    runs the standard ``evaluate()`` used by ``run_pipeline``.

    Parameters
    ----------
    checkpoint_dir:
        The directory ``spec.checkpoint_dir`` pointed at during the original run
        (or any directory with that same layout).
    spec:
        The ``PipelineSpec`` the checkpointed run used -- NOT necessarily the same
        object with ``checkpoint_dir`` set to this path; ``rescore`` reads records
        from ``checkpoint_dir`` regardless of what ``spec.checkpoint_dir`` says.
        Every field that determines a cell's identity (targets, arms, horizons)
        must match the original run, or cells will not be found.

    Returns
    -------
    PipelineReport
        The same report type ``run_pipeline`` returns, with the evaluation fields
        (``forecasts``, ``accuracy``, ``significance``, ``mcs``) populated exactly
        as a live run would produce from the same forecasts. Fields that require
        having actually EXECUTED the run are explicitly absent/best-effort:

        - ``interpretation`` is always ``None`` (interpretation needs the fitted
          model; re-fit via ``interpret_pipeline`` on a live run instead).
        - ``failed_cells`` is always empty -- a cell that failed during the
          original run wrote no checkpoint files and is indistinguishable here
          from a cell that never ran.
        - ``empty_cells`` is best-effort: a (target, horizon) is reported empty
          only when NONE of its arms produced any checkpoint rows; an arm that
          failed outright (vs. produced zero rows) cannot be distinguished from
          one that was simply never run with this checkpoint_dir.
        - ``provenance``/``leakage_audit`` carry a ``rescored_from`` marker and a
          note that they were not recomputed from a live run.

    Raises
    ------
    ValueError
        If no cell under ``checkpoint_dir`` yields any checkpoint records at all
        (an empty or entirely-mismatched directory) -- a clear, actionable error
        instead of a silently-empty report.
    """
    from macroforecast.pipeline.evaluate import evaluate
    from macroforecast.pipeline.spec import PipelineReport

    checkpoint_root = Path(checkpoint_dir)
    probe_spec = _with_checkpoint_dir(spec, checkpoint_root)

    frames: list[pd.DataFrame] = []
    empty_cells: list[dict[str, Any]] = []
    any_cell_dir_found = False
    for target in spec.targets:
        for arm in spec.arms:
            cell_dir = _cell_checkpoint_path(probe_spec, arm, target)
            arm_produced_any_horizon = False
            for h in spec.horizons:
                h_dir = cell_dir / f"h{int(h)}"
                if h_dir.exists():
                    any_cell_dir_found = True
                frame = load_checkpoint_frame(h_dir)
                if frame.empty:
                    continue
                arm_produced_any_horizon = True
                frame = frame.copy()
                frame["arm"] = arm.name
                frame["contender"] = arm.name
                if "target" not in frame.columns or frame["target"].isna().all():
                    frame["target"] = target.name
                frames.append(frame)
            if not arm_produced_any_horizon:
                empty_cells.append({"target": target.name, "arm": arm.name})

    if not frames:
        if not any_cell_dir_found:
            raise ValueError(
                f"rescore: no checkpoint directories found under {checkpoint_root!r} "
                "for any (target, arm, horizon) cell this spec describes -- check "
                "that checkpoint_dir points at the directory passed as "
                "spec.checkpoint_dir during the original run_pipeline(...) call, "
                "and that spec's targets/arms/horizons match that run."
            )
        raise ValueError(
            f"rescore: checkpoint directories exist under {checkpoint_root!r} but "
            "every one is empty (a partial/interrupted run?) -- nothing to re-score."
        )

    master = pd.concat(frames, ignore_index=True)
    results = evaluate(master, spec)

    provenance = {
        **dict(spec.provenance),
        "rescored_from": str(checkpoint_root),
        "rescore_note": (
            "This report was reassembled from saved checkpoints, not a live run: "
            "interpretation is unavailable, failed_cells could not be recovered "
            "(indistinguishable from never-run), and empty_cells is best-effort "
            "(an arm with zero checkpoint rows for every horizon)."
        ),
    }
    leakage_audit = {
        "rescored_from": str(checkpoint_root),
        "empty_cells": list(empty_cells),
    }

    return PipelineReport(
        forecasts=results["forecasts"],
        accuracy=results["accuracy"],
        significance=results["significance"],
        mcs=results["mcs"],
        provenance=provenance,
        leakage_audit=leakage_audit,
        interpretation=None,
        model_store=spec.model_store,
        spec=spec,
        failed_cells=(),
        empty_cells=tuple(empty_cells),
    )


def _with_checkpoint_dir(spec: "Any", checkpoint_dir: Path) -> "Any":
    """A shallow copy of *spec* pointing ``checkpoint_dir`` at *checkpoint_dir*.

    ``_cell_checkpoint_path`` reads ``spec.checkpoint_dir`` to build each cell's
    directory; ``rescore`` accepts a possibly-different directory than whatever
    ``spec.checkpoint_dir`` says (the caller may be re-scoring a copy, or a spec
    built without checkpointing enabled at all), so this substitutes it in a copy
    rather than requiring the caller to mutate their spec.
    """
    import dataclasses as _dc

    return _dc.replace(spec, checkpoint_dir=str(checkpoint_dir))


__all__ = ["rescore"]
