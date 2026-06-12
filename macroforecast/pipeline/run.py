"""Pipeline execution: run arms into the master forecast frame (Stage 1+)."""
from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd

from macroforecast.pipeline.spec import Arm, PipelineSpec, ResolvedTarget


def _safe_segment(value: str) -> str:
    """Make a target/arm name safe to use as a single filesystem path segment."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("_") or "x"


def _cell_checkpoint_path(spec: PipelineSpec, arm: Arm, target: ResolvedTarget):
    """Per-(target, arm) checkpoint directory, or None when disabled.

    ``run()`` appends an ``h<h>`` subdirectory for each horizon when the spec
    carries more than one horizon, so the final layout is
    ``<checkpoint_dir>/<target>__<arm>/h<h>/origin_<pos>.parquet``.
    """
    if spec.checkpoint_dir is None:
        return None
    cell = f"{_safe_segment(target.name)}__{_safe_segment(arm.name)}"
    return Path(spec.checkpoint_dir) / cell


def _run_one_arm_target(
    spec: PipelineSpec,
    arm: Arm,
    target: ResolvedTarget,
    preprocessing_cache=None,
    horizons: "Sequence[int] | None" = None,
) -> pd.DataFrame:
    """Run a single arm for a single resolved target.

    By default every horizon in ``spec.horizons`` is computed in one consolidated
    multi-horizon call (sharing one ``preprocessing_cache`` across horizons). When
    ``horizons`` is given, only those horizons are computed -- the parallel path
    passes a single horizon per work unit so independent processes each compute
    just their cell.
    """
    import dataclasses as _dc

    from macroforecast.forecasting import run

    run_horizons = list(spec.horizons if horizons is None else horizons)

    # A multi-target pipeline runs each arm for every target, but an arm's feature
    # spec carries a single target; align it (and its transform) to this target.
    features = arm.features
    if features is not None:
        needs_retarget = (
            getattr(features, "target", None) != target.name
            or bool(getattr(features, "targets", ()))
        )
        if needs_retarget:
            kwargs: dict[str, Any] = {"target": target.name}
            if getattr(features, "targets", None):
                kwargs["targets"] = ()
            try:
                features = _dc.replace(features, **kwargs)
            except Exception as exc:  # surface a real misconfiguration
                raise ValueError(
                    f"could not re-target feature spec of arm {arm.name!r} "
                    f"to {target.name!r}: {exc}"
                ) from exc

    result = run(
        spec.data,
        arm.model,
        window=spec.window,
        preprocessing=arm.preprocessing if arm.preprocessing is not None else spec.preprocessing,
        preprocessing_policy=(arm.preprocessing_policy if arm.preprocessing is not None
                              else spec.preprocessing_policy),
        features=features,
        feature_policy=arm.feature_policy,
        params=arm.params,
        model_selection=arm.model_selection,
        model_selection_metric=arm.model_selection_metric,
        target=target.name,
        horizons=run_horizons,
        forecast_policy=target.policy,
        target_transform=target.transform,
        save_models=spec.save_models,
        model_store=spec.model_store,
        preprocessing_cache=preprocessing_cache,
        checkpoint_path=_cell_checkpoint_path(spec, arm, target),
    )
    frame = result.to_frame().copy()
    frame["arm"] = arm.name
    # A contender IS exactly an arm (one model per arm), so the contender label is
    # always the arm name regardless of the underlying model row.
    frame["contender"] = arm.name
    # ensure the target column carries the resolved target name
    if "target" not in frame.columns:
        frame["target"] = target.name
    return frame


def _empty_arm_warning(arm_name: str, target_name: str) -> str:
    """The diagnostic emitted when an arm yields zero forecast rows for a target."""
    return (
        f"pipeline arm {arm_name!r} produced ZERO forecast rows for "
        f"target {target_name!r}; it will be absent from the evaluation. "
        "Check the arm's feature spec / preprocessing (an all-NaN feature "
        "block over every origin empties the fit sample)."
    )


# --------------------------------------------------------------------------- #
# Unified cell run-manager
# --------------------------------------------------------------------------- #
# The pipeline MANAGES atomic ``run()`` calls. The managed unit is a "cell":
# ``(target, arm, horizon-group)``. Both backends enumerate the SAME cells and
# execute them through one structure; they differ only in how horizons are grouped
# and whether a preprocessing cache is shared.
#
#   * serial (n_jobs == 1): one cell per ``(target, arm)`` covering ALL horizons in
#     a single multi-horizon ``run()`` so the per-origin preprocessing cache / EM
#     state is shared across horizons (the byte-for-byte prior behavior).
#   * parallel (n_jobs > 1): one cell per ``(target, arm, horizon)`` -- a single
#     horizon per ``run()`` -- so independent processes each compute just their
#     cell. No shared cache across processes (each recomputes its own EM); the
#     forecasts are numerically identical to the serial path.
#
# Per cell: respect the per-(target, arm, horizon) checkpoint dirs; if a cell
# raises, record the failure (cell identity + error) and CONTINUE the rest of the
# set rather than aborting. Failures are surfaced on ``PipelineReport.failed_cells``
# (and mirrored into ``leakage_audit``).


class _Cell(NamedTuple):
    """One managed work unit: a (target, arm, horizon-group) forecast cell."""

    target_idx: int
    arm_idx: int
    horizons: tuple[int, ...]


def _enumerate_cells(spec: PipelineSpec) -> list[_Cell]:
    """Enumerate cells in the deterministic (target -> arm -> horizon) visit order.

    Serial groups all horizons per (target, arm); parallel splits one horizon per
    cell. The reassembly order is the enumeration order, so the master frame's row
    order is identical regardless of backend or worker completion order.
    """
    horizons = tuple(int(h) for h in spec.horizons)
    cells: list[_Cell] = []
    for t_idx, _target in enumerate(spec.targets):
        for a_idx, _arm in enumerate(spec.arms):
            if spec.n_jobs > 1:
                cells.extend(_Cell(t_idx, a_idx, (h,)) for h in horizons)
            else:
                cells.append(_Cell(t_idx, a_idx, horizons))
    return cells


def _execute_cell(
    spec: PipelineSpec, cell: _Cell, *, preprocessing_cache=None
) -> pd.DataFrame:
    """Run ONE cell as a single (multi- or single-horizon) ``run()`` call."""
    arm = spec.arms[cell.arm_idx]
    target = spec.targets[cell.target_idx]
    return _run_one_arm_target(
        spec, arm, target,
        preprocessing_cache=preprocessing_cache,
        horizons=list(cell.horizons),
    )


def _parallel_cell_worker(
    args: "tuple[PipelineSpec, _Cell]",
) -> "tuple[_Cell, pd.DataFrame | None, str | None]":
    """Module-level worker: run ONE cell in a subprocess, returning any error text.

    Caps nested BLAS/OpenMP threads to one so a pool of ``n_jobs`` processes does
    not oversubscribe the cores. Returns ``(cell, frame, error)`` where exactly one
    of ``frame``/``error`` is set, so the parent isolates per-cell failures.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
    spec, cell = args
    try:
        # No shared cache across processes: each cell recomputes its own preprocessing.
        return cell, _execute_cell(spec, cell, preprocessing_cache=None), None
    except Exception as exc:  # isolate the failure; the parent records it
        return cell, None, f"{type(exc).__name__}: {exc}"


def _cell_failure(spec: PipelineSpec, cell: _Cell, error: str) -> dict[str, Any]:
    """A structured record of one failed cell (identity + error)."""
    arm = spec.arms[cell.arm_idx]
    target = spec.targets[cell.target_idx]
    return {
        "target": target.name,
        "arm": arm.name,
        "horizons": list(cell.horizons),
        "error": error,
    }


def _run_cells(spec: PipelineSpec) -> "tuple[pd.DataFrame, list[dict[str, Any]]]":
    """Execute every cell, isolating per-cell failures.

    Returns ``(master, failed_cells)``. The two backends share this one
    enumerate-then-execute structure; the master frame and ``failed_cells`` are
    assembled in the deterministic cell order independent of execution backend.
    """
    import warnings as _warnings

    cells = _enumerate_cells(spec)
    if not cells:
        return pd.DataFrame(), []

    results: dict[_Cell, pd.DataFrame] = {}
    failed: list[dict[str, Any]] = []

    if spec.n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor

        max_workers = min(spec.n_jobs, len(cells))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for cell, frame, error in executor.map(
                _parallel_cell_worker, [(spec, c) for c in cells]
            ):
                if error is not None:
                    failed.append(_cell_failure(spec, cell, error))
                elif frame is not None:
                    results[cell] = frame
    else:
        # One shared preprocessing cache per target: arms of the same target reuse
        # the per-origin FittedPreprocessor (spec-level preprocessing only -- arm
        # overrides opt out by getting their own cache). Removes per-arm EM redundancy.
        target_caches = (
            {t.name: {} for t in spec.targets} if spec.preprocessing is not None else {}
        )
        for cell in cells:
            arm = spec.arms[cell.arm_idx]
            target = spec.targets[cell.target_idx]
            cache = target_caches.get(target.name)
            arm_cache = cache if arm.preprocessing is None else None
            try:
                results[cell] = _execute_cell(spec, cell, preprocessing_cache=arm_cache)
            except Exception as exc:
                # Record the failure and CONTINUE the rest of the set rather than
                # aborting the whole pipeline on one bad cell.
                failed.append(
                    _cell_failure(spec, cell, f"{type(exc).__name__}: {exc}")
                )

    # Reassemble in enumeration order; warn once per (arm, target) that yielded no
    # rows (an empty arm hides a misconfiguration). A failed cell does NOT also warn
    # as empty -- the failure is already recorded on failed_cells.
    frames: list[pd.DataFrame] = []
    seen_empty: set[tuple[str, str]] = set()
    failed_keys = {(f["arm"], f["target"]) for f in failed}
    for cell in cells:
        arm = spec.arms[cell.arm_idx]
        target = spec.targets[cell.target_idx]
        frame = results.get(cell)
        if frame is not None and not frame.empty:
            frames.append(frame)
            continue
        key = (arm.name, target.name)
        if key in failed_keys or key in seen_empty:
            continue
        seen_empty.add(key)
        _warnings.warn(
            _empty_arm_warning(arm.name, target.name),
            RuntimeWarning,
            stacklevel=2,
        )

    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return master, failed


def run_arms(spec: PipelineSpec) -> pd.DataFrame:
    """Execute every (arm, target) and concatenate into one master forecast frame.

    Columns include arm, model, contender, target, horizon, origin, date,
    prediction, actual, target_transform, forecast_policy. Each arm is run with its
    own preprocessing/features/model, and each target with its resolved
    (forecast_policy, target_transform).

    The pipeline MANAGES atomic ``run()`` calls over (target, arm, horizon-group)
    cells. When ``spec.n_jobs > 1`` the cells run across a process pool, one horizon
    per cell; the result is numerically identical to the serial multi-horizon path.
    Per-cell failures are isolated -- see :func:`run_pipeline` /
    ``PipelineReport.failed_cells`` for how they are surfaced.
    """
    master, _failed = _run_cells(spec)
    return master


def _panel_index(data: Any):
    """Best-effort extraction of the panel DatetimeIndex from any data input."""
    panel = getattr(data, "panel", None)
    if panel is None and isinstance(data, tuple) and data:
        panel = data[0]
    if panel is None and isinstance(data, pd.DataFrame):
        panel = data
    return getattr(panel, "index", None)


def _audit(spec: PipelineSpec) -> tuple[dict, dict]:
    """Collect provenance and a leakage audit for the run."""
    import macroforecast as _mf

    provenance: dict = dict(spec.provenance)
    provenance.update({
        "package_version": getattr(_mf, "__version__", "unknown"),
        "seed": spec.seed,
        "targets": [
            {"name": t.name, "policy": t.policy, "transform": t.transform, "tcode": t.tcode}
            for t in spec.targets
        ],
        "horizons": list(spec.horizons),
        "arms": [a.name for a in spec.arms],
        "benchmark": spec.evaluation.benchmark,
        "combinations": [c.name for c in spec.combinations],
    })
    leakage: dict = {}
    # Surface window.validate() warnings (e.g. embargo < horizon-1) when available.
    # The runner injects each horizon into the window's test spec at execution
    # time, so validate the window once PER horizon (not the base test.horizon)
    # to surface the embargo / pseudo-OOS warning that only fires for horizon > 1.
    import dataclasses as _dc

    index = _panel_index(spec.data)
    warnings_seen: list[str] = []
    window_ok = True
    if index is not None and hasattr(spec.window, "validate"):
        for h in spec.horizons:
            try:
                if hasattr(spec.window, "test"):
                    test_spec = _dc.replace(spec.window.test, horizon=int(h))
                    wspec = _dc.replace(spec.window, test=test_spec)
                else:
                    wspec = spec.window
                report = wspec.validate(index)
            except Exception as exc:
                # A validation that crashes must NOT be read as leak-free; record
                # the failure and mark the window not-ok rather than staying silent.
                report = None
                window_ok = False
                msg = f"h={h}: window.validate raised {type(exc).__name__}: {exc}"
                if msg not in warnings_seen:
                    warnings_seen.append(msg)
            if isinstance(report, dict):
                window_ok = window_ok and bool(report.get("ok", True))
                for w in report.get("warnings", []):
                    tagged = w if str(h) in w else f"h={h}: {w}"
                    if tagged not in warnings_seen:
                        warnings_seen.append(tagged)
        leakage["window_warnings"] = warnings_seen
        leakage["window_ok"] = window_ok
    leakage["preprocessing_policies"] = {
        a.name: str(a.preprocessing_policy) for a in spec.arms
    }
    return provenance, leakage


def run_pipeline(spec: PipelineSpec):
    """Execute the full pipeline: run arms, evaluate, and assemble a PipelineReport."""
    from macroforecast.pipeline.evaluate import evaluate
    from macroforecast.pipeline.spec import PipelineReport

    master, failed_cells = _run_cells(spec)
    results = evaluate(master, spec)
    provenance, leakage = _audit(spec)
    # Mirror per-cell failures into the leakage audit so any consumer that reads
    # only the audit still sees that some arms failed to run.
    leakage = {**leakage, "failed_cells": list(failed_cells)}
    return PipelineReport(
        forecasts=results["forecasts"],
        accuracy=results["accuracy"],
        significance=results["significance"],
        mcs=results["mcs"],
        provenance=provenance,
        leakage_audit=leakage,
        interpretation=None,
        model_store=spec.model_store,
        spec=spec,
        failed_cells=tuple(failed_cells),
    )
