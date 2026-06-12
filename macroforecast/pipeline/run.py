"""Pipeline execution: run arms into the master forecast frame (Stage 1+)."""
from __future__ import annotations

import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from macroforecast.pipeline.spec import Arm, PipelineSpec, ResolvedTarget, _arm_model_names


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


def _contender_label(arm: Arm, model_value: Any) -> str:
    """arm name for a single-model arm, else ``arm:model``."""
    if len(_arm_model_names(arm)) <= 1:
        return arm.name
    return f"{arm.name}:{model_value}"


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
    if "model" in frame.columns:
        frame["contender"] = [_contender_label(arm, m) for m in frame["model"]]
    else:
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


def _parallel_unit_worker(
    args: "tuple[PipelineSpec, int, int, int]",
) -> "tuple[int, int, int, pd.DataFrame]":
    """Module-level worker: run ONE (arm, target, horizon) cell in a subprocess.

    Receives integer indices into ``spec.arms`` / ``spec.targets`` and the horizon
    rather than the (unpicklable-by-identity) objects themselves, so the unit is
    fully reconstructed from the pickled spec. Caps nested BLAS/OpenMP threads to
    one so that a pool of ``n_jobs`` processes does not oversubscribe the cores.
    Returns ``(arm_idx, target_idx, horizon, frame)`` so the parent reassembles the
    master frame in a deterministic order independent of completion order.
    """
    # Avoid nested thread oversubscription: each worker process runs single-threaded
    # BLAS so n_jobs processes map cleanly onto n_jobs cores.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    spec, arm_idx, target_idx, horizon = args
    arm = spec.arms[arm_idx]
    target = spec.targets[target_idx]
    # No shared cache across processes: each unit recomputes its own preprocessing.
    frame = _run_one_arm_target(spec, arm, target, preprocessing_cache=None,
                                horizons=[horizon])
    return arm_idx, target_idx, horizon, frame


def _run_arms_parallel(spec: PipelineSpec) -> pd.DataFrame:
    """Fan out (arm x target x horizon) units across a process pool.

    Numerically identical to the sequential path: every unit is deterministic in
    ``spec.seed`` and the per-cell computation does not depend on sibling cells.
    Units are reassembled in a fixed (target, arm, horizon) order so the master
    frame's row order is independent of which worker finishes first.
    """
    import warnings as _warnings
    from concurrent.futures import ProcessPoolExecutor

    # Build the work list in the SAME nesting order the sequential path visits
    # (target -> arm -> horizon) so the concatenated frame matches row-for-row.
    units: list[tuple[PipelineSpec, int, int, int]] = []
    for t_idx, _target in enumerate(spec.targets):
        for a_idx, _arm in enumerate(spec.arms):
            for h in spec.horizons:
                units.append((spec, a_idx, t_idx, int(h)))

    if not units:
        return pd.DataFrame()

    max_workers = min(spec.n_jobs, len(units))
    # Collect results keyed by their deterministic (target, arm, horizon) position.
    results: dict[tuple[int, int, int], pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for arm_idx, target_idx, horizon, frame in executor.map(
            _parallel_unit_worker, units
        ):
            results[(target_idx, arm_idx, horizon)] = frame

    frames: list[pd.DataFrame] = []
    horizons = [int(h) for h in spec.horizons]
    seen_empty: set[tuple[str, str]] = set()
    for t_idx, target in enumerate(spec.targets):
        for a_idx, arm in enumerate(spec.arms):
            for h in horizons:
                frame = results.get((t_idx, a_idx, h))
                if frame is not None and not frame.empty:
                    frames.append(frame)
                elif (arm.name, target.name) not in seen_empty:
                    # Warn once per (arm, target) -- matches the sequential path's
                    # single warning even though units are split by horizon here.
                    seen_empty.add((arm.name, target.name))
                    _warnings.warn(
                        _empty_arm_warning(arm.name, target.name),
                        RuntimeWarning,
                        stacklevel=2,
                    )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_arms(spec: PipelineSpec) -> pd.DataFrame:
    """Execute every (arm, target) and concatenate into one master forecast frame.

    Columns include arm, model, contender, target, horizon, origin, date,
    prediction, actual, target_transform, forecast_policy. Each arm is run with its
    own preprocessing/features/model, and each target with its resolved
    (forecast_policy, target_transform).

    When ``spec.n_jobs > 1`` the (arm x target x horizon) work units run across a
    process pool; the result is numerically identical to the sequential path.
    """
    if spec.n_jobs > 1:
        return _run_arms_parallel(spec)

    frames: list[pd.DataFrame] = []
    # One shared preprocessing cache per target: arms of the same target reuse the
    # per-origin FittedPreprocessor (spec-level preprocessing only -- arm overrides
    # opt out by getting their own cache). Removes the per-arm EM redundancy.
    target_caches = {t.name: {} for t in spec.targets} if spec.preprocessing is not None else {}
    for target in spec.targets:
        cache = target_caches.get(target.name)
        for arm in spec.arms:
            arm_cache = cache if arm.preprocessing is None else None
            frame = _run_one_arm_target(spec, arm, target, preprocessing_cache=arm_cache)
            if not frame.empty:
                frames.append(frame)
            else:
                # An arm that yields zero forecast rows is dropped from evaluation
                # entirely, which silently hides a misconfiguration (e.g. a feature
                # block that is all-NaN over every origin so the per-origin dropna
                # empties the fit sample). Surface it rather than letting the arm
                # disappear without a trace.
                import warnings as _warnings

                _warnings.warn(
                    _empty_arm_warning(arm.name, target.name),
                    RuntimeWarning,
                    stacklevel=2,
                )
    if not frames:
        return pd.DataFrame()
    master = pd.concat(frames, ignore_index=True)
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

    master = run_arms(spec)
    results = evaluate(master, spec)
    provenance, leakage = _audit(spec)
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
    )
