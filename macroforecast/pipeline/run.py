"""Pipeline execution: run arms into the master forecast frame (Stage 1+)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from macroforecast.pipeline.spec import Arm, PipelineSpec, ResolvedTarget, _arm_model_names


def _contender_label(arm: Arm, model_value: Any) -> str:
    """arm name for a single-model arm, else ``arm:model``."""
    if len(_arm_model_names(arm)) <= 1:
        return arm.name
    return f"{arm.name}:{model_value}"


def _run_one_arm_target(
    spec: PipelineSpec, arm: Arm, target: ResolvedTarget
) -> pd.DataFrame:
    """Run a single arm for a single resolved target across all horizons."""
    from macroforecast.forecasting import run

    result = run(
        spec.data,
        arm.model,
        window=spec.window,
        preprocessing=arm.preprocessing,
        preprocessing_policy=arm.preprocessing_policy,
        features=arm.features,
        feature_policy=arm.feature_policy,
        params=arm.params,
        model_selection=arm.model_selection,
        model_selection_metric=arm.model_selection_metric,
        target=target.name,
        horizons=list(spec.horizons),
        forecast_policy=target.policy,
        target_transform=target.transform,
        save_models=spec.save_models,
        model_store=spec.model_store,
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


def run_arms(spec: PipelineSpec) -> pd.DataFrame:
    """Execute every (arm, target) and concatenate into one master forecast frame.

    Columns include arm, model, contender, target, horizon, origin, date,
    prediction, actual, target_transform, forecast_policy. Each arm is run with its
    own preprocessing/features/model, and each target with its resolved
    (forecast_policy, target_transform).
    """
    frames: list[pd.DataFrame] = []
    for arm in spec.arms:
        for target in spec.targets:
            frame = _run_one_arm_target(spec, arm, target)
            if not frame.empty:
                frames.append(frame)
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
    try:
        index = _panel_index(spec.data)
        report = spec.window.validate(index) if (index is not None and hasattr(spec.window, "validate")) else None
    except Exception:
        report = None
    if isinstance(report, dict):
        leakage["window_warnings"] = list(report.get("warnings", []))
        leakage["window_ok"] = bool(report.get("ok", True))
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
    )
