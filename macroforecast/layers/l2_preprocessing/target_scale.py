from __future__ import annotations

from typing import Any

from .types import PreprocessContract


TARGET_SCALE_CONTRACT_VERSION = "target_scale_contract_v1"
_ORIGINAL_SCALE_ALIASES = {"raw_level", "original_scale"}


def build_target_scale_contract(
    contract: PreprocessContract,
    *,
    target_transformer: str = "none",
) -> dict[str, Any]:
    """Describe model, forecast, and evaluation target scales.

    The executable path fits target-normalization state inside each training
    window and writes model/transformed/original scale forecasts to prediction
    artifacts. Multi-step inverse for difference-style target transforms is
    still guarded at execution time because it requires an unobserved raw
    level between the origin and target date.
    """

    transform = str(getattr(contract, "target_transform", "level"))
    normalization = str(getattr(contract, "target_normalization", "none"))
    inverse_policy = str(getattr(contract, "inverse_transform_policy", "none"))
    evaluation_scale = str(getattr(contract, "evaluation_scale", "raw_level"))
    transformer = str(target_transformer or "none")

    model_scale = (
        "custom_transformer_scale"
        if transformer != "none"
        else "transformed_target_scale"
        if transform != "level" or normalization != "none"
        else "original_target_scale"
    )
    forecast_scale = (
        "original_target_scale"
        if transformer != "none" or inverse_policy in {"target_only", "forecast_scale_only"}
        else model_scale
    )
    blockers: list[str] = []
    if normalization not in {"none", "zscore_train_only", "robust_zscore", "minmax", "unit_variance"}:
        blockers.append(f"target_normalization={normalization!r} has no built-in runtime")
    if inverse_policy not in {"none", "target_only", "forecast_scale_only"}:
        blockers.append(f"inverse_transform_policy={inverse_policy!r} requires a custom inverse runtime")
    if evaluation_scale not in _ORIGINAL_SCALE_ALIASES | {"transformed_scale", "both"}:
        blockers.append(f"evaluation_scale={evaluation_scale!r} has no built-in metric artifact runtime")
    runtime_status = "operational" if not blockers else "contract_defined_gated"

    return {
        "schema_version": TARGET_SCALE_CONTRACT_VERSION,
        "target_transform": transform,
        "target_normalization": normalization,
        "target_transformer": transformer,
        "model_target_scale": model_scale,
        "forecast_scale": forecast_scale,
        "evaluation_scale": evaluation_scale,
        "inverse_transform_policy": inverse_policy,
        "normalization_fit_scope": (
            str(getattr(contract, "preprocess_fit_scope", "not_applicable"))
            if normalization != "none"
            else "not_applicable"
        ),
        "runtime_status": runtime_status,
        "blockers": blockers,
        "leakage_rule": "target-side fitted statistics must be estimated on the active training window only",
    }
