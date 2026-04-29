from __future__ import annotations

from ..registry.stage0.study_scope import get_study_scope_entry
from ..registry.naming import canonical_axis_value
from .types import ReplicationInput, VaryingDesign


def derive_design_shape(
    varying_design: VaryingDesign,
    study_scope: str | None = None,
) -> str:
    if study_scope is not None:
        study_scope = canonical_axis_value("study_scope", study_scope)
        entry = get_study_scope_entry(study_scope)
        if entry.compares_methods:
            return "one_fixed_env_controlled_axis_variation"

    n_models = len(varying_design.model_families)
    n_control_axes = sum(
        len(tuple(values)) > 1
        for values in (
            varying_design.feature_recipes,
            varying_design.preprocess_variants,
            varying_design.tuning_variants,
        )
    )

    if n_control_axes > 0:
        return "one_fixed_env_controlled_axis_variation"
    if n_models <= 1:
        return "one_fixed_env_one_tool_surface"
    return "one_fixed_env_multi_tool_surface"


def derive_execution_posture(
    design_shape: str,
    replication_input: ReplicationInput | None,
    study_scope: str | None = None,
) -> str:
    if study_scope is not None:
        study_scope = canonical_axis_value("study_scope", study_scope)
        entry = get_study_scope_entry(study_scope)
        return "comparison_sweep_plan" if entry.compares_methods else "comparison_cell"

    if design_shape == "one_fixed_env_controlled_axis_variation":
        return "comparison_sweep_plan"
    return "comparison_cell"


def derive_study_scope(
    execution_posture: str,
    forecast_task: str = "single_target",
) -> str | None:
    compares_methods = execution_posture == "comparison_sweep_plan"
    if forecast_task == "multi_target":
        return "multiple_targets_compare_methods" if compares_methods else "multiple_targets_one_method"
    return "one_target_compare_methods" if compares_methods else "one_target_one_method"
