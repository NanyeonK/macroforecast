from __future__ import annotations

from ..registry.stage0.experiment_unit import get_experiment_unit_entry
from ..registry.naming import canonical_axis_value
from .types import ComparisonContract, DesignShape, ExecutionPosture, ReplicationInput, VaryingDesign


def derive_design_shape(
    varying_design: VaryingDesign,
    experiment_unit: str | None = None,
) -> str:
    if experiment_unit is not None:
        experiment_unit = canonical_axis_value("experiment_unit", experiment_unit)
    if experiment_unit is not None:
        unit_entry = get_experiment_unit_entry(experiment_unit)
        if unit_entry.requires_wrapper or unit_entry.route_owner in {"wrapper", "orchestrator"}:
            return "wrapper_managed_multi_run_bundle"
        if experiment_unit == "single_target_generator_grid":
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
    experiment_unit: str | None = None,
) -> str:
    if experiment_unit is not None:
        experiment_unit = canonical_axis_value("experiment_unit", experiment_unit)
    if experiment_unit is not None:
        unit_entry = get_experiment_unit_entry(experiment_unit)
        if unit_entry.route_owner == "replication" or replication_input is not None:
            return "replication_locked_plan"
        if unit_entry.requires_wrapper or unit_entry.route_owner in {"wrapper", "orchestrator"}:
            return "wrapper_bundle_plan"
        if experiment_unit == "single_target_generator_grid":
            return "comparison_sweep_plan"
        return "comparison_cell"

    if replication_input is not None:
        return "replication_locked_plan"
    if design_shape == "wrapper_managed_multi_run_bundle":
        return "wrapper_bundle_plan"
    if design_shape == "one_fixed_env_controlled_axis_variation":
        return "comparison_sweep_plan"
    return "comparison_cell"


def derive_experiment_unit(
    execution_posture: str,
    forecast_task: str = "single_target",
) -> str | None:
    if execution_posture == "wrapper_bundle_plan":
        # multi_target_separate_runs is the supported wrapper fan-out path for
        # multi-target recipes. Single-target wrapper suites stay registered but
        # are not the default route.
        return "multi_target_separate_runs" if forecast_task == "multi_target" else "benchmark_suite"
    if execution_posture == "replication_locked_plan":
        return "replication_recipe"
    if forecast_task == "multi_target":
        # Default operational multi-target unit — shared_design is handled by
        # execute_recipe's multi-target path (single aggregated output). See
        # docs/user_guide/design.md 0.3.
        return "multi_target_shared_design"
    if execution_posture == "comparison_sweep_plan":
        return "single_target_generator_grid"
    return "single_target_single_generator"
