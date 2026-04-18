from __future__ import annotations

from ..registry.stage0.experiment_unit import get_experiment_unit_entry
from .derive import derive_design_shape, derive_execution_posture, derive_experiment_unit
from .errors import DesignCompletenessError, DesignRoutingError, DesignValidationError
from .normalize import (
    normalize_comparison_contract,
    normalize_fixed_design,
    normalize_replication_input,
    normalize_study_mode,
    normalize_varying_design,
)
from .types import ComparisonContract, FixedDesign, ReplicationInput, DesignFrame, VaryingDesign
from .validate import validate_stage0_frame


def build_design_frame(
    *,
    study_mode: str,
    fixed_design: FixedDesign | dict,
    comparison_contract: ComparisonContract | dict,
    varying_design: VaryingDesign | dict | None = None,
    replication_input: ReplicationInput | dict | None = None,
    experiment_unit: str | None = None,
) -> DesignFrame:
    normalized_study_mode = normalize_study_mode(study_mode)
    normalized_fixed_design = normalize_fixed_design(fixed_design)
    normalized_comparison_contract = normalize_comparison_contract(comparison_contract)
    normalized_varying_design = normalize_varying_design(varying_design)
    normalized_replication_input = normalize_replication_input(replication_input)

    resolved_experiment_unit = experiment_unit
    if resolved_experiment_unit is None:
        provisional_design_shape = derive_design_shape(
            normalized_study_mode,
            normalized_varying_design,
        )
        provisional_execution_posture = derive_execution_posture(
            normalized_study_mode,
            provisional_design_shape,
            normalized_replication_input,
        )
        resolved_experiment_unit = derive_experiment_unit(
            normalized_study_mode,
            provisional_execution_posture,
            normalized_fixed_design.forecast_task,
        )
    else:
        get_experiment_unit_entry(resolved_experiment_unit)

    if resolved_experiment_unit is None:
        raise DesignValidationError("experiment_unit could not be derived")

    unit_entry = get_experiment_unit_entry(resolved_experiment_unit)
    if unit_entry.requires_multi_target and normalized_fixed_design.forecast_task != "multi_target_point_forecast":
        raise DesignValidationError(
            f"experiment_unit={resolved_experiment_unit!r} requires forecast_task='multi_target_point_forecast'"
        )
    if not unit_entry.requires_multi_target and normalized_fixed_design.forecast_task == "multi_target_point_forecast" and resolved_experiment_unit not in {"single_target_model_grid", "single_target_single_model", "single_target_full_sweep", "replication_recipe"}:
        pass

    design_shape = derive_design_shape(
        normalized_study_mode,
        normalized_varying_design,
        resolved_experiment_unit,
    )
    execution_posture = derive_execution_posture(
        normalized_study_mode,
        design_shape,
        normalized_replication_input,
        resolved_experiment_unit,
    )

    stage0 = DesignFrame(
        study_mode=normalized_study_mode,
        fixed_design=normalized_fixed_design,
        comparison_contract=normalized_comparison_contract,
        varying_design=normalized_varying_design,
        execution_posture=execution_posture,
        design_shape=design_shape,
        replication_input=normalized_replication_input,
        experiment_unit=resolved_experiment_unit,
    )
    validate_stage0_frame(stage0)
    return stage0


def resolve_route_owner(stage0: DesignFrame) -> str:
    if stage0.experiment_unit is not None:
        return get_experiment_unit_entry(stage0.experiment_unit).route_owner
    if stage0.execution_posture == "wrapper_bundle_plan":
        return "wrapper"
    if stage0.execution_posture == "replication_locked_plan":
        return "replication"
    if stage0.execution_posture in {"single_run_recipe", "single_run_with_internal_sweep"}:
        return "single_run"
    raise DesignRoutingError(f"unknown execution_posture={stage0.execution_posture!r}")


def check_design_completeness(stage0: DesignFrame) -> None:
    if stage0.execution_posture in {"single_run_recipe", "single_run_with_internal_sweep"}:
        if not stage0.varying_design.model_families:
            raise DesignCompletenessError(
                "stage0 requires at least one model family for single-run execution"
            )


def design_summary(stage0: DesignFrame) -> str:
    models = ", ".join(stage0.varying_design.model_families) or "none"
    horizons = ", ".join(stage0.varying_design.horizons) or "none"
    return (
        f"study_mode={stage0.study_mode}; "
        f"experiment_unit={stage0.experiment_unit}; "
        f"dataset={stage0.fixed_design.dataset_adapter}; "
        f"route={resolve_route_owner(stage0)}; "
        f"execution_posture={stage0.execution_posture}; "
        f"design_shape={stage0.design_shape}; "
        f"models=[{models}]; horizons=[{horizons}]"
    )
