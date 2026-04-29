from __future__ import annotations

from ..registry.stage0.study_scope import get_study_scope_entry
from ..registry.naming import canonical_axis_value
from .derive import derive_design_shape, derive_execution_posture, derive_study_scope
from .errors import DesignCompletenessError, DesignRoutingError, DesignValidationError
from .normalize import (
    normalize_comparison_contract,
    normalize_fixed_design,
    normalize_replication_input,
    normalize_varying_design,
)
from .types import ComparisonContract, FixedDesign, ReplicationInput, DesignFrame, VaryingDesign
from .validate import validate_stage0_frame


def build_design_frame(
    *,
    fixed_design: FixedDesign | dict,
    comparison_contract: ComparisonContract | dict,
    varying_design: VaryingDesign | dict | None = None,
    replication_input: ReplicationInput | dict | None = None,
    study_scope: str | None = None,
) -> DesignFrame:
    normalized_fixed_design = normalize_fixed_design(fixed_design)
    normalized_comparison_contract = normalize_comparison_contract(comparison_contract)
    normalized_varying_design = normalize_varying_design(varying_design)
    normalized_replication_input = normalize_replication_input(replication_input)

    resolved_study_scope = (
        canonical_axis_value("study_scope", str(study_scope))
        if study_scope is not None
        else None
    )
    if resolved_study_scope is None:
        provisional_design_shape = derive_design_shape(
            normalized_varying_design,
        )
        provisional_execution_posture = derive_execution_posture(
            provisional_design_shape,
            normalized_replication_input,
        )
        resolved_study_scope = derive_study_scope(
            provisional_execution_posture,
            normalized_fixed_design.forecast_task,
        )
    else:
        get_study_scope_entry(resolved_study_scope)

    if resolved_study_scope is None:
        raise DesignValidationError("study_scope could not be derived")

    unit_entry = get_study_scope_entry(resolved_study_scope)
    if unit_entry.requires_multi_target and normalized_fixed_design.forecast_task != "multi_target":
        raise DesignValidationError(
            f"study_scope={resolved_study_scope!r} requires forecast_task='multi_target'"
        )
    if not unit_entry.requires_multi_target and normalized_fixed_design.forecast_task == "multi_target":
        raise DesignValidationError(
            f"study_scope={resolved_study_scope!r} is incompatible with forecast_task=\"multi_target\""
        )

    design_shape = derive_design_shape(
        normalized_varying_design,
        resolved_study_scope,
    )
    execution_posture = derive_execution_posture(
        design_shape,
        normalized_replication_input,
        resolved_study_scope,
    )

    stage0 = DesignFrame(
        fixed_design=normalized_fixed_design,
        comparison_contract=normalized_comparison_contract,
        varying_design=normalized_varying_design,
        execution_posture=execution_posture,
        design_shape=design_shape,
        replication_input=normalized_replication_input,
        study_scope=resolved_study_scope,
    )
    validate_stage0_frame(stage0)
    return stage0


def resolve_route_owner(stage0: DesignFrame) -> str:
    if stage0.study_scope is not None:
        return get_study_scope_entry(stage0.study_scope).route_owner
    if stage0.execution_posture in {"comparison_cell", "comparison_sweep_plan"}:
        return "comparison_sweep"
    raise DesignRoutingError(f"unknown execution_posture={stage0.execution_posture!r}")


def check_design_completeness(stage0: DesignFrame) -> None:
    if stage0.execution_posture in {"comparison_cell", "comparison_sweep_plan"}:
        if not stage0.varying_design.model_families:
            raise DesignCompletenessError(
                "stage0 requires at least one model family for comparison execution"
            )


def design_summary(stage0: DesignFrame) -> str:
    models = ", ".join(stage0.varying_design.model_families) or "none"
    horizons = ", ".join(stage0.varying_design.horizons) or "none"
    return (
        f"study_scope={stage0.study_scope}; "
        f"dataset={stage0.fixed_design.dataset_adapter}; "
        f"route={resolve_route_owner(stage0)}; "
        f"execution_posture={stage0.execution_posture}; "
        f"design_shape={stage0.design_shape}; "
        f"models=[{models}]; horizons=[{horizons}]"
    )
