from __future__ import annotations

from .errors import DesignValidationError
from .types import ComparisonContract, FixedDesign, ReplicationInput, DesignFrame, VaryingDesign


def _ensure_nonempty(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DesignValidationError(f"{field_name} must be a non-empty string")


def validate_fixed_design(fixed_design: FixedDesign) -> None:
    _ensure_nonempty(fixed_design.dataset_adapter, "fixed_design.dataset_adapter")
    _ensure_nonempty(fixed_design.information_set, "fixed_design.information_set")
    _ensure_nonempty(fixed_design.sample_split, "fixed_design.sample_split")
    _ensure_nonempty(fixed_design.benchmark, "fixed_design.benchmark")
    _ensure_nonempty(fixed_design.evaluation_protocol, "fixed_design.evaluation_protocol")
    _ensure_nonempty(fixed_design.forecast_task, "fixed_design.forecast_task")


def validate_varying_design(varying_design: VaryingDesign) -> None:
    for field_name in (
        "model_families",
        "feature_recipes",
        "preprocess_variants",
        "tuning_variants",
        "horizons",
    ):
        value = getattr(varying_design, field_name)
        if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
            raise DesignValidationError(f"varying_design.{field_name} must be a tuple of non-empty strings")


def validate_comparison_contract(contract: ComparisonContract) -> None:
    _ensure_nonempty(contract.information_set_policy, "comparison_contract.information_set_policy")
    _ensure_nonempty(contract.sample_split_policy, "comparison_contract.sample_split_policy")
    _ensure_nonempty(contract.benchmark_policy, "comparison_contract.benchmark_policy")
    _ensure_nonempty(contract.evaluation_policy, "comparison_contract.evaluation_policy")


def validate_replication_input(replication_input: ReplicationInput | None) -> None:
    if replication_input is None:
        return
    _ensure_nonempty(replication_input.source_type, "replication_input.source_type")
    _ensure_nonempty(replication_input.source_id, "replication_input.source_id")
    if not isinstance(replication_input.locked_constraints, tuple):
        raise DesignValidationError("replication_input.locked_constraints must be a tuple")


def validate_stage0_frame(stage0: DesignFrame) -> None:
    validate_fixed_design(stage0.fixed_design)
    validate_varying_design(stage0.varying_design)
    validate_comparison_contract(stage0.comparison_contract)
    validate_replication_input(stage0.replication_input)
    if stage0.experiment_unit is not None:
        _ensure_nonempty(stage0.experiment_unit, "experiment_unit")
