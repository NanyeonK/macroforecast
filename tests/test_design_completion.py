from __future__ import annotations

import pytest

from macrocast.design import (
    ComparisonContract,
    FixedDesign,
    ReplicationInput,
    DesignCompletenessError,
    DesignValidationError,
    VaryingDesign,
    build_design_frame,
    check_design_completeness,
    resolve_route_owner,
    design_from_dict,
    design_to_dict,
)


def _base_fixed() -> dict:
    return {
        "dataset_adapter": "fred_md",
        "information_set": "revised_monthly",
        "sample_split": "expanding_window_oos",
        "benchmark": "autoregressive_bic",
        "evaluation_protocol": "point_forecast_core",
        "forecast_task": "single_target",
    }


def _base_contract() -> dict:
    return {
        "information_set_policy": "identical",
        "sample_split_policy": "identical",
        "benchmark_policy": "identical",
        "evaluation_policy": "identical",
    }


def test_stage0_roundtrip_dict() -> None:
    stage0 = build_design_frame(
        fixed_design=_base_fixed(),
        comparison_contract=_base_contract(),
        varying_design={"model_families": ("ar", "ridge"), "horizons": ("h1", "h3")},
    )

    rebuilt = design_from_dict(design_to_dict(stage0))

    assert rebuilt == stage0


def test_replication_route_owner_and_execution_posture() -> None:
    stage0 = build_design_frame(
        experiment_unit="replication_recipe",
        fixed_design=_base_fixed(),
        comparison_contract=_base_contract(),
        varying_design={"model_families": ("ar",)},
        replication_input=ReplicationInput(source_type="paper_recipe", source_id="clss2021"),
    )

    assert stage0.execution_posture == "replication_locked_plan"
    assert resolve_route_owner(stage0) == "replication"
    assert stage0.experiment_unit == "replication_recipe"


def test_invalid_experiment_unit_raises_validation_error() -> None:
    with pytest.raises(Exception):
        build_design_frame(
            experiment_unit="unknown_unit",
            fixed_design=_base_fixed(),
            comparison_contract=_base_contract(),
            varying_design={"model_families": ("ar",)},
        )


def test_blank_fixed_design_field_raises_validation_error() -> None:
    fixed = _base_fixed()
    fixed["benchmark"] = ""

    with pytest.raises(DesignValidationError):
        build_design_frame(
            fixed_design=fixed,
            comparison_contract=_base_contract(),
            varying_design={"model_families": ("ar",)},
        )


def test_comparison_cell_without_model_families_raises_completeness_error() -> None:
    stage0 = build_design_frame(
        fixed_design=_base_fixed(),
        comparison_contract=_base_contract(),
        varying_design=VaryingDesign(),
    )

    with pytest.raises(DesignCompletenessError):
        check_design_completeness(stage0)


def test_wrapper_experiment_unit_route_owner_is_wrapper() -> None:
    stage0 = build_design_frame(
        experiment_unit="benchmark_suite",
        fixed_design=_base_fixed(),
        comparison_contract=_base_contract(),
        varying_design={"model_families": ("ar",)},
    )

    assert stage0.execution_posture == "wrapper_bundle_plan"
    assert resolve_route_owner(stage0) == "wrapper"
    assert stage0.experiment_unit == "benchmark_suite"
