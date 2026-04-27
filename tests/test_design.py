from __future__ import annotations

import pytest

from macrocast.design import (
    ComparisonContract,
    FixedDesign,
    DesignFrame,
    VaryingDesign,
    build_design_frame,
    check_design_completeness,
    resolve_route_owner,
    design_summary,
)


def test_build_stage0_frame_single_forecast_run() -> None:
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design=FixedDesign(
            dataset_adapter="fred_md",
            information_set="revised_monthly",
            sample_split="expanding_window_oos",
            benchmark="ar_bic",
            evaluation_protocol="point_forecast_core",
            forecast_task="single_target",
        ),
        comparison_contract=ComparisonContract(
            information_set_policy="identical",
            sample_split_policy="identical",
            benchmark_policy="identical",
            evaluation_policy="identical",
        ),
        varying_design=VaryingDesign(model_families=("ar", "ridge"), horizons=("h1", "h3")),
    )

    assert isinstance(stage0, DesignFrame)
    assert stage0.design_shape == "one_fixed_env_multi_tool_surface"
    assert stage0.execution_posture == "single_run_recipe"
    assert resolve_route_owner(stage0) == "single_run"
    check_design_completeness(stage0)


def test_build_stage0_frame_single_fixed_model_and_feature_is_one_tool_surface() -> None:
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design=FixedDesign(
            dataset_adapter="fred_md",
            information_set="revised_monthly",
            sample_split="expanding_window_oos",
            benchmark="ar_bic",
            evaluation_protocol="point_forecast_core",
            forecast_task="single_target",
        ),
        comparison_contract=ComparisonContract(
            information_set_policy="identical",
            sample_split_policy="identical",
            benchmark_policy="identical",
            evaluation_policy="identical",
        ),
        varying_design=VaryingDesign(model_families=("ar",), feature_recipes=("target_lag_features",), horizons=("h1",)),
    )

    assert stage0.design_shape == "one_fixed_env_one_tool_surface"
    assert stage0.execution_posture == "single_run_recipe"
    assert stage0.experiment_unit == "single_target_single_generator"


def test_build_stage0_frame_multiple_feature_recipes_is_controlled_variation() -> None:
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design=FixedDesign(
            dataset_adapter="fred_md",
            information_set="revised_monthly",
            sample_split="expanding_window_oos",
            benchmark="ar_bic",
            evaluation_protocol="point_forecast_core",
            forecast_task="single_target",
        ),
        comparison_contract=ComparisonContract(
            information_set_policy="identical",
            sample_split_policy="identical",
            benchmark_policy="identical",
            evaluation_policy="identical",
        ),
        varying_design=VaryingDesign(
            model_families=("ridge",),
            feature_recipes=("target_lag_features", "raw_feature_panel"),
            horizons=("h1",),
        ),
    )

    assert stage0.design_shape == "one_fixed_env_controlled_axis_variation"
    assert stage0.execution_posture == "single_run_with_internal_sweep"


def test_build_stage0_frame_bundle_route() -> None:
    stage0 = build_design_frame(
        research_design="study_bundle",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar", "ridge", "rf")},
    )

    assert stage0.design_shape == "wrapper_managed_multi_run_bundle"
    assert stage0.execution_posture == "wrapper_bundle_plan"
    assert resolve_route_owner(stage0) == "wrapper"


def test_check_stage0_completeness_rejects_empty_model_surface() -> None:
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
    )

    with pytest.raises(Exception):
        check_design_completeness(stage0)


def test_stage0_summary_contains_core_fields() -> None:
    stage0 = build_design_frame(
        research_design="single_forecast_run",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "horizons": ("h1",)},
    )

    summary = design_summary(stage0)

    assert "single_forecast_run" in summary
    assert "fred_md" in summary
    assert "single_run_recipe" in summary



def test_build_stage0_frame_explicit_experiment_unit_controls_wrapper_route() -> None:
    stage0 = build_design_frame(
        research_design="study_bundle",
        experiment_unit="benchmark_suite",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",)},
    )

    assert stage0.experiment_unit == "benchmark_suite"
    assert stage0.execution_posture == "wrapper_bundle_plan"
    assert resolve_route_owner(stage0) == "wrapper"
