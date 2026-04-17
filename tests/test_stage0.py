from __future__ import annotations

import pytest

from macrocast.stage0 import (
    ComparisonContract,
    FixedDesign,
    Stage0Frame,
    VaryingDesign,
    build_stage0_frame,
    check_stage0_completeness,
    resolve_route_owner,
    stage0_summary,
)


def test_build_stage0_frame_single_path_benchmark_study() -> None:
    stage0 = build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design=FixedDesign(
            dataset_adapter="fred_md",
            information_set="revised_monthly",
            sample_split="expanding_window_oos",
            benchmark="ar_bic",
            evaluation_protocol="point_forecast_core",
            forecast_task="single_target_point_forecast",
        ),
        comparison_contract=ComparisonContract(
            information_set_policy="identical",
            sample_split_policy="identical",
            benchmark_policy="identical",
            evaluation_policy="identical",
        ),
        varying_design=VaryingDesign(model_families=("ar", "ridge"), horizons=("h1", "h3")),
    )

    assert isinstance(stage0, Stage0Frame)
    assert stage0.design_shape == "one_fixed_env_multi_tool_surface"
    assert stage0.execution_posture == "single_run_recipe"
    assert resolve_route_owner(stage0) == "single_run"
    check_stage0_completeness(stage0)


def test_build_stage0_frame_bundle_route() -> None:
    stage0 = build_stage0_frame(
        study_mode="orchestrated_bundle_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target_point_forecast",
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
    stage0 = build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target_point_forecast",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
    )

    with pytest.raises(Exception):
        check_stage0_completeness(stage0)


def test_stage0_summary_contains_core_fields() -> None:
    stage0 = build_stage0_frame(
        study_mode="single_path_benchmark_study",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target_point_forecast",
        },
        comparison_contract={
            "information_set_policy": "identical",
            "sample_split_policy": "identical",
            "benchmark_policy": "identical",
            "evaluation_policy": "identical",
        },
        varying_design={"model_families": ("ar",), "horizons": ("h1",)},
    )

    summary = stage0_summary(stage0)

    assert "single_path_benchmark_study" in summary
    assert "fred_md" in summary
    assert "single_run_recipe" in summary



def test_build_stage0_frame_explicit_experiment_unit_controls_wrapper_route() -> None:
    stage0 = build_stage0_frame(
        study_mode="orchestrated_bundle_study",
        experiment_unit="benchmark_suite",
        fixed_design={
            "dataset_adapter": "fred_md",
            "information_set": "revised_monthly",
            "sample_split": "expanding_window_oos",
            "benchmark": "ar_bic",
            "evaluation_protocol": "point_forecast_core",
            "forecast_task": "single_target_point_forecast",
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
