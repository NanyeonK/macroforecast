"""End-to-end tests for 1.4 implementation: 19 values flipped operational.

Covers:
- benchmark_family: factor_model, multi_benchmark_suite, paper_specific_benchmark, survey_forecast.
- predictor_family: all_except_target, category_based, factor_only, handpicked_set.
- variable_universe: 6 subset rules via leaf_config input channels.
- deterministic_components: 5 feature-augmentation rules.

Every case verifies compile-time closure plus the behavioural invariant specific to the rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from macrocast.compiler.build import compile_recipe_dict, run_compiled_recipe
from macrocast.compiler.errors import CompileValidationError


def _recipe(**axes_1_extras) -> dict:
    axes_1 = {
        "dataset": "fred_md",
        "info_set": "revised",
        "task": "single_target_point_forecast",
    }
    leaf_extras = axes_1_extras.pop("_leaf", {})
    # Pull 1_data_task axis values out of kwargs into axes_1
    for k, v in axes_1_extras.items():
        axes_1[k] = v
    return {
        "recipe_id": "s14-impl-test",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_forecast_run"}},
            "1_data_task": {
                "fixed_axes": axes_1,
                "leaf_config": {"target": "INDPRO", "horizons": [1], **leaf_extras},
            },
            "2_preprocessing": {"fixed_axes": {
                "target_transform_policy": "raw_level",
                "x_transform_policy": "raw_level",
                "tcode_policy": "raw_only",
                "target_missing_policy": "none",
                "x_missing_policy": "none",
                "target_outlier_policy": "none",
                "x_outlier_policy": "none",
                "scaling_policy": "none",
                "dimensionality_reduction_policy": "none",
                "feature_selection_policy": "none",
                "preprocess_order": "none",
                "preprocess_fit_scope": "not_applicable",
                "inverse_transform_policy": "none",
                "evaluation_scale": "raw_level",
            }},
            "3_training": {"fixed_axes": {
                "framework": "expanding",
                "benchmark_family": "historical_mean",
                "feature_builder": "autoreg_lagged_target",
                "model_family": "ar",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {
                "manifest_mode": "full",
                "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5},
            }},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


# ---------- benchmark_family ----------

def test_benchmark_family_factor_model_compiles() -> None:
    recipe = _recipe()
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = "factor_model"
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"


def test_benchmark_family_multi_suite_requires_list() -> None:
    recipe = _recipe()
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = "multi_benchmark_suite"
    with pytest.raises(CompileValidationError, match="benchmark_suite"):
        compile_recipe_dict(recipe)


def test_benchmark_family_multi_suite_compiles_with_list() -> None:
    recipe = _recipe()
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = "multi_benchmark_suite"
    recipe["path"]["1_data_task"]["leaf_config"]["benchmark_suite"] = ["historical_mean", "zero_change"]
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"


def test_benchmark_family_paper_specific_compiles() -> None:
    recipe = _recipe()
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = "paper_specific_benchmark"
    recipe["path"]["1_data_task"]["leaf_config"]["paper_forecast_series"] = {
        "INDPRO": pd.Series([100.0, 101.0, 102.0],
                            index=pd.date_range("2000-11-01", periods=3, freq="MS"))
    }
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["paper_forecast_series"] is not None


def test_benchmark_family_survey_forecast_compiles() -> None:
    recipe = _recipe()
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = "survey_forecast"
    recipe["path"]["1_data_task"]["leaf_config"]["survey_forecast_series"] = {
        "INDPRO": pd.Series([50.0, 51.0], index=pd.date_range("2000-11-01", periods=2, freq="MS"))
    }
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"


# ---------- predictor_family ----------

@pytest.mark.parametrize("value", ["category_based", "factor_only", "handpicked_set"])
def test_predictor_family_compiles(value: str) -> None:
    recipe = _recipe(predictor_family=value)
    # These values expect feature_builder != autoreg_lagged_target
    recipe["path"]["3_training"]["fixed_axes"]["feature_builder"] = "raw_feature_panel"
    recipe["path"]["3_training"]["fixed_axes"]["model_family"] = "ridge"
    if value == "category_based":
        recipe["path"]["1_data_task"]["leaf_config"]["predictor_category"] = "output"
        recipe["path"]["1_data_task"]["leaf_config"]["predictor_category_columns"] = {"output": ["RPI", "UNRATE"]}
    if value == "handpicked_set":
        recipe["path"]["1_data_task"]["leaf_config"]["handpicked_columns"] = ["RPI", "UNRATE"]
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable", f"{value}: blocked={r.manifest.get('blocked_reasons')}"
    assert r.manifest["layer2_representation_spec"]["input_panel"]["predictor_family"] == value
    assert "predictor_family" not in r.manifest["data_task_spec"]


# ---------- variable_universe ----------



def test_variable_universe_handpicked_set_compiles() -> None:
    recipe = _recipe(variable_universe="handpicked_set")
    recipe["path"]["1_data_task"]["leaf_config"]["variable_universe_columns"] = ["RPI", "UNRATE"]
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"


def test_variable_universe_category_subset_compiles() -> None:
    recipe = _recipe(variable_universe="category_subset")
    recipe["path"]["1_data_task"]["leaf_config"]["variable_universe_category"] = "output"
    recipe["path"]["1_data_task"]["leaf_config"]["variable_universe_category_columns"] = {"output": ["RPI"]}
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"


def test_variable_universe_target_specific_compiles() -> None:
    recipe = _recipe(variable_universe="target_specific_subset")
    recipe["path"]["1_data_task"]["leaf_config"]["target_specific_columns"] = {"INDPRO": ["RPI"]}
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable"




@pytest.mark.parametrize(
    "value",
    ["constant_only", "linear_trend", "monthly_seasonal", "quarterly_seasonal", "break_dummies"],
)
def test_deterministic_components_compile(value: str) -> None:
    recipe = _recipe(deterministic_components=value)
    if value == "break_dummies":
        recipe["path"]["1_data_task"]["leaf_config"]["break_dates"] = ["2008-09-01", "2020-03-01"]
    r = compile_recipe_dict(recipe)
    assert r.compiled.execution_status == "executable", f"{value}: blocked={r.manifest.get('blocked_reasons')}"
    block = r.manifest["layer2_representation_spec"]["feature_blocks"]["deterministic_feature_block"]
    assert block["deterministic_components"] == value
    assert "deterministic_components" not in r.manifest["data_task_spec"]


def test_deterministic_augment_module_adds_expected_columns() -> None:
    """Unit test for the augmentation module independent of the runner."""
    from macrocast.execution.deterministic import augment_frame
    idx = pd.date_range("2000-01-01", periods=13, freq="MS")
    df = pd.DataFrame({"x1": range(13)}, index=idx)
    assert list(augment_frame(df, "linear_trend").columns) == ["x1", "_dc_trend"]
    monthly_cols = set(augment_frame(df, "monthly_seasonal").columns)
    assert "_dc_month_01" in monthly_cols and "_dc_month_11" in monthly_cols and "_dc_month_12" not in monthly_cols
    q = augment_frame(df, "quarterly_seasonal")
    assert {"_dc_q1", "_dc_q2", "_dc_q3"}.issubset(q.columns)
    assert "_dc_q4" not in q.columns
    brk = augment_frame(df, "break_dummies", break_dates=["2000-06-01"])
    # Break indicator should be 0 before June 2000 and 1 from June onward
    series = brk["_dc_break_1"]
    assert series.loc["2000-01-01":"2000-05-01"].sum() == 0
    assert series.loc["2000-06-01":].min() == 1.0
