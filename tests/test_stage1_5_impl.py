"""End-to-end tests for 1.5 implementation (9 values flipped operational).

Covers:
- contemporaneous_x_rule.allow_same_period_predictors vs. forbid_same_period_predictors (default).
- release_lag_rule.series_specific_lag via leaf_config.release_lag_per_series.
- missing_availability.keep_available_rows / impute_predictors_only (+ guards).
- structural_break_segmentation 2 presets (pre_post_crisis / pre_post_covid); user_break_dates was dropped as a duplicate of deterministic_components.break_dummies.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from macrocast.compiler.build import compile_recipe_dict


def _recipe(**data_task_axes) -> dict:
    axes_1 = {
        "dataset": "fred_md",
        "info_set": "final_revised_data",
        "task": "single_target",
    }
    leaf_extras = data_task_axes.pop("_leaf", {})
    for k, v in data_task_axes.items():
        axes_1[k] = v

    return {
        "recipe_id": "s15-impl-test",
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
                "feature_builder": "target_lag_features",
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


# ---------- contemporaneous_x_rule ----------

def test_contemporaneous_x_forbid_default_compiles() -> None:
    r = compile_recipe_dict(_recipe())
    assert r.compiled.execution_status == "executable"
    assert r.manifest["layer2_representation_spec"]["input_panel"]["contemporaneous_x_rule"] == "forbid_same_period_predictors"
    assert "contemporaneous_x_rule" not in r.manifest["data_task_spec"]


def test_contemporaneous_x_allow_compiles() -> None:
    r = compile_recipe_dict(_recipe(contemporaneous_x_rule="allow_same_period_predictors"))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["layer2_representation_spec"]["input_panel"]["contemporaneous_x_rule"] == "allow_same_period_predictors"
    assert "contemporaneous_x_rule" not in r.manifest["data_task_spec"]


# ---------- release_lag_rule ----------

def test_release_lag_series_specific_compiles_with_dict() -> None:
    r = compile_recipe_dict(_recipe(
        release_lag_rule="series_specific_lag",
        _leaf={"release_lag_per_series": {"INDPRO": 1, "RPI": 2}},
    ))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["release_lag_per_series"] == {"INDPRO": 1, "RPI": 2}


def test_release_lag_fixed_lag_all_series_executes() -> None:
    r = compile_recipe_dict(_recipe(release_lag_rule="fixed_lag_all_series"))
    assert r.compiled.execution_status == "executable"


# ---------- missing_availability ----------

def test_missing_availability_zero_fill_default_compiles() -> None:
    r = compile_recipe_dict(_recipe())
    assert r.manifest["data_task_spec"]["missing_availability"] == "zero_fill_leading_predictor_gaps"
    assert r.manifest["data_task_spec"]["raw_missing_policy"] == "preserve_raw_missing"
    assert r.manifest["data_task_spec"]["raw_outlier_policy"] == "preserve_raw_outliers"


def test_missing_availability_keep_available_rows_compiles() -> None:
    r = compile_recipe_dict(_recipe(missing_availability="keep_available_rows"))
    assert r.compiled.execution_status == "executable"


def test_missing_availability_impute_predictors_only_compiles_with_strategy() -> None:
    r = compile_recipe_dict(_recipe(
        missing_availability="impute_predictors_only",
        _leaf={"x_imputation": "ffill"},
    ))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["x_imputation"] == "ffill"


def test_raw_missing_policy_impute_raw_predictors_compiles_with_strategy() -> None:
    r = compile_recipe_dict(_recipe(
        raw_missing_policy="impute_raw_predictors",
        _leaf={"raw_x_imputation": "ffill"},
    ))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["raw_x_imputation"] == "ffill"


def test_raw_outlier_policy_compiles_with_optional_column_subset() -> None:
    r = compile_recipe_dict(_recipe(
        raw_outlier_policy="winsorize_raw",
        _leaf={"raw_outlier_columns": ["INDPRO"]},
    ))
    assert r.compiled.execution_status == "executable"
    assert r.manifest["data_task_spec"]["raw_outlier_columns"] == ["INDPRO"]


# ---------- structural_break_segmentation ----------

@pytest.mark.parametrize("value", ["pre_post_crisis", "pre_post_covid"])
def test_structural_break_presets_compile(value: str) -> None:
    r = compile_recipe_dict(_recipe(structural_break_segmentation=value))
    assert r.compiled.execution_status == "executable"
    block = r.manifest["layer2_representation_spec"]["feature_blocks"]["deterministic_feature_block"]
    assert block["structural_break_segmentation"] == value
    assert "structural_break_segmentation" not in r.manifest["data_task_spec"]




def test_structural_break_presets_resolve_to_expected_dates() -> None:
    # Unit test on the resolver helper — no end-to-end execution needed.
    from macrocast.execution.build import _resolve_structural_break_dates
    assert _resolve_structural_break_dates({"structural_break_segmentation": "none"}) is None
    assert _resolve_structural_break_dates({"structural_break_segmentation": "pre_post_crisis"}) == ["2008-09-01"]
    assert _resolve_structural_break_dates({"structural_break_segmentation": "pre_post_covid"}) == ["2020-03-01"]
