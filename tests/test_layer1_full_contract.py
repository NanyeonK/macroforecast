from __future__ import annotations

import pytest

from macrocast.compiler.build import compile_recipe_dict
from macrocast.compiler.errors import CompileValidationError


def _recipe() -> dict:
    return {
        "recipe_id": "layer1-full-contract",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "task": "single_target_point_forecast",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
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


def _drop_legacy_tcode_bridge(recipe: dict) -> dict:
    fixed_axes = recipe["path"]["2_preprocessing"]["fixed_axes"]
    for axis in (
        "target_transform_policy",
        "x_transform_policy",
        "tcode_policy",
        "preprocess_order",
        "representation_policy",
        "tcode_application_scope",
    ):
        fixed_axes.pop(axis, None)
    return recipe


def test_standalone_fred_sd_requires_explicit_frequency() -> None:
    recipe = _recipe()
    recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_sd"

    with pytest.raises(CompileValidationError, match="fred_sd.*frequency"):
        compile_recipe_dict(recipe)


def test_fred_sd_composite_rejects_wrong_frequency() -> None:
    recipe = _recipe()
    recipe["path"]["1_data_task"]["fixed_axes"].update({
        "dataset": "fred_qd+fred_sd",
        "frequency": "monthly",
    })

    with pytest.raises(CompileValidationError, match="fred_qd\\+fred_sd.*quarterly"):
        compile_recipe_dict(recipe)


@pytest.mark.parametrize(
    ("axis", "value", "message"),
    [
        ("variable_universe", "handpicked_set", "variable_universe_columns"),
        ("variable_universe", "category_subset", "variable_universe_category_columns"),
        ("variable_universe", "target_specific_subset", "target_specific_columns"),
        ("predictor_family", "handpicked_set", "handpicked_columns"),
        ("predictor_family", "category_based", "predictor_category_columns"),
        ("deterministic_components", "break_dummies", "break_dates"),
        ("missing_availability", "x_impute_only", "x_imputation"),
        ("raw_missing_policy", "x_impute_raw", "raw_x_imputation"),
        ("release_lag_rule", "series_specific_lag", "release_lag_per_series"),
    ],
)
def test_layer1_leaf_config_required_inputs_are_compile_time_contracts(
    axis: str,
    value: str,
    message: str,
) -> None:
    recipe = _recipe()
    recipe["path"]["1_data_task"]["fixed_axes"][axis] = value

    with pytest.raises(CompileValidationError, match=message):
        compile_recipe_dict(recipe)


@pytest.mark.parametrize(
    ("benchmark_family", "message"),
    [
        ("multi_benchmark_suite", "benchmark_suite"),
        ("paper_specific_benchmark", "paper_forecast_series"),
        ("survey_forecast", "survey_forecast_series"),
        ("expert_benchmark", "expert_callable"),
    ],
)
def test_layer1_benchmark_required_inputs_are_compile_time_contracts(
    benchmark_family: str,
    message: str,
) -> None:
    recipe = _recipe()
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = benchmark_family

    with pytest.raises(CompileValidationError, match=message):
        compile_recipe_dict(recipe)


def test_multi_benchmark_suite_rejects_unsupported_members() -> None:
    recipe = _recipe()
    recipe["path"]["3_training"]["fixed_axes"]["benchmark_family"] = "multi_benchmark_suite"
    recipe["path"]["1_data_task"]["leaf_config"]["benchmark_suite"] = ["historical_mean", "expert_benchmark"]

    with pytest.raises(CompileValidationError, match="unsupported members"):
        compile_recipe_dict(recipe)


def test_official_transform_defaults_are_derived_from_legacy_raw_preprocess_bridge() -> None:
    compiled = compile_recipe_dict(_recipe())

    data_task = compiled.manifest["data_task_spec"]
    assert data_task["official_transform_policy"] == "raw_official_frame"
    assert data_task["official_transform_scope"] == "apply_tcode_to_none"
    assert data_task["official_transform_source"]["policy_source"] == "legacy_layer2_tcode_bridge"
    assert data_task["official_transform_source"]["scope_source"] == "legacy_layer2_tcode_bridge"
    assert "tcode_policy" in data_task["official_transform_source"]["legacy_bridge_axes"]


def test_official_transform_axes_record_layer1_dataset_tcode_path() -> None:
    recipe = _drop_legacy_tcode_bridge(_recipe())
    recipe["path"]["1_data_task"]["fixed_axes"].update(
        {
            "official_transform_policy": "dataset_tcode",
            "official_transform_scope": "apply_tcode_to_both",
        }
    )

    compiled = compile_recipe_dict(recipe)

    data_task = compiled.manifest["data_task_spec"]
    axis_layers = compiled.compiled.tree_context["axis_layers"]
    preprocess = compiled.manifest["preprocess_contract"]
    assert data_task["official_transform_policy"] == "dataset_tcode"
    assert data_task["official_transform_scope"] == "apply_tcode_to_both"
    assert data_task["official_transform_source"] == {
        "policy_source": "layer1_axis",
        "scope_source": "layer1_axis",
        "legacy_bridge_axes": [],
    }
    assert preprocess["tcode_policy"] == "tcode_only"
    assert preprocess["target_transform_policy"] == "tcode_transformed"
    assert preprocess["x_transform_policy"] == "dataset_tcode_transformed"
    assert axis_layers["official_transform_policy"] == "1_data_task"
    assert axis_layers["official_transform_scope"] == "1_data_task"


def test_official_transform_scope_can_target_only_without_layer2_bridge() -> None:
    recipe = _drop_legacy_tcode_bridge(_recipe())
    recipe["path"]["1_data_task"]["fixed_axes"].update(
        {
            "official_transform_policy": "dataset_tcode",
            "official_transform_scope": "apply_tcode_to_target",
        }
    )

    compiled = compile_recipe_dict(recipe)

    preprocess = compiled.manifest["preprocess_contract"]
    assert preprocess["tcode_policy"] == "tcode_only"
    assert preprocess["target_transform_policy"] == "tcode_transformed"
    assert preprocess["x_transform_policy"] == "raw_level"
    assert preprocess["tcode_application_scope"] == "apply_tcode_to_target"


def test_official_transform_policy_rejects_legacy_layer2_conflict() -> None:
    recipe = _recipe()
    recipe["path"]["1_data_task"]["fixed_axes"].update(
        {
            "official_transform_policy": "dataset_tcode",
            "official_transform_scope": "apply_tcode_to_both",
        }
    )

    with pytest.raises(CompileValidationError, match="official_transform_policy conflicts"):
        compile_recipe_dict(recipe)


def test_official_transform_scope_rejects_legacy_layer2_conflict() -> None:
    recipe = _recipe()
    recipe["path"]["1_data_task"]["fixed_axes"]["official_transform_scope"] = "apply_tcode_to_target"

    with pytest.raises(CompileValidationError, match="raw_official_frame"):
        compile_recipe_dict(recipe)


def test_layer1_raw_missing_policy_records_before_tcode_contract() -> None:
    recipe = _drop_legacy_tcode_bridge(_recipe())
    recipe["path"]["1_data_task"]["fixed_axes"].update(
        {
            "official_transform_policy": "dataset_tcode",
            "official_transform_scope": "apply_tcode_to_both",
            "raw_missing_policy": "x_impute_raw",
        }
    )
    recipe["path"]["1_data_task"]["leaf_config"]["raw_x_imputation"] = "median"

    compiled = compile_recipe_dict(recipe)

    data_task = compiled.manifest["data_task_spec"]
    axis_layers = compiled.compiled.tree_context["axis_layers"]
    assert data_task["raw_missing_policy"] == "x_impute_raw"
    assert data_task["raw_x_imputation"] == "median"
    assert axis_layers["raw_missing_policy"] == "1_data_task"


def test_layer1_raw_outlier_policy_records_before_tcode_contract() -> None:
    recipe = _drop_legacy_tcode_bridge(_recipe())
    recipe["path"]["1_data_task"]["fixed_axes"].update(
        {
            "official_transform_policy": "dataset_tcode",
            "official_transform_scope": "apply_tcode_to_both",
            "raw_outlier_policy": "iqr_clip_raw",
        }
    )
    recipe["path"]["1_data_task"]["leaf_config"]["raw_outlier_columns"] = ["INDPRO"]

    compiled = compile_recipe_dict(recipe)

    data_task = compiled.manifest["data_task_spec"]
    axis_layers = compiled.compiled.tree_context["axis_layers"]
    assert data_task["raw_outlier_policy"] == "iqr_clip_raw"
    assert data_task["raw_outlier_columns"] == ["INDPRO"]
    assert axis_layers["raw_outlier_policy"] == "1_data_task"
