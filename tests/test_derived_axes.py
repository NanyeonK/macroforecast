"""Tests for axis_type='derived' support via derived_axes recipe section."""

from __future__ import annotations

import copy

import pytest

from macrocast.compiler.build import (
    DERIVATION_RULES,
    _resolve_derived_axes,
    _rule_experiment_unit_default,
    compile_recipe_dict,
)
from macrocast.compiler.errors import CompileValidationError
from macrocast.registry import AxisSelection


def _base_recipe() -> dict:
    """Full minimal recipe compatible with compile_recipe_dict."""
    return {
        "recipe_id": "derived-test",
        "path": {
            "0_meta": {"fixed_axes": {}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
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
                "framework": "rolling",
                "benchmark_family": "zero_change",
                "feature_builder": "raw_feature_panel",
                "model_family": "ridge",
            }},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"manifest_mode": "full", "benchmark_config": {"minimum_train_size": 5, "rolling_window_size": 5}}},
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }


def test_experiment_unit_default_rule_is_registered() -> None:
    assert "experiment_unit_default" in DERIVATION_RULES


def test_derived_axes_experiment_unit_compiles_with_consistent_value() -> None:
    recipe = _base_recipe()
    recipe["path"]["0_meta"]["derived_axes"] = {"experiment_unit": "experiment_unit_default"}
    result = compile_recipe_dict(recipe)
    assert result.compiled.execution_status == "executable"


def test_derived_axes_conflicts_with_fixed_axes_same_axis() -> None:
    recipe = _base_recipe()
    recipe["path"]["0_meta"]["fixed_axes"]["experiment_unit"] = "single_target_single_generator"
    recipe["path"]["0_meta"]["derived_axes"] = {"experiment_unit": "experiment_unit_default"}
    with pytest.raises(CompileValidationError, match="declared as derived but also appears"):
        compile_recipe_dict(recipe)


def test_derived_axes_unknown_rule_raises() -> None:
    recipe = _base_recipe()
    recipe["path"]["0_meta"]["derived_axes"] = {"experiment_unit": "totally_made_up"}
    with pytest.raises(CompileValidationError, match="unknown derivation rule"):
        compile_recipe_dict(recipe)


def test_derived_axes_unknown_axis_raises() -> None:
    recipe = _base_recipe()
    recipe["path"]["0_meta"]["derived_axes"] = {"not_a_real_axis": "experiment_unit_default"}
    with pytest.raises(CompileValidationError, match="unknown axis"):
        compile_recipe_dict(recipe)


def test_derived_axes_must_be_mapping() -> None:
    recipe = _base_recipe()
    # Call _resolve_derived_axes directly with list form to verify its validation error
    recipe["path"]["0_meta"]["derived_axes"] = ["experiment_unit", "experiment_unit_default"]
    with pytest.raises(CompileValidationError, match="derived_axes must be a mapping"):
        _resolve_derived_axes(recipe, selection_map={}, leaf_config={})


def test_derived_experiment_unit_rule_returns_model_grid_when_sweep() -> None:
    selection_map = {
        "model_family": AxisSelection(
            axis_name="model_family", layer="3_training", selection_mode="sweep",
            selected_values=("ols", "ridge"),
            selected_status={"ols": "operational", "ridge": "operational"},
        ),
        "feature_builder": AxisSelection(
            axis_name="feature_builder", layer="2_preprocessing", selection_mode="fixed",
            selected_values=("target_lag_features",),
            selected_status={"target_lag_features": "operational"},
        ),
    }
    leaf_config = {"target_structure": "single_target"}
    result = _rule_experiment_unit_default(selection_map=selection_map, leaf_config=leaf_config)
    assert result == "single_target_generator_grid"


def test_derived_experiment_unit_rule_returns_model_grid_when_feature_sweep() -> None:
    selection_map = {
        "model_family": AxisSelection(
            axis_name="model_family", layer="3_training", selection_mode="fixed",
            selected_values=("ridge",),
            selected_status={"ridge": "operational"},
        ),
        "feature_builder": AxisSelection(
            axis_name="feature_builder", layer="2_preprocessing", selection_mode="sweep",
            selected_values=("target_lag_features", "raw_feature_panel"),
            selected_status={"target_lag_features": "operational", "raw_feature_panel": "operational"},
        ),
    }
    leaf_config = {"target_structure": "single_target"}
    result = _rule_experiment_unit_default(selection_map=selection_map, leaf_config=leaf_config)
    assert result == "single_target_generator_grid"
