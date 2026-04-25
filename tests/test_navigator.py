from __future__ import annotations

from pathlib import Path

import yaml

from macrocast.navigator import (
    build_navigation_view,
    get_replication_entry,
    replication_recipe_yaml,
    write_replication_recipe,
)
from macrocast.navigator.cli import main as navigator_main


def _recipe(**training_overrides):
    training = {
        "framework": "expanding",
        "benchmark_family": "zero_change",
        "feature_builder": "raw_feature_panel",
        "model_family": "ridge",
        "forecast_type": "direct",
        "forecast_object": "point_mean",
    }
    training.update(training_overrides)
    return {
        "recipe_id": "navigator-test",
        "path": {
            "0_meta": {"fixed_axes": {"research_design": "single_path_benchmark"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "revised",
                    "target_structure": "single_target_point_forecast",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1]},
            },
            "2_preprocessing": {
                "fixed_axes": {
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
                    "target_normalization": "none",
                }
            },
            "3_training": {"fixed_axes": training},
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {"leaf_config": {"benchmark_config": {"minimum_train_size": 5}}},
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def _axis(view, axis):
    return next(item for item in view["tree"] if item["axis"] == axis)


def _option(axis_view, value):
    return next(item for item in axis_view["options"] if item["value"] == value)


def test_navigation_disables_tree_shap_for_non_tree_model():
    recipe = _recipe()
    recipe["path"]["7_importance"]["fixed_axes"]["importance_method"] = "tree_shap"
    view = build_navigation_view(recipe)

    model_axis = _axis(view, "model_family")
    assert _option(model_axis, "ridge")["enabled"] is False
    assert "tree_shap" in _option(model_axis, "ridge")["disabled_reason"]
    assert _option(model_axis, "randomforest")["enabled"] is True


def test_navigation_quantile_forecast_restricts_model_family():
    view = build_navigation_view(_recipe(forecast_object="quantile"))
    model_axis = _axis(view, "model_family")

    assert _option(model_axis, "ridge")["enabled"] is False
    assert _option(model_axis, "quantile_linear")["enabled"] is True
    assert view["compatibility"]["active_rules"][0]["rule"] == "quantile_requires_quantile_generator"


def test_navigation_shows_current_deep_sequence_gate():
    view = build_navigation_view(_recipe(model_family="lstm", feature_builder="autoreg_lagged_target"))
    feature_axis = _axis(view, "feature_builder")

    assert _option(feature_axis, "autoreg_lagged_target")["enabled"] is True
    assert _option(feature_axis, "sequence_tensor")["enabled"] is False
    assert "sequence_tensor remains gated" in _option(feature_axis, "sequence_tensor")["disabled_reason"]


def test_replication_library_writes_yaml(tmp_path: Path):
    out = write_replication_recipe("synthetic-replication-roundtrip", tmp_path / "synthetic.yaml")
    payload = yaml.safe_load(out.read_text())

    assert payload["recipe_id"] == "synthetic-replication-roundtrip-navigator"
    assert payload["path"]["0_meta"]["fixed_axes"]["research_design"] == "single_path_benchmark"
    assert payload["path"]["3_training"]["fixed_axes"]["forecast_type"] == "iterated"
    assert get_replication_entry("synthetic-replication-roundtrip")["expected_outputs"]
    assert "single_path_benchmark" in replication_recipe_yaml("synthetic-replication-roundtrip")


def test_navigator_cli_writes_replication_yaml(tmp_path: Path):
    out = tmp_path / "gc.yaml"
    rc = navigator_main(["replications", "goulet-coulombe-2021-fred-md-ridge", "--write-yaml", str(out)])

    assert rc == 0
    payload = yaml.safe_load(out.read_text())
    assert payload["path"]["3_training"]["fixed_axes"]["model_family"] == "ridge"
