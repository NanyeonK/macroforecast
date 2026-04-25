from __future__ import annotations

from pathlib import Path

import yaml

from macrocast.navigator import (
    build_navigation_view,
    get_replication_entry,
    navigator_ui_data,
    replication_recipe_yaml,
    write_navigator_ui_data,
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


def test_navigation_tree_exposes_downstream_layer_axes():
    view = build_navigation_view(_recipe())
    axes_by_layer = {}
    for item in view["tree"]:
        axes_by_layer.setdefault(item["layer"], set()).add(item["axis"])
    all_axes = {item["axis"] for item in view["tree"]}

    assert {
        "benchmark_window",
        "agg_time",
        "regime_definition",
        "oos_period",
    }.issubset(axes_by_layer["4_evaluation"])
    assert {
        "export_format",
        "saved_objects",
        "provenance_fields",
        "artifact_granularity",
    }.issubset(axes_by_layer["5_output_provenance"])
    assert {"test_scope", "overlap_handling"}.issubset(axes_by_layer["6_stat_tests"])
    assert {
        "importance_model_native",
        "importance_model_agnostic",
        "importance_partial_dependence",
        "importance_gradient_path",
    }.issubset(axes_by_layer["7_importance"])
    assert not {
        "model_native",
        "model_agnostic",
        "partial_dependence",
    }.intersection(all_axes)


def test_navigation_tree_populates_downstream_defaults():
    view = build_navigation_view(_recipe())

    assert _axis(view, "export_format")["selected"] == "json"
    assert _axis(view, "saved_objects")["selected"] == "full_bundle"
    assert _axis(view, "regime_definition")["selected"] == "none"
    assert _axis(view, "test_scope")["selected"] == "per_target"
    assert _axis(view, "overlap_handling")["selected"] == "allow_overlap"
    assert _axis(view, "importance_method")["selected"] == "none"
    assert _axis(view, "importance_scope")["selected"] == "global"
    assert _axis(view, "importance_temporal")["selected"] == "static_snapshot"
    assert _option(_axis(view, "export_format"), "parquet")["canonical_path_effect"].endswith(
        "fixed_axes.export_format = 'parquet'"
    )


def test_navigator_ui_data_tree_includes_output_layer():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))
    tree_axes = {
        item["axis"]
        for item in payload["samples"][0]["view"]["tree"]
        if item["layer"] == "5_output_provenance"
    }

    assert {"export_format", "saved_objects", "provenance_fields", "artifact_granularity"}.issubset(tree_axes)


def test_navigation_disables_tree_shap_for_non_tree_model():
    recipe = _recipe()
    recipe["path"]["7_importance"]["fixed_axes"]["importance_method"] = "tree_shap"
    view = build_navigation_view(recipe)

    model_axis = _axis(view, "model_family")
    assert _option(model_axis, "ridge")["enabled"] is False
    assert "tree_shap" in _option(model_axis, "ridge")["disabled_reason"]
    assert _option(model_axis, "randomforest")["enabled"] is True


def test_navigation_disables_split_tree_shap_for_non_tree_model():
    recipe = _recipe()
    recipe["path"]["7_importance"]["fixed_axes"]["importance_shap"] = "tree_shap"
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


def test_navigation_layer6_split_hac_compatibility():
    recipe = _recipe()
    recipe["path"]["6_stat_tests"]["fixed_axes"] = {
        "equal_predictive": "dm_hln",
        "dependence_correction": "nw_hac",
        "overlap_handling": "evaluate_with_hac",
    }
    view = build_navigation_view(recipe)

    equal_axis = _axis(view, "equal_predictive")
    correction_axis = _axis(view, "dependence_correction")
    overlap_axis = _axis(view, "overlap_handling")

    assert _option(equal_axis, "dm_hln")["enabled"] is True
    assert _option(correction_axis, "nw_hac")["enabled"] is True
    assert _option(overlap_axis, "evaluate_with_hac")["enabled"] is True
    assert _option(equal_axis, "dm")["enabled"] is False
    assert "HAC-capable" in _option(equal_axis, "dm")["disabled_reason"]


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


def test_navigator_ui_data_export_roundtrip(tmp_path: Path):
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))

    assert payload["schema_version"] == "navigator_ui_data_v1"
    assert payload["samples"][0]["view"]["schema_version"] == "navigator_view_v1"
    assert "model_family" in payload["axis_catalog"]
    assert payload["replications"]

    out = write_navigator_ui_data(tmp_path / "navigator_ui_data.json", sample_paths=("examples/recipes/model-benchmark.yaml",))
    assert out.exists()
    assert write_navigator_ui_data(out, sample_paths=("examples/recipes/model-benchmark.yaml",), check=True) == out


def test_navigator_cli_checks_ui_data(tmp_path: Path):
    out = tmp_path / "navigator_ui_data.json"

    assert navigator_main(["export-ui-data", "--output", str(out)]) == 0
    assert navigator_main(["export-ui-data", "--output", str(out), "--check"]) == 0
    out.write_text("{}\n", encoding="utf-8")
    assert navigator_main(["export-ui-data", "--output", str(out), "--check"]) == 1
