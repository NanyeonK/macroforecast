from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import yaml

from macrocast.navigator import (
    OPERATIONAL_NARROW_CONTRACTS,
    build_navigation_view,
    get_replication_entry,
    navigator_ui_data,
    replication_recipe_yaml,
    write_navigator_ui_data,
    write_replication_recipe,
)
from macrocast.navigator.cli import main as navigator_main
from macrocast.navigator.presentation import AXIS_PRESENTATION_SCHEMA_VERSION, AXIS_PRESENTATION_MAP


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
            "0_meta": {"fixed_axes": {"study_scope": "one_target_one_method"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "information_set_type": "final_revised_data",
                    "target_structure": "single_target",
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
            "6_stat_tests": {"fixed_axes": {}},
            "7_importance": {"fixed_axes": {"importance_method": "none"}},
        },
    }


def _axis(view, axis):
    return next(item for item in view["tree"] if item["axis"] == axis)


def _option(axis_view, value):
    return next(item for item in axis_view["options"] if item["value"] == value)


def _write_recipe(path: Path, recipe: dict) -> Path:
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return path


def _recipe_with_axis(recipe: dict, layer: str, axis: str, value: str) -> dict:
    out = yaml.safe_load(yaml.safe_dump(recipe, sort_keys=False))
    out["path"].setdefault(layer, {}).setdefault("fixed_axes", {})[axis] = value
    return out


def _js_state_snapshot(tmp_path: Path, recipe: dict, actions: list[tuple[str, str]]) -> dict:
    node = shutil.which("node")
    if node is None:
        import pytest

        pytest.skip("node is not installed")
    recipe_path = _write_recipe(tmp_path / "recipe.yaml", recipe)
    data_path = tmp_path / "navigator_ui_data.json"
    write_navigator_ui_data(data_path, sample_paths=(recipe_path,))
    script = """
const fs = require("fs");
const E = require("./docs/_extra/navigator_app/assets/state_engine.js");
const data = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
const actions = JSON.parse(process.argv[2]);
let state = E.createState(data, data.samples[0]);
for (const [axis, value] of actions) state = E.selectOption(data, state, axis, value);
const tree = E.buildTree(data, state);
function option(axisName, value) {
  const axis = tree.find((item) => item.axis === axisName);
  return axis.options.find((item) => item.value === value);
}
const source = data.samples[0];
const imported = E.recipeFromYaml(E.recipeYaml(data, source, state));
console.log(JSON.stringify({
  visible_axes: E.visibleTree(data, state).map((item) => item.axis),
  options: {
    model_ridge: option("model_family", "ridge"),
    model_midas_almon: option("model_family", "midas_almon"),
    model_midasr: option("model_family", "midasr"),
    model_midasr_nealmon: option("model_family", "midasr_nealmon"),
    midasr_weight_almonp: option("midasr_weight_family", "almonp"),
    model_random_forest: option("model_family", "random_forest"),
    model_quantile_linear: option("model_family", "quantile_linear"),
    equal_dm: option("equal_predictive", "dm"),
    equal_dm_hln: option("equal_predictive", "dm_hln"),
    density_pit: option("density_interval", "pit_uniformity"),
    direction_binomial: option("direction", "binomial_hit"),
    compute_parallel_by_model: option("compute_mode", "parallel_by_model"),
    compute_parallel_by_target: option("compute_mode", "parallel_by_target"),
    frequency_monthly: option("frequency", "monthly"),
    frequency_quarterly: option("frequency", "quarterly"),
    target_single: option("target_structure", "single_target"),
    target_multi: option("target_structure", "multi_target"),
    sd_state_west: option("fred_sd_state_group", "census_region_west"),
    sd_state_selected: option("state_selection", "selected_states"),
    sd_variable_labor: option("fred_sd_variable_group", "labor_market_core"),
    sd_series_selected: option("sd_variable_selection", "selected_sd_variables"),
    sd_mixed_drop_non_target: option("fred_sd_mixed_frequency_representation", "drop_non_target_native_frequency"),
    sd_mixed_feature_blocks: option("fred_sd_mixed_frequency_representation", "native_frequency_block_payload"),
    sd_mixed_adapter: option("fred_sd_mixed_frequency_representation", "mixed_frequency_model_adapter")
  },
  selected_disabled: E.selectedDisabledReasons(data, state),
  recipe_id: imported.recipe_id,
  edited_recipe: E.recipeWithEdits(data, source, state)
}));
"""
    result = subprocess.run(
        [node, "-e", script, str(data_path), json.dumps(actions)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


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


def test_navigator_ui_data_exports_layer0_presentation_contract():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))
    presentation = payload["axis_presentation"]

    assert payload["axis_presentation_schema_version"] == AXIS_PRESENTATION_SCHEMA_VERSION
    assert "research" + "_design" not in presentation
    assert presentation["study_scope"]["label"] == "Study Scope"
    assert presentation["study_scope"]["values"]["one_target_one_method"]["label"] == "One Target, One Method"
    assert presentation["study_scope"]["docs_url"].endswith("/detail/layer0/study_scope.html")
    assert presentation["study_scope"]["selection_kind"] == "user_choice"
    assert presentation["compute_mode"]["selection_kind"] == "defaulted_choice"
    assert presentation["compute_mode"]["default_value"] == "serial"
    assert presentation["compute_mode"]["values"]["serial"]["label"] == "Serial (Default)"
    assert presentation["compute_mode"]["values"]["parallel_by_model"]["label"] == "Parallelize Model Variants"
    assert "parallel_by_trial" not in presentation["compute_mode"]["values"]
    assert "distributed_cluster" not in presentation["compute_mode"]["values"]
    assert payload["state_engine"]["default_selections"]["compute_mode"] == "serial"
    assert AXIS_PRESENTATION_MAP["failure_policy"]["default_value"] == "fail_fast"
    assert "Default" in AXIS_PRESENTATION_MAP["failure_policy"]["values"]["fail_fast"]["label"]
    assert "retry_then_skip" not in AXIS_PRESENTATION_MAP["failure_policy"]["values"]
    assert "fallback_to_default_hp" not in AXIS_PRESENTATION_MAP["failure_policy"]["values"]
    assert payload["state_engine"]["default_selections"]["failure_policy"] == "fail_fast"
    assert AXIS_PRESENTATION_MAP["reproducibility_mode"]["default_value"] == "seeded_reproducible"
    assert "Default" in AXIS_PRESENTATION_MAP["reproducibility_mode"]["values"]["seeded_reproducible"]["label"]
    assert payload["state_engine"]["default_selections"]["reproducibility_mode"] == "seeded_reproducible"


def test_navigator_ui_data_exports_layer1_presentation_contract():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))
    presentation = payload["axis_presentation"]
    tree_axes = payload["tree_axes"]["1_data_task"]

    assert tree_axes == [
        "custom_source_policy",
        "dataset",
        "frequency",
        "information_set_type",
        "release_lag_rule",
        "contemporaneous_x_rule",
        "target_structure",
        "variable_universe",
        "fred_sd_frequency_policy",
        "fred_sd_state_group",
        "state_selection",
        "fred_sd_variable_group",
        "sd_variable_selection",
        "raw_missing_policy",
        "raw_outlier_policy",
        "official_transform_policy",
        "official_transform_scope",
        "missing_availability",
    ]
    layer1_groups = payload["layer_axis_groups"]["1_data_task"]
    assert [group["id"] for group in layer1_groups] == [
        "data_source_mode",
        "forecast_time_information",
        "target_y_definition",
        "predictor_x_definition",
        "fred_sd_source_scope",
        "raw_source_quality",
        "official_frame_policy",
    ]
    assert layer1_groups[-1]["label"] == "Official Transform / Frame Availability"
    assert layer1_groups[0]["level"] == "primary_decision"
    assert layer1_groups[0]["axes"] == [
        "custom_source_policy",
        "dataset",
        "frequency",
    ]
    assert layer1_groups[2]["parent_axis"] == "study_scope"
    assert layer1_groups[2]["level"] == "contract_derived"
    assert layer1_groups[4]["parent_axis"] == "dataset"
    assert layer1_groups[4]["level"] == "conditional_subgroup"
    sample_axes = {item["axis"]: item for item in payload["samples"][0]["view"]["tree"]}
    assert sample_axes["dataset"]["group_id"] == "data_source_mode"
    assert sample_axes["dataset"]["axis_level"] == "conditional_subdecision"
    assert sample_axes["custom_source_policy"]["group_id"] == "data_source_mode"
    assert sample_axes["custom_source_policy"]["axis_level"] == "primary_decision"
    assert "custom_source_format" not in sample_axes
    assert "custom_source_schema" not in sample_axes
    assert sample_axes["frequency"]["axis_level"] == "derived_or_required"
    assert sample_axes["information_set_type"]["group_id"] == "forecast_time_information"
    assert sample_axes["release_lag_rule"]["group_id"] == "forecast_time_information"
    assert sample_axes["contemporaneous_x_rule"]["group_id"] == "forecast_time_information"
    assert sample_axes["fred_sd_state_group"]["group_id"] == "fred_sd_source_scope"
    assert sample_axes["fred_sd_state_group"]["axis_level"] == "conditional_subdecision"
    assert sample_axes["state_selection"]["group_id"] == "fred_sd_source_scope"
    assert sample_axes["sd_variable_selection"]["group_id"] == "fred_sd_source_scope"
    assert sample_axes["target_structure"]["group_id"] == "target_y_definition"
    assert sample_axes["target_structure"]["parent_axis"] == "study_scope"
    assert sample_axes["target_structure"]["axis_level"] == "contract_derived"
    assert sample_axes["variable_universe"]["group_id"] == "predictor_x_definition"
    assert sample_axes["raw_missing_policy"]["group_id"] == "raw_source_quality"
    assert sample_axes["raw_missing_policy"]["axis_level"] == "secondary_policy"
    assert sample_axes["missing_availability"]["group_id"] == "official_frame_policy"
    assert presentation["dataset"]["label"] == "FRED Source Panel"
    assert presentation["dataset"]["values"]["fred_md+fred_sd"]["label"] == "FRED-MD + FRED-SD"
    assert "custom_csv" not in presentation["dataset"]["values"]
    assert presentation["custom_source_policy"]["label"] == "Data Source Mode"
    assert presentation["custom_source_policy"]["default_value"] == "official_only"
    assert presentation["custom_source_policy"]["values"]["custom_panel_only"]["label"] == "Custom Data Only"
    assert "custom_source_format" not in presentation
    assert "custom_source_schema" not in presentation
    assert presentation["frequency"]["selection_kind"] == "derived_or_required_choice"
    assert presentation["information_set_type"]["label"] == "Data Revision / Vintage Regime"
    assert presentation["release_lag_rule"]["label"] == "Publication Lag Rule"
    assert presentation["contemporaneous_x_rule"]["label"] == "Same-Period Predictor Rule"
    assert presentation["fred_sd_state_group"]["label"] == "FRED-SD State Scope"
    assert presentation["state_selection"]["label"] == "FRED-SD State List"
    assert presentation["fred_sd_variable_group"]["label"] == "FRED-SD Series Scope"
    assert presentation["sd_variable_selection"]["label"] == "FRED-SD Series List"
    assert presentation["target_structure"]["label"] == "Target (y) Definition"
    assert presentation["target_structure"]["contract"].startswith("Target-y cardinality")
    assert presentation["target_structure"]["default_value"] == "single_target"
    assert presentation["variable_universe"]["label"] == "FRED-MD/QD Predictor (x) Universe"
    assert presentation["raw_missing_policy"]["default_value"] == "preserve_raw_missing"
    assert presentation["raw_outlier_policy"]["default_value"] == "preserve_raw_outliers"
    assert presentation["official_transform_policy"]["default_value"] == "apply_official_tcode"
    assert presentation["official_transform_scope"]["default_value"] == "target_and_predictors"
    assert presentation["missing_availability"]["default_value"] == "zero_fill_leading_predictor_gaps"
    assert presentation["missing_availability"]["docs_url"].endswith("/detail/layer1/frame_availability.html")
    assert presentation["release_lag_rule"]["default_value"] == "ignore_release_lag"
    assert presentation["contemporaneous_x_rule"]["default_value"] == "forbid_same_period_predictors"
    assert payload["state_engine"]["default_selections"]["information_set_type"] == "final_revised_data"
    assert payload["state_engine"]["default_selections"]["target_structure"] == "single_target"
    assert payload["state_engine"]["default_selections"]["variable_universe"] == "all_variables"
    assert payload["state_engine"]["default_selections"]["state_selection"] == "all_states"
    assert payload["state_engine"]["default_selections"]["sd_variable_selection"] == "all_sd_variables"
    assert payload["state_engine"]["default_selections"]["raw_missing_policy"] == "preserve_raw_missing"
    assert payload["state_engine"]["default_selections"]["release_lag_rule"] == "ignore_release_lag"


def test_navigation_compute_layout_respects_study_scope():
    view = build_navigation_view(_recipe())
    compute_axis = _axis(view, "compute_mode")

    assert _option(compute_axis, "serial")["enabled"] is True
    assert _option(compute_axis, "parallel_by_horizon")["enabled"] is True
    assert _option(compute_axis, "parallel_by_oos_date")["enabled"] is True
    assert "compares methods" in _option(compute_axis, "parallel_by_model")["disabled_reason"]
    assert "multiple targets" in _option(compute_axis, "parallel_by_target")["disabled_reason"]
    assert {option["value"] for option in compute_axis["options"]} == {
        "serial",
        "parallel_by_model",
        "parallel_by_horizon",
        "parallel_by_target",
        "parallel_by_oos_date",
    }

    compare_methods_recipe = _recipe()
    compare_methods_recipe["path"]["0_meta"]["fixed_axes"]["study_scope"] = "one_target_compare_methods"
    compare_methods_axis = _axis(build_navigation_view(compare_methods_recipe), "compute_mode")
    assert _option(compare_methods_axis, "parallel_by_model")["enabled"] is True
    assert "multiple targets" in _option(compare_methods_axis, "parallel_by_target")["disabled_reason"]

    multi_target_recipe = _recipe()
    multi_target_recipe["path"]["0_meta"]["fixed_axes"]["study_scope"] = "multiple_targets_one_method"
    multi_target_axis = _axis(build_navigation_view(multi_target_recipe), "compute_mode")
    assert "compares methods" in _option(multi_target_axis, "parallel_by_model")["disabled_reason"]
    assert _option(multi_target_axis, "parallel_by_target")["enabled"] is True


def test_navigation_layer1_frequency_respects_dataset():
    view = build_navigation_view(_recipe())
    frequency_axis = _axis(view, "frequency")

    assert _option(frequency_axis, "monthly")["enabled"] is True
    assert "requires frequency=monthly" in _option(frequency_axis, "quarterly")["disabled_reason"]

    qd_recipe = _recipe()
    qd_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_qd+fred_sd"
    qd_axis = _axis(build_navigation_view(qd_recipe), "frequency")
    assert _option(qd_axis, "quarterly")["enabled"] is True
    assert "requires frequency=quarterly" in _option(qd_axis, "monthly")["disabled_reason"]

    sd_recipe = _recipe()
    sd_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_sd"
    sd_axis = _axis(build_navigation_view(sd_recipe), "frequency")
    assert _option(sd_axis, "monthly")["enabled"] is True
    assert _option(sd_axis, "quarterly")["enabled"] is True


def test_navigation_layer1_custom_only_leaves_frequency_free():
    recipe = _recipe()
    recipe["path"]["1_data_task"]["fixed_axes"]["custom_source_policy"] = "custom_panel_only"
    view = build_navigation_view(recipe)
    frequency_axis = _axis(view, "frequency")

    assert _option(frequency_axis, "monthly")["enabled"] is True
    assert _option(frequency_axis, "quarterly")["enabled"] is True


def test_navigation_layer1_target_structure_respects_study_scope():
    view = build_navigation_view(_recipe())
    target_axis = _axis(view, "target_structure")

    assert _option(target_axis, "single_target")["enabled"] is True
    assert "one-target Study Scope" in _option(target_axis, "multi_target")["disabled_reason"]

    multi_recipe = _recipe()
    multi_recipe["path"]["0_meta"]["fixed_axes"]["study_scope"] = "multiple_targets_compare_methods"
    multi_axis = _axis(build_navigation_view(multi_recipe), "target_structure")
    assert "multiple-target Study Scope" in _option(multi_axis, "single_target")["disabled_reason"]
    assert _option(multi_axis, "multi_target")["enabled"] is True


def test_navigator_ui_data_omits_rename_ledger():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))

    assert "naming_ledger_version" not in payload
    assert "rename_ledger" not in payload


def test_navigation_disables_tree_shap_for_non_tree_model():
    recipe = _recipe()
    recipe["path"]["7_importance"]["fixed_axes"]["importance_method"] = "tree_shap"
    view = build_navigation_view(recipe)

    model_axis = _axis(view, "model_family")
    assert _option(model_axis, "ridge")["enabled"] is False
    assert "tree_shap" in _option(model_axis, "ridge")["disabled_reason"]
    assert _option(model_axis, "random_forest")["enabled"] is True


def test_navigation_disables_split_tree_shap_for_non_tree_model():
    recipe = _recipe()
    recipe["path"]["7_importance"]["fixed_axes"]["importance_shap"] = "tree_shap"
    view = build_navigation_view(recipe)

    model_axis = _axis(view, "model_family")
    assert _option(model_axis, "ridge")["enabled"] is False
    assert "tree_shap" in _option(model_axis, "ridge")["disabled_reason"]
    assert _option(model_axis, "random_forest")["enabled"] is True


def test_navigation_quantile_forecast_restricts_model_family():
    view = build_navigation_view(_recipe(forecast_object="quantile"))
    model_axis = _axis(view, "model_family")

    assert _option(model_axis, "ridge")["enabled"] is False
    assert _option(model_axis, "quantile_linear")["enabled"] is True
    assert view["compatibility"]["active_rules"][0]["rule"] == "quantile_requires_quantile_generator"


def test_navigation_shows_current_deep_sequence_gate():
    view = build_navigation_view(_recipe(model_family="lstm", feature_builder="target_lag_features"))
    feature_axis = _axis(view, "feature_builder")

    assert _option(feature_axis, "target_lag_features")["enabled"] is True
    assert _option(feature_axis, "sequence_tensor")["enabled"] is False
    assert "sequence_tensor remains gated" in _option(feature_axis, "sequence_tensor")["disabled_reason"]


def test_navigation_layer6_forecast_object_family_gates(tmp_path: Path):
    recipe = _recipe()
    view = build_navigation_view(recipe)
    js = _js_state_snapshot(tmp_path, recipe, [])

    density_option = _option(_axis(view, "density_interval"), "pit_uniformity")
    direction_option = _option(_axis(view, "direction"), "binomial_hit")

    assert density_option["enabled"] is False
    assert direction_option["enabled"] is False
    assert "interval, density, or quantile" in density_option["disabled_reason"]
    assert "forecast_object=direction" in direction_option["disabled_reason"]
    assert js["options"]["density_pit"]["enabled"] == density_option["enabled"]
    assert js["options"]["direction_binomial"]["enabled"] == direction_option["enabled"]


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
    assert payload["path"]["0_meta"]["fixed_axes"]["study_scope"] == "one_target_one_method"
    assert "research" + "_design" not in payload["path"]["0_meta"]["fixed_axes"]
    assert payload["path"]["3_training"]["fixed_axes"]["forecast_type"] == "iterated"
    assert get_replication_entry("synthetic-replication-roundtrip")["expected_outputs"]
    assert "one_target_one_method" in replication_recipe_yaml("synthetic-replication-roundtrip")


def test_navigator_cli_writes_replication_yaml(tmp_path: Path):
    out = tmp_path / "gc.yaml"
    rc = navigator_main(["replications", "goulet-coulombe-2021-fred-md-ridge", "--write-yaml", str(out)])

    assert rc == 0
    payload = yaml.safe_load(out.read_text())
    assert payload["path"]["3_training"]["fixed_axes"]["model_family"] == "ridge"



def test_navigator_ui_data_exports_registered_layer_topology():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))
    topology = payload["layer_topology"]
    nodes = {node["id"]: node for node in topology["nodes"]}

    assert topology["schema_version"] == "navigator_layer_topology_v1"
    assert topology["main_flow"] == ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"]
    assert nodes["l3"]["ui_mode"] == "graph"
    assert nodes["l7"]["ui_mode"] == "graph"
    assert nodes["l1_5"]["category"] == "diagnostic"
    assert "L6_D_multiple_model" in nodes["l6"]["sub_layers"]
    assert "saved_objects" in nodes["l8"]["axes"]
    assert "l8_artifacts_v1" in nodes["l8"]["produces"]
    assert any(edge["from"] == "l4" and edge["to"] == "l5" for edge in topology["edges"])


def test_navigator_topology_uses_current_layer_specs_for_l0_l1_l2():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))
    nodes = {node["id"]: node for node in payload["layer_topology"]["nodes"]}

    assert nodes["l0"]["sub_layers"] == ["L0.A Execution policy"]
    assert nodes["l0"]["axes"] == ["failure_policy", "reproducibility_mode", "compute_mode"]
    assert "study_scope" not in nodes["l0"]["axes"]

    assert nodes["l1"]["sub_layers"] == [
        "L1.A Source selection",
        "L1.B Target definition",
        "L1.C Predictor universe",
        "L1.D Geography scope",
        "L1.E Sample window",
        "L1.F Horizon set",
        "L1.G Regime definition",
    ]
    assert nodes["l1"]["sub_layer_axes"]["L1.E Sample window"] == ["sample_start_rule", "sample_end_rule"]
    assert nodes["l1"]["sub_layer_axes"]["L1.F Horizon set"] == ["horizon_set"]
    assert "information_set_type" not in nodes["l1"]["axes"]
    assert "fred_sd_variable_group" not in nodes["l1"]["axes"]

    assert nodes["l2"]["sub_layers"] == [
        "L2.A FRED-SD frequency alignment",
        "L2.B Transform",
        "L2.C Outlier handling",
        "L2.D Imputation",
        "L2.E Frame edge",
    ]
    assert "scaling_policy" not in nodes["l2"]["axes"]
    assert nodes["l2"]["sub_layer_axes"]["L2.D Imputation"] == [
        "imputation_policy",
        "imputation_temporal_rule",
        "imputation_scope",
    ]


def test_navigator_generated_yaml_validates_all_current_layers(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        import pytest

        pytest.skip("node is not installed")

    data_path = tmp_path / "navigator_ui_data.json"
    write_navigator_ui_data(data_path, sample_paths=("examples/recipes/model-benchmark.yaml",))
    script = r"""
const fs = require("fs");
const data = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
let code = fs.readFileSync("docs/_extra/navigator_app/assets/app.js", "utf8");
code = code.replace(/boot\(\)\.catch\(\(error\) => \{[\s\S]*?\n\}\);/, "");
global.document = { getElementById: () => ({}), body: {} };
global.window = { addEventListener: () => {}, location: { search: "" } };
global.navigator = {};
global.URLSearchParams = class { get(){ return null; } };
global.fetch = async () => { throw new Error("fetch disabled"); };
eval(code + "\nstate.data = data; state.sampleIndex = 0; console.log(canonicalRecipeYaml());");
"""
    result = subprocess.run(
        [node, "-e", script, str(data_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    generated = yaml.safe_load(result.stdout)

    from macrocast.core.layers import l0, l1, l1_5, l2, l2_5, l3, l3_5, l4, l4_5, l5, l6, l7, l8

    modules = [
        ("0_meta", "l0", l0),
        ("1_data", "l1", l1),
        ("2_preprocessing", "l2", l2),
        ("3_feature_engineering", "l3", l3),
        ("4_forecasting_model", "l4", l4),
        ("5_evaluation", "l5", l5),
        ("6_statistical_tests", "l6", l6),
        ("7_interpretation", "l7", l7),
        ("8_output", "l8", l8),
        ("1_5_data_summary", "l1_5", l1_5),
        ("2_5_pre_post_preprocessing", "l2_5", l2_5),
        ("3_5_feature_diagnostics", "l3_5", l3_5),
        ("4_5_generator_diagnostics", "l4_5", l4_5),
    ]
    for yaml_key, layer_id, module in modules:
        layer_yaml = yaml.safe_dump({yaml_key: generated[yaml_key]}, sort_keys=False)
        layer = module.parse_layer_yaml(layer_yaml, layer_id)
        assert module.validate_layer(layer).has_hard_errors is False, yaml_key


def test_navigator_generated_yaml_validates_each_active_choice(tmp_path: Path):
    node = shutil.which("node")
    if node is None:
        import pytest

        pytest.skip("node is not installed")

    data_path = tmp_path / "navigator_ui_data.json"
    write_navigator_ui_data(data_path, sample_paths=("examples/recipes/model-benchmark.yaml",))
    script = r"""
const fs = require("fs");
const data = JSON.parse(fs.readFileSync(process.argv[1], "utf8"));
let code = fs.readFileSync("docs/_extra/navigator_app/assets/app.js", "utf8");
code = code.replace(/boot\(\)\.catch\(\(error\) => \{[\s\S]*?\n\}\);/, "");
global.document = { getElementById: () => ({}), body: {} };
global.window = { addEventListener: () => {}, location: { search: "" } };
global.navigator = {};
global.URLSearchParams = class { get(){ return null; } };
global.fetch = async () => { throw new Error("fetch disabled"); };
eval(code + `
state.data = data; state.sampleIndex = 0;
const layerKeys = {l0:"0_meta", l1:"1_data", l2:"2_preprocessing", l5:"5_evaluation", l6:"6_statistical_tests", l7:"7_interpretation", l8:"8_output", l1_5:"1_5_data_summary", l2_5:"2_5_pre_post_preprocessing", l3_5:"3_5_feature_diagnostics", l4_5:"4_5_generator_diagnostics"};
const cases = [];
const missing = [];
for (const n of data.layer_topology.nodes) {
  if (!layerKeys[n.id]) continue;
  for (const axis of n.axes || []) {
    const records = optionRecordsForAxis(axis, n.id);
    if (!records.length) missing.push([n.id, axis]);
    for (const rec of records) {
      if (rec.enabled === false) continue;
      state.canonicalSelections = {};
      state.dagSelections = {};
      state.canonicalSelections[axisSelectionKey(n.id, axis)] = isMultiSelectAxis(axis) ? [rec.value] : rec.value;
      cases.push({layer: n.id, yaml_key: layerKeys[n.id], axis, value: rec.value, yaml: canonicalRecipeYaml()});
    }
  }
}
console.log(JSON.stringify({missing, cases}));
`);
"""
    result = subprocess.run(
        [node, "-e", script, str(data_path)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["missing"] == []

    from macrocast.core.layers import l0, l1, l1_5, l2, l2_5, l3_5, l4_5, l5, l6, l7, l8

    modules = {
        "l0": l0,
        "l1": l1,
        "l2": l2,
        "l5": l5,
        "l6": l6,
        "l7": l7,
        "l8": l8,
        "l1_5": l1_5,
        "l2_5": l2_5,
        "l3_5": l3_5,
        "l4_5": l4_5,
    }
    failures = []
    for case in payload["cases"]:
        module = modules[case["layer"]]
        if hasattr(module, "parse_recipe_yaml") and hasattr(module, "validate_recipe"):
            report = module.validate_recipe(module.parse_recipe_yaml(case["yaml"]))
        else:
            generated = yaml.safe_load(case["yaml"])
            layer_yaml = yaml.safe_dump({case["yaml_key"]: generated[case["yaml_key"]]}, sort_keys=False)
            layer = module.parse_layer_yaml(layer_yaml, case["layer"])
            report = module.validate_layer(layer)
        if report.has_hard_errors:
            failures.append((case["layer"], case["axis"], case["value"]))
    assert failures == []


def test_navigator_ui_data_exports_runtime_support_metadata():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))
    support = payload["runtime_support"]

    assert support["schema_version"] == "navigator_runtime_support_v1"
    assert support["status_map"]["operational"] == "runtime_supported"
    assert support["status_map"]["registry_only"] == "schema_only"
    assert "runtime_supported" in support["legend"]
    assert support["layer_notes"]["6_stat_tests"]["label"] == "Lightweight runtime"


def test_navigator_ui_data_export_roundtrip(tmp_path: Path):
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))

    assert payload["schema_version"] == "navigator_ui_data_v1"
    assert payload["samples"][0]["view"]["schema_version"] == "navigator_view_v1"
    assert "model_family" in payload["axis_catalog"]
    assert payload["state_engine"]["schema_version"] == "navigator_state_engine_v1"
    assert payload["state_engine"]["default_selections"]["importance_scope"] == "global"
    assert payload["state_engine"]["importance"]["legacy_to_axis"]["tree_shap"]["axis"] == "importance_shap"
    assert payload["operational_narrow_contracts"]
    assert payload["replications"]
    assert payload["replications"][0]["recipe"]["path"]

    out = write_navigator_ui_data(tmp_path / "navigator_ui_data.json", sample_paths=("examples/recipes/model-benchmark.yaml",))
    assert out.exists()
    assert write_navigator_ui_data(out, sample_paths=("examples/recipes/model-benchmark.yaml",), check=True) == out


def test_navigator_operational_narrow_contracts_are_source_of_truth():
    payload = navigator_ui_data(("examples/recipes/model-benchmark.yaml",))
    contracts = {item["axis"]: item for item in payload["operational_narrow_contracts"]}

    assert tuple(OPERATIONAL_NARROW_CONTRACTS[0]["values"]) == tuple(contracts["feature_block_set"]["values"])
    assert contracts["feature_block_set"]["contract"] == "feature_block_set_public_axis_v1"
    assert contracts["fred_sd_mixed_frequency_representation"]["contract"] == (
        "fred_sd_native_frequency_block_payload_v1"
    )
    assert contracts["exogenous_x_path_policy"]["contract"] == "exogenous_x_path_contract_v1"
    assert "scheduled_known_future_x_columns" in " ".join(
        contracts["exogenous_x_path_policy"]["required_companions"]
    )
    assert "recursive_x_model_family=ar1" in " ".join(
        contracts["exogenous_x_path_policy"]["required_companions"]
    )

    axis_catalog = payload["axis_catalog"]
    assert axis_catalog["exogenous_x_path_policy"]["current_status"]["unavailable"] == "gated_named"
    assert axis_catalog["exogenous_x_path_policy"]["current_status"]["recursive_x_model"] == "operational_narrow"
    assert axis_catalog["recursive_x_model_family"]["current_status"]["none"] == "gated_named"
    assert axis_catalog["recursive_x_model_family"]["current_status"]["ar1"] == "operational_narrow"


def test_navigator_cli_checks_ui_data(tmp_path: Path):
    out = tmp_path / "navigator_ui_data.json"

    assert navigator_main(["export-ui-data", "--output", str(out)]) == 0
    assert navigator_main(["export-ui-data", "--output", str(out), "--check"]) == 0
    out.write_text("{}\n", encoding="utf-8")
    assert navigator_main(["export-ui-data", "--output", str(out), "--check"]) == 1


def test_browser_state_engine_matches_python_tree_shap_model_gate(tmp_path: Path):
    recipe = _recipe()
    python_view = build_navigation_view(_recipe_with_axis(recipe, "7_importance", "importance_shap", "tree_shap"))
    js = _js_state_snapshot(tmp_path, recipe, [("importance_shap", "tree_shap")])

    assert js["options"]["model_ridge"]["enabled"] == _option(_axis(python_view, "model_family"), "ridge")["enabled"]
    assert js["options"]["model_random_forest"]["enabled"] == _option(
        _axis(python_view, "model_family"), "random_forest"
    )["enabled"]
    assert "tree_shap" in js["options"]["model_ridge"]["disabled_reason"]
    assert js["edited_recipe"]["path"]["7_importance"]["fixed_axes"]["importance_shap"] == "tree_shap"
    assert js["recipe_id"] == recipe["recipe_id"]


def test_browser_state_engine_matches_python_quantile_model_gate(tmp_path: Path):
    recipe = _recipe()
    python_view = build_navigation_view(_recipe_with_axis(recipe, "3_training", "forecast_object", "quantile"))
    js = _js_state_snapshot(tmp_path, recipe, [("forecast_object", "quantile")])

    assert js["options"]["model_ridge"]["enabled"] == _option(_axis(python_view, "model_family"), "ridge")["enabled"]
    assert js["options"]["model_quantile_linear"]["enabled"] == _option(
        _axis(python_view, "model_family"), "quantile_linear"
    )["enabled"]
    assert "quantile_linear" in js["options"]["model_ridge"]["disabled_reason"]


def test_browser_state_engine_matches_python_hac_gate(tmp_path: Path):
    recipe = _recipe()
    hac_recipe = _recipe_with_axis(recipe, "6_stat_tests", "equal_predictive", "dm_hln")
    hac_recipe = _recipe_with_axis(hac_recipe, "6_stat_tests", "dependence_correction", "nw_hac")
    hac_recipe = _recipe_with_axis(hac_recipe, "6_stat_tests", "overlap_handling", "evaluate_with_hac")
    python_view = build_navigation_view(hac_recipe)
    js = _js_state_snapshot(
        tmp_path,
        recipe,
        [
            ("equal_predictive", "dm_hln"),
            ("dependence_correction", "nw_hac"),
            ("overlap_handling", "evaluate_with_hac"),
        ],
    )

    assert js["options"]["equal_dm"]["enabled"] == _option(_axis(python_view, "equal_predictive"), "dm")["enabled"]
    assert js["options"]["equal_dm_hln"]["enabled"] == _option(
        _axis(python_view, "equal_predictive"), "dm_hln"
    )["enabled"]
    assert "HAC-capable" in js["options"]["equal_dm"]["disabled_reason"]


def test_browser_state_engine_matches_compute_layout_scope_gate(tmp_path: Path):
    recipe = _recipe()
    python_view = build_navigation_view(recipe)
    js = _js_state_snapshot(tmp_path, recipe, [])

    assert js["options"]["compute_parallel_by_model"]["enabled"] == _option(
        _axis(python_view, "compute_mode"), "parallel_by_model"
    )["enabled"]
    assert js["options"]["compute_parallel_by_target"]["enabled"] == _option(
        _axis(python_view, "compute_mode"), "parallel_by_target"
    )["enabled"]
    assert "compares methods" in js["options"]["compute_parallel_by_model"]["disabled_reason"]
    assert "multiple targets" in js["options"]["compute_parallel_by_target"]["disabled_reason"]

    compare_js = _js_state_snapshot(tmp_path, recipe, [("study_scope", "one_target_compare_methods")])
    assert compare_js["options"]["compute_parallel_by_model"]["enabled"] is True
    assert "multiple targets" in compare_js["options"]["compute_parallel_by_target"]["disabled_reason"]


def test_browser_state_engine_matches_layer1_frequency_and_target_gates(tmp_path: Path):
    recipe = _recipe()
    python_view = build_navigation_view(recipe)
    js = _js_state_snapshot(tmp_path, recipe, [])

    assert js["options"]["frequency_monthly"]["enabled"] == _option(
        _axis(python_view, "frequency"), "monthly"
    )["enabled"]
    assert js["options"]["frequency_quarterly"]["enabled"] == _option(
        _axis(python_view, "frequency"), "quarterly"
    )["enabled"]
    assert "requires frequency=monthly" in js["options"]["frequency_quarterly"]["disabled_reason"]
    assert js["options"]["target_single"]["enabled"] == _option(
        _axis(python_view, "target_structure"), "single_target"
    )["enabled"]
    assert "one-target Study Scope" in js["options"]["target_multi"]["disabled_reason"]

    qd_js = _js_state_snapshot(tmp_path, recipe, [("dataset", "fred_qd")])
    assert qd_js["options"]["frequency_quarterly"]["enabled"] is True
    assert "requires frequency=quarterly" in qd_js["options"]["frequency_monthly"]["disabled_reason"]

    sd_js = _js_state_snapshot(tmp_path, recipe, [("dataset", "fred_sd")])
    assert sd_js["options"]["frequency_monthly"]["enabled"] is True
    assert sd_js["options"]["frequency_quarterly"]["enabled"] is True
    assert "variable_universe" not in sd_js["visible_axes"]
    assert "official_transform_policy" not in sd_js["visible_axes"]
    assert "official_transform_scope" not in sd_js["visible_axes"]
    assert "fred_sd_state_group" in sd_js["visible_axes"]
    assert "state_selection" in sd_js["visible_axes"]
    assert "variable_universe" not in sd_js["edited_recipe"]["path"]["1_data_task"]["fixed_axes"]

    custom_js = _js_state_snapshot(tmp_path, recipe, [("custom_source_policy", "custom_panel_only")])
    assert "dataset" not in custom_js["visible_axes"]
    assert "information_set_type" not in custom_js["visible_axes"]
    assert "release_lag_rule" not in custom_js["visible_axes"]
    assert "variable_universe" not in custom_js["visible_axes"]
    assert "official_transform_policy" not in custom_js["visible_axes"]
    assert "official_transform_scope" not in custom_js["visible_axes"]
    assert "contemporaneous_x_rule" in custom_js["visible_axes"]
    assert custom_js["options"]["frequency_monthly"]["enabled"] is True
    assert custom_js["options"]["frequency_quarterly"]["enabled"] is True
    assert "dataset" not in custom_js["edited_recipe"]["path"]["1_data_task"]["fixed_axes"]
    assert "information_set_type" not in custom_js["edited_recipe"]["path"]["1_data_task"]["fixed_axes"]

    multi_js = _js_state_snapshot(tmp_path, recipe, [("study_scope", "multiple_targets_compare_methods")])
    assert "multiple-target Study Scope" in multi_js["options"]["target_single"]["disabled_reason"]
    assert multi_js["options"]["target_multi"]["enabled"] is True


def test_browser_state_engine_matches_python_fred_sd_group_gate(tmp_path: Path):
    recipe = _recipe()
    python_view = build_navigation_view(recipe)
    js = _js_state_snapshot(tmp_path, recipe, [])

    assert js["options"]["sd_state_west"]["enabled"] == _option(
        _axis(python_view, "fred_sd_state_group"), "census_region_west"
    )["enabled"]
    assert js["options"]["sd_state_selected"]["enabled"] == _option(
        _axis(python_view, "state_selection"), "selected_states"
    )["enabled"]
    assert js["options"]["sd_variable_labor"]["enabled"] == _option(
        _axis(python_view, "fred_sd_variable_group"), "labor_market_core"
    )["enabled"]
    assert js["options"]["sd_series_selected"]["enabled"] == _option(
        _axis(python_view, "sd_variable_selection"), "selected_sd_variables"
    )["enabled"]
    assert js["options"]["sd_mixed_drop_non_target"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "drop_non_target_native_frequency"
    )["enabled"]
    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "native_frequency_block_payload"
    )["enabled"]
    assert "fred_sd" in js["options"]["sd_state_west"]["disabled_reason"]
    assert "fred_sd" in js["options"]["sd_state_selected"]["disabled_reason"]
    assert "fred_sd" in js["options"]["sd_mixed_drop_non_target"]["disabled_reason"]
    assert "midasr" in js["options"]["midasr_weight_almonp"]["disabled_reason"]

    fred_sd_recipe = _recipe()
    fred_sd_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_md+fred_sd"
    python_view = build_navigation_view(fred_sd_recipe)
    js = _js_state_snapshot(tmp_path, fred_sd_recipe, [])

    assert js["options"]["sd_state_west"]["enabled"] == _option(
        _axis(python_view, "fred_sd_state_group"), "census_region_west"
    )["enabled"]
    assert js["options"]["sd_state_selected"]["enabled"] == _option(
        _axis(python_view, "state_selection"), "selected_states"
    )["enabled"]
    assert js["options"]["sd_variable_labor"]["enabled"] == _option(
        _axis(python_view, "fred_sd_variable_group"), "labor_market_core"
    )["enabled"]
    assert js["options"]["sd_series_selected"]["enabled"] == _option(
        _axis(python_view, "sd_variable_selection"), "selected_sd_variables"
    )["enabled"]
    assert js["options"]["sd_mixed_drop_non_target"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "drop_non_target_native_frequency"
    )["enabled"]
    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "native_frequency_block_payload"
    )["enabled"]
    assert "custom model" in js["options"]["sd_mixed_feature_blocks"]["disabled_reason"]
    assert "custom model" in js["options"]["sd_mixed_adapter"]["disabled_reason"]

    midas_recipe = _recipe(model_family="midas_almon")
    midas_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_md+fred_sd"
    python_view = build_navigation_view(midas_recipe)
    js = _js_state_snapshot(tmp_path, midas_recipe, [])

    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "native_frequency_block_payload"
    )["enabled"]
    assert js["options"]["sd_mixed_adapter"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "mixed_frequency_model_adapter"
    )["enabled"]
    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] is True
    assert js["options"]["sd_mixed_adapter"]["enabled"] is True
    assert "advanced FRED-SD" in js["options"]["sd_mixed_drop_non_target"]["disabled_reason"]

    generic_midasr_recipe = _recipe(model_family="midasr")
    generic_midasr_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_md+fred_sd"
    python_view = build_navigation_view(generic_midasr_recipe)
    js = _js_state_snapshot(tmp_path, generic_midasr_recipe, [])

    assert js["options"]["model_midasr"]["enabled"] == _option(
        _axis(python_view, "model_family"), "midasr"
    )["enabled"]
    assert js["options"]["midasr_weight_almonp"]["enabled"] == _option(
        _axis(python_view, "midasr_weight_family"), "almonp"
    )["enabled"]
    assert js["options"]["midasr_weight_almonp"]["enabled"] is True
    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] is True
    assert js["options"]["sd_mixed_adapter"]["enabled"] is True

    midasr_recipe = _recipe(model_family="midasr_nealmon")
    midasr_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_md+fred_sd"
    python_view = build_navigation_view(midasr_recipe)
    js = _js_state_snapshot(tmp_path, midasr_recipe, [])

    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "native_frequency_block_payload"
    )["enabled"]
    assert js["options"]["sd_mixed_adapter"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "mixed_frequency_model_adapter"
    )["enabled"]
    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] is True
    assert js["options"]["sd_mixed_adapter"]["enabled"] is True
