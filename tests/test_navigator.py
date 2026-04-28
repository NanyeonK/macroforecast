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
            "0_meta": {"fixed_axes": {"experiment_unit": "single_target_single_generator"}},
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
    sd_state_west: option("fred_sd_state_group", "census_region_west"),
    sd_variable_labor: option("fred_sd_variable_group", "labor_market_core"),
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
    assert presentation["experiment_unit"]["label"] == "Execution Unit"
    assert presentation["experiment_unit"]["values"]["single_target_single_generator"]["label"] == "One Target, One Forecasting Path"
    assert presentation["experiment_unit"]["docs_url"].endswith("/detail/layer0/experiment_unit.html")
    assert presentation["experiment_unit"]["selection_kind"] == "user_choice"
    assert presentation["compute_mode"]["values"]["parallel_by_model"]["label"] == "Parallelize Models"
    assert AXIS_PRESENTATION_MAP["failure_policy"]["values"]["fail_fast"]["label"] == "Stop on First Failure"


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
    assert payload["path"]["0_meta"]["fixed_axes"]["experiment_unit"] == "single_target_single_generator"
    assert "research" + "_design" not in payload["path"]["0_meta"]["fixed_axes"]
    assert payload["path"]["3_training"]["fixed_axes"]["forecast_type"] == "iterated"
    assert get_replication_entry("synthetic-replication-roundtrip")["expected_outputs"]
    assert "single_target_single_generator" in replication_recipe_yaml("synthetic-replication-roundtrip")


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


def test_browser_state_engine_matches_python_fred_sd_group_gate(tmp_path: Path):
    recipe = _recipe()
    python_view = build_navigation_view(recipe)
    js = _js_state_snapshot(tmp_path, recipe, [])

    assert js["options"]["sd_state_west"]["enabled"] == _option(
        _axis(python_view, "fred_sd_state_group"), "census_region_west"
    )["enabled"]
    assert js["options"]["sd_variable_labor"]["enabled"] == _option(
        _axis(python_view, "fred_sd_variable_group"), "labor_market_core"
    )["enabled"]
    assert js["options"]["sd_mixed_drop_non_target"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "drop_non_target_native_frequency"
    )["enabled"]
    assert js["options"]["sd_mixed_feature_blocks"]["enabled"] == _option(
        _axis(python_view, "fred_sd_mixed_frequency_representation"), "native_frequency_block_payload"
    )["enabled"]
    assert "fred_sd" in js["options"]["sd_state_west"]["disabled_reason"]
    assert "fred_sd" in js["options"]["sd_mixed_drop_non_target"]["disabled_reason"]
    assert "midasr" in js["options"]["midasr_weight_almonp"]["disabled_reason"]

    fred_sd_recipe = _recipe()
    fred_sd_recipe["path"]["1_data_task"]["fixed_axes"]["dataset"] = "fred_md+fred_sd"
    python_view = build_navigation_view(fred_sd_recipe)
    js = _js_state_snapshot(tmp_path, fred_sd_recipe, [])

    assert js["options"]["sd_state_west"]["enabled"] == _option(
        _axis(python_view, "fred_sd_state_group"), "census_region_west"
    )["enabled"]
    assert js["options"]["sd_variable_labor"]["enabled"] == _option(
        _axis(python_view, "fred_sd_variable_group"), "labor_market_core"
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
