from pathlib import Path

from macrocast.core.layers.l8 import (
    make_recipe_with_glmboost,
    make_recipe_with_l6_l7_active,
    make_recipe_without_ensemble,
    normalize_to_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    validate_layer,
    validate_recipe,
)
from macrocast.core.ops import get_op

ROOT = Path(__file__).resolve().parents[2]


def _example(name: str) -> str:
    return (ROOT / "examples" / "recipes" / name).read_text()


def test_l8_minimal_yaml_parses_to_defaults():
    layer = parse_layer_yaml("8_output:\n  fixed_axes: {}", "l8")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l8"))
    assert resolved["export_format"] == "json_csv"
    assert resolved["compression"] == "none"
    assert resolved["manifest_format"] == "json"
    assert resolved["artifact_granularity"] == "per_cell"
    assert resolved["naming_convention"] == "descriptive"


def test_l8_default_saved_objects_minimal_recipe():
    resolved = resolve_axes(parse_recipe_yaml("8_output:\n  fixed_axes: {}").layers["l8"].dag)
    assert {"forecasts", "metrics", "ranking"} <= set(resolved["saved_objects"])


def test_l8_default_saved_objects_active_components():
    resolved = resolve_axes(parse_recipe_yaml(make_recipe_with_l6_l7_active()).layers["l8"].dag)
    assert "tests" in resolved["saved_objects"]
    assert "importance" in resolved["saved_objects"]


def test_l8_default_saved_objects_active_diagnostics():
    resolved = resolve_axes(
        parse_recipe_yaml(
            "1_5_data_summary:\n"
            "  enabled: true\n"
            "2_5_pre_post_preprocessing:\n"
            "  enabled: true\n"
            "3_5_feature_diagnostics:\n"
            "  enabled: true\n"
            "4_5_generator_diagnostics:\n"
            "  enabled: true\n"
            "8_output:\n"
            "  fixed_axes: {}\n"
        ).layers["l8"].dag
    )
    assert {
        "diagnostics_l1_5",
        "diagnostics_l2_5",
        "diagnostics_l3_5",
        "diagnostics_l4_5",
    } <= set(resolved["saved_objects"])


def test_l8_axes_not_sweepable():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    export_format: {sweep: [json_csv, json_parquet]}\n", "l8")
    assert validate_layer(layer).has_hard_errors


def test_l8_latex_tables_requires_l5_active():
    recipe = parse_recipe_yaml("5_evaluation:\n  fixed_axes: {}\n8_output:\n  fixed_axes:\n    export_format: latex_tables\n")
    assert validate_recipe(recipe).has_hard_errors is False


def test_l8_latex_tables_rejects_explicit_l5_disabled():
    recipe = parse_recipe_yaml("5_evaluation:\n  enabled: false\n8_output:\n  fixed_axes:\n    export_format: latex_tables\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l8_model_artifacts_format_pickle_default():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    saved_objects: [forecasts, metrics, ranking, model_artifacts]\n", "l8")
    assert resolve_axes(normalize_to_dag_form(layer, "l8"))["model_artifacts_format"] == "pickle"


def test_l8_model_artifacts_format_inactive_without_artifact_object():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    saved_objects: [forecasts, metrics, ranking]\n    model_artifacts_format: pickle\n", "l8")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l8"))
    assert resolved.get_active("model_artifacts_format") is False


def test_l8_onnx_format_rejected_as_future():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    saved_objects: [forecasts, model_artifacts]\n    model_artifacts_format: onnx\n", "l8")
    assert validate_layer(layer).has_hard_errors


def test_l8_pmml_format_rejected_as_future():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    saved_objects: [forecasts, model_artifacts]\n    model_artifacts_format: pmml\n", "l8")
    assert validate_layer(layer).has_hard_errors


def test_l8_provenance_fields_default_all():
    resolved = resolve_axes(normalize_to_dag_form(parse_layer_yaml("8_output:\n  fixed_axes: {}", "l8"), "l8"))
    expected = {"recipe_yaml_full", "recipe_hash", "package_version", "python_version", "dependency_lockfile", "data_revision_tag", "random_seed_used", "runtime_environment", "runtime_duration", "cell_resolved_axes"}
    assert expected <= set(resolved["provenance_fields"])


def test_l8_descriptive_naming_default():
    resolved = resolve_axes(normalize_to_dag_form(parse_layer_yaml("8_output:\n  fixed_axes: {}", "l8"), "l8"))
    assert resolved["naming_convention"] == "descriptive"
    assert resolved["leaf_config"]["descriptive_naming_template"] == "{model_family}_{forecast_strategy}_h{horizon}"


def test_l8_custom_naming_requires_callable_path():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    naming_convention: custom\n", "l8")
    assert validate_layer(layer).has_hard_errors


def test_l8_descriptive_template_invalid_placeholder():
    recipe = parse_recipe_yaml("8_output:\n  fixed_axes:\n    naming_convention: descriptive\n  leaf_config:\n    descriptive_naming_template: \"{nonexistent_axis}_{horizon}\"\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l8_per_target_horizon_granularity():
    assert validate_layer(parse_layer_yaml("8_output:\n  fixed_axes:\n    artifact_granularity: per_target_horizon\n", "l8")).has_hard_errors is False


def test_l8_flat_granularity_with_descriptive_naming():
    assert validate_layer(parse_layer_yaml("8_output:\n  fixed_axes:\n    artifact_granularity: flat\n    naming_convention: descriptive\n", "l8")).has_hard_errors is False


def test_l8_compression_zip():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    compression: zip\n", "l8")
    assert resolve_axes(normalize_to_dag_form(layer, "l8"))["compression"] == "zip"


def test_l8_manifest_format_json_lines():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    manifest_format: json_lines\n", "l8")
    assert resolve_axes(normalize_to_dag_form(layer, "l8"))["manifest_format"] == "json_lines"


def test_l8_diagnostics_all_shortcut():
    layer = parse_layer_yaml("8_output:\n  fixed_axes:\n    saved_objects: [forecasts, diagnostics_all]\n", "l8")
    assert {"forecasts", "diagnostics_l1_5", "diagnostics_l2_5", "diagnostics_l3_5", "diagnostics_l4_5"} <= set(resolve_axes(normalize_to_dag_form(layer, "l8"))["saved_objects"])


def test_l8_paper_replication_mode_yaml_parses():
    assert validate_layer(parse_layer_yaml(_example("l8_paper_replication.yaml"), "l8")).has_hard_errors is False


def test_l8_compact_mode_yaml_parses():
    assert validate_layer(parse_layer_yaml(_example("l8_compact_mode.yaml"), "l8")).has_hard_errors is False


def test_l8_latex_paper_export_yaml_parses():
    assert validate_layer(parse_layer_yaml(_example("l8_latex_paper.yaml"), "l8")).has_hard_errors is False


def test_l8_state_metrics_inactive_without_fred_sd():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    dataset: fred_md\n8_output:\n  fixed_axes:\n    saved_objects: [forecasts, metrics, ranking, state_metrics]\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l8_regime_metrics_inactive_without_regime():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    regime_definition: none\n8_output:\n  fixed_axes:\n    saved_objects: [forecasts, metrics, ranking, regime_metrics]\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l8_combination_weights_inactive_without_ensemble():
    root = make_recipe_without_ensemble()
    root["8_output"]["fixed_axes"]["saved_objects"] = ["forecasts", "metrics", "combination_weights"]
    assert validate_recipe(parse_recipe_yaml(root)).has_hard_errors


def test_l8_dependency_lockfile_includes_uv_lock():
    resolved = resolve_axes(parse_recipe_yaml("8_output:\n  fixed_axes: {}").layers["l8"].dag)
    assert "dependency_lockfile" in resolved["provenance_fields"]


def test_l8_r_version_auto_populated_for_r_models():
    resolved = resolve_axes(parse_recipe_yaml(make_recipe_with_glmboost()).layers["l8"].dag)
    assert "r_version" in resolved["provenance_fields"]


def test_l8_recipe_hash_in_manifest():
    resolved = resolve_axes(parse_recipe_yaml("8_output:\n  fixed_axes: {}").layers["l8"].dag)
    assert "recipe_hash" in resolved["provenance_fields"]


def test_l8_runtime_environment_field():
    from macrocast.core.types import RuntimeEnvironment

    assert {"os_name", "python_version", "cpu_info"} <= set(RuntimeEnvironment.__dataclass_fields__)


def test_l8_output_directory_default():
    resolved = resolve_axes(parse_recipe_yaml("8_output:\n  fixed_axes: {}").layers["l8"].dag)
    output_dir = resolved["leaf_config"]["output_directory"]
    assert output_dir.startswith("./macrocast_output/")
    assert "/macrocast_output/" in output_dir


def test_l8_registered_with_spec_correct_class():
    from macrocast.core.layers.registry import get_layer
    from macrocast.core.layers.l8 import L8Output

    spec = get_layer("l8")
    assert spec.cls is L8Output
    assert spec.produces == ("l8_artifacts_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "consumption"


def test_l8_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l8" in LAYER_SINKS
    assert "l8_artifacts_v1" in LAYER_SINKS["l8"]


def test_l8_sub_layer_count():
    from macrocast.core.layers.registry import get_layer

    assert len(get_layer("l8").cls.sub_layers) == 4


def test_l8_axis_count_per_sub_layer():
    from macrocast.core.layers.registry import get_layer

    counts = {"L8_A_export_format": 2, "L8_B_saved_objects": 2, "L8_C_provenance": 2, "L8_D_artifact_granularity": 2}
    for sub_name, expected_count in counts.items():
        assert len(get_layer("l8").cls.sub_layers[sub_name].axes) == expected_count


def test_l8_total_axis_count_8():
    from macrocast.core.layers.registry import get_layer

    assert sum(len(sub.axes) for sub in get_layer("l8").cls.sub_layers.values()) == 8


def test_l8_inverse_msfe_alias_unchanged():
    methods = get_op("weighted_average_forecast").params_schema["weights_method"]["options"]
    assert "dmsfe" in methods
    assert "inverse_msfe" in methods
