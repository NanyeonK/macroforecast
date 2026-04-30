from macrocast.core.layers.l5 import (
    make_l5_yaml,
    make_recipe_with_benchmark,
    make_recipe_with_l3_metadata,
    make_recipe_without_benchmark,
    normalize_to_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    validate_layer,
    validate_recipe,
)
from macrocast.core.ops import get_op


def test_l5_minimal_yaml_parses_to_defaults():
    layer = parse_layer_yaml("5_evaluation:\n  fixed_axes: {}", "l5")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l5"))
    assert resolved["primary_metric"] == "mse"
    assert resolved["point_metrics"] == ["mse", "mae"]
    assert resolved["agg_time"] == "mean"
    assert resolved["agg_horizon"] == "per_horizon_separate"
    assert resolved["oos_period"] == "full_oos"
    assert resolved["decomposition_target"] == "none"
    assert resolved["ranking"] == "by_primary_metric"
    assert resolved["report_style"] == "single_table"


def test_l5_relative_metric_requires_benchmark():
    recipe = parse_recipe_yaml(make_l5_yaml(primary_metric="relative_mse"))
    assert validate_recipe(recipe).has_hard_errors


def test_l5_log_score_requires_density_forecast():
    recipe = parse_recipe_yaml(make_l5_yaml(primary_metric="log_score"))
    assert validate_recipe(recipe).has_hard_errors


def test_l5_density_metrics_gated_inactive_for_point():
    recipe = parse_recipe_yaml("5_evaluation:\n  fixed_axes:\n    density_metrics: [log_score]\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_axes_not_sweepable():
    layer = parse_layer_yaml("5_evaluation:\n  fixed_axes:\n    primary_metric: {sweep: [mse, mae]}\n", "l5")
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("sweep" in i.message.lower() for i in report.hard_errors)


def test_l5_relative_metrics_default_when_benchmark_present():
    recipe = make_recipe_with_benchmark()
    resolved = resolve_axes(recipe.layers["l5"].dag)
    assert resolved["relative_metrics"] == ["relative_mse", "r2_oos"]


def test_l5_relative_metrics_inactive_without_benchmark():
    recipe = parse_recipe_yaml(make_recipe_without_benchmark())
    resolved = resolve_axes(recipe.layers["l5"].dag)
    assert resolved.get_active("relative_metrics") is False


def test_l5_agg_target_inactive_for_single_target():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    target_structure: single_target\n5_evaluation:\n  fixed_axes:\n    agg_target: per_target_separate\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_agg_state_inactive_without_fred_sd():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    dataset: fred_md\n5_evaluation:\n  fixed_axes:\n    agg_state: per_state_separate\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_decomposition_by_state_requires_fred_sd():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    dataset: fred_md\n5_evaluation:\n  fixed_axes:\n    decomposition_target: by_state\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_decomposition_by_regime_requires_regime_active():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    regime_definition: none\n5_evaluation:\n  fixed_axes:\n    decomposition_target: by_regime\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_decomposition_order_inactive_when_target_none():
    layer = parse_layer_yaml("5_evaluation:\n  fixed_axes:\n    decomposition_target: none\n    decomposition_order: shapley\n", "l5")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l5"))
    assert resolved.get_active("decomposition_order") is False


def test_l5_regime_use_inactive_when_regime_none():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    regime_definition: none\n5_evaluation:\n  fixed_axes:\n    regime_use: per_regime\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_regime_metrics_inactive_when_regime_use_pooled():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    regime_definition: external_nber\n5_evaluation:\n  fixed_axes:\n    regime_use: pooled\n    regime_metrics: [mse]\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_oos_period_fixed_dates_requires_dates():
    assert validate_layer(parse_layer_yaml("5_evaluation:\n  fixed_axes:\n    oos_period: fixed_dates\n", "l5")).has_hard_errors


def test_l5_subperiod_requires_list():
    assert validate_layer(parse_layer_yaml("5_evaluation:\n  fixed_axes:\n    oos_period: multiple_subperiods\n", "l5")).has_hard_errors


def test_l5_ranking_mcs_inclusion_requires_l6_mcs():
    recipe = parse_recipe_yaml("6_statistical_tests:\n  enabled: false\n5_evaluation:\n  fixed_axes:\n    ranking: mcs_inclusion\n")
    assert validate_recipe(recipe).has_hard_errors


def test_l5_ranking_by_relative_metric_requires_benchmark():
    root = make_recipe_without_benchmark()
    root["5_evaluation"]["fixed_axes"]["ranking"] = "by_relative_metric"
    recipe = parse_recipe_yaml(root)
    assert validate_recipe(recipe).has_hard_errors


def test_l5_latex_table_requires_caption_and_label():
    assert validate_layer(parse_layer_yaml("5_evaluation:\n  fixed_axes:\n    report_style: latex_table\n", "l5")).has_hard_errors


def test_l5_msfe_renamed_to_mse():
    report = validate_layer(parse_layer_yaml("5_evaluation:\n  fixed_axes:\n    primary_metric: msfe\n", "l5"))
    assert report.has_hard_errors


def test_l5_dmsfe_kept_as_separate_combine_method():
    weights_methods = get_op("weighted_average_forecast").params_schema["weights_method"]["options"]
    assert "dmsfe" in weights_methods


def test_l5_decomposition_by_predictor_block_uses_l3_metadata():
    root = make_recipe_with_l3_metadata()
    root["5_evaluation"]["fixed_axes"]["decomposition_target"] = "by_predictor_block"
    recipe = parse_recipe_yaml(root)
    sources = [node for node in recipe.layers["l5"].dag.nodes.values() if node.type == "source"]
    assert "l3_metadata_v1" in [source.selector.sink_name for source in sources]


def test_l5_economic_metrics_does_not_exist():
    from macrocast.core.layers.registry import get_layer

    assert "economic_metrics" not in get_layer("l5").cls.list_axes()


def test_l5_registered_with_spec_correct_class():
    from macrocast.core.layers.registry import get_layer
    from macrocast.core.layers.l5 import L5Evaluation

    spec = get_layer("l5")
    assert spec.cls is L5Evaluation
    assert spec.produces == ("l5_evaluation_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "consumption"


def test_l5_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l5" in LAYER_SINKS
    assert "l5_evaluation_v1" in LAYER_SINKS["l5"]


def test_l5_evaluation_artifact_records_resolved_axes():
    from macrocast.core.types import L5EvaluationArtifact

    assert "l5_axis_resolved" in L5EvaluationArtifact.__dataclass_fields__


def test_l5_axis_count():
    from macrocast.core.layers.registry import get_layer

    expected_axes = {
        "primary_metric", "point_metrics", "density_metrics",
        "direction_metrics", "relative_metrics",
        "benchmark_window", "benchmark_scope",
        "agg_time", "agg_horizon", "agg_target", "agg_state",
        "oos_period", "regime_use", "regime_metrics",
        "decomposition_target", "decomposition_order",
        "ranking", "report_style",
    }
    assert set(get_layer("l5").cls.list_axes()) == expected_axes
