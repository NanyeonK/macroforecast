from pathlib import Path

import pytest

from macrocast.core.layers.l7 import (
    execute_layer,
    make_l7_yaml,
    make_l7_yaml_with_lineage_attribution,
    normalize_to_dag_form,
    parse_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    validate_layer,
    validate_recipe,
)
from macrocast.core.ops import get_op, list_ops
from macrocast.core.ops.l7_ops import FIGURE_TYPES, PRE_DEFINED_BLOCKS
from macrocast.core.validator import validate_dag

ROOT = Path(__file__).resolve().parents[2]


def _example(name: str) -> str:
    return (ROOT / "examples" / "recipes" / name).read_text()


def test_l7_disabled_by_default():
    layer = parse_layer_yaml("7_interpretation:\n  enabled: false", "l7")
    assert resolve_axes(normalize_to_dag_form(layer, "l7"))["enabled"] is False


def test_l7_minimal_shap_parses():
    layer = parse_layer_yaml(_example("l7_minimal_shap.yaml"), "l7")
    assert validate_dag(parse_dag_form(layer["nodes"])).valid


def test_l7_multi_method_parses():
    assert validate_layer(parse_layer_yaml(_example("l7_multi_method.yaml"), "l7")).has_hard_errors is False


def test_l7_coulombe_groups_parses():
    assert validate_layer(parse_layer_yaml(_example("l7_coulombe_groups.yaml"), "l7")).has_hard_errors is False


def test_l7_transformation_attribution_parses():
    assert validate_layer(parse_layer_yaml(_example("l7_transformation_attribution.yaml"), "l7")).has_hard_errors is False


def test_l7_temporal_parses():
    assert validate_layer(parse_layer_yaml(_example("l7_temporal.yaml"), "l7")).has_hard_errors is False


def test_l7_shap_tree_rejects_linear_model():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="shap_tree", model_family="ridge"), "l7")).has_hard_errors


def test_l7_shap_linear_rejects_tree_model():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="shap_linear", model_family="xgboost"), "l7")).has_hard_errors


def test_l7_shap_deep_requires_nn_model():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="shap_deep", model_family="ridge"), "l7")).has_hard_errors


def test_l7_shap_kernel_works_on_any_model():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="shap_kernel", model_family="xgboost"), "l7")).has_hard_errors is False


def test_l7_shap_kernel_works_on_linear():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="shap_kernel", model_family="ridge"), "l7")).has_hard_errors is False


def test_l7_mrf_gtvp_only_for_mrf():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="mrf_gtvp", model_family="xgboost"), "l7")).has_hard_errors


def test_l7_gradient_methods_reject_tree_model():
    for op in ["integrated_gradients", "saliency_map", "deep_lift", "gradient_shap"]:
        assert validate_layer(parse_layer_yaml(make_l7_yaml(op=op, model_family="xgboost"), "l7")).has_hard_errors


def test_l7_var_specific_ops_reject_non_var():
    for op in ["fevd", "historical_decomposition", "generalized_irf"]:
        assert validate_layer(parse_layer_yaml(make_l7_yaml(op=op, model_family="ridge"), "l7")).has_hard_errors


def test_l7_lasso_inclusion_rejects_xgboost():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="lasso_inclusion_frequency", model_family="xgboost"), "l7")).has_hard_errors


def test_l7_bvar_pip_rejects_non_bvar():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="bvar_pip", model_family="var"), "l7")).has_hard_errors


def test_l7_mccracken_ng_md_groups_requires_fred_md():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    dataset: fred_qd\n" + make_l7_yaml(op="group_aggregate", model_family="xgboost").replace("params: {model_family: xgboost}", "params: {grouping: mccracken_ng_md_groups, aggregation: sum}"))
    assert validate_recipe(recipe).has_hard_errors


def test_l7_fred_sd_states_requires_fred_sd():
    recipe = parse_recipe_yaml("1_data:\n  fixed_axes:\n    dataset: fred_md\n" + make_l7_yaml(op="group_aggregate").replace("params: {model_family: xgboost}", "params: {grouping: fred_sd_states, aggregation: sum}"))
    assert validate_recipe(recipe).has_hard_errors


def test_l7_user_groups_works_in_leaf_config():
    layer = parse_layer_yaml(_example("l7_coulombe_groups.yaml"), "l7")
    user_groups = resolve_axes(normalize_to_dag_form(layer, "l7"))["leaf_config"]["user_groups"]
    assert "real_activity" in user_groups
    assert "INDPRO" in user_groups["real_activity"]


def test_l7_theme_block_missing_series_hard_error():
    yaml_text = """
1_data:
  fixed_axes:
    variable_universe: explicit_variable_list
  leaf_config:
    variable_universe_columns: [INDPRO, PAYEMS]
7_interpretation:
  enabled: true
  nodes:
    - {id: shap_taylor, type: step, op: group_aggregate, params: {grouping: taylor_rule_block, aggregation: sum}, inputs: [shap]}
  sinks:
    l7_importance_v1: {group: shap_taylor}
"""
    assert validate_recipe(parse_recipe_yaml(yaml_text)).has_hard_errors


def test_l7_mcs_inclusion_requires_l6_mcs_active():
    yaml_text = """
6_statistical_tests:
  enabled: true
  sub_layers:
    L6_D_multiple_model: {enabled: false}
7_interpretation:
  enabled: true
  nodes:
    - {id: src_mcs, type: source, selector: {layer_ref: l6, sink_name: l6_tests_v1, subset: {family: multiple_model, name: mcs_inclusion}}}
    - {id: shap_filtered, type: step, op: shap_tree, params: {model_family: xgboost}, inputs: [src_mcs]}
  sinks:
    l7_importance_v1: {global: shap_filtered}
"""
    assert validate_recipe(parse_recipe_yaml(yaml_text)).has_hard_errors


def test_l7_lineage_attribution_requires_l3_metadata():
    layer = parse_layer_yaml(make_l7_yaml_with_lineage_attribution(), "l7")
    sink_refs = [source["selector"]["sink_name"] for source in layer["nodes"] if source.get("type") == "source"]
    assert "l3_metadata_v1" in sink_refs


def test_l7_attention_weights_rejected_as_future():
    report = validate_layer(parse_layer_yaml(make_l7_yaml(op="attention_weights", model_family="transformer"), "l7"))
    assert report.has_hard_errors
    assert any("future" in issue.message.lower() for issue in report.hard_errors)


def test_l7_lstm_hidden_state_rejected_as_future():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="lstm_hidden_state", model_family="lstm"), "l7")).has_hard_errors


def test_l7_boruta_selection_rejected_as_future():
    assert validate_layer(parse_layer_yaml(make_l7_yaml(op="boruta_selection"), "l7")).has_hard_errors


def test_l7_mrf_gtvp_schema_valid_runtime_stub():
    layer = parse_layer_yaml(make_l7_yaml(op="mrf_gtvp", model_family="macroeconomic_random_forest"), "l7")
    assert validate_layer(layer).has_hard_errors is False
    with pytest.raises(NotImplementedError):
        execute_layer(layer)


def test_l7_default_figure_mapping_for_shap_tree():
    assert get_op("shap_tree").default_figure_type == ["beeswarm", "force_plot"]


def test_l7_default_figure_mapping_for_partial_dependence():
    assert get_op("partial_dependence").default_figure_type == "pdp_line"


def test_l7_default_figure_mapping_for_transformation_attribution():
    assert get_op("transformation_attribution").default_figure_type == "shapley_waterfall"


def test_l7_figure_type_auto_uses_step_default():
    layer = parse_layer_yaml(_example("l7_minimal_shap.yaml"), "l7")
    assert resolve_axes(normalize_to_dag_form(layer, "l7"))["figure_type"] == "auto"


def test_l7_figure_type_override():
    layer = parse_layer_yaml("7_interpretation:\n  enabled: true\n  nodes:\n    - {id: shap, type: step, op: shap_tree, params: {model_family: xgboost}, inputs: []}\n  sinks:\n    l7_importance_v1: {global: shap}\n  fixed_axes:\n    figure_type: heatmap\n", "l7")
    assert resolve_axes(normalize_to_dag_form(layer, "l7"))["figure_type"] == "heatmap"


def test_l7_top_k_features_default():
    layer = parse_layer_yaml("7_interpretation:\n  enabled: true\n  nodes: []\n", "l7")
    assert resolve_axes(normalize_to_dag_form(layer, "l7"))["top_k_features_to_show"] == 20


def test_l7_group_aggregate_sums_correctly():
    assert "sum" in get_op("group_aggregate").params_schema["aggregation"]["options"]


def test_l7_lineage_attribution_levels():
    levels = get_op("lineage_attribution").params_schema["level"]["options"]
    assert "pipeline_name" in levels
    assert "step_op" in levels
    assert "source_node" in levels


def test_l7_rolling_recompute_window_options():
    windows = get_op("rolling_recompute").params_schema["window"]["options"]
    assert "expanding" in windows
    assert "rolling" in windows


def test_l7_two_sinks_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS
    assert "l7_importance_v1" in LAYER_SINKS["l7"]
    assert "l7_transformation_attribution_v1" in LAYER_SINKS["l7"]


def test_l7_transformation_attribution_sink_empty_when_step_unused():
    sinks = parse_layer_yaml(_example("l7_minimal_shap.yaml"), "l7")["sinks"]
    assert "l7_importance_v1" in sinks
    assert "l7_transformation_attribution_v1" not in sinks or sinks["l7_transformation_attribution_v1"] is None


def test_l7_29_operational_ops_registered():
    operational = [op for op in list_ops().values() if "l7" in op.layer_scope and op.status == "operational"]
    assert len(operational) == 29


def test_l7_6_future_ops_registered():
    future_ops = [op for op in list_ops().values() if "l7" in op.layer_scope and op.status == "future"]
    assert len(future_ops) == 6


def test_l7_18_figure_types():
    assert len(FIGURE_TYPES) == 18


def test_l7_7_pre_defined_blocks():
    expected = {"mccracken_ng_md_groups", "mccracken_ng_qd_groups", "fred_sd_states", "nber_real_activity", "taylor_rule_block", "term_structure_block", "credit_spread_block", "financial_conditions_block"}
    assert set(PRE_DEFINED_BLOCKS.keys()) == expected


def test_l7_registered_with_spec_correct_class():
    from macrocast.core.layers.registry import get_layer
    from macrocast.core.layers.l7 import L7Interpretation
    spec = get_layer("l7")
    assert spec.cls is L7Interpretation
    assert "l7_importance_v1" in spec.produces
    assert "l7_transformation_attribution_v1" in spec.produces
    assert spec.ui_mode == "graph"
    assert spec.category == "consumption"


def test_l7_axes_not_sweepable():
    layer = parse_layer_yaml("7_interpretation:\n  enabled: true\n  nodes: []\n  fixed_axes:\n    figure_type: {sweep: [bar_global, beeswarm]}\n", "l7")
    assert validate_layer(layer).has_hard_errors


def test_l7_does_not_register_forecast_combination_ops():
    for op_name in ["weighted_average_forecast", "median_forecast", "trimmed_mean_forecast", "bma_forecast", "bivariate_ardl_combination"]:
        assert "l7" not in get_op(op_name).layer_scope
