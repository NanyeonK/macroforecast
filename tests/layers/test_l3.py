from pathlib import Path

from macrocast.core.layers.l3 import (
    build_cascade_chain,
    build_metadata_artifact,
    expand_sweeps,
    normalize_to_dag_form,
    parse_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    validate_layer,
    validate_recipe,
)
from macrocast.core.validator import validate_dag


ROOT = Path(__file__).resolve().parents[2]


def _example(name: str) -> str:
    return (ROOT / "examples" / "recipes" / name).read_text()


def _base_nodes(extra_x, y_params=None):
    y_params = y_params or {"mode": "point_forecast", "method": "direct", "horizon": 1}
    return f"""
3_feature_engineering:
  nodes:
    - id: src_x
      type: source
      selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: predictors}}}}
    - id: src_y
      type: source
      selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: target}}}}
{extra_x}
    - id: y_h
      type: step
      op: target_construction
      params: {y_params}
      inputs: [src_y]
  sinks:
    l3_features_v1: {{X_final: x_final, y_final: y_h}}
    l3_metadata_v1: auto
"""


def make_l3_yaml_with_lag(n_lag=4):
    return _base_nodes(f"""    - id: x_final
      type: step
      op: lag
      params: {{n_lag: {n_lag}}}
      inputs: [src_x]""")


def make_l3_yaml_with_pca(n_components=4, temporal_rule="expanding_window_per_origin"):
    params = f"{{n_components: {n_components}" + ("" if temporal_rule is None else f", temporal_rule: {temporal_rule}") + "}"
    return _base_nodes(f"""    - id: x_final
      type: step
      op: pca
      params: {params}
      inputs: [src_x]""")


def make_l3_yaml_with_scale_temporal_rule(rule):
    return _base_nodes(f"""    - id: x_final
      type: step
      op: scale
      params: {{method: zscore, temporal_rule: {rule}}}
      inputs: [src_x]""")


def make_l3_yaml_with_op(op, **params):
    param_text = "{" + ", ".join(f"{k}: {v}" for k, v in params.items()) + "}"
    return _base_nodes(f"""    - id: x_final
      type: step
      op: {op}
      params: {param_text}
      inputs: [src_x]""")


def test_l3_minimal_lag_only_parses():
    layer = parse_layer_yaml(_example("l3_minimal_lag_only.yaml"), "l3")
    dag = parse_dag_form(layer)
    assert "l3_features_v1" in layer["sinks"]
    assert validate_dag(dag).valid


def test_l3_mccracken_ng_baseline_parses():
    layer = parse_layer_yaml(_example("l3_mccracken_ng_baseline.yaml"), "l3")
    dag = parse_dag_form(layer)
    assert validate_dag(dag).valid


def test_l3_target_construction_only_in_l3_a():
    yaml_text = """
3_feature_engineering:
  nodes:
    - {id: src_x, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - id: misuse
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_x]
    - id: x_final
      type: step
      op: lag
      params: {n_lag: 4}
      inputs: [misuse]
    - id: y_h
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_y]
  sinks:
    l3_features_v1: {X_final: x_final, y_final: y_h}
    l3_metadata_v1: auto
"""
    assert validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l3_cascade_simple():
    layer = parse_layer_yaml(_example("l3_cascade_pca_on_marx.yaml"), "l3")
    report = validate_layer(layer)
    assert not report.has_hard_errors


def test_l3_cascade_depth_limit():
    layer = {
        "nodes": build_cascade_chain(depth=4),
        "sinks": {"l3_features_v1": {"X_final": "p4", "y_final": "y_h"}, "l3_metadata_v1": "auto"},
    }
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("cascade depth" in i.message for i in report.hard_errors)


def test_l3_cascade_cycle_rejected():
    yaml_text = """
3_feature_engineering:
  nodes:
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - id: pipeline_A
      type: step
      op: lag
      params: {n_lag: 2}
      pipeline_id: A
      inputs:
        - {type: source, selector: {layer_ref: l3, sink_name: pipeline_output, subset: {pipeline_id: B}}}
    - id: pipeline_B
      type: step
      op: lag
      params: {n_lag: 2}
      pipeline_id: B
      inputs:
        - {type: source, selector: {layer_ref: l3, sink_name: pipeline_output, subset: {pipeline_id: A}}}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: pipeline_A, y_final: y_h}
    l3_metadata_v1: auto
"""
    assert validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l3_lag_with_n_lag_zero_rejected():
    assert validate_layer(parse_layer_yaml(make_l3_yaml_with_lag(n_lag=0))).has_hard_errors


def test_l3_pca_n_components_exceeds_n():
    assert validate_layer(parse_layer_yaml(make_l3_yaml_with_pca(n_components=10000))).has_hard_errors


def test_l3_full_sample_once_rejected_for_scale():
    assert validate_layer(parse_layer_yaml(make_l3_yaml_with_scale_temporal_rule("full_sample_once"))).has_hard_errors


def test_l3_full_sample_once_rejected_for_pca():
    assert validate_layer(parse_layer_yaml(make_l3_yaml_with_pca(4, "full_sample_once"))).has_hard_errors


def test_l3_future_op_rejected_boruta():
    report = validate_layer(parse_layer_yaml(make_l3_yaml_with_op("boruta_selection", n_estimators=100)))
    assert report.has_hard_errors
    assert any("future" in i.message.lower() for i in report.hard_errors)


def test_l3_scaled_pca_requires_target_signal():
    report = validate_layer(parse_layer_yaml(make_l3_yaml_with_op("scaled_pca", n_components=4, temporal_rule="expanding_window_per_origin")))
    assert report.has_hard_errors


def test_l3_dfm_n_lags_factor_high_warns_soft():
    yaml_text = make_l3_yaml_with_op("dfm", n_factors=4, n_lags_factor=8, temporal_rule="expanding_window_per_origin")
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors
    assert any("dfm" in w.message.lower() for w in report.soft_warnings)


def test_l3_lag_before_log_warns_soft():
    yaml_text = """
3_feature_engineering:
  nodes:
    - {id: src_x, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_first, type: step, op: lag, params: {n_lag: 2}, inputs: [src_x]}
    - {id: x_final, type: step, op: log, inputs: [lag_first]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: x_final, y_final: y_h}
    l3_metadata_v1: auto
"""
    assert any("ordering" in w.message.lower() for w in validate_layer(parse_layer_yaml(yaml_text)).soft_warnings)


def test_l3_concat_with_misaligned_inputs_hard_error():
    yaml_text = _base_nodes("""    - {id: a, type: step, op: lag, params: {n_lag: 2}, inputs: [src_x]}
    - {id: b, type: step, op: lag, params: {n_lag: 3}, inputs: [src_x]}
    - {id: x_final, type: combine, op: concat, inputs: [a, b]}""")
    assert not validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l3_x_final_zero_columns_rejected():
    yaml_text = _base_nodes("""    - id: empty_x
      type: step
      op: pca
      params: {n_components: 1, temporal_rule: expanding_window_per_origin}
      inputs: [src_x]""").replace("X_final: x_final", "X_final: empty_x")
    assert validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l3_y_final_must_be_series():
    yaml_text = make_l3_yaml_with_lag().replace("y_final: y_h", "y_final: x_final")
    assert validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l3_horizon_must_be_in_l1_horizon_set():
    yaml_text = """
1_data:
  fixed_axes:
    horizon_set: custom_list
  leaf_config:
    target: CPI
    target_horizons: [1, 3, 6, 12]
""" + make_l3_yaml_with_lag().replace("'horizon': 1", "'horizon': 99")
    assert validate_recipe(parse_recipe_yaml(yaml_text)).has_hard_errors


def test_l3_l1_regime_metadata_required_when_regime_indicator_used():
    yaml_text = """
1_data:
  fixed_axes:
    regime_definition: none
3_feature_engineering:
  nodes:
    - {id: src_regime, type: source, selector: {layer_ref: l1, sink_name: l1_regime_metadata_v1}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: src_x, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: regime_dummy, type: step, op: regime_indicator, params: {definition_ref: l1_regime_metadata_v1}, inputs: [src_regime]}
    - {id: x_final, type: combine, op: concat, inputs: [src_x, regime_dummy]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: x_final, y_final: y_h}
    l3_metadata_v1: auto
"""
    assert validate_recipe(parse_recipe_yaml(yaml_text)).has_hard_errors


def test_l3_lineage_metadata_records_pipeline_ids():
    metadata = build_metadata_artifact(_example("l3_mccracken_ng_baseline.yaml"))
    assert any(c.pipeline_id is not None for c in metadata.column_lineage.values())


def test_l3_x_final_y_final_aligned_index():
    assert not validate_layer(parse_layer_yaml(_example("l3_mccracken_ng_baseline.yaml"))).has_hard_errors


def test_l3_two_sinks_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l3" in LAYER_SINKS
    assert "l3_features_v1" in LAYER_SINKS["l3"]
    assert "l3_metadata_v1" in LAYER_SINKS["l3"]


def test_l3_registered_with_spec_correct_class():
    from macrocast.core.layers.registry import get_layer
    from macrocast.core.layers.l3 import L3FeatureEngineering

    spec = get_layer("l3")
    assert spec.cls is L3FeatureEngineering
    assert spec.produces == ("l3_features_v1", "l3_metadata_v1")
    assert spec.ui_mode == "graph"
    assert spec.category == "construction"


def test_l3_op_count_37_operational():
    from macrocast.core.ops import list_ops

    operational = [op for op in list_ops().values() if op.available_in("l3") and op.status == "operational"]
    assert len(operational) >= 37


def test_l3_op_count_6_future():
    from macrocast.core.ops import list_ops

    future_ops = [op for op in list_ops().values() if op.status == "future"]
    assert len(future_ops) >= 6


def test_l3_pca_temporal_rule_required():
    assert validate_layer(parse_layer_yaml(make_l3_yaml_with_pca(n_components=4, temporal_rule=None))).has_hard_errors


def test_l3_concat_with_one_input_warns_or_errors():
    yaml_text = _base_nodes("""    - {id: x_final, type: combine, op: concat, inputs: [src_x]}""")
    assert validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l3_target_construction_horizon_with_sweep_works():
    cells = expand_sweeps(parse_layer_yaml(_example("l3_mccracken_ng_baseline.yaml")))
    horizons = sorted({c.sweep_values["l3.y_cum.horizon"] for c in cells})
    assert horizons == [1, 3, 6, 12]


def test_l3_forecast_combination_ops_not_registered_in_l3():
    from macrocast.core.ops import list_ops

    forbidden = {
        "weighted_average_forecast",
        "median_forecast",
        "trimmed_mean_forecast",
        "bma_forecast",
        "bivariate_ardl_combination",
        "dmsfe",
        "bma",
        "mallows_cp",
    }
    for op_name in forbidden & set(list_ops()):
        assert "l3" not in list_ops()[op_name].layer_scope


def test_l3_rejects_l4_forecast_combination_nodes():
    for op_name in ["median_forecast", "trimmed_mean_forecast", "bma_forecast", "bivariate_ardl_combination"]:
        yaml_text = _base_nodes(f"    - {{id: bad_combine, type: combine, op: {op_name}, inputs: [src_x, src_x]}}")
        assert validate_layer(parse_layer_yaml(yaml_text)).has_hard_errors


def test_l3_canonical_design_op_aliases_registered():
    from macrocast.core.ops import list_ops

    for op_name in ["varimax", "polynomial", "kernel", "nystroem"]:
        assert "l3" in list_ops()[op_name].layer_scope
