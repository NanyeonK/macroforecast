from macrocast.core.layers.l3_5 import (
    normalize_to_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    validate_layer,
    validate_recipe,
)
from macrocast.core.layers.registry import get_layer


def test_l3_5_disabled_by_default():
    recipe = parse_recipe_yaml("")
    assert "l3_5" not in recipe.layers or recipe.layers["l3_5"].enabled is False


def test_l3_5_minimal_enabled_parses():
    yaml_text = "3_5_feature_diagnostics:\n  enabled: true\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text, "l3_5")
    assert validate_layer(layer).has_hard_errors is False
    resolved = resolve_axes(normalize_to_dag_form(layer, "l3_5"))
    assert resolved["comparison_stages"] == "cleaned_vs_features"
    assert resolved["comparison_output_form"] == "multi"
    assert resolved["feature_correlation"] == "cross_block"


def test_l3_5_axes_not_sweepable():
    yaml_text = """
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        comparison_stages: {sweep: [cleaned_vs_features, raw_vs_cleaned_vs_features]}
    """
    layer = parse_layer_yaml(yaml_text, "l3_5")
    assert validate_layer(layer).has_hard_errors


def test_l3_5_factor_view_inactive_without_factor_step():
    yaml_text = """
    3_feature_engineering:
      nodes:
        - {id: src_X, type: source}
        - {id: src_y, type: source}
        - {id: lag_x, type: step, op: lag, params: {n_lag: 4}, inputs: [src_X]}
        - {id: y_h, type: step, op: target_construction}
      sinks: {}
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        factor_view: multi
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l3_5_dfm_diagnostics_inactive_without_dfm():
    yaml_text = """
    3_feature_engineering:
      nodes:
        - {id: pca_x, type: step, op: pca, params: {n_components: 4}}
      sinks: {}
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        dfm_diagnostics: multi
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l3_5_lag_view_inactive_without_lag_step():
    yaml_text = """
    3_feature_engineering:
      nodes:
        - {id: pca_x, type: step, op: pca, params: {n_components: 8}, inputs: [src_X]}
      sinks: {}
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        lag_view: multi
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l3_5_marx_view_inactive_without_ma_increasing_order():
    yaml_text = """
    3_feature_engineering:
      nodes:
        - {id: lag_x, type: step, op: lag, params: {n_lag: 4}}
      sinks: {}
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        marx_view: weight_decay_visualization
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l3_5_selection_view_inactive_without_feature_selection():
    yaml_text = """
    3_feature_engineering:
      nodes:
        - {id: pca_x, type: step, op: pca, params: {n_components: 8}}
      sinks: {}
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        selection_view: multi
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l3_5_correlation_method_inactive_when_no_correlation():
    yaml_text = """
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        feature_correlation: none
        correlation_method: spearman
    """
    layer = parse_layer_yaml(yaml_text, "l3_5")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l3_5"))
    assert resolved.get_active("correlation_method") is False


def test_l3_5_stability_metric_inactive_without_selection_stability():
    yaml_text = """
    3_5_feature_diagnostics:
      enabled: true
      fixed_axes:
        selection_view: selected_list
        stability_metric: jaccard
    """
    layer = parse_layer_yaml(yaml_text, "l3_5")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l3_5"))
    assert resolved.get_active("stability_metric") is False


def test_l3_5_z_diagnostic_format_default():
    layer = parse_layer_yaml("3_5_feature_diagnostics:\n  enabled: true", "l3_5")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l3_5"))
    assert resolved["diagnostic_format"] == "pdf"


def test_l3_5_registered_with_spec_correct_class():
    spec = get_layer("l3_5")
    from macrocast.core.layers.l3_5 import L3_5FeatureDiagnostics

    assert spec.cls is L3_5FeatureDiagnostics
    assert spec.produces == ("l3_5_diagnostic_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "diagnostic"


def test_l3_5_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l3_5" in LAYER_SINKS
    assert "l3_5_diagnostic_v1" in LAYER_SINKS["l3_5"]


def test_l3_5_sub_layer_count():
    layer_class = get_layer("l3_5").cls
    assert len(layer_class.sub_layers) == 6
