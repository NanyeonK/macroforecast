from macroforecast.core.layers.l4_5 import (
    normalize_to_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    validate_layer,
    validate_recipe,
)
from macroforecast.core.layers.registry import get_layer


def test_l4_5_disabled_by_default():
    recipe = parse_recipe_yaml("")
    assert "l4_5" not in recipe.layers or recipe.layers["l4_5"].enabled is False


def test_l4_5_minimal_enabled_parses():
    yaml_text = "4_5_generator_diagnostics:\n  enabled: true\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text, "l4_5")
    assert validate_layer(layer).has_hard_errors is False
    resolved = resolve_axes(normalize_to_dag_form(layer, "l4_5"))
    assert resolved["fit_view"] == "multi"
    assert resolved["fit_per_origin"] == "last_origin_only"
    assert resolved["forecast_scale_view"] == "both_overlay"
    assert resolved["window_view"] == "multi"


def test_l4_5_axes_not_sweepable():
    yaml_text = """
    4_5_generator_diagnostics:
      enabled: true
      fixed_axes:
        fit_view: {sweep: [multi, residual_time]}
    """
    layer = parse_layer_yaml(yaml_text, "l4_5")
    assert validate_layer(layer).has_hard_errors


def test_l4_5_fit_n_origins_step_required_for_every_n_origins():
    yaml_text = """
    4_5_generator_diagnostics:
      enabled: true
      fixed_axes:
        fit_per_origin: every_n_origins
    """
    layer = parse_layer_yaml(yaml_text, "l4_5")
    assert validate_layer(layer).has_hard_errors is False


def test_l4_5_back_transform_method_manual_requires_function():
    yaml_text = """
    4_5_generator_diagnostics:
      enabled: true
      fixed_axes:
        back_transform_method: manual_function
    """
    layer = parse_layer_yaml(yaml_text, "l4_5")
    assert validate_layer(layer).has_hard_errors


def test_l4_5_coef_view_models_inactive_without_linear_model():
    yaml_text = """
    4_forecasting_model:
      nodes:
        - {id: fit_xgb, type: step, op: fit_model, params: {family: xgboost}}
      sinks: {}
    4_5_generator_diagnostics:
      enabled: true
      fixed_axes:
        coef_view_models: all_linear_models
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l4_5_tuning_view_inactive_without_search():
    yaml_text = """
    4_forecasting_model:
      nodes:
        - {id: fit_xgb, type: step, op: fit_model, params: {family: xgboost, search_algorithm: none}}
      sinks: {}
    4_5_generator_diagnostics:
      enabled: true
      fixed_axes:
        tuning_view: multi
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l4_5_ensemble_view_inactive_without_ensemble():
    yaml_text = """
    4_forecasting_model:
      nodes:
        - {id: fit_ridge, type: step, op: fit_model, params: {family: ridge}}
        - {id: predict_ridge, type: step, op: predict}
      sinks: {}
    4_5_generator_diagnostics:
      enabled: true
      fixed_axes:
        ensemble_view: multi
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l4_5_weights_over_time_method_inactive_without_weights_view():
    yaml_text = """
    4_5_generator_diagnostics:
      enabled: true
      fixed_axes:
        ensemble_view: weight_concentration
        weights_over_time_method: stacked_area
    """
    layer = parse_layer_yaml(yaml_text, "l4_5")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l4_5"))
    assert resolved.get_active("weights_over_time_method") is False


def test_l4_5_z_diagnostic_format_default():
    layer = parse_layer_yaml("4_5_generator_diagnostics:\n  enabled: true", "l4_5")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l4_5"))
    assert resolved["diagnostic_format"] == "pdf"


def test_l4_5_registered_with_spec_correct_class():
    spec = get_layer("l4_5")
    from macroforecast.core.layers.l4_5 import L4_5GeneratorDiagnostics

    assert spec.cls is L4_5GeneratorDiagnostics
    assert spec.produces == ("l4_5_diagnostic_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "diagnostic"


def test_l4_5_sink_in_layer_sinks():
    from macroforecast.core.types import LAYER_SINKS

    assert "l4_5" in LAYER_SINKS
    assert "l4_5_diagnostic_v1" in LAYER_SINKS["l4_5"]


def test_l4_5_sub_layer_count():
    layer_class = get_layer("l4_5").cls
    assert len(layer_class.sub_layers) == 6
