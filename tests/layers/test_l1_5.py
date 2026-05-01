from macrocast.core.layers.l1_5 import (
    normalize_to_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    validate_layer,
    validate_recipe,
)
from macrocast.core.layers.registry import get_layer


def test_l1_5_disabled_by_default():
    recipe = parse_recipe_yaml("")
    assert "l1_5" not in recipe.layers or recipe.layers["l1_5"].enabled is False


def test_l1_5_explicit_disabled():
    layer = parse_layer_yaml("1_5_data_summary:\n  enabled: false")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l1_5"))
    assert resolved["enabled"] is False


def test_l1_5_minimal_enabled_parses():
    yaml_text = "1_5_data_summary:\n  enabled: true\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text)
    assert validate_layer(layer).has_hard_errors is False
    resolved = resolve_axes(normalize_to_dag_form(layer, "l1_5"))
    assert resolved["coverage_view"] == "multi"
    assert resolved["summary_metrics"] == ["mean", "sd", "min", "max", "n_missing"]
    assert resolved["stationarity_test"] == "none"
    assert resolved["outlier_view"] == "iqr_flag"
    assert resolved["correlation_view"] == "none"


def test_l1_5_axes_not_sweepable():
    yaml_text = """
    1_5_data_summary:
      enabled: true
      fixed_axes:
        coverage_view: {sweep: [multi, observation_count]}
    """
    layer = parse_layer_yaml(yaml_text)
    assert validate_layer(layer).has_hard_errors


def test_l1_5_per_regime_split_requires_regime():
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: none
    1_5_data_summary:
      enabled: true
      fixed_axes:
        summary_split: per_regime
    """
    recipe = parse_recipe_yaml(yaml_text)
    assert validate_recipe(recipe).has_hard_errors


def test_l1_5_stationarity_test_inactive_when_none():
    yaml_text = """
    1_5_data_summary:
      enabled: true
      fixed_axes:
        stationarity_test: none
        stationarity_test_scope: target_only
    """
    layer = parse_layer_yaml(yaml_text)
    resolved = resolve_axes(normalize_to_dag_form(layer, "l1_5"))
    assert resolved.get_active("stationarity_test_scope") is False


def test_l1_5_outlier_threshold_default_iqr():
    yaml_text = """
    1_5_data_summary:
      enabled: true
      fixed_axes:
        outlier_view: iqr_flag
    """
    layer = parse_layer_yaml(yaml_text)
    resolved = resolve_axes(normalize_to_dag_form(layer, "l1_5"))
    assert resolved["leaf_config"]["outlier_threshold_iqr"] == 10.0


def test_l1_5_correlation_top_k_requires_top_k_view():
    yaml_text = """
    1_5_data_summary:
      enabled: true
      fixed_axes:
        correlation_view: top_k_per_target
    """
    layer = parse_layer_yaml(yaml_text)
    resolved = resolve_axes(normalize_to_dag_form(layer, "l1_5"))
    assert resolved["leaf_config"]["correlation_top_k"] == 20


def test_l1_5_z_diagnostic_format_default():
    layer = parse_layer_yaml("1_5_data_summary:\n  enabled: true")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l1_5"))
    assert resolved["diagnostic_format"] == "pdf"
    assert resolved["attach_to_manifest"] is True
    assert resolved["figure_dpi"] == 300


def test_l1_5_registered_with_spec_correct_class():
    spec = get_layer("l1_5")
    from macrocast.core.layers.l1_5 import L1_5DataSummary

    assert spec.cls is L1_5DataSummary
    assert spec.produces == ("l1_5_diagnostic_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "diagnostic"


def test_l1_5_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l1_5" in LAYER_SINKS
    assert "l1_5_diagnostic_v1" in LAYER_SINKS["l1_5"]
