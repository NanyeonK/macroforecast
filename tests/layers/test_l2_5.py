from macrocast.core.layers.l2_5 import normalize_to_dag_form, parse_layer_yaml, parse_recipe_yaml, resolve_axes, validate_layer
from macrocast.core.layers.registry import get_layer


def test_l2_5_disabled_by_default():
    recipe = parse_recipe_yaml("")
    assert "l2_5" not in recipe.layers or recipe.layers["l2_5"].enabled is False


def test_l2_5_explicit_disabled():
    layer = parse_layer_yaml("2_5_pre_post_preprocessing:\n  enabled: false")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l2_5"))
    assert resolved["enabled"] is False


def test_l2_5_minimal_enabled_parses():
    yaml_text = "2_5_pre_post_preprocessing:\n  enabled: true\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text)
    assert validate_layer(layer).has_hard_errors is False
    resolved = resolve_axes(normalize_to_dag_form(layer, "l2_5"))
    assert resolved["comparison_pair"] == "raw_vs_final_clean"
    assert resolved["comparison_output_form"] == "multi"
    assert resolved["distribution_metric"] == ["mean_change", "sd_change", "ks_statistic"]
    assert resolved["correlation_shift"] == "none"
    assert resolved["t_code_application_log"] == "summary"


def test_l2_5_multi_stage_comparison_parses():
    yaml_text = """
    2_5_pre_post_preprocessing:
      enabled: true
      fixed_axes:
        comparison_pair: multi_stage
    """
    layer = parse_layer_yaml(yaml_text)
    assert validate_layer(layer).has_hard_errors is False


def test_l2_5_axes_not_sweepable():
    yaml_text = """
    2_5_pre_post_preprocessing:
      enabled: true
      fixed_axes:
        comparison_pair: {sweep: [raw_vs_final_clean, raw_vs_tcoded]}
    """
    layer = parse_layer_yaml(yaml_text)
    assert validate_layer(layer).has_hard_errors


def test_l2_5_correlation_method_inactive_when_no_correlation_shift():
    yaml_text = """
    2_5_pre_post_preprocessing:
      enabled: true
      fixed_axes:
        correlation_shift: none
        correlation_method: spearman
    """
    layer = parse_layer_yaml(yaml_text)
    resolved = resolve_axes(normalize_to_dag_form(layer, "l2_5"))
    assert resolved.get_active("correlation_method") is False


def test_l2_5_distribution_metric_list_validation():
    yaml_text = """
    2_5_pre_post_preprocessing:
      enabled: true
      fixed_axes:
        distribution_metric: [invalid_metric]
    """
    layer = parse_layer_yaml(yaml_text)
    assert validate_layer(layer).has_hard_errors


def test_l2_5_z_diagnostic_format_default():
    layer = parse_layer_yaml("2_5_pre_post_preprocessing:\n  enabled: true")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l2_5"))
    assert resolved["diagnostic_format"] == "pdf"


def test_l2_5_registered_with_spec_correct_class():
    spec = get_layer("l2_5")
    from macrocast.core.layers.l2_5 import L2_5PrePostPreprocessing

    assert spec.cls is L2_5PrePostPreprocessing
    assert spec.produces == ("l2_5_diagnostic_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "diagnostic"


def test_l2_5_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS

    assert "l2_5" in LAYER_SINKS
    assert "l2_5_diagnostic_v1" in LAYER_SINKS["l2_5"]
