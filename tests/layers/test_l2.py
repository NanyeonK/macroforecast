from macrocast.core.layers.l2 import (
    L2Preprocessing,
    normalize_to_dag_form,
    parse_layer_yaml,
    parse_recipe_yaml,
    resolve_axes,
    topological_order,
    validate_layer,
    validate_recipe,
)


def test_l2_minimal_yaml_parses_to_mccracken_ng_path():
    yaml_text = "2_preprocessing:\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text, "l2")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l2"))
    assert resolved["transform_policy"] == "apply_official_tcode"
    assert resolved["outlier_policy"] == "mccracken_ng_iqr"
    assert resolved["outlier_action"] == "flag_as_nan"
    assert resolved["imputation_policy"] == "em_factor"
    assert resolved["frame_edge_policy"] == "truncate_to_balanced"


def test_l2_no_scaling_axis_exists():
    axes = L2Preprocessing.list_axes()
    assert not any("scaling" in axis.lower() for axis in axes)
    assert not any("scale" == axis for axis in axes)


def test_l2_dynamic_scope_default_for_outlier_none():
    yaml_text = """
    2_preprocessing:
      fixed_axes:
        outlier_policy: none
    """
    layer = parse_layer_yaml(yaml_text, "l2")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l2"))
    assert resolved["outlier_scope"] == "not_applicable"
    assert resolved.source["outlier_scope"] == "derived"


def test_l2_explicit_scope_overrides_default():
    yaml_text = """
    2_preprocessing:
      fixed_axes:
        outlier_policy: mccracken_ng_iqr
        outlier_scope: target_and_predictors
    """
    layer = parse_layer_yaml(yaml_text, "l2")
    resolved = resolve_axes(normalize_to_dag_form(layer, "l2"))
    assert resolved["outlier_scope"] == "target_and_predictors"
    assert resolved.source["outlier_scope"] == "explicit"


def test_l2_full_sample_once_rejected_for_imputation():
    yaml_text = """
    2_preprocessing:
      fixed_axes:
        imputation_temporal_rule: full_sample_once
    """
    layer = parse_layer_yaml(yaml_text, "l2")
    report = validate_layer(layer)
    assert report.has_hard_errors


def test_l2_chow_lin_rejected_as_future():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: fred_md+fred_sd
        frequency: monthly
    2_preprocessing:
      fixed_axes:
        sd_series_frequency_filter: both
        quarterly_to_monthly_rule: chow_lin
    """
    recipe = parse_recipe_yaml(yaml_text)
    report = validate_recipe(recipe)
    assert report.has_hard_errors
    assert any("future" in issue.message.lower() for issue in report.hard_errors)


def test_l2_keep_with_indicator_rejected_as_future():
    yaml_text = """
    2_preprocessing:
      fixed_axes:
        outlier_action: keep_with_indicator
    """
    layer = parse_layer_yaml(yaml_text, "l2")
    report = validate_layer(layer)
    assert report.has_hard_errors


def test_l2_a_inactive_when_no_fred_sd():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: fred_md
    2_preprocessing:
      fixed_axes:
        sd_series_frequency_filter: both
    """
    recipe = parse_recipe_yaml(yaml_text)
    report = validate_recipe(recipe)
    assert report.has_hard_errors


def test_l2_a_quarterly_to_monthly_only_active_for_monthly_frequency():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: fred_qd+fred_sd
        frequency: quarterly
    2_preprocessing:
      fixed_axes:
        quarterly_to_monthly_rule: step_backward
    """
    recipe = parse_recipe_yaml(yaml_text)
    report = validate_recipe(recipe)
    assert report.has_hard_errors


def test_l2_pipeline_order_in_dag():
    yaml_text = "2_preprocessing:\n  fixed_axes: {}"
    layer = parse_layer_yaml(yaml_text, "l2")
    dag = normalize_to_dag_form(layer, "l2")
    order = topological_order(dag)
    transform_idx = order.index("step:transform")
    outlier_idx = order.index("step:outlier_handle")
    imputation_idx = order.index("step:imputation")
    edge_idx = order.index("step:frame_edge")
    assert transform_idx < outlier_idx < imputation_idx < edge_idx


def test_l2_replace_with_cap_value_requires_winsorize():
    yaml_text = """
    2_preprocessing:
      fixed_axes:
        outlier_policy: mccracken_ng_iqr
        outlier_action: replace_with_cap_value
    """
    layer = parse_layer_yaml(yaml_text, "l2")
    report = validate_layer(layer)
    assert report.has_hard_errors


def test_l2_custom_tcode_requires_map():
    yaml_text = """
    2_preprocessing:
      fixed_axes:
        transform_policy: custom_tcode
    """
    layer = parse_layer_yaml(yaml_text, "l2")
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("custom_tcode_map" in issue.message for issue in report.hard_errors)


def test_l2_registered_with_spec_correct_class():
    from macrocast.core.types import L2CleanPanelArtifact, LAYER_SINKS
    from macrocast.core.layers.registry import get_layer

    spec = get_layer("l2")
    assert spec.cls is L2Preprocessing
    assert spec.produces == ("l2_clean_panel_v1",)
    assert spec.ui_mode == "list"
    assert spec.category == "construction"
    assert LAYER_SINKS["l2"] == {"l2_clean_panel_v1": L2CleanPanelArtifact}


def test_l2_only_reads_l1_data_definition_not_regime():
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: external_nber
    2_preprocessing:
      fixed_axes: {}
    """
    recipe = parse_recipe_yaml(yaml_text)
    dag = normalize_to_dag_form(recipe["2_preprocessing"], "l2")
    sources = [node for node in dag.nodes.values() if node.type == "source"]
    sink_refs = [source.selector.sink_name for source in sources]
    assert "l1_data_definition_v1" in sink_refs
    assert "l1_regime_metadata_v1" not in sink_refs
