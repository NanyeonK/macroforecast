from macrocast.core.layers.l1 import (
    L1Data,
    build_recipe_with_l1_only,
    execute_recipe,
    normalize_to_dag_form,
    parse_layer_yaml,
    resolve_axes,
    validate_layer,
)


def test_l1_registered_with_spec_correct_class():
    from macrocast.core.layers.registry import get_layer

    spec = get_layer("l1")
    assert spec.cls is L1Data
    assert spec.produces == ("l1_data_definition_v1", "l1_regime_metadata_v1")
    assert spec.ui_mode == "list"
    assert spec.category == "construction"


def test_l1_sink_in_layer_sinks():
    from macrocast.core.types import LAYER_SINKS, L1DataDefinitionArtifact, SeriesMetadata

    assert LAYER_SINKS["l1"]["l1_data_definition_v1"] is L1DataDefinitionArtifact
    assert LAYER_SINKS["l1"]["l1_regime_metadata_v1"] is SeriesMetadata


def test_l1_standard_fred_md_yaml_parses():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: official_only
        dataset: fred_md
        target_structure: single_target
        variable_universe: all_variables
      leaf_config:
        target: CPIAUCSL
    """
    layer = parse_layer_yaml(yaml_text)
    dag = normalize_to_dag_form(layer)
    resolved = resolve_axes(dag)
    assert resolved["frequency"] == "monthly"
    assert resolved["horizon_set"] == "standard_md"
    assert "l1_data_definition_v1" in dag.sinks
    assert "target_geography_scope" not in resolved or resolved["target_geography_scope"] is None


def test_l1_fred_qd_derives_quarterly_and_standard_qd():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: fred_qd
      leaf_config:
        target: GDPC1
    """
    resolved = resolve_axes(normalize_to_dag_form(parse_layer_yaml(yaml_text)))
    assert resolved["frequency"] == "quarterly"
    assert resolved["horizon_set"] == "standard_qd"


def test_l1_frequency_mismatch_fails():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: fred_md
        frequency: quarterly
      leaf_config:
        target: CPIAUCSL
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any("frequency" in issue.message for issue in report.hard_errors)


def test_l1_fred_sd_requires_frequency_and_geography():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: fred_sd
        frequency: monthly
        target_geography_scope: selected_states
        predictor_geography_scope: selected_states
      leaf_config:
        target: PAYEMS
        target_states: [CA, NY]
        predictor_states: [CA, NY, TX]
    """
    layer = parse_layer_yaml(yaml_text)
    report = validate_layer(layer)
    resolved = resolve_axes(normalize_to_dag_form(layer))
    assert not report.has_hard_errors
    assert resolved["target_geography_scope"] == "selected_states"
    assert resolved["predictor_geography_scope"] == "selected_states"
    assert resolved["variable_universe"] is None


def test_l1_fred_sd_without_frequency_fails():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: fred_sd
      leaf_config:
        target: PAYEMS
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any("frequency" in issue.message for issue in report.hard_errors)


def test_l1_custom_panel_requires_path_and_frequency():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
      leaf_config:
        target: y
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    messages = " ".join(issue.message for issue in report.hard_errors)
    assert "custom_source_path" in messages
    assert "frequency" in messages


def test_l1_official_plus_custom_requires_merge_keys():
    yaml_text = """
    1_data:
      fixed_axes:
        custom_source_policy: official_plus_custom
      leaf_config:
        target: CPIAUCSL
        custom_source_path: custom.csv
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any("custom_merge_rule" in issue.message for issue in report.hard_errors)


def test_l1_realtime_alfred_future_fails():
    yaml_text = """
    1_data:
      fixed_axes:
        vintage_policy: real_time_alfred
      leaf_config:
        target: CPIAUCSL
        vintage_date_or_tag: "2020-01-01"
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any("future" in issue.message for issue in report.hard_errors)


def test_l1_target_structure_rules():
    single_missing = """
    1_data:
      fixed_axes:
        target_structure: single_target
      leaf_config: {}
    """
    multi_valid = """
    1_data:
      fixed_axes:
        target_structure: multi_series_target
      leaf_config:
        targets: [CPIAUCSL, INDPRO]
    """
    assert validate_layer(parse_layer_yaml(single_missing)).has_hard_errors
    assert not validate_layer(parse_layer_yaml(multi_valid)).has_hard_errors


def test_l1_variable_universe_required_keys():
    yaml_text = """
    1_data:
      fixed_axes:
        variable_universe: explicit_variable_list
      leaf_config:
        target: CPIAUCSL
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any("variable_universe_columns" in issue.message for issue in report.hard_errors)


def test_l1_target_specific_must_cover_targets():
    yaml_text = """
    1_data:
      fixed_axes:
        target_structure: multi_series_target
        variable_universe: target_specific_variables
      leaf_config:
        targets: [CPIAUCSL, INDPRO]
        target_specific_columns:
          CPIAUCSL: [UNRATE]
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any("cover all targets" in issue.message for issue in report.hard_errors)


def test_l1_fixed_sample_window_date_rules():
    yaml_text = """
    1_data:
      fixed_axes:
        sample_start_rule: fixed_date
        sample_end_rule: fixed_date
      leaf_config:
        target: CPIAUCSL
        sample_start_date: "2020-01-01"
        sample_end_date: "2019-01-01"
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any(">=" in issue.message for issue in report.hard_errors)


def test_l1_horizon_rules_and_resolution():
    yaml_text = """
    1_data:
      fixed_axes:
        horizon_set: range_up_to_h
      leaf_config:
        target: CPIAUCSL
        max_horizon: 3
    """
    recipe = build_recipe_with_l1_only(yaml_text)
    manifest = execute_recipe(recipe)
    artifact = manifest.layer_execution_log["l1"].artifact
    assert artifact.target_horizons == (1, 2, 3)


def test_l1_no_sweep_except_dataset():
    yaml_text = """
    1_data:
      fixed_axes:
        frequency: {sweep: [monthly, quarterly]}
      leaf_config:
        target: CPIAUCSL
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    assert any("not sweepable" in issue.message for issue in report.hard_errors)


def test_l1_dataset_sweep_allowed():
    yaml_text = """
    1_data:
      fixed_axes:
        dataset: {sweep: [fred_md, fred_qd]}
      leaf_config:
        target: CPIAUCSL
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not any("not sweepable" in issue.message for issue in report.hard_errors)


def test_l1_regime_sink_gated_by_non_none_regime():
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: custom
      leaf_config:
        target: CPIAUCSL
        regime_source: nber
    """
    dag = normalize_to_dag_form(parse_layer_yaml(yaml_text))
    assert "l1_regime_metadata_v1" in dag.sinks


def test_l1_manifest_records_resolved_defaults():
    yaml_text = """
    1_data:
      fixed_axes: {}
      leaf_config:
        target: CPIAUCSL
    """
    manifest = execute_recipe(build_recipe_with_l1_only(yaml_text))
    record = manifest.layer_execution_log["l1"]
    assert record.resolved_axes["custom_source_policy"].source == "package_default"
    assert record.resolved_axes["frequency"].value == "monthly"
    assert record.produced_sinks == ("l1_data_definition_v1",)
