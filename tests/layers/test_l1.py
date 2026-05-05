from macroforecast.core.layers.l1 import (
    L1Data,
    build_recipe_with_l1_only,
    execute_recipe,
    normalize_to_dag_form,
    parse_layer_yaml,
    resolve_axes,
    validate_layer,
    validate_regime_source_reference,
)


def test_l1_registered_with_spec_correct_class():
    from macroforecast.core.layers.registry import get_layer

    spec = get_layer("l1")
    assert spec.cls is L1Data
    assert spec.produces == ("l1_data_definition_v1", "l1_regime_metadata_v1")
    assert spec.ui_mode == "list"
    assert spec.category == "construction"


def test_l1_sink_in_layer_sinks():
    from macroforecast.core.types import LAYER_SINKS, L1DataDefinitionArtifact, L1RegimeMetadataArtifact

    assert LAYER_SINKS["l1"]["l1_data_definition_v1"] is L1DataDefinitionArtifact
    assert LAYER_SINKS["l1"]["l1_regime_metadata_v1"] is L1RegimeMetadataArtifact


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
        regime_definition: external_user_provided
      leaf_config:
        target: CPIAUCSL
        regime_indicator_path: regimes.csv
        n_regimes: 2
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
    assert record.produced_sinks == ("l1_data_definition_v1", "l1_regime_metadata_v1")


def test_l1_g_default_is_none_and_sink_is_inactive():
    from macroforecast.core import SourceSelector

    yaml_text = """
    1_data:
      fixed_axes: {}
      leaf_config:
        target: CPIAUCSL
    """
    manifest = execute_recipe(build_recipe_with_l1_only(yaml_text))
    record = manifest.layer_execution_log["l1"]
    assert record.regime_artifact.definition == "none"
    assert record.regime_artifact.regime_label_series is None
    assert record.regime_artifact.estimation_temporal_rule is None
    assert record.produced_sinks == ("l1_data_definition_v1", "l1_regime_metadata_v1")
    report = validate_regime_source_reference(
        parse_layer_yaml(yaml_text),
        SourceSelector(layer_ref="l1", sink_name="l1_regime_metadata_v1"),
    )
    assert report.has_hard_errors


def test_l1_g_external_nber_loads_usrec_metadata():
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: external_nber
      leaf_config:
        target: CPIAUCSL
    """
    manifest = execute_recipe(build_recipe_with_l1_only(yaml_text))
    regime = manifest.layer_execution_log["l1"].regime_artifact
    assert regime.definition == "external_nber"
    assert regime.estimation_metadata["source_series"] == "USREC"


def test_l1_g_estimated_markov_switching_operational_after_hamilton_landing():
    # Issue #195 lands the real Hamilton (1989) Markov regression via
    # ``statsmodels.tsa.regime_switching.MarkovRegression``; the validator
    # should accept the family. Note ``regime_estimation_temporal_rule``
    # must be one of the leakage-safe options.
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: estimated_markov_switching
        regime_estimation_temporal_rule: expanding_window_per_origin
      leaf_config:
        target: CPIAUCSL
        n_regimes: 2
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors


def test_l1_g_full_sample_once_temporal_rule_still_rejected_as_leakage():
    # full_sample_once leakage check fires *before* ``_validate_regime``
    # runs (it's a separate validator), so the leakage message remains
    # the user-facing failure mode for this combination.
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: estimated_markov_switching
        regime_estimation_temporal_rule: full_sample_once
      leaf_config:
        target: CPIAUCSL
        n_regimes: 2
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert report.has_hard_errors
    # Either the leakage check or the future-status check is acceptable;
    # both convey that the recipe is invalid.
    assert any(
        "full_sample_once" in issue.message or "future" in issue.message.lower()
        for issue in report.hard_errors
    )


def test_l1_g_estimated_threshold_operational_after_setar_landing():
    # Issue #196 lands a Tong (1990) SETAR quantile-split estimator; the
    # validator must accept the family.
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: estimated_threshold
        regime_estimation_temporal_rule: expanding_window_per_origin
      leaf_config:
        target: CPIAUCSL
        threshold_variable: INDPRO
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors


def test_l1_g_estimated_structural_break_operational_after_bai_perron_landing():
    # Issue #197 lands a Bai-Perron (1998) global LSE break detector; the
    # validator must accept the family.
    yaml_text = """
    1_data:
      fixed_axes:
        regime_definition: estimated_structural_break
        regime_estimation_temporal_rule: expanding_window_per_origin
      leaf_config:
        target: CPIAUCSL
        max_breaks: 3
    """
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors
