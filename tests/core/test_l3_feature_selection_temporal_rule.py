"""Cycle 14 J-5: feature_selection temporal_rule schema validation tests.

Verifies that:
1. Default temporal_rule (expanding_window_per_origin) is accepted.
2. Explicit expanding_window_per_origin is accepted.
3. Setting temporal_rule=full_sample_once raises a hard validation error.

Uses validate_layer(parse_layer_yaml(...)) matching the existing scale/pca
test pattern (tests/layers/test_l3.py::test_l3_full_sample_once_rejected_for_scale).
"""
from macroforecast.core.layers.l3 import (
    parse_layer_yaml,
    validate_layer,
)


def _make_l3_yaml_with_feature_selection(temporal_rule=None):
    """Build a minimal L3 YAML with a feature_selection node."""
    rule_part = ""
    if temporal_rule is not None:
        rule_part = f", temporal_rule: {temporal_rule}"
    return f"""
3_feature_engineering:
  nodes:
    - id: src_x
      type: source
      selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: predictors}}}}
    - id: src_y
      type: source
      selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: target}}}}
    - id: x_final
      type: step
      op: feature_selection
      params: {{n_features: 0.5{rule_part}}}
      inputs: [src_x]
    - id: y_h
      type: step
      op: target_construction
      params: {{mode: point_forecast, method: direct, horizon: 1}}
      inputs: [src_y]
  sinks:
    l3_features_v1: {{X_final: x_final, y_final: y_h}}
    l3_metadata_v1: auto
"""


def test_feature_selection_default_temporal_rule_accepted():
    """Default (no temporal_rule specified) should have no hard errors."""
    report = validate_layer(parse_layer_yaml(_make_l3_yaml_with_feature_selection()))
    assert not report.has_hard_errors, (
        f"Unexpected hard errors with default temporal_rule: "
        f"{[i.message for i in report.hard_errors]}"
    )


def test_feature_selection_explicit_expanding_accepted():
    """Explicit expanding_window_per_origin should have no hard errors."""
    report = validate_layer(
        parse_layer_yaml(
            _make_l3_yaml_with_feature_selection("expanding_window_per_origin")
        )
    )
    assert not report.has_hard_errors, (
        f"Unexpected hard errors with expanding_window_per_origin: "
        f"{[i.message for i in report.hard_errors]}"
    )


def test_feature_selection_full_sample_once_hard_rejected():
    """Setting temporal_rule=full_sample_once must produce a hard validation error."""
    report = validate_layer(
        parse_layer_yaml(_make_l3_yaml_with_feature_selection("full_sample_once"))
    )
    assert report.has_hard_errors, (
        "Expected hard error for full_sample_once but none were raised"
    )
    msgs = [i.message for i in report.hard_errors]
    assert any(
        "full_sample_once" in m or "temporal_rule" in m or "lookahead" in m
        for m in msgs
    ), f"Hard error messages do not mention full_sample_once/temporal_rule/lookahead: {msgs}"
