"""Regression tests for the v0.1 ``planned`` model family status (closes #166).

``macroeconomic_random_forest`` (Coulombe 2024) and
``dfm_mixed_mariano_murasawa`` were marked ``operational`` in v0.1 even
though their wrappers are honest approximations of the published methods.
This batch downgrades them to ``planned`` so users can detect via
``get_family_status`` that the runtime path is an approximation, while
keeping existing recipes runnable (the L4 validator accepts both
operational and planned families).
"""
from __future__ import annotations

import pytest

from macrocast.core.layers.l4 import parse_layer_yaml, validate_layer
from macrocast.core.ops.l4_ops import (
    FUTURE_MODEL_FAMILIES,
    MODEL_FAMILY_STATUS,
    OPERATIONAL_MODEL_FAMILIES,
    PLANNED_MODEL_FAMILIES,
    get_family_status,
)


def test_planned_families_are_published():
    assert "macroeconomic_random_forest" in PLANNED_MODEL_FAMILIES
    assert "dfm_mixed_mariano_murasawa" in PLANNED_MODEL_FAMILIES


def test_planned_families_have_planned_status():
    for family in PLANNED_MODEL_FAMILIES:
        assert get_family_status(family) == "planned"


def test_planned_families_not_in_operational_set():
    overlap = set(PLANNED_MODEL_FAMILIES) & set(OPERATIONAL_MODEL_FAMILIES)
    assert not overlap, f"families appear in both operational and planned: {overlap}"


def test_planned_families_distinct_from_future():
    overlap = set(PLANNED_MODEL_FAMILIES) & set(FUTURE_MODEL_FAMILIES)
    assert not overlap, f"families appear in both planned and future: {overlap}"


def test_status_dict_is_partition_of_three_buckets():
    """Every family in the status dict belongs to exactly one of operational
    / planned / future."""

    op = set(OPERATIONAL_MODEL_FAMILIES)
    plan = set(PLANNED_MODEL_FAMILIES)
    fut = set(FUTURE_MODEL_FAMILIES)
    assert set(MODEL_FAMILY_STATUS) == op | plan | fut
    assert not (op & plan)
    assert not (op & fut)
    assert not (plan & fut)


@pytest.mark.parametrize("family", PLANNED_MODEL_FAMILIES)
def test_planned_family_passes_l4_validator(family):
    """A recipe selecting a planned family must validate cleanly so existing
    recipes don't break."""

    yaml_text = f"""
nodes:
  - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
  - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
  - id: fit
    type: step
    op: fit_model
    params:
      family: {family}
      forecast_strategy: direct
      training_start_rule: expanding
      refit_policy: every_origin
      search_algorithm: none
    inputs: [src_X, src_y]
  - {{id: predict, type: step, op: predict, inputs: [fit, src_X]}}
sinks:
  l4_forecasts_v1: predict
  l4_model_artifacts_v1: fit
  l4_training_metadata_v1: auto
"""
    layer = parse_layer_yaml(yaml_text)
    report = validate_layer(layer)
    assert not report.has_hard_errors, report.hard_errors


def test_future_families_still_rejected_by_validator():
    yaml_text = """
nodes:
  - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
  - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
  - id: fit
    type: step
    op: fit_model
    params:
      family: midas_almon
      n_lag: 12
      polynomial_degree: 3
      forecast_strategy: direct
      training_start_rule: expanding
      refit_policy: every_origin
      search_algorithm: none
    inputs: [src_X, src_y]
  - {id: predict, type: step, op: predict, inputs: [fit, src_X]}
sinks:
  l4_forecasts_v1: predict
  l4_model_artifacts_v1: fit
  l4_training_metadata_v1: auto
"""
    layer = parse_layer_yaml(yaml_text)
    report = validate_layer(layer)
    assert report.has_hard_errors
    assert any("future" in issue.message.lower() for issue in report.hard_errors)
