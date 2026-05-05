"""Regression tests for the v0.1 honesty pass demotion of formerly-planned
L4 families to ``future`` (closes #166, finalised in PR-B).

Background
==========

v0.1 introduced ``PLANNED_MODEL_FAMILIES`` for ``macroeconomic_random_forest``
and ``dfm_mixed_mariano_murasawa`` -- families whose runtime ran but used
acknowledged approximations of the Coulombe 2024 GTVP local-linear forest
and Mariano-Murasawa Kalman state-space EM. The L4 validator accepted both
``operational`` and ``planned`` to keep recipes runnable.

PR-B of the v0.1 honesty pass collapses ``planned`` into ``future``: any
family whose runtime does not match the design's named procedure is
hard-rejected by the validator. Real implementations land on a per-family
basis through the v0.2 issue tracker; the ``planned`` bucket stays empty.

This file pins:

* ``PLANNED_MODEL_FAMILIES`` is empty (back-compat tuple kept so external
  imports do not crash).
* MRF / DFM-MM / FAVAR / BVAR x2 are in ``FUTURE_MODEL_FAMILIES`` and
  report ``get_family_status() == "future"``.
* The L4 validator rejects every demoted family with a clear message
  pointing at the v0.2 implementation tracker.
* Future families (``midas_*``) remain rejected (no regression).
"""
from __future__ import annotations

import pytest

from macroforecast.core.layers.l4 import parse_layer_yaml, validate_layer
from macroforecast.core.ops.l4_ops import (
    FUTURE_MODEL_FAMILIES,
    MODEL_FAMILY_STATUS,
    OPERATIONAL_MODEL_FAMILIES,
    PLANNED_MODEL_FAMILIES,
    get_family_status,
)
from macroforecast.core.status import FUTURE, OPERATIONAL


# v0.2 follow-up: every L4 honesty-pass demotion is now re-promoted
# (#184 / #185 / #186 / #187 / #188). Empty tuple keeps the meta-test
# parametrisation in place even when there are no L4 demotions left.
_HONESTY_DEMOTED: tuple[str, ...] = ()


def test_planned_bucket_is_empty_after_v0_1_honesty_pass():
    assert PLANNED_MODEL_FAMILIES == ()


@pytest.mark.parametrize("family", _HONESTY_DEMOTED)
def test_demoted_families_are_in_future_bucket(family):
    assert family in FUTURE_MODEL_FAMILIES
    assert family not in OPERATIONAL_MODEL_FAMILIES


@pytest.mark.parametrize("family", _HONESTY_DEMOTED)
def test_demoted_families_report_future_status(family):
    assert get_family_status(family) == FUTURE


def test_status_dict_partition_is_two_buckets():
    op = set(OPERATIONAL_MODEL_FAMILIES)
    fut = set(FUTURE_MODEL_FAMILIES)
    # Every entry in MODEL_FAMILY_STATUS belongs to exactly one of the two
    # canonical buckets.
    assert set(MODEL_FAMILY_STATUS) == op | fut
    assert not (op & fut)


def test_operational_families_report_operational_status():
    for family in OPERATIONAL_MODEL_FAMILIES:
        assert get_family_status(family) == OPERATIONAL


@pytest.mark.parametrize("family", _HONESTY_DEMOTED)
def test_demoted_family_rejected_by_l4_validator(family):
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
    assert report.has_hard_errors, f"{family} should be rejected as future"
    assert any("future or unknown" in issue.message.lower() for issue in report.hard_errors)


def test_midas_future_families_still_rejected():
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
