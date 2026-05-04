"""Honesty-pass meta-test battery (PR-G).

Cross-cuts PR-A through PR-D: pins that every item demoted to ``future``
in the v0.1 honesty pass is consistently rejected by every layer's
validator, and that ``get_*_status`` helpers return the canonical
2-value vocabulary defined in :mod:`macrocast.core.status`.

If a future v0.2 PR re-implements one of the demoted items and promotes
it back to ``operational``, the corresponding parameter row in this
file should be removed (and the per-layer schema test reactivated).
"""
from __future__ import annotations

from textwrap import dedent

import pytest

from macrocast.core import FUTURE, OPERATIONAL, ItemStatus, is_future, is_runnable
from macrocast.core.layers.l1 import parse_layer_yaml as l1_parse, validate_layer as l1_validate
from macrocast.core.layers.l4 import parse_layer_yaml as l4_parse, validate_layer as l4_validate
from macrocast.core.layers.l7 import parse_layer_yaml as l7_parse, validate_layer as l7_validate
from macrocast.core.ops import list_ops
from macrocast.core.ops.l4_ops import (
    FUTURE_MODEL_FAMILIES,
    OPERATIONAL_MODEL_FAMILIES,
    PLANNED_MODEL_FAMILIES,
    get_family_status,
)
from macrocast.core.ops.l7_ops import HONESTY_DEMOTED_L7_OPS


# Items demoted from operational/planned to future during the v0.1
# honesty pass (PR-A..D). The list must stay in sync with the
# ``HONESTY_DEMOTED_*`` exports in the corresponding module.
_DEMOTED_L4_FAMILIES = (
    "factor_augmented_var",
    "bvar_minnesota",
    "bvar_normal_inverse_wishart",
    "macroeconomic_random_forest",
    "dfm_mixed_mariano_murasawa",
)

_DEMOTED_L1_REGIMES = (
    "estimated_markov_switching",
    "estimated_threshold",
    "estimated_structural_break",
)


# ---------------------------------------------------------------------------
# PR-A: vocabulary
# ---------------------------------------------------------------------------

def test_status_vocabulary_is_two_value_only():
    """No ``planned`` / ``approximation`` / ``simplified`` / ``registry_only``
    appearing in any item's ``status`` field (callers should only ever
    observe ``operational`` / ``future``)."""

    bad: list[tuple[str, str]] = []
    for op_name, spec in list_ops().items():
        if spec.status not in {OPERATIONAL, FUTURE}:
            bad.append((op_name, spec.status))
    assert not bad, f"ops with non-canonical status: {bad}"


# ---------------------------------------------------------------------------
# PR-B: L4 family demotions
# ---------------------------------------------------------------------------

def test_planned_l4_bucket_is_empty():
    assert PLANNED_MODEL_FAMILIES == ()


@pytest.mark.parametrize("family", _DEMOTED_L4_FAMILIES)
def test_demoted_l4_family_is_in_future_bucket(family):
    assert family in FUTURE_MODEL_FAMILIES
    assert family not in OPERATIONAL_MODEL_FAMILIES
    assert get_family_status(family) == FUTURE
    assert is_future(family is None or get_family_status(family))


@pytest.mark.parametrize("family", _DEMOTED_L4_FAMILIES)
def test_demoted_l4_family_rejected_by_validator(family):
    yaml_text = dedent(
        f"""
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
    )
    report = l4_validate(l4_parse(yaml_text))
    assert report.has_hard_errors, f"{family} should be rejected"


# ---------------------------------------------------------------------------
# PR-C: L7 op demotions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op_name", HONESTY_DEMOTED_L7_OPS)
def test_demoted_l7_op_status_is_future(op_name):
    spec = list_ops()[op_name]
    assert spec.status == FUTURE, f"{op_name} should have status=future"
    assert not is_runnable(spec.status)


@pytest.mark.parametrize("op_name", HONESTY_DEMOTED_L7_OPS)
def test_demoted_l7_op_rejected_by_validator(op_name):
    # Build a minimal L7 recipe selecting the op. Use a benign
    # model_family that does not trigger the L7 op compatibility rule
    # (which would itself reject independently).
    family_hint = {
        "shap_tree": "xgboost",
        "fevd": "var",
        "historical_decomposition": "var",
        "generalized_irf": "var",
        "mrf_gtvp": "macroeconomic_random_forest",
        "lasso_inclusion_frequency": "lasso",
        "accumulated_local_effect": "ridge",
        "friedman_h_interaction": "ridge",
        "gradient_shap": "mlp",
        "integrated_gradients": "mlp",
        "saliency_map": "mlp",
        "deep_lift": "mlp",
    }.get(op_name, "ridge")
    yaml_text = dedent(
        f"""
        nodes:
          - {{id: src_model, type: source, selector: {{layer_ref: l4, sink_name: l4_model_artifacts_v1, subset: {{model_id: fit_model}}}}}}
          - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
          - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
          - {{id: imp, type: step, op: {op_name}, params: {{model_family: {family_hint}}}, inputs: [src_model, src_X, src_y]}}
        sinks:
          l7_importance_v1: imp
        """
    )
    report = l7_validate(l7_parse(yaml_text, "l7"), recipe_context={})
    assert report.has_hard_errors, f"L7 op {op_name} should be rejected by validator"


# ---------------------------------------------------------------------------
# PR-D: L1 estimated regime demotions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime", _DEMOTED_L1_REGIMES)
def test_demoted_l1_regime_rejected_by_validator(regime):
    yaml_text = dedent(
        f"""
        1_data:
          fixed_axes:
            regime_definition: {regime}
          leaf_config:
            target: CPIAUCSL
            n_regimes: 2
            threshold_variable: INDPRO
        """
    )
    report = l1_validate(l1_parse(yaml_text))
    assert report.has_hard_errors, f"L1 regime {regime} should be rejected"
    # The validator's message must point at the future status (so the user
    # finds the v0.2 implementation tracker).
    messages = [issue.message for issue in report.hard_errors]
    assert any("future" in m.lower() for m in messages), (
        f"L1 {regime}: expected a 'future' rejection message, got {messages}"
    )


# ---------------------------------------------------------------------------
# Cross-cutting honesty invariant
# ---------------------------------------------------------------------------

def test_count_of_demotions_matches_documentation():
    """The CLAUDE.md "v0.1 honesty-pass demotions" table lists 8 row
    groups. The runtime exports those groups via three constants. Pin
    the totals so a stray addition / removal triggers a failing test."""

    # 5 L4 families
    assert len(_DEMOTED_L4_FAMILIES) == 5
    # 11 L7 ops
    assert len(HONESTY_DEMOTED_L7_OPS) == 11
    # 3 L1 estimated regimes
    assert len(_DEMOTED_L1_REGIMES) == 3
    # Combined honest count: 19 demoted items across 3 layers.
    assert len(_DEMOTED_L4_FAMILIES) + len(HONESTY_DEMOTED_L7_OPS) + len(_DEMOTED_L1_REGIMES) == 19
