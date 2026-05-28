"""tests/layers/test_l2_temporal_dispatch_audit.py

TDD tests for L2 temporal dispatch audit (PR7).

Covers:
- Gap 1/2: rolling_window_per_origin + linear_interpolation -> hard error (validator)
- Gap 2: rolling_window_per_origin + forward_fill -> no hard error (safe causal combo)
- Gap 2: rolling_window_per_origin + non-leaky policy -> no hard error
- Gap 3: block_recompute + stateful policies -> SOFT warning (not hard error)
- Gap 3: block_recompute + causal-safe policies -> no hard error, no SOFT warning
- Dispatch matrix: expanding_window_per_origin is unaffected
- Regression: full_sample_once still hard-rejected
- Regression: expanding_window_per_origin + linear_interpolation is NOT rejected
"""
from __future__ import annotations

import pytest

from macroforecast.preprocessing.schema import (
    parse_layer_yaml,
    validate_layer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yaml(imputation_policy: str, temporal_rule: str) -> str:
    return f"""
preprocessing:
  fixed_axes:
    imputation_policy: {imputation_policy}
    imputation_temporal_rule: {temporal_rule}
"""


def _make_yaml_with_outlier(outlier_policy: str, temporal_rule: str) -> str:
    return f"""
preprocessing:
  fixed_axes:
    outlier_policy: {outlier_policy}
    imputation_temporal_rule: {temporal_rule}
"""


# ---------------------------------------------------------------------------
# Gap 1 / Gap 2 — rolling_window_per_origin + linear_interpolation: HARD reject
# ---------------------------------------------------------------------------

class TestRollingWindowLinearInterpolationHardReject:
    """rolling_window_per_origin + linear_interpolation must be a hard error.

    The option name implies per-origin behavior but the runtime applies
    full-sample linear_interpolation (bidirectional by pandas default),
    causing a silent lookahead leak.
    """

    def test_hard_error_raised(self):
        yaml_text = _make_yaml("linear_interpolation", "rolling_window_per_origin")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert report.has_hard_errors, (
            "rolling_window_per_origin + linear_interpolation must produce a hard "
            "validation error because the implementation falls into full-sample "
            "imputation while the option name implies per-origin safety."
        )

    def test_error_message_references_rolling_window(self):
        yaml_text = _make_yaml("linear_interpolation", "rolling_window_per_origin")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        hard_msgs = [issue.message for issue in report.hard_errors]
        assert any(
            "rolling_window_per_origin" in msg or "lookahead" in msg.lower()
            for msg in hard_msgs
        ), f"Expected 'rolling_window_per_origin' or 'lookahead' in error messages, got: {hard_msgs}"

    def test_alternative_mentioned_in_message(self):
        """The error message should mention the working alternative."""
        yaml_text = _make_yaml("linear_interpolation", "rolling_window_per_origin")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        hard_msgs = " ".join(issue.message for issue in report.hard_errors)
        # Should mention the safe alternative or forward_fill
        assert (
            "expanding_window_per_origin" in hard_msgs
            or "forward_fill" in hard_msgs
        ), f"Error message should mention the safe alternative, got: {hard_msgs}"


# ---------------------------------------------------------------------------
# Gap 2 — rolling_window_per_origin + safe policies: no hard error
# ---------------------------------------------------------------------------

class TestRollingWindowSafeCombosPasses:
    """rolling_window_per_origin with causal-safe policies must NOT be rejected."""

    def test_forward_fill_passes(self):
        yaml_text = _make_yaml("forward_fill", "rolling_window_per_origin")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert not report.has_hard_errors, (
            "forward_fill is inherently causal (uses only past values). "
            f"rolling_window_per_origin + forward_fill must not be rejected. "
            f"Hard errors: {[i.message for i in report.hard_errors]}"
        )

    def test_none_propagate_passes(self):
        yaml_text = _make_yaml("none_propagate", "rolling_window_per_origin")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert not report.has_hard_errors, (
            "none_propagate performs no imputation. "
            f"Hard errors: {[i.message for i in report.hard_errors]}"
        )


# ---------------------------------------------------------------------------
# Gap 3 — block_recompute + stateful policies: SOFT warning, NOT hard error
# ---------------------------------------------------------------------------

class TestBlockRecomputeSoftWarning:
    """block_recompute + stateful policies emits SOFT warning, not hard error.

    block_recompute is a legitimate full-sample-at-block-boundary approach.
    The name does not imply per-origin behavior. Only a soft warning is
    appropriate because the semantics are documented and user opt-in.
    """

    @pytest.mark.parametrize("stateful_policy", [
        "mean",
        "em_factor",
        "em_multivariate",
    ])
    def test_stateful_imputation_policy_soft_warning(self, stateful_policy: str):
        yaml_text = _make_yaml(stateful_policy, "block_recompute")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        # Must NOT be a hard error
        assert not report.has_hard_errors, (
            f"block_recompute + {stateful_policy} must NOT be a hard error — "
            f"it is a documented user opt-in. "
            f"Hard errors: {[i.message for i in report.hard_errors]}"
        )
        # Must be a soft warning
        soft_msgs = [issue.message for issue in report.soft_warnings]
        assert any(
            "block_recompute" in msg for msg in soft_msgs
        ), (
            f"block_recompute + {stateful_policy} must emit a SOFT warning "
            f"mentioning 'block_recompute'. Soft warnings: {soft_msgs}"
        )

    @pytest.mark.parametrize("stateful_outlier_policy", [
        "mccracken_ng_iqr",
        "zscore_threshold",
        "winsorize",
    ])
    def test_stateful_outlier_policy_soft_warning(self, stateful_outlier_policy: str):
        yaml_text = _make_yaml_with_outlier(stateful_outlier_policy, "block_recompute")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert not report.has_hard_errors, (
            f"block_recompute + {stateful_outlier_policy} must NOT be hard rejected. "
            f"Hard errors: {[i.message for i in report.hard_errors]}"
        )
        soft_msgs = [issue.message for issue in report.soft_warnings]
        assert any(
            "block_recompute" in msg for msg in soft_msgs
        ), (
            f"block_recompute + {stateful_outlier_policy} must emit a SOFT warning. "
            f"Soft warnings: {soft_msgs}"
        )

    def test_forward_fill_no_hard_no_soft_block_recompute(self):
        """block_recompute + forward_fill (causal) should not warn."""
        yaml_text = _make_yaml("forward_fill", "block_recompute")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert not report.has_hard_errors, (
            f"block_recompute + forward_fill must not be hard rejected. "
            f"Hard errors: {[i.message for i in report.hard_errors]}"
        )
        soft_msgs = [issue.message for issue in report.soft_warnings]
        block_warns = [m for m in soft_msgs if "block_recompute" in m]
        assert not block_warns, (
            f"block_recompute + forward_fill (causal) should not emit a "
            f"block_recompute SOFT warning. Got: {block_warns}"
        )

    def test_none_propagate_no_soft_warning_block_recompute(self):
        """block_recompute + none_propagate: no imputation, so no warning."""
        yaml_text = _make_yaml("none_propagate", "block_recompute")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert not report.has_hard_errors
        soft_msgs = [issue.message for issue in report.soft_warnings]
        block_warns = [m for m in soft_msgs if "block_recompute" in m]
        assert not block_warns, (
            f"block_recompute + none_propagate should not emit a block_recompute "
            f"warning. Got: {block_warns}"
        )


# ---------------------------------------------------------------------------
# Dispatch matrix completeness — expanding_window_per_origin unaffected
# ---------------------------------------------------------------------------

class TestExpandingWindowPerOriginUnaffected:
    """expanding_window_per_origin (default) must never be rejected or warned."""

    @pytest.mark.parametrize("policy", [
        "linear_interpolation",
        "mean",
        "em_factor",
        "em_multivariate",
        "forward_fill",
        "none_propagate",
    ])
    def test_expanding_window_per_origin_always_passes(self, policy: str):
        yaml_text = _make_yaml(policy, "expanding_window_per_origin")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert not report.has_hard_errors, (
            f"expanding_window_per_origin + {policy} must never be hard rejected. "
            f"Hard errors: {[i.message for i in report.hard_errors]}"
        )


# ---------------------------------------------------------------------------
# Regression — full_sample_once still hard-rejected
# ---------------------------------------------------------------------------

class TestFullSampleOnceRegressionGuard:
    def test_full_sample_once_still_rejected(self):
        yaml_text = _make_yaml("mean", "full_sample_once")
        layer = parse_layer_yaml(yaml_text)
        report = validate_layer(layer)
        assert report.has_hard_errors, (
            "full_sample_once must remain hard-rejected (regression guard)"
        )
