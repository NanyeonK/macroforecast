"""test(c51): Chow-Lin disaggregation op — Flag-A regression guard.

Verifies that the chow_lin_disaggregation op body:
1. Is registered with status="operational" (no future ops remain post-C51).
2. Dispatches to _chow_lin_disaggregate and produces results matching direct
   helper calls on a deterministic seed-42 fixture.
3. Raises ValueError when chow_lin_indicator is absent and no second input
   is provided.

These tests are fully independent of implementation knowledge: they verify
behavioral contracts specified in test-spec.md TC-FlagA-2, TC-FlagA-3.

Scope: pytest -m "not slow and not heavy and not deep"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Deterministic fixture (seed 42, from test-spec.md TC-FlagA-3)
# ---------------------------------------------------------------------------

def _make_fixture_seed42():
    """Return (q_series, m_indicator, q_df, m_df) for seed 42."""
    rng = np.random.default_rng(42)
    # Quarterly dates: 2010-Q1 through 2014-Q4 (20 quarters)
    q_dates = pd.date_range("2010-01-01", periods=20, freq="QE")
    q_series = pd.Series(rng.standard_normal(20), index=q_dates)

    # Monthly indicator: 60 months (2010-01 through 2014-12)
    m_dates = pd.date_range("2010-01-01", periods=60, freq="MS")
    m_indicator = pd.Series(rng.standard_normal(60), index=m_dates)

    q_df = q_series.to_frame("gdp")
    m_df = m_indicator.to_frame("ip")
    return q_series, m_indicator, q_df, m_df


# ---------------------------------------------------------------------------
# TC-FlagA-2: status is "operational" in registry
# ---------------------------------------------------------------------------

def test_chow_lin_disaggregation_status_operational() -> None:
    """chow_lin_disaggregation must be registered with status="operational"."""
    from macroforecast.core.ops import list_ops

    ops = list_ops()
    assert "chow_lin_disaggregation" in ops, (
        "chow_lin_disaggregation not found in registry — was it de-registered?"
    )
    op = ops["chow_lin_disaggregation"]
    assert op.status == "operational", (
        f"Expected status='operational', got status={op.status!r}. "
        "C51 Flag-A must have promoted the op."
    )


# ---------------------------------------------------------------------------
# TC-FlagA-2 (extended): no future ops remain post-C51
# ---------------------------------------------------------------------------

def test_no_future_ops_post_c51() -> None:
    """Post-C51 registry must have zero future ops."""
    from macroforecast.core.ops import list_ops

    future_ops = [op for op in list_ops().values() if op.status == "future"]
    assert len(future_ops) == 0, (
        f"Expected 0 future ops post-C51; found {len(future_ops)}: "
        f"{[op.name for op in future_ops]}"
    )


# ---------------------------------------------------------------------------
# TC-FlagA-3: op body result matches _chow_lin_disaggregate directly
# ---------------------------------------------------------------------------

def test_chow_lin_op_matches_helper_directly() -> None:
    """chow_lin_disaggregation([q_df, m_df], {}) must match _chow_lin_disaggregate(q, m).

    Tolerance: atol=1e-10 (test-spec.md).
    """
    from macroforecast.core.runtime import _chow_lin_disaggregate
    from macroforecast.layers.l3_features.ops import chow_lin_disaggregation

    q_series, m_indicator, q_df, m_df = _make_fixture_seed42()

    # Step 1: direct helper call
    expected = _chow_lin_disaggregate(q_series, m_indicator)

    # Step 2: op body call
    result = chow_lin_disaggregation([q_df, m_df], params={})

    # Assertion 1: result is a pd.Series or single-column pd.DataFrame
    assert isinstance(result, (pd.Series, pd.DataFrame)), (
        f"Expected pd.Series or pd.DataFrame, got {type(result)}"
    )

    # Flatten to Series for comparison
    if isinstance(result, pd.DataFrame):
        result_values = result.iloc[:, 0].values
        result_index = result.index
    else:
        result_values = result.values
        result_index = result.index

    # Assertion 2: values match helper output (atol=1e-10 per test-spec.md)
    expected_aligned = expected.reindex(result_index)
    assert np.allclose(result_values, expected_aligned.values, atol=1e-10), (
        f"chow_lin_disaggregation op does not match _chow_lin_disaggregate helper.\n"
        f"Max abs diff: {np.max(np.abs(result_values - expected_aligned.values)):.6e}\n"
        f"Tolerance: atol=1e-10"
    )

    # Assertion 3: index is monthly (60 entries for 20-quarter input)
    assert len(result_index) == 60, (
        f"Expected 60 monthly entries (for 20 quarters), got {len(result_index)}"
    )


# ---------------------------------------------------------------------------
# TC-FlagA-3 (edge case): ValueError when no second input and no param
# ---------------------------------------------------------------------------

def test_chow_lin_op_raises_on_missing_indicator() -> None:
    """chow_lin_disaggregation([q_df], {}) must raise ValueError about chow_lin_indicator."""
    from macroforecast.layers.l3_features.ops import chow_lin_disaggregation

    _, _, q_df, _ = _make_fixture_seed42()

    with pytest.raises(ValueError, match="chow_lin_indicator"):
        chow_lin_disaggregation([q_df], params={})


# ---------------------------------------------------------------------------
# Additional: op does not raise on valid two-input call (smoke test)
# ---------------------------------------------------------------------------

def test_chow_lin_op_no_error_two_inputs() -> None:
    """chow_lin_disaggregation([q_df, m_df], {}) must not raise any error."""
    from macroforecast.layers.l3_features.ops import chow_lin_disaggregation

    _, _, q_df, m_df = _make_fixture_seed42()
    # Must not raise
    result = chow_lin_disaggregation([q_df, m_df], params={})
    assert result is not None, "Expected non-None result from chow_lin_disaggregation"
