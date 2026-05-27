"""Independent validation suite — C63.1 chow_lin_disaggregate.

Tests T1-T12 from test-spec.md Section 2. Written by tester, isolated from
builder's implementation details. Verifies behavioral contracts from
test-spec.md Section 1.

Test IDs map to test-spec.md:
    T1  — Conservation: aggregation='mean' (atol=1e-8)
    T2  — Conservation: aggregation='sum'  (atol=1e-8)
    T3  — Rho estimation accuracy (MSE < var(y_h_true))
    T4  — rho_method='fixed' with explicit rho=0.0
    T5  — explicit rho=0.5
    T6  — rho_method='max_likelihood'
    T7  — Invalid aggregation raises ValueError
    T8  — rho out of range raises ValueError
    T9  — rho_method invalid value raises ValueError
    T10 — Non-DatetimeIndex input: graceful fallback
    T11 — Backward compatibility: two-positional signature
    T12 — Return type and index alignment
    I1  — Property invariant: conservation (sum)
    I2  — Property invariant: conservation (mean)
    R2  — Regression: private runtime helper still importable
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.features.transforms import chow_lin_disaggregate


# ---------------------------------------------------------------------------
# Shared synthetic data fixture (from test-spec.md Section 2)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data() -> dict:
    """Quarterly-to-monthly synthetic data with true rho=0.7 AR(1) latent."""
    np.random.seed(42)
    idx_m = pd.date_range("2010-01-31", periods=60, freq="ME")
    idx_q = pd.date_range("2010-03-31", periods=20, freq="QE")

    # True high-frequency AR(1) latent series
    rho_true = 0.7
    eps = np.random.randn(60)
    y_h_true = np.zeros(60)
    for t in range(1, 60):
        y_h_true[t] = rho_true * y_h_true[t - 1] + eps[t]

    # Low-frequency sum aggregation (3 months per quarter)
    y_l_sum = np.array([y_h_true[3 * i : 3 * i + 3].sum() for i in range(20)])
    y_l_sum_s = pd.Series(y_l_sum, index=idx_q, name="y_l")

    # Low-frequency mean aggregation
    y_l_mean = np.array([y_h_true[3 * i : 3 * i + 3].mean() for i in range(20)])
    y_l_mean_s = pd.Series(y_l_mean, index=idx_q, name="y_l")

    # High-frequency indicator correlated with latent series
    indicator = pd.Series(
        0.6 * y_h_true + 0.4 * np.random.randn(60),
        index=idx_m,
        name="indicator",
    )

    return {
        "idx_m": idx_m,
        "idx_q": idx_q,
        "y_h_true": y_h_true,
        "y_l_sum_s": y_l_sum_s,
        "y_l_mean_s": y_l_mean_s,
        "indicator": indicator,
    }


# ---------------------------------------------------------------------------
# T1 — Conservation: aggregation='mean' (atol=1e-8)
# ---------------------------------------------------------------------------

def test_T1_conservation_mean(synthetic_data: dict) -> None:
    """T1: Re-aggregating via mean recovers low-freq input to atol=1e-8."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    result = chow_lin_disaggregate(y_l_mean_s, indicator, aggregation="mean")

    # Must return a pd.Series of length 60
    assert isinstance(result, pd.Series)
    assert len(result) == 60

    # Re-aggregate: monthly mean -> quarterly
    reaggregated = result.resample("QE").mean()
    aligned_lf = y_l_mean_s.reindex(reaggregated.index)

    # Conservation check: atol=1e-8 (from test-spec.md T1)
    assert np.allclose(
        reaggregated.values, aligned_lf.values, atol=1e-8
    ), (
        f"T1 FAIL: max abs diff = "
        f"{np.max(np.abs(reaggregated.values - aligned_lf.values)):.2e} "
        f"(tolerance atol=1e-8)"
    )


# ---------------------------------------------------------------------------
# T2 — Conservation: aggregation='sum' (atol=1e-8)
# ---------------------------------------------------------------------------

def test_T2_conservation_sum(synthetic_data: dict) -> None:
    """T2: Re-aggregating via sum recovers low-freq input to atol=1e-8."""
    y_l_sum_s = synthetic_data["y_l_sum_s"]
    indicator = synthetic_data["indicator"]

    result = chow_lin_disaggregate(y_l_sum_s, indicator, aggregation="sum")

    assert isinstance(result, pd.Series)
    assert len(result) == 60

    # Re-aggregate: monthly sum -> quarterly
    reaggregated = result.resample("QE").sum()
    aligned_lf = y_l_sum_s.reindex(reaggregated.index)

    # Conservation check: atol=1e-8 (from test-spec.md T2)
    assert np.allclose(
        reaggregated.values, aligned_lf.values, atol=1e-8
    ), (
        f"T2 FAIL: max abs diff = "
        f"{np.max(np.abs(reaggregated.values - aligned_lf.values)):.2e} "
        f"(tolerance atol=1e-8)"
    )


# ---------------------------------------------------------------------------
# T3 — Rho estimation accuracy (MSE < var(y_h_true))
# ---------------------------------------------------------------------------

def test_T3_rho_estimation_accuracy(synthetic_data: dict) -> None:
    """T3: Disaggregated MSE vs true latent is < variance of the latent series."""
    y_l_sum_s = synthetic_data["y_l_sum_s"]
    indicator = synthetic_data["indicator"]
    y_h_true = synthetic_data["y_h_true"]

    result = chow_lin_disaggregate(
        y_l_sum_s, indicator, aggregation="sum", rho_method="min_chi_squared"
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 60

    mse = float(np.mean((result.values - y_h_true) ** 2))
    var_y_h = float(np.var(y_h_true))

    assert mse < var_y_h, (
        f"T3 FAIL: MSE={mse:.6f} >= var(y_h_true)={var_y_h:.6f}. "
        "Disaggregation quality is worse than naive benchmark."
    )


# ---------------------------------------------------------------------------
# T4 — rho_method='fixed' with explicit rho=0.0 satisfies conservation
# ---------------------------------------------------------------------------

def test_T4_fixed_rho_zero_conservation(synthetic_data: dict) -> None:
    """T4: rho=0.0 satisfies conservation atol=1e-8 and differs from rho=0.5."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    result_rho0 = chow_lin_disaggregate(
        y_l_mean_s, indicator, aggregation="mean", rho=0.0
    )

    assert isinstance(result_rho0, pd.Series)

    # Conservation at atol=1e-8
    reagg = result_rho0.resample("QE").mean()
    aligned = y_l_mean_s.reindex(reagg.index)
    assert np.allclose(reagg.values, aligned.values, atol=1e-8), (
        f"T4 conservation FAIL: max abs diff = "
        f"{np.max(np.abs(reagg.values - aligned.values)):.2e}"
    )

    # Must differ from rho=0.5 (non-trivial effect)
    result_rho05 = chow_lin_disaggregate(
        y_l_mean_s, indicator, aggregation="mean", rho=0.5
    )
    assert not np.allclose(result_rho0.values, result_rho05.values, atol=1e-6), (
        "T4: rho=0.0 and rho=0.5 produce identical results — unexpected."
    )


# ---------------------------------------------------------------------------
# T5 — explicit rho=0.5 satisfies conservation and differs from rho=0.0
# ---------------------------------------------------------------------------

def test_T5_explicit_rho_05_conservation(synthetic_data: dict) -> None:
    """T5: rho=0.5 satisfies conservation atol=1e-8 and differs from rho=0.0."""
    y_l_sum_s = synthetic_data["y_l_sum_s"]
    indicator = synthetic_data["indicator"]

    result = chow_lin_disaggregate(
        y_l_sum_s, indicator, aggregation="sum", rho=0.5
    )

    assert isinstance(result, pd.Series)

    # Conservation check
    reagg = result.resample("QE").sum()
    aligned = y_l_sum_s.reindex(reagg.index)
    assert np.allclose(reagg.values, aligned.values, atol=1e-8), (
        f"T5 conservation FAIL: max abs diff = "
        f"{np.max(np.abs(reagg.values - aligned.values)):.2e}"
    )

    # Differs from rho=0.0
    result_rho0 = chow_lin_disaggregate(
        y_l_sum_s, indicator, aggregation="sum", rho=0.0
    )
    assert not np.allclose(result.values, result_rho0.values, atol=1e-6), (
        "T5: rho=0.5 and rho=0.0 produce identical results — unexpected."
    )


# ---------------------------------------------------------------------------
# T6 — rho_method='max_likelihood' completes and conserves
# ---------------------------------------------------------------------------

def test_T6_max_likelihood_conservation(synthetic_data: dict) -> None:
    """T6: rho_method='max_likelihood' completes without error and conserves."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    result = chow_lin_disaggregate(
        y_l_mean_s, indicator, aggregation="mean", rho_method="max_likelihood"
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 60

    # Conservation check
    reagg = result.resample("QE").mean()
    aligned = y_l_mean_s.reindex(reagg.index)
    assert np.allclose(reagg.values, aligned.values, atol=1e-8), (
        f"T6 conservation FAIL: max abs diff = "
        f"{np.max(np.abs(reagg.values - aligned.values)):.2e}"
    )


# ---------------------------------------------------------------------------
# T7 — Invalid aggregation raises ValueError containing "aggregation"
# ---------------------------------------------------------------------------

def test_T7_invalid_aggregation_raises(synthetic_data: dict) -> None:
    """T7: aggregation='average' raises ValueError with 'aggregation' in message."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    with pytest.raises(ValueError, match="aggregation"):
        chow_lin_disaggregate(y_l_mean_s, indicator, aggregation="average")


# ---------------------------------------------------------------------------
# T8 — rho out of range raises ValueError
# ---------------------------------------------------------------------------

def test_T8_rho_out_of_range_above(synthetic_data: dict) -> None:
    """T8a: rho=1.5 raises ValueError containing 'rho'."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    with pytest.raises(ValueError, match="rho"):
        chow_lin_disaggregate(y_l_mean_s, indicator, rho=1.5)


def test_T8_rho_boundary_minus_one(synthetic_data: dict) -> None:
    """T8b: rho=-1.0 raises ValueError (open interval: -1 excluded)."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    with pytest.raises(ValueError):
        chow_lin_disaggregate(y_l_mean_s, indicator, rho=-1.0)


# ---------------------------------------------------------------------------
# T9 — rho_method invalid value raises ValueError containing "rho_method"
# ---------------------------------------------------------------------------

def test_T9_invalid_rho_method_raises(synthetic_data: dict) -> None:
    """T9: rho_method='ols' raises ValueError containing 'rho_method'."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    with pytest.raises(ValueError, match="rho_method"):
        chow_lin_disaggregate(y_l_mean_s, indicator, rho_method="ols")


# ---------------------------------------------------------------------------
# T10 — Non-DatetimeIndex input: graceful fallback (no exception)
# ---------------------------------------------------------------------------

def test_T10_non_datetimeindex_graceful_fallback() -> None:
    """T10: Integer-indexed inputs do not raise; return a pd.Series."""
    y_l_int = pd.Series([1.0, 2.0, 3.0, 4.0], index=[0, 3, 6, 9])
    ind_int = pd.Series(np.arange(12.0), index=range(12))

    # Must not raise
    result = chow_lin_disaggregate(y_l_int, ind_int)

    assert isinstance(result, pd.Series), (
        f"T10 FAIL: expected pd.Series, got {type(result)}"
    )


# ---------------------------------------------------------------------------
# T11 — Backward compatibility: two-positional signature
# ---------------------------------------------------------------------------

def test_T11_backward_compat_two_positional(synthetic_data: dict) -> None:
    """T11: Calling with only two positional args works and conserves."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    # Should not raise TypeError
    result = chow_lin_disaggregate(y_l_mean_s, indicator)

    assert isinstance(result, pd.Series)

    # Default aggregation='mean' applies — conservation check
    reagg = result.resample("QE").mean()
    aligned = y_l_mean_s.reindex(reagg.index)
    assert np.allclose(reagg.values, aligned.values, atol=1e-8), (
        f"T11 conservation FAIL: max abs diff = "
        f"{np.max(np.abs(reagg.values - aligned.values)):.2e}"
    )


# ---------------------------------------------------------------------------
# T12 — Return type and index alignment
# ---------------------------------------------------------------------------

def test_T12_return_type_and_index(synthetic_data: dict) -> None:
    """T12: Returns pd.Series of length 60 aligned with indicator index."""
    y_l_mean_s = synthetic_data["y_l_mean_s"]
    indicator = synthetic_data["indicator"]

    result = chow_lin_disaggregate(y_l_mean_s, indicator, aggregation="mean")

    assert isinstance(result, pd.Series), (
        f"T12 FAIL: isinstance(result, pd.Series) is False; got {type(result)}"
    )
    assert len(result) == 60, (
        f"T12 FAIL: len(result)={len(result)}, expected 60"
    )
    # Index aligned with indicator's first n_l*m elements
    assert result.index.equals(indicator.index[:60]), (
        "T12 FAIL: result.index does not match indicator.index[:60]"
    )


# ---------------------------------------------------------------------------
# I1 — Property invariant: conservation sum (multiple random inputs)
# ---------------------------------------------------------------------------

def test_I1_property_conservation_sum() -> None:
    """I1: Conservation (sum) holds for 5 different random low-freq inputs."""
    rng = np.random.RandomState(7)
    idx_m = pd.date_range("2015-01-31", periods=48, freq="ME")
    idx_q = pd.date_range("2015-03-31", periods=16, freq="QE")

    indicator = pd.Series(rng.randn(48), index=idx_m, name="ind")

    for trial in range(5):
        lf_vals = rng.randn(16)
        lf = pd.Series(lf_vals, index=idx_q, name="y")

        result = chow_lin_disaggregate(lf, indicator, aggregation="sum")
        reagg = result.resample("QE").sum()
        aligned = lf.reindex(reagg.index)

        max_err = float(np.max(np.abs(reagg.values - aligned.values)))
        assert max_err < 1e-8, (
            f"I1 FAIL trial={trial}: max abs conservation error = {max_err:.2e}"
        )


# ---------------------------------------------------------------------------
# I2 — Property invariant: conservation mean (multiple random inputs)
# ---------------------------------------------------------------------------

def test_I2_property_conservation_mean() -> None:
    """I2: Conservation (mean) holds for 5 different random low-freq inputs."""
    rng = np.random.RandomState(13)
    idx_m = pd.date_range("2015-01-31", periods=48, freq="ME")
    idx_q = pd.date_range("2015-03-31", periods=16, freq="QE")

    indicator = pd.Series(rng.randn(48), index=idx_m, name="ind")

    for trial in range(5):
        lf_vals = rng.randn(16)
        lf = pd.Series(lf_vals, index=idx_q, name="y")

        result = chow_lin_disaggregate(lf, indicator, aggregation="mean")
        reagg = result.resample("QE").mean()
        aligned = lf.reindex(reagg.index)

        max_err = float(np.max(np.abs(reagg.values - aligned.values)))
        assert max_err < 1e-8, (
            f"I2 FAIL trial={trial}: max abs conservation error = {max_err:.2e}"
        )


# ---------------------------------------------------------------------------
# R2 — Regression guard: private runtime helper still importable
# ---------------------------------------------------------------------------

def test_R2_private_runtime_helper_importable(synthetic_data: dict) -> None:
    """R2: _chow_lin_disaggregate in core.runtime imports and runs unchanged."""
    from macroforecast.core.runtime import _chow_lin_disaggregate  # type: ignore[attr-defined]

    y_l_sum_s = synthetic_data["y_l_sum_s"]
    indicator = synthetic_data["indicator"]

    # Original two-arg call signature must still work
    result = _chow_lin_disaggregate(y_l_sum_s, indicator)
    assert isinstance(result, pd.Series), (
        "R2 FAIL: _chow_lin_disaggregate did not return pd.Series"
    )
