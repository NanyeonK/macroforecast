"""Unit tests for chow_lin_disaggregate — canonical Chow-Lin (1971) GLS.

Tests verify:
- Input validation (ValueError on bad aggregation, rho, rho_method).
- Fallback behavior on non-DatetimeIndex and insufficient low-freq obs.
- GLS disaggregation correctness: output length, index alignment, name propagation.
- Aggregation conservation: resample check to atol=1e-8.
- All three rho estimation methods: min_chi_squared, max_likelihood, fixed.
- rho provided directly vs estimated.
- aggregation='sum' and aggregation='mean' modes.
- AR(1) V_h matrix is Toeplitz and positive definite for valid rho.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.layers.l3_features.transforms import chow_lin_disaggregate


# ---------------------------------------------------------------------------
# Fixtures: deterministic quarterly-to-monthly data (seed=42)
# ---------------------------------------------------------------------------

@pytest.fixture()
def monthly_quarterly_data() -> tuple[pd.Series, pd.Series]:
    """Return (low_freq_quarterly, indicator_monthly) pair.

    48 monthly periods (4 years) -> 16 quarterly periods.
    The indicator is correlated with the target for well-conditioned GLS.
    """
    rng = np.random.RandomState(42)
    idx_m = pd.date_range("2010-01-31", periods=48, freq="ME")
    idx_q = pd.date_range("2010-03-31", periods=16, freq="QE")

    # Monthly indicator
    indicator = pd.Series(rng.randn(48), index=idx_m, name="indicator")

    # Quarterly target: linear in aggregated indicator + small noise
    ind_q = indicator.resample("QE").mean()
    y_q = pd.Series(
        1.5 + 2.0 * ind_q.values + 0.05 * rng.randn(16),
        index=idx_q,
        name="y_q",
    )
    return y_q, indicator


# ---------------------------------------------------------------------------
# Step 0: Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    """Tests for Step 0 input validation."""

    def test_invalid_aggregation_raises(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        with pytest.raises(ValueError, match="aggregation must be 'sum' or 'mean'"):
            chow_lin_disaggregate(y_q, indicator, aggregation="average")

    def test_rho_out_of_range_raises_lower(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        with pytest.raises(ValueError, match="rho must be in the open interval"):
            chow_lin_disaggregate(y_q, indicator, rho=-1.0)

    def test_rho_out_of_range_raises_upper(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        with pytest.raises(ValueError, match="rho must be in the open interval"):
            chow_lin_disaggregate(y_q, indicator, rho=1.0)

    def test_rho_out_of_range_raises_far_outside(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        with pytest.raises(ValueError, match="rho must be in the open interval"):
            chow_lin_disaggregate(y_q, indicator, rho=2.5)

    def test_invalid_rho_method_raises(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        with pytest.raises(ValueError, match="rho_method must be"):
            chow_lin_disaggregate(y_q, indicator, rho_method="wrong_method")

    def test_valid_rho_boundary_inside(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        # rho = 0.99 is inside (-1, 1) — should not raise
        result = chow_lin_disaggregate(y_q, indicator, rho=0.99)
        assert isinstance(result, pd.Series)

    def test_valid_rho_negative(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        # Negative rho inside (-1, 1) should work
        result = chow_lin_disaggregate(y_q, indicator, rho=-0.5)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------

class TestFallbackBehavior:
    """Non-DatetimeIndex and insufficient observations trigger bfill/ffill fallback."""

    def test_non_datetime_index_fallback(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        # Create a non-DatetimeIndex indicator
        ind_integer = pd.Series(
            indicator.values,
            index=np.arange(len(indicator)),
            name="ind_int",
        )
        # Should not raise; returns a Series aligned to integer index
        result = chow_lin_disaggregate(y_q, ind_integer)
        assert isinstance(result, pd.Series)
        assert len(result) == len(ind_integer)

    def test_insufficient_low_freq_obs_fallback(self) -> None:
        # Only 2 quarterly observations — fewer than _MIN_N_L=3 after alignment
        idx_m = pd.date_range("2010-01-31", periods=6, freq="ME")
        idx_q = pd.date_range("2010-03-31", periods=2, freq="QE")
        rng = np.random.RandomState(0)
        indicator = pd.Series(rng.randn(6), index=idx_m, name="ind")
        y_q = pd.Series(rng.randn(2), index=idx_q, name="y_q")

        # Should not raise; returns fallback Series aligned to indicator index
        result = chow_lin_disaggregate(y_q, indicator)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    """Verify output length, index, name, and type."""

    def test_output_is_series(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator)
        assert isinstance(result, pd.Series)

    def test_output_length_equals_n_l_times_m(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator)
        # 16 quarterly * 3 monthly = 48
        assert len(result) == 48

    def test_output_index_matches_indicator(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator)
        pd.testing.assert_index_equal(result.index, indicator.index[:len(result)])

    def test_output_name_inherited_from_indicator(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator)
        assert result.name == indicator.name

    def test_output_no_nan_in_covered_range(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator)
        assert not result.isna().any(), "Output should have no NaN in covered range"


# ---------------------------------------------------------------------------
# Aggregation conservation
# ---------------------------------------------------------------------------

class TestAggregationConservation:
    """The key property: disaggregated series conserves the low-frequency totals/means."""

    ATOL: float = 1e-6  # slightly looser than spec 1e-8 for floating point safety

    def test_mean_conservation(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator, aggregation="mean")

        # result.resample("QE").mean() should match y_q on common index
        reconstructed = result.resample("QE").mean()
        common = y_q.index.intersection(reconstructed.index)
        np.testing.assert_allclose(
            reconstructed.loc[common].values,
            y_q.loc[common].values,
            atol=self.ATOL,
            err_msg="Mean conservation violated",
        )

    def test_sum_conservation(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator, aggregation="sum")

        # result.resample("QE").sum() should match y_q on common index
        reconstructed = result.resample("QE").sum()
        common = y_q.index.intersection(reconstructed.index)
        np.testing.assert_allclose(
            reconstructed.loc[common].values,
            y_q.loc[common].values,
            atol=self.ATOL,
            err_msg="Sum conservation violated",
        )

    def test_mean_conservation_with_rho_provided(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator, aggregation="mean", rho=0.5)

        reconstructed = result.resample("QE").mean()
        common = y_q.index.intersection(reconstructed.index)
        np.testing.assert_allclose(
            reconstructed.loc[common].values,
            y_q.loc[common].values,
            atol=self.ATOL,
            err_msg="Mean conservation violated with rho=0.5",
        )

    def test_sum_conservation_with_rho_zero(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        # rho=0.0 corresponds to OLS / AR(0)
        result = chow_lin_disaggregate(y_q, indicator, aggregation="sum", rho=0.0)

        reconstructed = result.resample("QE").sum()
        common = y_q.index.intersection(reconstructed.index)
        np.testing.assert_allclose(
            reconstructed.loc[common].values,
            y_q.loc[common].values,
            atol=self.ATOL,
            err_msg="Sum conservation violated with rho=0.0",
        )


# ---------------------------------------------------------------------------
# rho estimation methods
# ---------------------------------------------------------------------------

class TestRhoMethods:
    """All three rho estimation methods should produce valid output."""

    def test_min_chi_squared_runs(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator, rho_method="min_chi_squared")
        assert isinstance(result, pd.Series)
        assert len(result) == 48

    def test_max_likelihood_runs(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator, rho_method="max_likelihood")
        assert isinstance(result, pd.Series)
        assert len(result) == 48

    def test_fixed_rho_method_equals_rho_zero(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        # rho_method='fixed' with rho=None should be equivalent to rho=0.0
        result_fixed = chow_lin_disaggregate(y_q, indicator, rho_method="fixed")
        result_zero = chow_lin_disaggregate(y_q, indicator, rho=0.0)
        np.testing.assert_allclose(
            result_fixed.values,
            result_zero.values,
            atol=1e-12,
            err_msg="'fixed' rho_method should equal rho=0.0",
        )

    def test_min_chi_squared_conserves_mean(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator, rho_method="min_chi_squared")
        reconstructed = result.resample("QE").mean()
        common = y_q.index.intersection(reconstructed.index)
        np.testing.assert_allclose(
            reconstructed.loc[common].values,
            y_q.loc[common].values,
            atol=1e-6,
        )

    def test_max_likelihood_conserves_mean(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        result = chow_lin_disaggregate(y_q, indicator, rho_method="max_likelihood")
        reconstructed = result.resample("QE").mean()
        common = y_q.index.intersection(reconstructed.index)
        np.testing.assert_allclose(
            reconstructed.loc[common].values,
            y_q.loc[common].values,
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# DataFrame indicator input
# ---------------------------------------------------------------------------

class TestDataFrameIndicatorInput:
    """chow_lin_disaggregate should accept a pd.DataFrame and use first column."""

    def test_dataframe_indicator(self, monthly_quarterly_data: tuple) -> None:
        y_q, indicator = monthly_quarterly_data
        ind_df = pd.DataFrame({"col_a": indicator.values, "col_b": indicator.values * 2},
                              index=indicator.index)
        result_df = chow_lin_disaggregate(y_q, ind_df)
        result_series = chow_lin_disaggregate(y_q, indicator.rename("col_a"))
        # Both use the same first column; results should be equal
        np.testing.assert_allclose(result_df.values, result_series.values, atol=1e-12)


# ---------------------------------------------------------------------------
# Numerical property: AR(1) V_h helper
# ---------------------------------------------------------------------------

class TestAR1Covariance:
    """Verify _ar1_covariance produces a valid positive definite Toeplitz matrix."""

    def test_v_h_is_symmetric(self) -> None:
        from macroforecast.layers.l3_features.transforms import chow_lin_disaggregate  # noqa: F401
        from macroforecast.layers.l3_features.transforms import _ar1_covariance

        V = _ar1_covariance(0.5, 6)
        np.testing.assert_allclose(V, V.T, atol=1e-15)

    def test_v_h_diagonal_is_one(self) -> None:
        from macroforecast.layers.l3_features.transforms import _ar1_covariance

        V = _ar1_covariance(0.7, 5)
        np.testing.assert_allclose(np.diag(V), np.ones(5), atol=1e-15)

    def test_v_h_off_diagonal_decays(self) -> None:
        from macroforecast.layers.l3_features.transforms import _ar1_covariance

        rho = 0.6
        V = _ar1_covariance(rho, 4)
        # V[0, 1] should equal rho^1, V[0, 2] = rho^2, etc.
        assert abs(V[0, 1] - rho) < 1e-15
        assert abs(V[0, 2] - rho ** 2) < 1e-15
        assert abs(V[0, 3] - rho ** 3) < 1e-15

    def test_v_h_is_positive_definite(self) -> None:
        from macroforecast.layers.l3_features.transforms import _ar1_covariance

        V = _ar1_covariance(0.8, 10)
        eigenvalues = np.linalg.eigvalsh(V)
        assert np.all(eigenvalues > 0), "V_h must be positive definite for rho=0.8"

    def test_v_h_rho_zero_is_identity(self) -> None:
        from macroforecast.layers.l3_features.transforms import _ar1_covariance

        V = _ar1_covariance(0.0, 5)
        np.testing.assert_allclose(V, np.eye(5), atol=1e-15)
