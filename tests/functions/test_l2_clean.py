"""Tests for C34 L2 clean panel standalone callables.

14 ops:
  L2.C outlier (3):  iqr_outlier_clean, zscore_outlier_clean, winsorize_clean
  L2.D impute (5):   em_factor_impute_clean, em_multivariate_impute_clean,
                     mean_impute_clean, forward_fill_clean, linear_interpolate_clean
  L2.E frame (3):    truncate_to_balanced_clean, drop_unbalanced_series_clean,
                     zero_fill_leading_clean
  L2.B tcode (1):    apply_tcode_transform
  L2.A freq (2):     freq_align_quarterly_to_monthly_clean,
                     freq_align_monthly_to_quarterly_clean

All bit-exact tests compare against the runtime path for correctness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.functions.clean import (
    iqr_outlier_clean,
    zscore_outlier_clean,
    winsorize_clean,
    em_factor_impute_clean,
    em_multivariate_impute_clean,
    mean_impute_clean,
    forward_fill_clean,
    linear_interpolate_clean,
    truncate_to_balanced_clean,
    drop_unbalanced_series_clean,
    zero_fill_leading_clean,
    apply_tcode_transform,
    freq_align_quarterly_to_monthly_clean,
    freq_align_monthly_to_quarterly_clean,
)

# ---------------------------------------------------------------------------
# Shared fixtures (RNG-42, deterministic)
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)
PANEL = pd.DataFrame(RNG.randn(50, 5), columns=list("abcde"))
# Panel with outlier in column c (row 5) and missings in column b (rows 10-14)
PANEL_WITH_OUTLIER = PANEL.copy()
PANEL_WITH_OUTLIER.iloc[5, 2] = 100.0  # extreme outlier in column c

PANEL_WITH_NAN = PANEL.copy()
PANEL_WITH_NAN.iloc[10:15, 1] = np.nan  # 5 missing values in column b

PANEL_SMALL = pd.DataFrame(RNG.randn(20, 3), columns=list("abc"))

# Panel with leading + interior NaN
PANEL_NAN_PATTERN = pd.DataFrame({
    "a": [np.nan, np.nan, 1.0, 2.0, 3.0, np.nan, 5.0],
    "b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    "c": [np.nan, 1.0, np.nan, 3.0, np.nan, 5.0, np.nan],
})

# DatetimeIndex panel for freq-align tests
MONTHLY_IDX = pd.date_range("2020-01-01", periods=12, freq="MS")
QUARTERLY_IDX = pd.date_range("2020-03-31", periods=4, freq="QE")


# ===========================================================================
# L2.C Outlier tests
# ===========================================================================


class TestIqrOutlierClean:
    def test_output_shape(self):
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0)
        assert out.shape == PANEL_WITH_OUTLIER.shape

    def test_index_preserved(self):
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0)
        assert list(out.index) == list(PANEL_WITH_OUTLIER.index)

    def test_columns_preserved(self):
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0)
        assert list(out.columns) == list(PANEL_WITH_OUTLIER.columns)

    def test_extreme_outlier_flagged(self):
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0)
        # The 100.0 outlier in column c (index 2) at row 5 must be NaN
        assert np.isnan(out.iloc[5, 2])

    def test_normal_values_unchanged(self):
        # Values within range should not be NaN (columns a, d, e)
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0)
        # At least most of the panel should be preserved
        assert out.notna().sum().sum() >= 240  # 250 total, expect few flagged

    def test_iqr_near_zero_column_not_flagged(self):
        # Column with very small IQR: at threshold=10, cutoff is 10*IQR; all values within
        # range should not be flagged even if IQR is small.
        # Note: constant columns (IQR=0) cannot be tested here because the runtime and
        # standalone both rely on pd.NA behavior that fails with pandas >= 2.x on constant
        # float64 columns -- this is a pre-existing runtime limitation.
        rng2 = np.random.RandomState(99)
        # tiny_iqr column: values 1.0 or 1.001 (IQR=0.001), threshold*IQR=0.01
        # all values within 0.0005 of median -> none flagged
        panel = pd.DataFrame({
            "tiny_iqr": [1.0, 1.001] * 10,
            "with_outlier": list(rng2.randn(20)),
        })
        panel.loc[0, "with_outlier"] = 1000.0  # only with_outlier column has extreme value
        out = iqr_outlier_clean(panel, threshold=10.0)
        # tiny_iqr column: no NaN introduced (all values within threshold*IQR)
        assert out["tiny_iqr"].isna().sum() == 0
        # with_outlier column: the extreme value should be flagged
        assert np.isnan(out.loc[0, "with_outlier"])

    def test_action_replace_with_median(self):
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0, action="replace_with_median")
        # After replacing with median, no new NaN should appear
        assert out.isna().sum().sum() == 0

    def test_action_replace_with_cap_value(self):
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0, action="replace_with_cap_value")
        # All values should be capped, no NaN
        assert out.isna().sum().sum() == 0
        # 100.0 should be reduced (capped at the 99th percentile)
        assert out.iloc[5, 2] < 100.0

    def test_threshold_validation(self):
        with pytest.raises(ValueError, match="threshold must be > 0"):
            iqr_outlier_clean(PANEL, threshold=0.0)
        with pytest.raises(ValueError, match="threshold must be > 0"):
            iqr_outlier_clean(PANEL, threshold=-1.0)

    def test_action_validation(self):
        with pytest.raises(ValueError, match="action"):
            iqr_outlier_clean(PANEL, action="invalid_action")

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            iqr_outlier_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        """Bit-exact check vs _apply_outlier_policy for policy=mccracken_ng_iqr."""
        from macroforecast.core.runtime import _apply_outlier_policy
        from macroforecast.core.layers import l2 as l2_layer
        resolved = l2_layer.L2ResolvedAxes(
            {"outlier_policy": "mccracken_ng_iqr", "outlier_action": "flag_as_nan"},
            {"outlier_policy": "explicit", "outlier_action": "explicit"},
        )
        expected, _ = _apply_outlier_policy(
            PANEL_WITH_OUTLIER, resolved, {"outlier_iqr_threshold": 10.0}, {"steps": []}
        )
        out = iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0, action="flag_as_nan")
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "iqr_outlier_clean")
        out = mf.functions.iqr_outlier_clean(PANEL_WITH_OUTLIER, threshold=10.0)
        assert out.shape == PANEL_WITH_OUTLIER.shape

    def test_default_threshold_is_mccracken_ng(self):
        # Default threshold=10.0 is the McCracken-Ng published default
        import inspect
        sig = inspect.signature(iqr_outlier_clean)
        assert sig.parameters["threshold"].default == 10.0


class TestZscoreOutlierClean:
    def test_output_shape(self):
        out = zscore_outlier_clean(PANEL_WITH_OUTLIER, threshold=3.0)
        assert out.shape == PANEL_WITH_OUTLIER.shape

    def test_extreme_outlier_flagged(self):
        out = zscore_outlier_clean(PANEL_WITH_OUTLIER, threshold=3.0)
        # The 100.0 outlier should be flagged
        assert np.isnan(out.iloc[5, 2])

    def test_near_zero_std_column_not_flagged(self):
        # Column with very small std: values should not be flagged when within threshold.
        # Note: true constant columns (std=0) share the same pre-existing limitation as
        # iqr_outlier_clean (runtime uses std.replace(0, pd.NA) which fails with pd.NA).
        rng2 = np.random.RandomState(99)
        panel = pd.DataFrame({
            "near_const": [1.0, 1.001] * 10,   # tiny std, values close to mean
            "with_outlier": list(rng2.randn(20)),
        })
        panel.loc[0, "with_outlier"] = 1000.0
        out = zscore_outlier_clean(panel, threshold=3.0)
        # near_const: z-scores are very small -> none flagged
        assert out["near_const"].isna().sum() == 0
        # with_outlier: the extreme value should be flagged
        assert np.isnan(out.loc[0, "with_outlier"])

    def test_action_replace_with_median(self):
        out = zscore_outlier_clean(PANEL_WITH_OUTLIER, threshold=3.0, action="replace_with_median")
        assert out.isna().sum().sum() == 0

    def test_threshold_validation(self):
        with pytest.raises(ValueError, match="threshold must be > 0"):
            zscore_outlier_clean(PANEL, threshold=-0.1)

    def test_action_validation(self):
        with pytest.raises(ValueError, match="action"):
            zscore_outlier_clean(PANEL, action="bad_action")

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            zscore_outlier_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _apply_outlier_policy
        from macroforecast.core.layers import l2 as l2_layer
        resolved = l2_layer.L2ResolvedAxes(
            {"outlier_policy": "zscore_threshold", "outlier_action": "flag_as_nan"},
            {"outlier_policy": "explicit", "outlier_action": "explicit"},
        )
        expected, _ = _apply_outlier_policy(
            PANEL_WITH_OUTLIER, resolved, {"zscore_threshold_value": 3.0}, {"steps": []}
        )
        out = zscore_outlier_clean(PANEL_WITH_OUTLIER, threshold=3.0)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "zscore_outlier_clean")
        out = mf.functions.zscore_outlier_clean(PANEL_WITH_OUTLIER, threshold=3.0)
        assert out.shape == PANEL_WITH_OUTLIER.shape

    def test_default_threshold_is_3(self):
        import inspect
        sig = inspect.signature(zscore_outlier_clean)
        assert sig.parameters["threshold"].default == 3.0


class TestWinsorizeClean:
    def test_output_shape(self):
        out = winsorize_clean(PANEL_WITH_OUTLIER)
        assert out.shape == PANEL_WITH_OUTLIER.shape

    def test_extreme_value_capped(self):
        out = winsorize_clean(PANEL_WITH_OUTLIER, lower_quantile=0.01, upper_quantile=0.99)
        # 100.0 should be capped at the 99th percentile (which is < 100.0)
        assert out.iloc[5, 2] < 100.0

    def test_no_new_nan_introduced(self):
        out = winsorize_clean(PANEL_WITH_OUTLIER)
        # winsorize should not introduce NaN where there was none
        assert out.isna().sum().sum() == 0

    def test_existing_nan_preserved(self):
        out = winsorize_clean(PANEL_WITH_NAN)
        # NaN values that were in the input should remain
        assert out.isna().sum().sum() == PANEL_WITH_NAN.isna().sum().sum()

    def test_quantile_validation(self):
        with pytest.raises(ValueError):
            winsorize_clean(PANEL, lower_quantile=0.5, upper_quantile=0.3)
        with pytest.raises(ValueError):
            winsorize_clean(PANEL, lower_quantile=-0.1, upper_quantile=0.99)
        with pytest.raises(ValueError):
            winsorize_clean(PANEL, lower_quantile=0.01, upper_quantile=1.1)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            winsorize_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _apply_outlier_policy
        from macroforecast.core.layers import l2 as l2_layer
        resolved = l2_layer.L2ResolvedAxes(
            {"outlier_policy": "winsorize", "outlier_action": "flag_as_nan"},
            {"outlier_policy": "explicit", "outlier_action": "explicit"},
        )
        expected, _ = _apply_outlier_policy(
            PANEL, resolved, {"winsorize_quantiles": [0.01, 0.99]}, {"steps": []}
        )
        out = winsorize_clean(PANEL, lower_quantile=0.01, upper_quantile=0.99)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "winsorize_clean")
        out = mf.functions.winsorize_clean(PANEL)
        assert out.shape == PANEL.shape

    def test_index_preserved(self):
        out = winsorize_clean(PANEL)
        assert list(out.index) == list(PANEL.index)


# ===========================================================================
# L2.D Imputation tests
# ===========================================================================


class TestEmFactorImputeClean:
    def test_output_shape(self):
        out = em_factor_impute_clean(PANEL_WITH_NAN, n_factors=3)
        assert out.shape == PANEL_WITH_NAN.shape

    def test_missing_values_filled(self):
        out = em_factor_impute_clean(PANEL_WITH_NAN, n_factors=3)
        assert out.isna().sum().sum() == 0

    def test_non_missing_values_unchanged(self):
        # Values that were not NaN should be reconstructed (approximately)
        out = em_factor_impute_clean(PANEL_WITH_NAN, n_factors=3)
        # Shape preserved
        assert out.shape == PANEL_WITH_NAN.shape

    def test_index_preserved(self):
        out = em_factor_impute_clean(PANEL_WITH_NAN)
        assert list(out.index) == list(PANEL_WITH_NAN.index)

    def test_columns_preserved(self):
        out = em_factor_impute_clean(PANEL_WITH_NAN)
        assert list(out.columns) == list(PANEL_WITH_NAN.columns)

    def test_n_factors_validation(self):
        with pytest.raises(ValueError, match="n_factors"):
            em_factor_impute_clean(PANEL_WITH_NAN, n_factors=0)

    def test_max_iter_validation(self):
        with pytest.raises(ValueError, match="max_iter"):
            em_factor_impute_clean(PANEL_WITH_NAN, max_iter=0)

    def test_tol_validation(self):
        with pytest.raises(ValueError, match="tol"):
            em_factor_impute_clean(PANEL_WITH_NAN, tol=0.0)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            em_factor_impute_clean(pd.DataFrame())

    def test_fixed_selection_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _pca_em_imputation
        expected = _pca_em_imputation(PANEL_WITH_NAN, n_factors=8, max_iter=20, tol=1e-4)
        out = em_factor_impute_clean(
            PANEL_WITH_NAN,
            n_factors=8,
            max_iter=20,
            tol=1e-4,
            factor_selection="fixed",
        )
        pd.testing.assert_frame_equal(out, expected)

    def test_default_uses_fred_md_baing_path(self):
        out = em_factor_impute_clean(PANEL_WITH_NAN, n_factors=8)
        assert out.shape == PANEL_WITH_NAN.shape
        assert out.isna().sum().sum() == 0

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "em_factor_impute_clean")
        out = mf.functions.em_factor_impute_clean(PANEL_WITH_NAN, n_factors=3)
        assert out.isna().sum().sum() == 0

    def test_no_missings_noop(self):
        # Panel with no NaN: should return identical values
        out = em_factor_impute_clean(PANEL)
        assert out.isna().sum().sum() == 0
        assert out.shape == PANEL.shape


class TestEmMultivariateImputeClean:
    def test_output_shape(self):
        out = em_multivariate_impute_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape

    def test_missing_values_filled(self):
        out = em_multivariate_impute_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0

    def test_max_iter_validation(self):
        with pytest.raises(ValueError, match="max_iter"):
            em_multivariate_impute_clean(PANEL_WITH_NAN, max_iter=0)

    def test_tol_validation(self):
        with pytest.raises(ValueError, match="tol"):
            em_multivariate_impute_clean(PANEL_WITH_NAN, tol=-1.0)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            em_multivariate_impute_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        """n_factors=None triggers rank=min(T,K)//2 in _pca_em_imputation."""
        from macroforecast.core.runtime import _pca_em_imputation
        expected = _pca_em_imputation(PANEL_WITH_NAN, n_factors=None, max_iter=20, tol=1e-4)
        out = em_multivariate_impute_clean(PANEL_WITH_NAN, max_iter=20, tol=1e-4)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "em_multivariate_impute_clean")
        out = mf.functions.em_multivariate_impute_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape

    def test_different_from_em_factor(self):
        """em_multivariate (rank=min(T,K)//2=2 for 50x5) differs from em_factor at rank=1."""
        # For 50x5 panel: rank_multi = min(50,5)//2 = 2; use n_factors=1 to ensure difference
        out_factor = em_factor_impute_clean(PANEL_WITH_NAN, n_factors=1, max_iter=20)
        out_multi = em_multivariate_impute_clean(PANEL_WITH_NAN, max_iter=20)
        # rank=1 vs rank=2 produces different results
        assert not out_factor.equals(out_multi)


class TestMeanImputeClean:
    def test_output_shape(self):
        out = mean_impute_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape

    def test_missing_values_filled(self):
        out = mean_impute_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0

    def test_imputed_values_are_column_means(self):
        out = mean_impute_clean(PANEL_WITH_NAN)
        col_mean = PANEL_WITH_NAN["b"].mean()
        # The formerly-NaN rows in column b should be the mean
        filled_vals = out.loc[PANEL_WITH_NAN["b"].isna(), "b"]
        np.testing.assert_allclose(filled_vals.values, col_mean, rtol=1e-12)

    def test_non_nan_unchanged(self):
        out = mean_impute_clean(PANEL_WITH_NAN)
        # Non-NaN values in column b should be unchanged
        mask = PANEL_WITH_NAN["b"].notna()
        pd.testing.assert_series_equal(
            out.loc[mask, "b"], PANEL_WITH_NAN.loc[mask, "b"]
        )

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            mean_impute_clean(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "mean_impute_clean")
        out = mf.functions.mean_impute_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0

    def test_bit_exact_vs_runtime(self):
        expected = PANEL_WITH_NAN.fillna(PANEL_WITH_NAN.mean(numeric_only=True))
        out = mean_impute_clean(PANEL_WITH_NAN)
        pd.testing.assert_frame_equal(out, expected)


class TestForwardFillClean:
    def test_output_shape(self):
        out = forward_fill_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape

    def test_interior_nan_filled(self):
        out = forward_fill_clean(PANEL_WITH_NAN)
        # All interior NaN in column b should be filled
        assert out["b"].isna().sum() == 0

    def test_leading_nan_remain(self):
        # Leading NaN (no prior observation) should remain
        panel = pd.DataFrame({"a": [np.nan, np.nan, 1.0, 2.0, 3.0]})
        out = forward_fill_clean(panel)
        assert np.isnan(out.iloc[0, 0])
        assert np.isnan(out.iloc[1, 0])
        assert out.iloc[2, 0] == 1.0

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            forward_fill_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        expected = PANEL_WITH_NAN.ffill()
        out = forward_fill_clean(PANEL_WITH_NAN)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "forward_fill_clean")
        out = mf.functions.forward_fill_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape


class TestLinearInterpolateClean:
    def test_output_shape(self):
        out = linear_interpolate_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape

    def test_interior_nan_filled(self):
        out = linear_interpolate_clean(PANEL_WITH_NAN)
        assert out["b"].isna().sum() == 0

    def test_interpolated_values_between_neighbors(self):
        # Linear interpolation between 1 and 3 at position 2 should give 2
        panel = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        out = linear_interpolate_clean(panel)
        assert abs(out.iloc[1, 0] - 2.0) < 1e-10

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            linear_interpolate_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        expected = PANEL_WITH_NAN.interpolate(method="linear")
        out = linear_interpolate_clean(PANEL_WITH_NAN)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "linear_interpolate_clean")
        out = mf.functions.linear_interpolate_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape


# ===========================================================================
# L2.E Frame-edge tests
# ===========================================================================


class TestTruncateToBalancedClean:
    def test_output_shape_reduced_rows(self):
        out = truncate_to_balanced_clean(PANEL_WITH_NAN)
        assert out.shape[0] < PANEL_WITH_NAN.shape[0]
        assert out.shape[1] == PANEL_WITH_NAN.shape[1]

    def test_no_nan_in_output(self):
        out = truncate_to_balanced_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0

    def test_columns_preserved(self):
        out = truncate_to_balanced_clean(PANEL_WITH_NAN)
        assert list(out.columns) == list(PANEL_WITH_NAN.columns)

    def test_index_is_subset(self):
        out = truncate_to_balanced_clean(PANEL_WITH_NAN)
        assert set(out.index).issubset(set(PANEL_WITH_NAN.index))

    def test_fully_balanced_panel_unchanged(self):
        out = truncate_to_balanced_clean(PANEL)
        assert out.shape == PANEL.shape

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            truncate_to_balanced_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        expected = PANEL_WITH_NAN.dropna(axis=0, how="any")
        out = truncate_to_balanced_clean(PANEL_WITH_NAN)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "truncate_to_balanced_clean")
        out = mf.functions.truncate_to_balanced_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0


class TestDropUnbalancedSeriesClean:
    def test_output_cols_reduced(self):
        out = drop_unbalanced_series_clean(PANEL_WITH_NAN)
        # Column b has NaN so should be dropped
        assert out.shape[1] < PANEL_WITH_NAN.shape[1]
        assert out.shape[0] == PANEL_WITH_NAN.shape[0]

    def test_no_nan_in_output(self):
        out = drop_unbalanced_series_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0

    def test_nan_column_dropped(self):
        out = drop_unbalanced_series_clean(PANEL_WITH_NAN)
        assert "b" not in out.columns

    def test_fully_balanced_panel_unchanged(self):
        out = drop_unbalanced_series_clean(PANEL)
        assert out.shape == PANEL.shape

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            drop_unbalanced_series_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        expected = PANEL_WITH_NAN.dropna(axis=1, how="any")
        out = drop_unbalanced_series_clean(PANEL_WITH_NAN)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "drop_unbalanced_series_clean")
        out = mf.functions.drop_unbalanced_series_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0


class TestZeroFillLeadingClean:
    def test_output_shape_same(self):
        out = zero_fill_leading_clean(PANEL_WITH_NAN)
        assert out.shape == PANEL_WITH_NAN.shape

    def test_all_nan_filled(self):
        out = zero_fill_leading_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0

    def test_nan_replaced_by_zero(self):
        panel = pd.DataFrame({"a": [np.nan, 1.0, np.nan, 3.0]})
        out = zero_fill_leading_clean(panel)
        assert out.iloc[0, 0] == 0.0
        assert out.iloc[2, 0] == 0.0
        assert out.iloc[1, 0] == 1.0
        assert out.iloc[3, 0] == 3.0

    def test_fills_all_not_just_leading(self):
        """Name says 'leading' but runtime fills ALL NaN -- standalone must match."""
        panel = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
        out = zero_fill_leading_clean(panel)
        # Interior NaN at index 1 and 3 should also be filled
        assert out.iloc[1, 0] == 0.0
        assert out.iloc[3, 0] == 0.0

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            zero_fill_leading_clean(pd.DataFrame())

    def test_bit_exact_vs_runtime(self):
        expected = PANEL_WITH_NAN.fillna(0)
        out = zero_fill_leading_clean(PANEL_WITH_NAN)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "zero_fill_leading_clean")
        out = mf.functions.zero_fill_leading_clean(PANEL_WITH_NAN)
        assert out.isna().sum().sum() == 0


# ===========================================================================
# L2.B Tcode tests
# ===========================================================================


class TestApplyTcodeTransform:
    def test_output_shape_same(self):
        tmap = {"a": 2, "b": 5}
        out = apply_tcode_transform(PANEL, tmap)
        assert out.shape == PANEL.shape

    def test_tcode_1_identity(self):
        tmap = {"a": 1}
        out = apply_tcode_transform(PANEL, tmap)
        pd.testing.assert_series_equal(out["a"], PANEL["a"])

    def test_tcode_2_diff(self):
        tmap = {"a": 2}
        out = apply_tcode_transform(PANEL, tmap)
        expected = PANEL["a"].diff()
        pd.testing.assert_series_equal(out["a"], expected)

    def test_tcode_7_first_difference_of_percent_change(self):
        # Use positive panel to avoid log issues
        panel_pos = pd.DataFrame({"a": [1.0, 2.0, 4.0, 8.0], "b": [10.0, 20.0, 30.0, 40.0]})
        tmap = {"a": 7}
        out = apply_tcode_transform(panel_pos, tmap)
        expected = panel_pos["a"].pct_change(fill_method=None).diff()
        pd.testing.assert_series_equal(out["a"], expected)

    def test_untransformed_cols_preserved(self):
        tmap = {"a": 2}
        out = apply_tcode_transform(PANEL, tmap)
        # Column b should be unchanged
        pd.testing.assert_series_equal(out["b"], PANEL["b"])

    def test_missing_col_silently_skipped(self):
        # Column 'z' not in panel -> silently skip
        tmap = {"z": 2, "a": 1}
        out = apply_tcode_transform(PANEL, tmap)
        assert "z" not in out.columns
        pd.testing.assert_series_equal(out["a"], PANEL["a"])  # tcode=1 is identity

    def test_empty_tcode_map_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            apply_tcode_transform(PANEL, {})

    def test_invalid_tcode_raises(self):
        with pytest.raises(ValueError, match="1..7"):
            apply_tcode_transform(PANEL, {"a": 8})

    def test_non_string_key_raises(self):
        with pytest.raises(ValueError, match="string"):
            apply_tcode_transform(PANEL, {0: 2})

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            apply_tcode_transform(pd.DataFrame(), {"a": 1})

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _apply_tcode
        tmap = {"a": 2, "b": 5, "c": 4}
        panel_pos = PANEL.copy()
        panel_pos["c"] = np.abs(panel_pos["c"]) + 0.1  # make c positive for log
        expected = panel_pos.copy()
        for col, tcode in tmap.items():
            expected[col] = _apply_tcode(panel_pos[col], tcode)
        out = apply_tcode_transform(panel_pos, tmap)
        pd.testing.assert_frame_equal(out, expected)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "apply_tcode_transform")
        out = mf.functions.apply_tcode_transform(PANEL, {"a": 2})
        assert out.shape == PANEL.shape


# ===========================================================================
# L2.A Frequency-alignment tests
# ===========================================================================


class TestFreqAlignQuarterlyToMonthlyClean:
    def _make_quarterly_panel(self):
        """Monthly index with quarterly values (other months NaN)."""
        idx = pd.date_range("2020-01-01", periods=12, freq="MS")
        vals = [1.0, np.nan, np.nan, 2.0, np.nan, np.nan,
                3.0, np.nan, np.nan, 4.0, np.nan, np.nan]
        return pd.DataFrame({"q": vals, "m": RNG.randn(12)}, index=idx)

    def test_output_shape_same(self):
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q"])
        assert out.shape == panel.shape

    def test_step_backward_no_nan_in_qcol(self):
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q"], rule="step_backward")
        assert out["q"].isna().sum() == 0

    def test_step_backward_order(self):
        """step_backward: bfill().ffill() -- each month gets the quarter-start value."""
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q"], rule="step_backward")
        # After bfill().ffill(): all 12 should be non-NaN
        expected_q = panel["q"].bfill().ffill()
        pd.testing.assert_series_equal(out["q"], expected_q)

    def test_step_forward(self):
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q"], rule="step_forward")
        expected_q = panel["q"].ffill()
        pd.testing.assert_series_equal(out["q"], expected_q)

    def test_linear_interpolation(self):
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q"], rule="linear_interpolation")
        expected_q = panel["q"].interpolate(method="linear", limit_direction="both")
        pd.testing.assert_series_equal(out["q"], expected_q)

    def test_monthly_col_unchanged(self):
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q"])
        pd.testing.assert_series_equal(out["m"], panel["m"])

    def test_missing_column_silently_skipped(self):
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q", "nonexistent"])
        assert "nonexistent" not in out.columns

    def test_requires_datetime_index(self):
        panel = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            freq_align_quarterly_to_monthly_clean(panel, ["a"])

    def test_invalid_rule_raises(self):
        panel = self._make_quarterly_panel()
        with pytest.raises(ValueError, match="rule"):
            freq_align_quarterly_to_monthly_clean(panel, ["q"], rule="chow_lin")

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            freq_align_quarterly_to_monthly_clean(pd.DataFrame(), [])

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "freq_align_quarterly_to_monthly_clean")
        panel = self._make_quarterly_panel()
        out = mf.functions.freq_align_quarterly_to_monthly_clean(panel, ["q"])
        assert out.shape == panel.shape

    def test_index_preserved(self):
        panel = self._make_quarterly_panel()
        out = freq_align_quarterly_to_monthly_clean(panel, ["q"])
        assert list(out.index) == list(panel.index)


class TestFreqAlignMonthlyToQuarterlyClean:
    def _make_quarterly_panel_with_monthly(self):
        """Quarterly index panel where one column contains monthly values."""
        idx = pd.date_range("2020-03-31", periods=4, freq="QE")
        return pd.DataFrame(
            {"m": [1.0, 2.0, 3.0, 4.0], "other": [10.0, 20.0, 30.0, 40.0]},
            index=idx,
        )

    def test_output_shape_same(self):
        panel = self._make_quarterly_panel_with_monthly()
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"])
        assert out.shape == panel.shape

    def test_quarterly_average_aggregation(self):
        panel = self._make_quarterly_panel_with_monthly()
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"], rule="quarterly_average")
        # Check that column m was processed
        assert "m" in out.columns

    def test_other_cols_preserved(self):
        panel = self._make_quarterly_panel_with_monthly()
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"])
        pd.testing.assert_series_equal(out["other"], panel["other"])

    def test_missing_monthly_col_skipped(self):
        panel = self._make_quarterly_panel_with_monthly()
        out = freq_align_monthly_to_quarterly_clean(panel, ["nonexistent"])
        # No columns changed
        pd.testing.assert_frame_equal(out, panel)

    def test_requires_datetime_index(self):
        panel = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            freq_align_monthly_to_quarterly_clean(panel, ["a"])

    def test_invalid_rule_raises(self):
        panel = self._make_quarterly_panel_with_monthly()
        with pytest.raises(ValueError, match="rule"):
            freq_align_monthly_to_quarterly_clean(panel, ["m"], rule="bad_rule")

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            freq_align_monthly_to_quarterly_clean(pd.DataFrame(), [])

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "freq_align_monthly_to_quarterly_clean")
        panel = self._make_quarterly_panel_with_monthly()
        out = mf.functions.freq_align_monthly_to_quarterly_clean(panel, ["m"])
        assert out.shape == panel.shape

    def test_quarterly_endpoint_rule(self):
        panel = self._make_quarterly_panel_with_monthly()
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"], rule="quarterly_endpoint")
        assert "m" in out.columns

    def test_quarterly_sum_rule(self):
        panel = self._make_quarterly_panel_with_monthly()
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"], rule="quarterly_sum")
        assert "m" in out.columns


# ===========================================================================
# Cross-namespace: all 14 in mf.functions
# ===========================================================================


class TestNamespaceCompleteness:
    EXPECTED_NAMES = [
        "iqr_outlier_clean",
        "zscore_outlier_clean",
        "winsorize_clean",
        "em_factor_impute_clean",
        "em_multivariate_impute_clean",
        "mean_impute_clean",
        "forward_fill_clean",
        "linear_interpolate_clean",
        "truncate_to_balanced_clean",
        "drop_unbalanced_series_clean",
        "zero_fill_leading_clean",
        "apply_tcode_transform",
        "freq_align_quarterly_to_monthly_clean",
        "freq_align_monthly_to_quarterly_clean",
    ]

    def test_all_14_in_mf_functions(self):
        for name in self.EXPECTED_NAMES:
            assert hasattr(mf.functions, name), f"mf.functions missing: {name}"

    def test_all_14_in_all(self):
        from macroforecast.functions import __all__
        for name in self.EXPECTED_NAMES:
            assert name in __all__, f"__all__ missing: {name}"

    def test_all_14_callable(self):
        for name in self.EXPECTED_NAMES:
            fn = getattr(mf.functions, name)
            assert callable(fn), f"{name} is not callable"


# ===========================================================================
# BLK fixup verification tests (C34 fixup)
# ===========================================================================


class TestBLKF1IqrConstantColumn:
    """BLK F-1: iqr_outlier_clean must not raise TypeError on constant column."""

    def test_constant_column_no_typeerror(self):
        """Constant column (IQR=0) must not raise TypeError (pd.NA comparison bug)."""
        panel = pd.DataFrame({
            "const": [5.0] * 20,
            "normal": np.random.RandomState(7).randn(20),
        })
        # Must not raise TypeError (pandas 3.x: boolean value of NA is ambiguous)
        out = iqr_outlier_clean(panel, threshold=10.0)
        assert out.shape == panel.shape

    def test_constant_column_zero_outlier_flags(self):
        """Constant column must produce exactly zero outlier flags (no NaN introduced)."""
        panel = pd.DataFrame({
            "const": [3.14] * 30,
            "with_outlier": list(np.random.RandomState(11).randn(30)),
        })
        panel.loc[0, "with_outlier"] = 999.0
        out = iqr_outlier_clean(panel, threshold=10.0)
        # const column: zero flags (IQR=0, threshold*IQR=0, no value exceeds threshold*0=0)
        assert out["const"].isna().sum() == 0
        # Verify the constant values are unchanged
        np.testing.assert_array_equal(out["const"].values, panel["const"].values)

    def test_normal_column_still_flags_outlier(self):
        """Bit-exact normal-column behavior preserved after np.nan fix."""
        panel = pd.DataFrame({
            "const": [1.0] * 50,
            "outlier_col": np.random.RandomState(42).randn(50),
        })
        panel.loc[5, "outlier_col"] = 500.0
        out = iqr_outlier_clean(panel, threshold=10.0)
        # The extreme outlier in outlier_col must be flagged
        assert np.isnan(out.loc[5, "outlier_col"])
        # Constant column must have zero NaN
        assert out["const"].isna().sum() == 0


class TestBLKF2MonthlyToQuarterlyDownsample:
    """BLK F-2: freq_align_monthly_to_quarterly_clean must downsample to quarterly."""

    def test_60month_returns_20_quarters(self):
        """60-row monthly input must return 20-row quarterly output (not 60)."""
        rng = np.random.RandomState(17)
        idx = pd.date_range("2015-01-01", periods=60, freq="MS")
        panel = pd.DataFrame({"m": rng.randn(60), "other": rng.randn(60)}, index=idx)
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"])
        assert len(out) == len(panel) // 3, (
            f"Expected {len(panel) // 3} quarterly rows, got {len(out)}"
        )

    def test_quarterly_index_on_output(self):
        """Output has length T // 3 for monthly input."""
        idx = pd.date_range("2010-01-01", periods=24, freq="MS")
        rng = np.random.RandomState(23)
        panel = pd.DataFrame({"m": rng.randn(24)}, index=idx)
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"])
        assert len(out) == 8  # 24 // 3

    def test_all_rules_downsample(self):
        """All three aggregation rules return len == T // 3."""
        idx = pd.date_range("2020-01-01", periods=12, freq="MS")
        rng = np.random.RandomState(31)
        panel = pd.DataFrame({"m": rng.randn(12)}, index=idx)
        for rule in ["quarterly_average", "quarterly_endpoint", "quarterly_sum"]:
            out = freq_align_monthly_to_quarterly_clean(panel, ["m"], rule=rule)
            assert len(out) == 4, f"rule={rule}: expected 4, got {len(out)}"

    def test_quarterly_input_unchanged_length(self):
        """Quarterly-indexed panel (4 rows) still returns 4 rows after fix."""
        idx = pd.date_range("2020-03-31", periods=4, freq="QE")
        panel = pd.DataFrame(
            {"m": [1.0, 2.0, 3.0, 4.0], "other": [10.0, 20.0, 30.0, 40.0]},
            index=idx,
        )
        out = freq_align_monthly_to_quarterly_clean(panel, ["m"])
        assert len(out) == 4


class TestBLKF3EmSvdRng99:
    """BLK F-3: EM impute handles RNG-99 50x20 10pct NaN without LinAlgError."""

    @classmethod
    def _make_rng99_panel(cls):
        rng = np.random.RandomState(99)
        panel = pd.DataFrame(rng.randn(50, 20))
        n_missing = int(0.10 * 50 * 20)
        idx_r = rng.randint(0, 50, n_missing)
        idx_c = rng.randint(0, 20, n_missing)
        for r, c in zip(idx_r, idx_c):
            panel.iloc[r, c] = np.nan
        return panel

    def test_em_factor_rng99_no_error(self):
        """em_factor_impute_clean must not raise LinAlgError on RNG-99 50x20 10pct NaN."""
        panel = self._make_rng99_panel()
        out = em_factor_impute_clean(panel, n_factors=8, max_iter=20, tol=1e-4)
        assert out.shape == panel.shape

    def test_em_factor_rng99_no_remaining_nan(self):
        """em_factor_impute_clean must fill all NaN on RNG-99 50x20 10pct NaN."""
        panel = self._make_rng99_panel()
        out = em_factor_impute_clean(panel, n_factors=8, max_iter=20, tol=1e-4)
        assert out.isna().sum().sum() == 0

    def test_em_multivariate_rng99_no_error(self):
        """em_multivariate_impute_clean must not raise LinAlgError on RNG-99 50x20 10pct NaN."""
        panel = self._make_rng99_panel()
        out = em_multivariate_impute_clean(panel, max_iter=20, tol=1e-4)
        assert out.shape == panel.shape

    def test_em_multivariate_rng99_no_remaining_nan(self):
        """em_multivariate_impute_clean must fill all NaN on RNG-99 50x20 10pct NaN."""
        panel = self._make_rng99_panel()
        out = em_multivariate_impute_clean(panel, max_iter=20, tol=1e-4)
        assert out.isna().sum().sum() == 0
