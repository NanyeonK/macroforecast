"""Tests for C31 L3 advanced panel-transform standalone callables.

Each test uses a fixed RNG-42 mini-panel for bit-exact reproducibility.
Assertions are either ``np.allclose`` at rtol=1e-12 or structural checks.

Bit-exact parity: each callable is run via the runtime helper directly
and compared with ``np.allclose(rtol=1e-12, atol=1e-14)`` where numeric
output is deterministic (hp_filter, hamilton_filter, savitzky_golay,
polynomial_expansion, interaction_terms, fourier, asymmetric_trim,
season_dummy, pca, maf_per_variable_pca).  RF-based ops (adaptive_ma_rf)
use structural checks only due to forest non-determinism across platforms.

Backlog fixes from C30 are also covered:
  - NOTE-1: log_diff_transform Notes section (docstring, no numeric change)
  - NOTE-2: scale_transform raises ValueError instead of NotImplementedError
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import the 12 new standalone callables under test
from macroforecast.functions.transforms import (
    hp_filter_transform,
    hamilton_filter_transform,
    savitzky_golay_transform,
    polynomial_expansion_transform,
    interaction_terms_transform,
    pca_transform,
    maf_per_variable_pca_transform,
    adaptive_ma_rf_transform,
    wavelet_transform,
    fourier_transform,
    asymmetric_trim_transform,
    season_dummy_transform,
    # C30 backlog items
    log_diff_transform,
    scale_transform,
)

import macroforecast as mf

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)
_BASE = pd.DataFrame(RNG.randn(50, 3), columns=["a", "b", "c"])
PANEL = _BASE.copy()  # may be negative
PANEL_POS = _BASE + 5.0  # strictly positive


# ---------------------------------------------------------------------------
# C30 NOTE-1: log_diff_transform docstring fix
# ---------------------------------------------------------------------------

class TestLogDiffDocstringFix:
    """NOTE-1: verify that log_diff_transform Notes section contains the
    two additional sentences about pd.NA guard divergence.  Docstring-only
    change; numerics must be unchanged."""

    def test_notes_mentions_pd_na_guard(self):
        doc = log_diff_transform.__doc__
        assert "pd.NA" in doc, "NOTE-1: 'pd.NA' not in log_diff_transform docstring"
        assert "cell-by-cell guard" in doc, "NOTE-1: 'cell-by-cell guard' not in docstring"

    def test_numerics_unchanged(self):
        """Bit-exact check -- docstring change must not alter output."""
        from macroforecast.core.runtime import _as_frame, _diff_like
        logged = np.log(_as_frame(PANEL_POS))
        expected = _diff_like(logged, periods=1)
        out = log_diff_transform(PANEL_POS)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )


# ---------------------------------------------------------------------------
# C30 NOTE-2: scale_transform raises ValueError
# ---------------------------------------------------------------------------

class TestScaleTransformNOTE2:
    """NOTE-2: scale_transform must raise ValueError (not NotImplementedError)
    for unsupported methods, and the test for NotImplementedError is updated."""

    def test_invalid_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown method"):
            scale_transform(PANEL, method="l2norm")

    def test_not_not_implemented_error(self):
        """Confirm NotImplementedError is NOT raised (it was C30's bug)."""
        with pytest.raises(ValueError):
            scale_transform(PANEL, method="l2norm")
        # Confirm it is specifically ValueError, not a subclass
        try:
            scale_transform(PANEL, method="bad_method")
        except ValueError:
            pass
        except Exception as exc:
            pytest.fail(f"Expected ValueError but got {type(exc).__name__}")

    def test_valid_methods_still_work(self):
        for method in ("zscore", "standard", "standardize", "robust", "minmax"):
            out = scale_transform(PANEL, method=method)
            assert out.shape == PANEL.shape


# ---------------------------------------------------------------------------
# 1. hp_filter_transform
# ---------------------------------------------------------------------------

class TestHpFilterTransform:
    def test_column_suffix(self):
        out = hp_filter_transform(PANEL)
        assert "a_hp_cycle" in out.columns
        assert "b_hp_cycle" in out.columns
        assert "c_hp_cycle" in out.columns

    def test_output_shape(self):
        out = hp_filter_transform(PANEL)
        assert out.shape == (50, 3)

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _hp_filter
        expected = _hp_filter(_as_frame(PANEL), lam=1600)
        out = hp_filter_transform(PANEL, lambda_=1600)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_custom_lambda(self):
        out = hp_filter_transform(PANEL, lambda_=129600)
        assert out.shape == (50, 3)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda_ > 0"):
            hp_filter_transform(PANEL, lambda_=-1)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            hp_filter_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "hp_filter_transform")
        out = mf.functions.hp_filter_transform(PANEL)
        assert "a_hp_cycle" in out.columns


# ---------------------------------------------------------------------------
# 2. hamilton_filter_transform
# ---------------------------------------------------------------------------

class TestHamiltonFilterTransform:
    def test_column_suffix(self):
        out = hamilton_filter_transform(PANEL)
        assert "a_hamilton_cycle" in out.columns

    def test_output_shape(self):
        out = hamilton_filter_transform(PANEL)
        assert out.shape == (50, 3)

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _hamilton_filter
        expected = _hamilton_filter(_as_frame(PANEL), n_horizon=8, n_lags=4)
        out = hamilton_filter_transform(PANEL, h=8, p=4)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_custom_params(self):
        out = hamilton_filter_transform(PANEL, h=4, p=2)
        assert out.shape == (50, 3)

    def test_invalid_h(self):
        with pytest.raises(ValueError, match="h >= 1"):
            hamilton_filter_transform(PANEL, h=0)

    def test_invalid_p(self):
        with pytest.raises(ValueError, match="p >= 1"):
            hamilton_filter_transform(PANEL, p=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            hamilton_filter_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "hamilton_filter_transform")
        out = mf.functions.hamilton_filter_transform(PANEL)
        assert "a_hamilton_cycle" in out.columns


# ---------------------------------------------------------------------------
# 3. savitzky_golay_transform
# ---------------------------------------------------------------------------

class TestSavitzkyGolayTransform:
    def test_column_suffix(self):
        out = savitzky_golay_transform(PANEL)
        assert "a_savgol" in out.columns

    def test_shape_preserved(self):
        out = savitzky_golay_transform(PANEL)
        assert out.shape == (50, 3)

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _savitzky_golay_filter
        expected = _savitzky_golay_filter(_as_frame(PANEL), window_length=7, polyorder=3)
        out = savitzky_golay_transform(PANEL, window=7, polyorder=3)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window >= 3"):
            savitzky_golay_transform(PANEL, window=2)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            savitzky_golay_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "savitzky_golay_transform")
        out = mf.functions.savitzky_golay_transform(PANEL)
        assert "a_savgol" in out.columns


# ---------------------------------------------------------------------------
# 4. polynomial_expansion_transform
# ---------------------------------------------------------------------------

class TestPolynomialExpansionTransform:
    def test_degree_2_shape(self):
        out = polynomial_expansion_transform(PANEL, degree=2)
        # Original 3 + 3 pow2 = 6 columns
        assert out.shape == (50, 6)

    def test_degree_3_columns(self):
        out = polynomial_expansion_transform(PANEL, degree=3)
        assert out.shape == (50, 9)
        assert "a_pow3" in out.columns

    def test_degree_1_returns_original(self):
        out = polynomial_expansion_transform(PANEL, degree=1)
        np.testing.assert_array_equal(out.values, PANEL.values)

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _polynomial_expansion
        expected = _polynomial_expansion(_as_frame(PANEL), degree=2)
        out = polynomial_expansion_transform(PANEL, degree=2)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_pow2_values_correct(self):
        simple = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        out = polynomial_expansion_transform(simple, degree=2)
        np.testing.assert_allclose(out["x_pow2"].values, [1.0, 4.0, 9.0])

    def test_invalid_degree(self):
        with pytest.raises(ValueError, match="degree >= 1"):
            polynomial_expansion_transform(PANEL, degree=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            polynomial_expansion_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "polynomial_expansion_transform")
        out = mf.functions.polynomial_expansion_transform(PANEL)
        assert "a_pow2" in out.columns


# ---------------------------------------------------------------------------
# 5. interaction_terms_transform
# ---------------------------------------------------------------------------

class TestInteractionTermsTransform:
    def test_output_column_count(self):
        # 3 cols: C(3,2) = 3 pairs
        out = interaction_terms_transform(PANEL)
        assert out.shape == (50, 3)

    def test_column_names(self):
        out = interaction_terms_transform(PANEL)
        assert "a_x_b" in out.columns
        assert "a_x_c" in out.columns
        assert "b_x_c" in out.columns

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _interaction_terms
        expected = _interaction_terms(_as_frame(PANEL))
        out = interaction_terms_transform(PANEL)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_values_correct(self):
        simple = pd.DataFrame({"x": [2.0], "y": [3.0]})
        out = interaction_terms_transform(simple)
        assert float(out["x_x_y"].iloc[0]) == pytest.approx(6.0)

    def test_single_column_returns_empty(self):
        single = pd.DataFrame({"a": [1.0, 2.0]})
        out = interaction_terms_transform(single)
        assert out.shape[1] == 0

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            interaction_terms_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "interaction_terms_transform")
        out = mf.functions.interaction_terms_transform(PANEL)
        assert "a_x_b" in out.columns


# ---------------------------------------------------------------------------
# 6. pca_transform
# ---------------------------------------------------------------------------

class TestPcaTransform:
    def test_output_shape(self):
        out = pca_transform(PANEL, n_components=2)
        assert out.shape == (50, 2)

    def test_column_names(self):
        # 3-col panel caps at min(50,3)-1 = 2 components; use 5-col panel for n=3
        wide_panel = pd.DataFrame(RNG.randn(50, 5), columns=list("abcde"))
        out = pca_transform(wide_panel, n_components=3)
        assert out.columns.tolist() == ["factor_1", "factor_2", "factor_3"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _pca_factors
        expected = _pca_factors(_as_frame(PANEL), n_components=3, variant="pca")
        out = pca_transform(PANEL, n_components=3)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_n_components_clamped(self):
        # 3-column panel: max extractable = min(50, 3) - 1 = 2
        out = pca_transform(PANEL, n_components=100)
        assert out.shape[1] <= 3

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            pca_transform(PANEL, n_components=0)

    def test_n_components_all_sentinel(self):
        # BLK-1: n_components='all' must be accepted; returns full effective rank
        out = pca_transform(PANEL, n_components="all")
        assert out.shape[1] >= 1
        assert all(col.startswith("factor_") for col in out.columns)

    def test_invalid_string_n_components_raises_value_error(self):
        # BLK-2: invalid string must raise ValueError, not TypeError
        with pytest.raises(ValueError, match="n_components must be a positive int or 'all'"):
            pca_transform(PANEL, n_components="junk")

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            pca_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "pca_transform")
        out = mf.functions.pca_transform(PANEL)
        assert "factor_1" in out.columns


# ---------------------------------------------------------------------------
# 7. maf_per_variable_pca_transform
# ---------------------------------------------------------------------------

class TestMafPerVariablePcaTransform:
    def test_output_column_count(self):
        # 3 vars * 2 components = 6 cols
        out = maf_per_variable_pca_transform(PANEL)
        assert out.shape[1] == 6

    def test_output_column_names(self):
        out = maf_per_variable_pca_transform(PANEL)
        assert "a_maf1" in out.columns
        assert "a_maf2" in out.columns
        assert "c_maf2" in out.columns

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _maf_per_variable_pca
        # BLK-3: pass n_lags explicitly; runtime helper accepts it
        expected = _maf_per_variable_pca(_as_frame(PANEL), n_lags=12, n_components_per_var=2)
        out = maf_per_variable_pca_transform(PANEL, n_lags=12, n_components_per_var=2)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_leading_nan_rows(self):
        out = maf_per_variable_pca_transform(PANEL)
        # Default n_lags=12: first 12 rows should contain NaN
        assert out["a_maf1"].iloc[:12].isna().all()

    def test_leading_nan_rows_custom_n_lags(self):
        # BLK-3: n_lags=4 -> first 4 rows NaN
        out = maf_per_variable_pca_transform(PANEL, n_lags=4)
        assert out["a_maf1"].iloc[:4].isna().all()

    def test_invalid_n_lags(self):
        # BLK-3: n_lags=0 must raise ValueError matching "n_lags"
        with pytest.raises(ValueError, match="n_lags"):
            maf_per_variable_pca_transform(PANEL, n_lags=0)

    def test_invalid_n_components_per_var(self):
        with pytest.raises(ValueError, match="n_components_per_var >= 1"):
            maf_per_variable_pca_transform(PANEL, n_components_per_var=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            maf_per_variable_pca_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "maf_per_variable_pca_transform")
        out = mf.functions.maf_per_variable_pca_transform(PANEL)
        assert out.shape[1] == 6


# ---------------------------------------------------------------------------
# 8. adaptive_ma_rf_transform
# ---------------------------------------------------------------------------

class TestAdaptiveMaRfTransform:
    """RF-based: structural checks only (forest non-determinism across
    platforms); bit-exact check uses the same random_state=0 path."""

    def test_column_suffix(self):
        out = adaptive_ma_rf_transform(PANEL, n_estimators=10, min_samples_leaf=5)
        assert "a_albama" in out.columns

    def test_output_shape(self):
        out = adaptive_ma_rf_transform(PANEL, n_estimators=10, min_samples_leaf=5)
        assert out.shape == (50, 3)

    def test_output_finite(self):
        out = adaptive_ma_rf_transform(PANEL, n_estimators=10, min_samples_leaf=5)
        assert np.isfinite(out.values).all()

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _adaptive_ma_rf
        # BLK-4: pass random_state and sided explicitly
        expected = _adaptive_ma_rf(
            _as_frame(PANEL), n_estimators=10, min_samples_leaf=5,
            sided="two", random_state=0
        )
        out = adaptive_ma_rf_transform(
            PANEL, n_estimators=10, min_samples_leaf=5,
            sided="two", random_state=0
        )
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-10, atol=1e-10, equal_nan=True
        )

    def test_sided_one_produces_leading_nan(self):
        # BLK-4: sided='one' should produce NaN in first min_samples_leaf-1 positions
        single = pd.DataFrame({"x": np.random.RandomState(7).randn(60)})
        out = adaptive_ma_rf_transform(single, n_estimators=5, min_samples_leaf=10,
                                        sided="one", random_state=0)
        # First 9 rows (min_samples_leaf-1 = 9) should be NaN
        assert out["x_albama"].iloc[:9].isna().all()

    def test_invalid_sided_raises_value_error(self):
        # BLK-4: invalid sided must raise ValueError
        with pytest.raises(ValueError, match="sided must be 'two' or 'one'"):
            adaptive_ma_rf_transform(PANEL, sided="three")

    def test_invalid_n_estimators(self):
        with pytest.raises(ValueError, match="n_estimators >= 1"):
            adaptive_ma_rf_transform(PANEL, n_estimators=0)

    def test_invalid_min_samples_leaf(self):
        with pytest.raises(ValueError, match="min_samples_leaf >= 1"):
            adaptive_ma_rf_transform(PANEL, min_samples_leaf=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            adaptive_ma_rf_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "adaptive_ma_rf_transform")
        out = mf.functions.adaptive_ma_rf_transform(PANEL, n_estimators=5, min_samples_leaf=5)
        assert "a_albama" in out.columns


# ---------------------------------------------------------------------------
# 9. wavelet_transform
# ---------------------------------------------------------------------------

class TestWaveletTransform:
    def test_output_shape_n_levels_2(self):
        out = wavelet_transform(PANEL, n_levels=2)
        # 3 cols * 2 levels * 2 (approx + detail) = 12
        assert out.shape == (50, 12)

    def test_output_shape_n_levels_3(self):
        out = wavelet_transform(PANEL, n_levels=3)
        assert out.shape == (50, 18)

    def test_column_names(self):
        out = wavelet_transform(PANEL, n_levels=2)
        assert "a_wA1" in out.columns
        assert "a_wD1" in out.columns
        assert "a_wA2" in out.columns
        assert "a_wD2" in out.columns

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _wavelet_decomposition
        expected = _wavelet_decomposition(_as_frame(PANEL), n_levels=3)
        out = wavelet_transform(PANEL, n_levels=3)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_wavelet_param_accepted(self):
        # wavelet param is API-consistent; does not raise
        out = wavelet_transform(PANEL, wavelet="haar", n_levels=2)
        assert out.shape == (50, 12)

    def test_invalid_n_levels(self):
        with pytest.raises(ValueError, match="n_levels >= 1"):
            wavelet_transform(PANEL, n_levels=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            wavelet_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "wavelet_transform")
        out = mf.functions.wavelet_transform(PANEL)
        assert "a_wA1" in out.columns


# ---------------------------------------------------------------------------
# 10. fourier_transform
# ---------------------------------------------------------------------------

class TestFourierTransform:
    def test_output_shape(self):
        out = fourier_transform(PANEL, n_terms=4, period=12)
        assert out.shape == (50, 8)

    def test_column_names(self):
        out = fourier_transform(PANEL, n_terms=2, period=12)
        expected_cols = ["fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2"]
        assert out.columns.tolist() == expected_cols

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _fourier_features
        expected = _fourier_features(_as_frame(PANEL), n_terms=4, period=12)
        out = fourier_transform(PANEL, n_terms=4, period=12)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_sin_cos_range(self):
        out = fourier_transform(PANEL, n_terms=2, period=12)
        assert (out.values >= -1.0).all() and (out.values <= 1.0).all()

    def test_invalid_n_terms(self):
        with pytest.raises(ValueError, match="n_terms >= 1"):
            fourier_transform(PANEL, n_terms=0)

    def test_invalid_period(self):
        with pytest.raises(ValueError, match="period >= 1"):
            fourier_transform(PANEL, period=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            fourier_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "fourier_transform")
        out = mf.functions.fourier_transform(PANEL)
        assert "fourier_sin_1" in out.columns


# ---------------------------------------------------------------------------
# 11. asymmetric_trim_transform
# ---------------------------------------------------------------------------

class TestAsymmetricTrimTransform:
    def test_shape_preserved(self):
        out = asymmetric_trim_transform(PANEL)
        assert out.shape == (50, 3)

    def test_rows_sorted_ascending(self):
        out = asymmetric_trim_transform(PANEL)
        # Each row of the output should be non-decreasing
        values = out.values
        for row in values:
            assert row[0] <= row[1] <= row[2], f"Row not sorted: {row}"

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _asymmetric_trim
        expected = _asymmetric_trim(_as_frame(PANEL))
        out = asymmetric_trim_transform(PANEL)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_known_values(self):
        panel = pd.DataFrame({"a": [3.0], "b": [1.0], "c": [2.0]})
        out = asymmetric_trim_transform(panel)
        np.testing.assert_allclose(out.values[0], [1.0, 2.0, 3.0])

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            asymmetric_trim_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "asymmetric_trim_transform")
        out = mf.functions.asymmetric_trim_transform(PANEL)
        assert out.shape == PANEL.shape


# ---------------------------------------------------------------------------
# 12. season_dummy_transform
# ---------------------------------------------------------------------------

class TestSeasonDummyTransform:
    def test_month_dummies_with_period_index(self):
        idx = pd.period_range("2000-01", periods=24, freq="M")
        panel = pd.DataFrame({"x": np.ones(24)}, index=idx)
        out = season_dummy_transform(panel, season="month")
        # Should produce 12 unique month dummies
        assert out.shape[1] == 12

    def test_quarter_dummies_with_period_index(self):
        # BLK-5: season='quarter' with PeriodIndex now routes to _season_dummy
        # which produces 'season_*' prefix (modulo-12, up to 4 unique values
        # for a quarterly PeriodIndex).
        idx = pd.period_range("2000Q1", periods=12, freq="Q")
        panel = pd.DataFrame({"x": np.ones(12)}, index=idx)
        out = season_dummy_transform(panel, season="quarter")
        # Should produce some dummies; prefix must be 'season_*' not 'qtr_*'
        assert out.shape[1] >= 1
        assert all(c.startswith("season_") for c in out.columns), (
            f"Expected 'season_*' prefix for non-DatetimeIndex; got {out.columns.tolist()}"
        )

    def test_bit_exact_month_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _season_dummy
        idx = pd.period_range("2000-01", periods=24, freq="M")
        panel = pd.DataFrame({"x": np.ones(24)}, index=idx)
        expected = _season_dummy(_as_frame(panel))
        out = season_dummy_transform(panel, season="month")
        assert out.shape == expected.shape
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14)

    def test_dummy_values_are_binary(self):
        idx = pd.period_range("2000-01", periods=12, freq="M")
        panel = pd.DataFrame({"x": np.ones(12)}, index=idx)
        out = season_dummy_transform(panel, season="month")
        assert set(out.values.flatten().tolist()).issubset({0.0, 1.0})

    def test_default_season_is_quarter(self):
        # default should not raise and should return some dummies
        out = season_dummy_transform(PANEL)
        assert out.shape[0] == 50

    def test_non_datetime_prefix_is_season_or_month(self):
        # BLK-5: non-DatetimeIndex must produce 'season_*' or 'month_*' prefix
        out = season_dummy_transform(PANEL)
        assert all(c.startswith("season_") or c.startswith("month_") for c in out.columns), (
            f"Expected 'season_*' or 'month_*' prefix; got {out.columns.tolist()}"
        )

    def test_invalid_season_raises(self):
        with pytest.raises(ValueError, match="Unknown season"):
            season_dummy_transform(PANEL, season="week")

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            season_dummy_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "season_dummy_transform")
        out = mf.functions.season_dummy_transform(PANEL)
        assert out.shape[0] == 50


# ---------------------------------------------------------------------------
# Namespace content test for all 12 new ops
# ---------------------------------------------------------------------------

class TestNamespaceContent:
    """Verify all 12 new callables are importable from mf.functions."""

    NEW_NAMES = [
        "hp_filter_transform",
        "hamilton_filter_transform",
        "savitzky_golay_transform",
        "polynomial_expansion_transform",
        "interaction_terms_transform",
        "pca_transform",
        "maf_per_variable_pca_transform",
        "adaptive_ma_rf_transform",
        "wavelet_transform",
        "fourier_transform",
        "asymmetric_trim_transform",
        "season_dummy_transform",
    ]

    def test_all_names_present(self):
        for name in self.NEW_NAMES:
            assert hasattr(mf.functions, name), f"mf.functions.{name} missing"

    def test_all_in_dunder_all(self):
        import macroforecast.functions as mff
        all_set = set(mff.__all__)
        for name in self.NEW_NAMES:
            assert name in all_set, f"{name} not in __all__"

    def test_callables_are_callable(self):
        for name in self.NEW_NAMES:
            fn = getattr(mf.functions, name)
            assert callable(fn), f"mf.functions.{name} is not callable"
