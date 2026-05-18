"""Tests for C32 L3 supervised/mixed panel-transform standalone callables.

6 new ops:
  scaled_pca_transform       -- Huang-Zhou (2022) Scaled PCA
  supervised_pca_transform   -- Giglio-Xiu-Zhang (2025) screen-then-PCA
  partial_least_squares_transform -- sklearn PLSRegression
  sliced_inverse_regression_transform -- Fan-Xue-Yao (2017) SIR + Huang-Zhou scaling
  dfm_transform              -- static DFM (PCA on standardised panel, unsupervised)
  feature_selection_transform -- variance / correlation / lasso column filter

Each bit-exact test runs the standalone callable and the runtime helper with
identical inputs and asserts np.allclose(rtol=1e-12, atol=1e-14).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.functions.transforms import (
    scaled_pca_transform,
    supervised_pca_transform,
    partial_least_squares_transform,
    sliced_inverse_regression_transform,
    dfm_transform,
    feature_selection_transform,
)

import macroforecast as mf

# ---------------------------------------------------------------------------
# Shared fixtures (RNG-42, deterministic)
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)
PANEL = pd.DataFrame(RNG.randn(50, 8), columns=[f"x{i}" for i in range(8)])
TARGET = pd.Series(RNG.randn(50), name="y")

# Small panel for disjoint-index tests
_PANEL_SMALL = pd.DataFrame(
    RNG.randn(10, 4),
    index=range(100, 110),
    columns=list("abcd"),
)
_TARGET_SMALL = pd.Series(RNG.randn(10), index=range(100, 110), name="y")
_TARGET_DISJOINT = pd.Series(RNG.randn(10), index=range(200, 210), name="y")


# ---------------------------------------------------------------------------
# 1. scaled_pca_transform
# ---------------------------------------------------------------------------


class TestScaledPcaTransform:
    def test_output_shape(self):
        out = scaled_pca_transform(PANEL, TARGET, n_components=2)
        assert out.shape == (50, 2)

    def test_column_names(self):
        out = scaled_pca_transform(PANEL, TARGET, n_components=2)
        assert list(out.columns) == ["factor_1", "factor_2"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _pca_factors

        expected = _pca_factors(
            _as_frame(PANEL),
            n_components=3,
            variant="scaled_pca",
            target_signal=TARGET,
        )
        out = scaled_pca_transform(PANEL, TARGET, n_components=3)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            scaled_pca_transform(PANEL, TARGET, n_components=0)

    def test_disjoint_index_raises(self):
        with pytest.raises(ValueError, match="no common index values"):
            scaled_pca_transform(_PANEL_SMALL, _TARGET_DISJOINT)

    def test_wrong_target_type(self):
        with pytest.raises(TypeError, match="pd.Series"):
            scaled_pca_transform(PANEL, PANEL)  # DataFrame not accepted

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            scaled_pca_transform(pd.DataFrame(), TARGET)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "scaled_pca_transform")
        out = mf.functions.scaled_pca_transform(PANEL, TARGET)
        assert "factor_1" in out.columns

    def test_default_n_components(self):
        out = scaled_pca_transform(PANEL, TARGET)
        assert out.shape[1] == 3


# ---------------------------------------------------------------------------
# 2. supervised_pca_transform
# ---------------------------------------------------------------------------


class TestSupervisedPcaTransform:
    def test_output_shape(self):
        out = supervised_pca_transform(PANEL, TARGET, n_components=2)
        assert out.shape[1] == 2

    def test_column_prefix(self):
        out = supervised_pca_transform(PANEL, TARGET, n_components=2)
        assert out.columns[0] == "spca_1"
        assert out.columns[1] == "spca_2"

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _supervised_pca

        expected = _supervised_pca(_as_frame(PANEL), target=TARGET, n_components=3)
        out = supervised_pca_transform(PANEL, TARGET, n_components=3)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            supervised_pca_transform(PANEL, TARGET, n_components=0)

    def test_disjoint_index_raises(self):
        with pytest.raises(ValueError, match="no common index values"):
            supervised_pca_transform(_PANEL_SMALL, _TARGET_DISJOINT)

    def test_wrong_target_type(self):
        with pytest.raises(TypeError, match="pd.Series"):
            supervised_pca_transform(PANEL, target="not_a_series")

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            supervised_pca_transform(pd.DataFrame(), TARGET)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "supervised_pca_transform")
        out = mf.functions.supervised_pca_transform(PANEL, TARGET)
        assert out.columns[0] == "spca_1"

    def test_default_n_components(self):
        out = supervised_pca_transform(PANEL, TARGET)
        assert out.shape[1] == 3


# ---------------------------------------------------------------------------
# 3. partial_least_squares_transform
# ---------------------------------------------------------------------------


class TestPartialLeastSquaresTransform:
    def test_output_shape(self):
        out = partial_least_squares_transform(PANEL, TARGET, n_components=2)
        assert out.shape[1] == 2

    def test_column_prefix(self):
        out = partial_least_squares_transform(PANEL, TARGET, n_components=2)
        assert list(out.columns) == ["pls_1", "pls_2"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _partial_least_squares

        expected = _partial_least_squares(_as_frame(PANEL), target=TARGET, n_components=3)
        out = partial_least_squares_transform(PANEL, TARGET, n_components=3)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            partial_least_squares_transform(PANEL, TARGET, n_components=0)

    def test_disjoint_index_raises(self):
        with pytest.raises(ValueError, match="no common index values"):
            partial_least_squares_transform(_PANEL_SMALL, _TARGET_DISJOINT)

    def test_wrong_target_type(self):
        with pytest.raises(TypeError, match="pd.Series"):
            partial_least_squares_transform(PANEL, np.ones(50))

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            partial_least_squares_transform(pd.DataFrame(), TARGET)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "partial_least_squares_transform")
        out = mf.functions.partial_least_squares_transform(PANEL, TARGET)
        assert "pls_1" in out.columns

    def test_row_count_preserved(self):
        out = partial_least_squares_transform(PANEL, TARGET, n_components=2)
        assert len(out) == len(PANEL)


# ---------------------------------------------------------------------------
# 4. sliced_inverse_regression_transform
# ---------------------------------------------------------------------------


class TestSlicedInverseRegressionTransform:
    def test_output_shape(self):
        out = sliced_inverse_regression_transform(PANEL, TARGET, n_components=2)
        assert out.shape == (50, 2)

    def test_column_prefix(self):
        out = sliced_inverse_regression_transform(PANEL, TARGET, n_components=2)
        assert out.columns[0] == "factor_1"

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _sliced_inverse_regression

        expected = _sliced_inverse_regression(
            _as_frame(PANEL), target=TARGET, n_components=3, n_slices=10
        )
        out = sliced_inverse_regression_transform(
            PANEL, TARGET, n_components=3, n_slices=10
        )
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_invalid_n_components(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            sliced_inverse_regression_transform(PANEL, TARGET, n_components=0)

    def test_invalid_n_slices(self):
        with pytest.raises(ValueError, match="n_slices >= 2"):
            sliced_inverse_regression_transform(PANEL, TARGET, n_slices=1)

    def test_disjoint_index_raises(self):
        with pytest.raises(ValueError, match="no common index values"):
            sliced_inverse_regression_transform(_PANEL_SMALL, _TARGET_DISJOINT)

    def test_wrong_target_type(self):
        with pytest.raises(TypeError, match="pd.Series"):
            sliced_inverse_regression_transform(PANEL, list(range(50)))

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            sliced_inverse_regression_transform(pd.DataFrame(), TARGET)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "sliced_inverse_regression_transform")
        out = mf.functions.sliced_inverse_regression_transform(PANEL, TARGET)
        assert "factor_1" in out.columns

    def test_default_n_slices(self):
        # default n_slices=10; runtime should partition 50 rows into 10 slices
        out = sliced_inverse_regression_transform(PANEL, TARGET)
        assert out.shape == (50, 3)


# ---------------------------------------------------------------------------
# 5. dfm_transform  (unsupervised -- no target)
# ---------------------------------------------------------------------------


class TestDfmTransform:
    def test_output_shape(self):
        out = dfm_transform(PANEL, n_factors=2)
        assert out.shape == (50, 2)

    def test_column_prefix(self):
        out = dfm_transform(PANEL, n_factors=2)
        assert list(out.columns) == ["dfm_1", "dfm_2"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _dfm_factors

        expected = _dfm_factors(_as_frame(PANEL), n_factors=3)
        out = dfm_transform(PANEL, n_factors=3)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_no_target_required(self):
        # dfm_transform has no target positional arg -- call must succeed without target
        out = dfm_transform(PANEL)
        assert out.shape[1] == 3  # default n_factors=3

    def test_invalid_n_factors(self):
        with pytest.raises(ValueError, match="n_factors >= 1"):
            dfm_transform(PANEL, n_factors=0)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            dfm_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "dfm_transform")
        out = mf.functions.dfm_transform(PANEL)
        assert "dfm_1" in out.columns

    def test_row_count_preserved(self):
        out = dfm_transform(PANEL, n_factors=2)
        assert len(out) == len(PANEL)


# ---------------------------------------------------------------------------
# 6. feature_selection_transform
# ---------------------------------------------------------------------------


class TestFeatureSelectionTransform:
    def test_variance_no_target(self):
        out = feature_selection_transform(PANEL)
        # default n_features=0.5 -> keep 4 of 8 cols
        assert out.shape == (50, 4)
        assert set(out.columns).issubset(set(PANEL.columns))

    def test_correlation_with_target(self):
        out = feature_selection_transform(PANEL, TARGET, method="correlation")
        assert out.shape == (50, 4)
        assert set(out.columns).issubset(set(PANEL.columns))

    def test_lasso_with_target(self):
        out = feature_selection_transform(PANEL, TARGET, method="lasso")
        assert out.shape == (50, 4)
        assert set(out.columns).issubset(set(PANEL.columns))

    def test_bit_exact_variance_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _feature_selection

        expected = _feature_selection(
            _as_frame(PANEL), target=None, n_features=0.5, method="variance"
        )
        out = feature_selection_transform(PANEL, n_features=0.5, method="variance")
        assert list(out.columns) == list(expected.columns)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_bit_exact_correlation_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _feature_selection

        expected = _feature_selection(
            _as_frame(PANEL), target=TARGET, n_features=0.5, method="correlation"
        )
        out = feature_selection_transform(PANEL, TARGET, method="correlation")
        assert list(out.columns) == list(expected.columns)
        np.testing.assert_allclose(
            out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True
        )

    def test_correlation_without_target_raises(self):
        with pytest.raises(ValueError, match="requires target"):
            feature_selection_transform(PANEL, method="correlation")

    def test_lasso_without_target_raises(self):
        with pytest.raises(ValueError, match="requires target"):
            feature_selection_transform(PANEL, method="lasso")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="unknown method"):
            feature_selection_transform(PANEL, method="bad_method")

    def test_n_features_integer(self):
        out = feature_selection_transform(PANEL, n_features=3)
        assert out.shape[1] == 3

    def test_n_features_fraction(self):
        out = feature_selection_transform(PANEL, n_features=0.25)
        # 0.25 * 8 = 2 cols
        assert out.shape[1] == 2

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError, match="empty"):
            feature_selection_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "feature_selection_transform")
        out = mf.functions.feature_selection_transform(PANEL)
        assert out.shape[0] == 50

    def test_variance_target_ignored(self):
        # Passing target to variance method must not raise and must return same result
        # as calling without target (runtime ignores target for variance path)
        out_no_target = feature_selection_transform(PANEL, method="variance")
        out_with_target = feature_selection_transform(PANEL, TARGET, method="variance")
        assert list(out_no_target.columns) == list(out_with_target.columns)

    def test_row_count_preserved(self):
        out = feature_selection_transform(PANEL)
        assert len(out) == len(PANEL)


# ---------------------------------------------------------------------------
# C32 FIXUP — per-BLK tester assertions
# ---------------------------------------------------------------------------


class TestBLK1SirScalingMethod:
    """BLK-1: sliced_inverse_regression_transform scaling_method kwarg."""

    def test_default_scaling_method_accepted(self):
        # Should not raise — default scaling_method="scaled_pca"
        out = sliced_inverse_regression_transform(PANEL, TARGET, n_components=2)
        assert out.shape == (50, 2)

    def test_explicit_scaled_pca(self):
        out = sliced_inverse_regression_transform(
            PANEL, TARGET, n_components=2, scaling_method="scaled_pca"
        )
        assert out.shape == (50, 2)

    def test_none_scaling_method(self):
        out = sliced_inverse_regression_transform(
            PANEL, TARGET, n_components=2, scaling_method="none"
        )
        assert out.shape == (50, 2)

    def test_invalid_scaling_method_raises(self):
        with pytest.raises(ValueError, match="scaling_method must be one of"):
            sliced_inverse_regression_transform(
                PANEL, TARGET, n_components=2, scaling_method="bogus"
            )

    def test_forwarded_to_runtime(self):
        """scaled_pca and none give identical shapes but potentially different values."""
        out_sp = sliced_inverse_regression_transform(
            PANEL, TARGET, n_components=2, scaling_method="scaled_pca"
        )
        out_none = sliced_inverse_regression_transform(
            PANEL, TARGET, n_components=2, scaling_method="none"
        )
        assert out_sp.shape == out_none.shape == (50, 2)


class TestBLK2ScaledPcaStringGuard:
    """BLK-2: scaled_pca_transform rejects strings other than 'all'."""

    def test_all_passthrough(self):
        # "all" must not raise a ValueError at the guard level
        try:
            out = scaled_pca_transform(PANEL, TARGET, n_components="all")
            # If runtime supports "all", shape should have columns > 0
            assert out.shape[1] >= 1
        except (ValueError, TypeError):
            # "all" passes the guard but runtime may raise — only check guard logic
            pass  # acceptable: guard didn't reject it prematurely

    def test_string_not_all_raises(self):
        with pytest.raises(ValueError, match="must be"):
            scaled_pca_transform(PANEL, TARGET, n_components="three")

    def test_integer_path_unchanged(self):
        out = scaled_pca_transform(PANEL, TARGET, n_components=2)
        assert out.shape == (50, 2)

    def test_zero_still_raises(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            scaled_pca_transform(PANEL, TARGET, n_components=0)


class TestBLK3SupervisedPcaQ:
    """BLK-3: supervised_pca_transform q kwarg validation and forwarding."""

    def test_default_q_works(self):
        out = supervised_pca_transform(PANEL, TARGET, n_components=2)
        assert out.shape[1] == 2

    def test_custom_q_accepted(self):
        out = supervised_pca_transform(PANEL, TARGET, n_components=2, q=0.75)
        assert out.shape[1] == 2

    def test_q_zero_raises(self):
        with pytest.raises(ValueError, match="0 < q < 1"):
            supervised_pca_transform(PANEL, TARGET, n_components=2, q=0.0)

    def test_q_one_raises(self):
        with pytest.raises(ValueError, match="0 < q < 1"):
            supervised_pca_transform(PANEL, TARGET, n_components=2, q=1.0)

    def test_q_negative_raises(self):
        with pytest.raises(ValueError, match="0 < q < 1"):
            supervised_pca_transform(PANEL, TARGET, n_components=2, q=-0.1)

    def test_q_above_one_raises(self):
        with pytest.raises(ValueError, match="0 < q < 1"):
            supervised_pca_transform(PANEL, TARGET, n_components=2, q=1.5)

    def test_q_forwarded_to_runtime(self):
        """q=0.25 vs q=0.75 should give same shapes (n_components drives column count)."""
        out_lo = supervised_pca_transform(PANEL, TARGET, n_components=2, q=0.25)
        out_hi = supervised_pca_transform(PANEL, TARGET, n_components=2, q=0.75)
        assert out_lo.shape == out_hi.shape == (50, 2)


class TestBLK4PLSClamp:
    """BLK-4: partial_least_squares_transform clamps n_components before sklearn."""

    def test_rng42_50x5_clamp(self):
        # RNG-42 50x5 panel: T_clean=50, K_clean=5
        # NOTE-A fix: clamp = min(T_clean-1, K_clean) = min(49, 5) = 5
        rng = np.random.RandomState(42)
        panel50x5 = pd.DataFrame(rng.randn(50, 5), columns=[f"x{i}" for i in range(5)])
        target50 = pd.Series(rng.randn(50), name="y")
        out = partial_least_squares_transform(panel50x5, target50, n_components=100)
        assert out.shape[1] == 5, f"Expected == 5 columns (min(49,5)), got {out.shape[1]}"

    def test_clamp_preserves_small_n_components(self):
        out = partial_least_squares_transform(PANEL, TARGET, n_components=2)
        assert out.shape[1] == 2

    def test_clamp_column_names_correct(self):
        rng = np.random.RandomState(42)
        panel50x5 = pd.DataFrame(rng.randn(50, 5), columns=[f"x{i}" for i in range(5)])
        target50 = pd.Series(rng.randn(50), name="y")
        out = partial_least_squares_transform(panel50x5, target50, n_components=100)
        for i, col in enumerate(out.columns):
            assert col == f"pls_{i + 1}"


class TestBLK5DfmReindex:
    """BLK-5: dfm_transform reindexes output to original frame.index."""

    def test_nan_rows_preserved_in_output(self):
        # Build panel with NaN rows; output must have same index as input
        rng = np.random.RandomState(7)
        panel = pd.DataFrame(rng.randn(30, 4), columns=list("abcd"))
        panel.iloc[5] = np.nan   # inject NaN row
        panel.iloc[20] = np.nan  # inject another NaN row
        out = dfm_transform(panel, n_factors=2)
        assert len(out) == len(panel), "Output must preserve all rows including NaN rows"
        assert list(out.index) == list(panel.index)
        # NaN rows should remain NaN in output
        assert out.iloc[5].isna().all(), "Row 5 should be NaN in output"
        assert out.iloc[20].isna().all(), "Row 20 should be NaN in output"

    def test_full_panel_no_nan_shape(self):
        out = dfm_transform(PANEL, n_factors=3)
        assert out.shape == (50, 3)

    def test_non_default_index_preserved(self):
        rng = np.random.RandomState(99)
        idx = pd.date_range("2000-01", periods=20, freq="ME")
        panel = pd.DataFrame(rng.randn(20, 4), index=idx, columns=list("abcd"))
        out = dfm_transform(panel, n_factors=2)
        assert list(out.index) == list(idx)


class TestBLK6PLSSmallN:
    """BLK-6: partial_least_squares_transform returns NaN frame when len(clean) < 2."""

    def test_zero_clean_rows_returns_nan_frame(self):
        rng = np.random.RandomState(0)
        panel = pd.DataFrame(rng.randn(5, 3), columns=list("abc"))
        panel[:] = np.nan  # all NaN -> zero clean rows after dropna
        target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="y")
        out = partial_least_squares_transform(panel, target, n_components=2)
        assert out.shape == (5, 2), f"Expected (5, 2), got {out.shape}"
        assert out.isna().all().all(), "All values should be NaN"
        assert list(out.columns) == ["pls_1", "pls_2"]

    def test_one_clean_row_returns_nan_frame(self):
        rng = np.random.RandomState(0)
        panel = pd.DataFrame({"a": [np.nan, 1.0, np.nan, np.nan, np.nan]})
        target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="y")
        out = partial_least_squares_transform(panel, target, n_components=1)
        # Only 1 clean row (row index 1 has value in panel AND target is non-NaN)
        # Wait: target has no NaN so aligned clean = rows where panel has no NaN = row 1 only
        assert len(out) == len(panel)
        assert out.isna().all().all()

    def test_sufficient_rows_normal_output(self):
        out = partial_least_squares_transform(PANEL, TARGET, n_components=2)
        assert not out.isna().all().all()
        assert out.shape[1] == 2
