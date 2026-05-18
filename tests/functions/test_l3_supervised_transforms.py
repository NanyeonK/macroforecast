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
