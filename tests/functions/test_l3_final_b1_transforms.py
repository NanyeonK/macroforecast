"""Tests for C33 L3 final B1 panel-transform standalone callables.

8 new ops:
  sparse_pca_transform              -- sklearn SparsePCA (Zou-Hastie-Tibshirani 2006)
  sparse_pca_chen_rohe_transform    -- Chen-Rohe (2023) SCA non-diagonal-D variant
  varimax_transform                 -- orthogonal varimax rotation
  random_projection_transform       -- Johnson-Lindenstrauss Gaussian projection
  kernel_features_transform         -- exact RBF/polynomial Gram matrix
  nystroem_transform                -- Nystroem low-rank kernel approximation
  time_trend_transform              -- deterministic linear time trend
  holiday_transform                 -- US federal holiday indicator

Each bit-exact test runs the standalone callable and the runtime helper with
identical inputs and asserts np.allclose(rtol=1e-12, atol=1e-14).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.functions.transforms import (
    sparse_pca_transform,
    sparse_pca_chen_rohe_transform,
    varimax_transform,
    random_projection_transform,
    kernel_features_transform,
    nystroem_transform,
    time_trend_transform,
    holiday_transform,
)

import macroforecast as mf

# ---------------------------------------------------------------------------
# Shared fixtures (RNG-42, deterministic)
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)
PANEL = pd.DataFrame(RNG.randn(50, 8), columns=[f"x{i}" for i in range(8)])
PANEL_SMALL = pd.DataFrame(RNG.randn(20, 3), columns=list("abc"))

# Panel with DatetimeIndex for holiday tests
DATE_IDX = pd.date_range("2023-01-01", periods=50, freq="D")
PANEL_DATES = pd.DataFrame(RNG.randn(50, 3), index=DATE_IDX, columns=list("abc"))


# ---------------------------------------------------------------------------
# 1. sparse_pca_transform
# ---------------------------------------------------------------------------

class TestSparsePcaTransform:
    def test_output_shape(self):
        # n_components=3 but clamped to min(50,8)-1=7 at most; 3 < 7 so stays 3
        out = sparse_pca_transform(PANEL, n_components=3)
        assert out.shape == (50, 3)

    def test_column_names(self):
        out = sparse_pca_transform(PANEL, n_components=2)
        assert list(out.columns) == ["factor_1", "factor_2"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _pca_factors
        frame = _as_frame(PANEL)
        expected = _pca_factors(frame, n_components=2, variant="sparse_pca")
        out = sparse_pca_transform(PANEL, n_components=2)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_index_preserved(self):
        out = sparse_pca_transform(PANEL, n_components=2)
        assert list(out.index) == list(PANEL.index)

    def test_n_components_ge_1(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            sparse_pca_transform(PANEL, n_components=0)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            sparse_pca_transform(pd.DataFrame())

    def test_default_n_components(self):
        # default n_components=8; panel has 8 cols -> clamped to min(50,8)-1=7
        out = sparse_pca_transform(PANEL_SMALL)
        assert out.shape[0] == 20
        assert out.shape[1] >= 1

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "sparse_pca_transform")
        out = mf.functions.sparse_pca_transform(PANEL, n_components=2)
        assert out.shape == (50, 2)


# ---------------------------------------------------------------------------
# 2. sparse_pca_chen_rohe_transform
# ---------------------------------------------------------------------------

class TestSparsePcaChenRoheTransform:
    def test_output_shape(self):
        out = sparse_pca_chen_rohe_transform(PANEL, n_components=2)
        assert out.shape == (50, 2)

    def test_column_names(self):
        out = sparse_pca_chen_rohe_transform(PANEL, n_components=3)
        assert list(out.columns) == ["sca_1", "sca_2", "sca_3"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _sparse_pca_chen_rohe
        frame = _as_frame(PANEL)
        expected = _sparse_pca_chen_rohe(
            frame, n_components=2, zeta=0.0, max_iter=200,
            var_innovations=False, random_state=0,
        )
        out = sparse_pca_chen_rohe_transform(PANEL, n_components=2)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_zeta_custom(self):
        out = sparse_pca_chen_rohe_transform(PANEL, n_components=2, zeta=2.0)
        assert out.shape == (50, 2)

    def test_var_innovations(self):
        out = sparse_pca_chen_rohe_transform(PANEL, n_components=2, var_innovations=True)
        assert out.shape == (50, 2)

    def test_n_components_ge_1(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            sparse_pca_chen_rohe_transform(PANEL, n_components=0)

    def test_zeta_ge_0(self):
        with pytest.raises(ValueError, match="zeta >= 0"):
            sparse_pca_chen_rohe_transform(PANEL, n_components=2, zeta=-1.0)

    def test_max_iter_ge_1(self):
        with pytest.raises(ValueError, match="max_iter >= 1"):
            sparse_pca_chen_rohe_transform(PANEL, n_components=2, max_iter=0)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            sparse_pca_chen_rohe_transform(pd.DataFrame())

    def test_index_preserved(self):
        out = sparse_pca_chen_rohe_transform(PANEL, n_components=2)
        assert list(out.index) == list(PANEL.index)

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "sparse_pca_chen_rohe_transform")
        out = mf.functions.sparse_pca_chen_rohe_transform(PANEL, n_components=2)
        assert out.shape == (50, 2)


# ---------------------------------------------------------------------------
# 3. varimax_transform
# ---------------------------------------------------------------------------

class TestVarimaxTransform:
    def test_output_shape(self):
        out = varimax_transform(PANEL)
        assert out.shape == (50, 8)

    def test_column_names(self):
        out = varimax_transform(PANEL_SMALL)
        # Expect varimax_1, varimax_2, varimax_3
        assert all(c.startswith("varimax_") for c in out.columns)

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _varimax_rotation
        frame = _as_frame(PANEL_SMALL)
        expected = _varimax_rotation(frame)
        out = varimax_transform(PANEL_SMALL)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_index_preserved(self):
        out = varimax_transform(PANEL)
        assert list(out.index) == list(PANEL.index)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            varimax_transform(pd.DataFrame())

    def test_nan_rows_propagated(self):
        panel = PANEL.copy()
        panel.iloc[5] = np.nan
        out = varimax_transform(panel)
        assert out.shape[0] == 50

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "varimax_transform")
        out = mf.functions.varimax_transform(PANEL_SMALL)
        assert out.shape[0] == 20


# ---------------------------------------------------------------------------
# 4. random_projection_transform
# ---------------------------------------------------------------------------

class TestRandomProjectionTransform:
    def test_output_shape(self):
        out = random_projection_transform(PANEL, n_components=4)
        assert out.shape == (50, 4)

    def test_column_names(self):
        out = random_projection_transform(PANEL, n_components=3)
        assert list(out.columns) == ["rp_1", "rp_2", "rp_3"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _random_projection
        frame = _as_frame(PANEL)
        expected = _random_projection(frame, n_components=3)
        out = random_projection_transform(PANEL, n_components=3)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_n_components_clamped_to_K(self):
        # n_components=100 > K=8 -> clamped to 8
        out = random_projection_transform(PANEL, n_components=100)
        assert out.shape[1] <= 8

    def test_n_components_ge_1(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            random_projection_transform(PANEL, n_components=0)

    def test_index_preserved(self):
        out = random_projection_transform(PANEL, n_components=3)
        assert list(out.index) == list(PANEL.index)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            random_projection_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "random_projection_transform")
        out = mf.functions.random_projection_transform(PANEL, n_components=3)
        assert out.shape == (50, 3)


# ---------------------------------------------------------------------------
# 5. kernel_features_transform
# ---------------------------------------------------------------------------

class TestKernelFeaturesTransform:
    def test_rbf_output_shape_square(self):
        # Output is T_clean x T_clean Gram matrix
        out = kernel_features_transform(PANEL, kind="rbf", gamma=1.0)
        T_clean = len(PANEL.dropna())
        assert out.shape == (T_clean, T_clean)

    def test_polynomial_output_shape_square(self):
        out = kernel_features_transform(PANEL_SMALL, kind="polynomial", gamma=0.5)
        T_clean = len(PANEL_SMALL.dropna())
        assert out.shape == (T_clean, T_clean)

    def test_column_names(self):
        out = kernel_features_transform(PANEL_SMALL, kind="rbf", gamma=1.0)
        assert all(c.startswith("kernel_") for c in out.columns)

    def test_bit_exact_vs_runtime_rbf(self):
        from macroforecast.core.runtime import _as_frame, _kernel_features
        frame = _as_frame(PANEL_SMALL)
        expected = _kernel_features(frame, kind="rbf", gamma=0.5)
        out = kernel_features_transform(PANEL_SMALL, kind="rbf", gamma=0.5)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="unknown kind"):
            kernel_features_transform(PANEL, kind="laplacian", gamma=1.0)

    def test_gamma_le_0_raises(self):
        with pytest.raises(ValueError, match="gamma > 0"):
            kernel_features_transform(PANEL, gamma=0.0)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            kernel_features_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "kernel_features_transform")
        out = mf.functions.kernel_features_transform(PANEL_SMALL, kind="rbf", gamma=1.0)
        assert out.shape[0] == out.shape[1]  # square Gram matrix


# ---------------------------------------------------------------------------
# 6. nystroem_transform
# ---------------------------------------------------------------------------

class TestNystroemTransform:
    def test_output_shape(self):
        out = nystroem_transform(PANEL, n_components=10)
        assert out.shape == (50, 10)

    def test_column_names(self):
        out = nystroem_transform(PANEL_SMALL, n_components=5)
        assert list(out.columns) == ["nystroem_1", "nystroem_2", "nystroem_3",
                                      "nystroem_4", "nystroem_5"]

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _nystroem_features
        frame = _as_frame(PANEL_SMALL)
        expected = _nystroem_features(frame, n_components=5)
        out = nystroem_transform(PANEL_SMALL, n_components=5)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_n_components_clamped_to_T(self):
        # n_components=1000 > T=20 -> clamped to 20
        out = nystroem_transform(PANEL_SMALL, n_components=1000)
        assert out.shape[1] <= 20

    def test_n_components_ge_1(self):
        with pytest.raises(ValueError, match="n_components >= 1"):
            nystroem_transform(PANEL, n_components=0)

    def test_index_preserved(self):
        out = nystroem_transform(PANEL, n_components=5)
        assert list(out.index) == list(PANEL.index)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            nystroem_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "nystroem_transform")
        out = mf.functions.nystroem_transform(PANEL, n_components=5)
        assert out.shape == (50, 5)


# ---------------------------------------------------------------------------
# 7. time_trend_transform
# ---------------------------------------------------------------------------

class TestTimeTrendTransform:
    def test_output_shape(self):
        out = time_trend_transform(PANEL)
        assert out.shape == (50, 1)

    def test_column_name(self):
        out = time_trend_transform(PANEL)
        assert list(out.columns) == ["time_trend"]

    def test_trend_values(self):
        out = time_trend_transform(PANEL)
        expected = np.arange(1, 51, dtype=float)
        assert np.allclose(out["time_trend"].values, expected)

    def test_bit_exact_inline(self):
        # time_trend_transform uses inline np.arange - no runtime helper
        out = time_trend_transform(PANEL_SMALL)
        assert list(out["time_trend"]) == list(range(1, 21))

    def test_dtype_float(self):
        out = time_trend_transform(PANEL)
        assert out["time_trend"].dtype == np.float64

    def test_index_preserved(self):
        out = time_trend_transform(PANEL)
        assert list(out.index) == list(PANEL.index)

    def test_small_panel(self):
        panel = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        out = time_trend_transform(panel)
        assert list(out["time_trend"]) == [1.0, 2.0, 3.0]

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            time_trend_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "time_trend_transform")
        out = mf.functions.time_trend_transform(PANEL)
        assert out.shape == (50, 1)


# ---------------------------------------------------------------------------
# 8. holiday_transform
# ---------------------------------------------------------------------------

class TestHolidayTransform:
    def test_output_shape_integer_index(self):
        out = holiday_transform(PANEL)
        assert out.shape == (50, 1)

    def test_column_name(self):
        out = holiday_transform(PANEL)
        assert list(out.columns) == ["is_holiday"]

    def test_integer_index_all_zeros(self):
        # Non-DatetimeIndex -> all zeros
        out = holiday_transform(PANEL)
        assert (out["is_holiday"] == 0.0).all()

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _holiday_indicator
        frame = _as_frame(PANEL)
        expected = _holiday_indicator(frame)
        out = holiday_transform(PANEL)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_datetime_index_flag(self):
        # New Year's Day 2023 is a US federal holiday (observed Monday Jan 2)
        idx = pd.DatetimeIndex(["2023-01-02", "2023-01-03", "2023-01-04"])
        panel = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=idx)
        out = holiday_transform(panel)
        # 2023-01-02 is New Year's Day (observed) - should be 1.0
        assert out.loc["2023-01-02", "is_holiday"] == 1.0
        # Jan 3 and 4 are not holidays
        assert out.loc["2023-01-03", "is_holiday"] == 0.0
        assert out.loc["2023-01-04", "is_holiday"] == 0.0

    def test_datetime_panel_shape(self):
        out = holiday_transform(PANEL_DATES)
        assert out.shape == (50, 1)
        assert list(out.columns) == ["is_holiday"]

    def test_bit_exact_datetime_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _holiday_indicator
        frame = _as_frame(PANEL_DATES)
        expected = _holiday_indicator(frame)
        out = holiday_transform(PANEL_DATES)
        assert np.allclose(out.fillna(0), expected.fillna(0), rtol=1e-12, atol=1e-14)

    def test_index_preserved(self):
        out = holiday_transform(PANEL)
        assert list(out.index) == list(PANEL.index)

    def test_empty_panel_raises(self):
        with pytest.raises(ValueError):
            holiday_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "holiday_transform")
        out = mf.functions.holiday_transform(PANEL)
        assert out.shape == (50, 1)


# ---------------------------------------------------------------------------
# OD-3: Alias OptionDoc entries must have op_page=False
# ---------------------------------------------------------------------------

class TestOD3AliasOpPageFalse:
    def test_OD_3_aliases_have_op_page_false(self):
        from macroforecast.scaffold.option_docs.l3 import (
            _OP_VARIMAX_ROTATION,
            _OP_KERNEL,
            _OP_NYSTROEM_FEATURES,
            _OP_POLYNOMIAL,
        )
        for od in [_OP_VARIMAX_ROTATION, _OP_KERNEL, _OP_NYSTROEM_FEATURES, _OP_POLYNOMIAL]:
            assert od.op_page is False, f"{od.option!r} alias must have op_page=False"
