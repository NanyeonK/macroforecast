"""Tests for C30 L3 basic panel-transform standalone callables.

Each test uses a fixed RNG-42 mini-panel (100 rows x 3 cols) for
bit-exact reproducibility.  Assertions are either ``np.allclose`` at
rtol=1e-12 or ``DataFrame.equals`` for integer/structure checks.

Bit-exact parity: each callable is also run via the runtime recipe
path (calling the same runtime helper directly) and compared with
``np.allclose(rtol=1e-12, atol=1e-14)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Import the 10 standalone callables under test
from macroforecast.functions.transforms import (
    diff_transform,
    log_transform,
    log_diff_transform,
    pct_change_transform,
    cumsum_transform,
    ma_window_transform,
    lag_matrix,
    seasonal_lag_matrix,
    ma_increasing_order_transform,
    scale_transform,
)

# Also import via mf.functions namespace to verify __init__ wiring
import macroforecast as mf


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(42)
_BASE = pd.DataFrame(
    RNG.randn(100, 3),
    columns=["a", "b", "c"],
)
# Positive panel for log-transform tests (shift to > 0)
PANEL = _BASE + 5.0
# Plain panel (may be negative) for non-log tests
PANEL_RAW = _BASE.copy()


def _runtime_helpers():
    """Lazy-import all runtime helpers used in bit-exact checks."""
    from macroforecast.core.runtime import (  # noqa: PLC0415
        _as_frame,
        _diff_like,
        _pct_change_like,
        _lagged_predictors,
        _seasonal_lagged_predictors,
        _ma_increasing_order,
        _scale_frame,
    )
    return (
        _as_frame,
        _diff_like,
        _pct_change_like,
        _lagged_predictors,
        _seasonal_lagged_predictors,
        _ma_increasing_order,
        _scale_frame,
    )


# ---------------------------------------------------------------------------
# 1. diff_transform
# ---------------------------------------------------------------------------

class TestDiffTransform:
    def test_shape_preserved(self):
        out = diff_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape

    def test_first_row_nan(self):
        out = diff_transform(PANEL_RAW)
        assert out.iloc[0].isna().all()

    def test_default_periods_one(self):
        out = diff_transform(PANEL_RAW)
        expected_a = PANEL_RAW["a"].diff(1)
        np.testing.assert_allclose(out["a"].values, expected_a.values, rtol=1e-12, equal_nan=True)

    def test_periods_2(self):
        out = diff_transform(PANEL_RAW, periods=2)
        assert out.iloc[:2]["a"].isna().all()
        assert out.shape == PANEL_RAW.shape

    def test_bit_exact_vs_runtime(self):
        (_, _diff_like, *_rest) = _runtime_helpers()
        from macroforecast.core.runtime import _as_frame
        expected = _diff_like(_as_frame(PANEL_RAW), periods=1)
        out = diff_transform(PANEL_RAW)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, equal_nan=True)

    def test_invalid_periods(self):
        with pytest.raises(ValueError, match="periods >= 1"):
            diff_transform(PANEL_RAW, periods=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            diff_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "diff_transform")
        out = mf.functions.diff_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape


# ---------------------------------------------------------------------------
# 2. log_transform
# ---------------------------------------------------------------------------

class TestLogTransform:
    def test_shape_preserved(self):
        out = log_transform(PANEL)
        assert out.shape == PANEL.shape

    def test_values_correct(self):
        out = log_transform(PANEL)
        np.testing.assert_allclose(out.values, np.log(PANEL.values), rtol=1e-12)

    def test_bit_exact_vs_numpy(self):
        # log_transform uses np.log(_as_frame(panel)); compare directly
        out = log_transform(PANEL)
        expected = np.log(PANEL)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14)

    def test_columns_index_preserved(self):
        out = log_transform(PANEL)
        assert list(out.columns) == list(PANEL.columns)
        assert list(out.index) == list(PANEL.index)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            log_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "log_transform")
        out = mf.functions.log_transform(PANEL)
        assert out.shape == PANEL.shape


# ---------------------------------------------------------------------------
# 3. log_diff_transform
# ---------------------------------------------------------------------------

class TestLogDiffTransform:
    def test_shape_preserved(self):
        out = log_diff_transform(PANEL)
        assert out.shape == PANEL.shape

    def test_first_row_nan(self):
        out = log_diff_transform(PANEL)
        assert out.iloc[0].isna().all()

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _diff_like
        logged = np.log(_as_frame(PANEL))
        expected = _diff_like(logged, periods=1)
        out = log_diff_transform(PANEL)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True)

    def test_periods_2(self):
        out = log_diff_transform(PANEL, periods=2)
        assert out.iloc[:2]["a"].isna().all()

    def test_invalid_periods(self):
        with pytest.raises(ValueError, match="periods >= 1"):
            log_diff_transform(PANEL, periods=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            log_diff_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "log_diff_transform")
        out = mf.functions.log_diff_transform(PANEL)
        assert out.shape == PANEL.shape


# ---------------------------------------------------------------------------
# 4. pct_change_transform
# ---------------------------------------------------------------------------

class TestPctChangeTransform:
    def test_shape_preserved(self):
        out = pct_change_transform(PANEL)
        assert out.shape == PANEL.shape

    def test_first_row_nan(self):
        out = pct_change_transform(PANEL)
        assert out.iloc[0].isna().all()

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _pct_change_like
        expected = _pct_change_like(_as_frame(PANEL), periods=1)
        out = pct_change_transform(PANEL)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True)

    def test_periods_3(self):
        out = pct_change_transform(PANEL, periods=3)
        assert out.iloc[:3]["a"].isna().all()
        assert out.shape == PANEL.shape

    def test_invalid_periods(self):
        with pytest.raises(ValueError, match="periods >= 1"):
            pct_change_transform(PANEL, periods=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            pct_change_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "pct_change_transform")
        out = mf.functions.pct_change_transform(PANEL)
        assert out.shape == PANEL.shape


# ---------------------------------------------------------------------------
# 5. cumsum_transform
# ---------------------------------------------------------------------------

class TestCumsumTransform:
    def test_shape_preserved(self):
        out = cumsum_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame
        expected = _as_frame(PANEL_RAW).cumsum()
        out = cumsum_transform(PANEL_RAW)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14)

    def test_last_value_equals_sum(self):
        out = cumsum_transform(PANEL_RAW)
        expected_last_a = float(PANEL_RAW["a"].sum())
        assert abs(float(out["a"].iloc[-1]) - expected_last_a) < 1e-10

    def test_monotone_for_positive_input(self):
        pos_panel = pd.DataFrame({"x": np.ones(10)})
        out = cumsum_transform(pos_panel)
        assert (out["x"].diff().iloc[1:] >= 0).all()

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            cumsum_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "cumsum_transform")
        out = mf.functions.cumsum_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape


# ---------------------------------------------------------------------------
# 6. ma_window_transform
# ---------------------------------------------------------------------------

class TestMaWindowTransform:
    def test_shape_preserved(self):
        out = ma_window_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape

    def test_first_rows_nan_with_window_3(self):
        out = ma_window_transform(PANEL_RAW, window=3)
        assert out.iloc[:2]["a"].isna().all()
        assert not np.isnan(float(out["a"].iloc[2]))

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame
        expected = _as_frame(PANEL_RAW).rolling(window=3, min_periods=3).mean()
        out = ma_window_transform(PANEL_RAW, window=3)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True)

    def test_window_1_is_identity(self):
        out = ma_window_transform(PANEL_RAW, window=1)
        np.testing.assert_allclose(out.values, PANEL_RAW.values, rtol=1e-12)

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window >= 1"):
            ma_window_transform(PANEL_RAW, window=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ma_window_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "ma_window_transform")
        out = mf.functions.ma_window_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape


# ---------------------------------------------------------------------------
# 7. lag_matrix
# ---------------------------------------------------------------------------

class TestLagMatrix:
    def test_output_columns_count(self):
        out = lag_matrix(PANEL_RAW, n_lag=4)
        # 3 columns * 4 lags = 12
        assert out.shape[1] == 12

    def test_column_names(self):
        out = lag_matrix(PANEL_RAW, n_lag=2)
        assert "a_lag1" in out.columns
        assert "a_lag2" in out.columns
        assert "b_lag1" in out.columns

    def test_include_contemporaneous(self):
        out = lag_matrix(PANEL_RAW, n_lag=2, include_contemporaneous=True)
        # 3 columns * (1 contemporaneous + 2 lags) = 9
        assert out.shape[1] == 9
        assert "a_lag0" in out.columns

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _lagged_predictors
        expected = _lagged_predictors(_as_frame(PANEL_RAW), 4, include_contemporaneous=False)
        out = lag_matrix(PANEL_RAW, n_lag=4)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True)

    def test_first_rows_nan(self):
        out = lag_matrix(PANEL_RAW, n_lag=3)
        # lag3 column should have NaN in first 3 rows
        assert out["a_lag3"].iloc[:3].isna().all()

    def test_invalid_n_lag(self):
        with pytest.raises(ValueError, match="n_lag >= 1"):
            lag_matrix(PANEL_RAW, n_lag=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            lag_matrix(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "lag_matrix")
        out = mf.functions.lag_matrix(PANEL_RAW)
        assert out.shape[1] == 12


# ---------------------------------------------------------------------------
# 8. seasonal_lag_matrix
# ---------------------------------------------------------------------------

class TestSeasonalLagMatrix:
    def test_output_columns_with_1_lag(self):
        out = seasonal_lag_matrix(PANEL_RAW, seasonal_period=12, n_seasonal_lags=1)
        assert out.shape == (100, 3)
        assert "a_s12_lag1" in out.columns

    def test_output_columns_with_2_lags(self):
        out = seasonal_lag_matrix(PANEL_RAW, seasonal_period=12, n_seasonal_lags=2)
        assert out.shape == (100, 6)
        assert "a_s12_lag2" in out.columns

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _seasonal_lagged_predictors
        expected = _seasonal_lagged_predictors(
            _as_frame(PANEL_RAW),
            seasonal_period=12,
            n_seasonal_lags=1,
        )
        out = seasonal_lag_matrix(PANEL_RAW, seasonal_period=12, n_seasonal_lags=1)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True)

    def test_first_seasonal_period_rows_nan(self):
        out = seasonal_lag_matrix(PANEL_RAW, seasonal_period=12, n_seasonal_lags=1)
        assert out["a_s12_lag1"].iloc[:12].isna().all()
        assert not np.isnan(float(out["a_s12_lag1"].iloc[12]))

    def test_invalid_seasonal_period(self):
        with pytest.raises(ValueError, match="seasonal_period >= 2"):
            seasonal_lag_matrix(PANEL_RAW, seasonal_period=1)

    def test_invalid_n_seasonal_lags(self):
        with pytest.raises(ValueError, match="n_seasonal_lags >= 1"):
            seasonal_lag_matrix(PANEL_RAW, n_seasonal_lags=0)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            seasonal_lag_matrix(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "seasonal_lag_matrix")
        out = mf.functions.seasonal_lag_matrix(PANEL_RAW)
        assert "a_s12_lag1" in out.columns


# ---------------------------------------------------------------------------
# 9. ma_increasing_order_transform
# ---------------------------------------------------------------------------

class TestMaIncreasingOrderTransform:
    def test_output_columns(self):
        out = ma_increasing_order_transform(PANEL_RAW, max_order=4)
        # 3 cols * (orders 2,3,4) = 9 columns
        assert out.shape[1] == 9
        assert "a_ma2" in out.columns
        assert "a_ma4" in out.columns

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _ma_increasing_order
        expected = _ma_increasing_order(_as_frame(PANEL_RAW), max_order=12)
        out = ma_increasing_order_transform(PANEL_RAW, max_order=12)
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14, equal_nan=True)

    def test_ma2_nan_first_row(self):
        out = ma_increasing_order_transform(PANEL_RAW, max_order=5)
        assert np.isnan(float(out["a_ma2"].iloc[0]))
        assert not np.isnan(float(out["a_ma2"].iloc[1]))

    def test_ma12_nan_first_11_rows(self):
        out = ma_increasing_order_transform(PANEL_RAW, max_order=12)
        assert out["a_ma12"].iloc[:11].isna().all()
        assert not np.isnan(float(out["a_ma12"].iloc[11]))

    def test_invalid_max_order(self):
        with pytest.raises(ValueError, match="max_order >= 2"):
            ma_increasing_order_transform(PANEL_RAW, max_order=1)

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ma_increasing_order_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "ma_increasing_order_transform")
        out = mf.functions.ma_increasing_order_transform(PANEL_RAW)
        assert "a_ma2" in out.columns


# ---------------------------------------------------------------------------
# 10. scale_transform
# ---------------------------------------------------------------------------

class TestScaleTransform:
    def test_zscore_zero_mean(self):
        out = scale_transform(PANEL_RAW, method="zscore")
        np.testing.assert_allclose(out.mean().values, np.zeros(3), atol=1e-10)

    def test_zscore_unit_std(self):
        out = scale_transform(PANEL_RAW, method="zscore")
        np.testing.assert_allclose(out.std(ddof=0).values, np.ones(3), atol=1e-10)

    def test_standard_alias(self):
        out_z = scale_transform(PANEL_RAW, method="zscore")
        out_s = scale_transform(PANEL_RAW, method="standard")
        np.testing.assert_allclose(out_z.values, out_s.values, rtol=1e-12, atol=1e-14)

    def test_minmax_range(self):
        out = scale_transform(PANEL_RAW, method="minmax")
        np.testing.assert_allclose(out.min().values, np.zeros(3), atol=1e-10)
        np.testing.assert_allclose(out.max().values, np.ones(3), atol=1e-10)

    def test_robust_median_zero(self):
        out = scale_transform(PANEL_RAW, method="robust")
        np.testing.assert_allclose(out.median().values, np.zeros(3), atol=1e-10)

    def test_bit_exact_vs_runtime(self):
        from macroforecast.core.runtime import _as_frame, _scale_frame
        expected = _scale_frame(_as_frame(PANEL_RAW), method="zscore")
        out = scale_transform(PANEL_RAW, method="zscore")
        np.testing.assert_allclose(out.values, expected.values, rtol=1e-12, atol=1e-14)

    def test_invalid_method(self):
        with pytest.raises(NotImplementedError, match="method="):
            scale_transform(PANEL_RAW, method="l2norm")

    def test_shape_preserved(self):
        out = scale_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape

    def test_empty_panel_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            scale_transform(pd.DataFrame())

    def test_namespace_wiring(self):
        assert hasattr(mf.functions, "scale_transform")
        out = mf.functions.scale_transform(PANEL_RAW)
        assert out.shape == PANEL_RAW.shape


# ---------------------------------------------------------------------------
# Namespace content test
# ---------------------------------------------------------------------------

class TestNamespaceContent:
    """Verify all 10 callables are importable from mf.functions."""

    EXPECTED_NAMES = [
        "diff_transform",
        "log_transform",
        "log_diff_transform",
        "pct_change_transform",
        "cumsum_transform",
        "ma_window_transform",
        "lag_matrix",
        "seasonal_lag_matrix",
        "ma_increasing_order_transform",
        "scale_transform",
    ]

    def test_all_names_present(self):
        for name in self.EXPECTED_NAMES:
            assert hasattr(mf.functions, name), f"mf.functions.{name} missing"

    def test_all_in_dunder_all(self):
        import macroforecast.functions as mff
        all_set = set(mff.__all__)
        for name in self.EXPECTED_NAMES:
            assert name in all_set, f"{name} not in __all__"

    def test_callables_are_callable(self):
        for name in self.EXPECTED_NAMES:
            fn = getattr(mf.functions, name)
            assert callable(fn), f"mf.functions.{name} is not callable"
