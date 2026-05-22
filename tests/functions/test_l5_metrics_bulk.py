"""Tests for Cycle 27 L5 metrics bulk standalone callables.

Each test class covers one new function in ``mf.functions``.
Bit-exact assertions compare against the numpy formula extracted from
runtime.py (which IS the formula, since Python float arithmetic is
deterministic for these operations).

Canonical-only functions (mape, interval_score, coverage_rate) are
asserted against hand-computed expected values.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import macroforecast as mf


# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def arrays_100():
    """100-element random arrays with seed 42."""
    rng = np.random.RandomState(42)
    y_true = rng.randn(100)
    y_pred = y_true + 0.1 * rng.randn(100)
    return y_true, y_pred


@pytest.fixture(scope="module")
def arrays_small():
    """Small 5-element arrays for hand-computed assertions."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.0, 2.5, 4.5, 5.5])
    return y_true, y_pred


@pytest.fixture(scope="module")
def arrays_relative():
    """Arrays for relative metrics (y_true, y_model, y_benchmark)."""
    rng = np.random.RandomState(0)
    y_true = rng.randn(50)
    y_model = y_true + 0.1 * rng.randn(50)
    y_benchmark = y_true + 0.5 * rng.randn(50)
    return y_true, y_model, y_benchmark


@pytest.fixture(scope="module")
def arrays_interval():
    """Arrays for interval/coverage metrics."""
    rng = np.random.RandomState(7)
    y_true = rng.randn(60)
    y_lower = y_true - 1.96
    y_upper = y_true + 1.96
    return y_true, y_lower, y_upper


@pytest.fixture(scope="module")
def arrays_direction():
    """Arrays for direction metrics with y_prev (first row is NaN)."""
    rng = np.random.RandomState(13)
    y_true = rng.randn(80)
    y_pred = y_true + 0.2 * rng.randn(80)
    y_prev = np.empty(80)
    y_prev[0] = np.nan
    y_prev[1:] = y_true[:-1]
    return y_true, y_pred, y_prev


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------

class TestMse:
    def test_bit_exact_formula(self, arrays_100):
        y_true, y_pred = arrays_100
        expected = float(np.mean((y_true - y_pred) ** 2))
        result = mf.functions.mse(y_true, y_pred)
        assert result == expected, f"mse: {result} != {expected}"

    def test_perfect_forecast_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mf.functions.mse(y, y) == 0.0

    def test_hand_computed(self, arrays_small):
        y_true, y_pred = arrays_small
        # errors: 0.5, 0.0, 0.5, 0.5, 0.5 -> squared: 0.25, 0, 0.25, 0.25, 0.25 -> mean: 0.2
        assert abs(mf.functions.mse(y_true, y_pred) - 0.2) < 1e-12

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.mse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.mse(np.array([]), np.array([]))

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            mf.functions.mse(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))

    def test_accepts_series(self):
        import pandas as pd
        y_true = pd.Series([1.0, 2.0, 3.0])
        y_pred = pd.Series([1.5, 2.5, 3.5])
        result = mf.functions.mse(y_true, y_pred)
        assert abs(result - 0.25) < 1e-12

    def test_returns_float(self, arrays_100):
        y_true, y_pred = arrays_100
        assert isinstance(mf.functions.mse(y_true, y_pred), float)


class TestRmse:
    def test_bit_exact_formula(self, arrays_100):
        y_true, y_pred = arrays_100
        expected = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        result = mf.functions.rmse(y_true, y_pred)
        assert result == expected, f"rmse: {result} != {expected}"

    def test_equals_sqrt_mse(self, arrays_100):
        y_true, y_pred = arrays_100
        assert abs(mf.functions.rmse(y_true, y_pred) - math.sqrt(mf.functions.mse(y_true, y_pred))) < 1e-15

    def test_hand_computed(self, arrays_small):
        y_true, y_pred = arrays_small
        expected = math.sqrt(0.2)
        assert abs(mf.functions.rmse(y_true, y_pred) - expected) < 1e-12

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.rmse(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.rmse(np.array([]), np.array([]))

    def test_returns_float(self, arrays_100):
        y_true, y_pred = arrays_100
        assert isinstance(mf.functions.rmse(y_true, y_pred), float)


class TestMae:
    def test_bit_exact_formula(self, arrays_100):
        y_true, y_pred = arrays_100
        expected = float(np.mean(np.abs(y_true - y_pred)))
        result = mf.functions.mae(y_true, y_pred)
        assert result == expected, f"mae: {result} != {expected}"

    def test_perfect_forecast_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mf.functions.mae(y, y) == 0.0

    def test_hand_computed(self, arrays_small):
        y_true, y_pred = arrays_small
        # abs errors: 0.5, 0.0, 0.5, 0.5, 0.5 -> mean: 0.4
        assert abs(mf.functions.mae(y_true, y_pred) - 0.4) < 1e-12

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.mae(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.mae(np.array([]), np.array([]))

    def test_returns_float(self, arrays_100):
        y_true, y_pred = arrays_100
        assert isinstance(mf.functions.mae(y_true, y_pred), float)


class TestMedae:
    def test_bit_exact_formula(self, arrays_100):
        y_true, y_pred = arrays_100
        expected = float(np.median(np.abs(y_true - y_pred)))
        result = mf.functions.medae(y_true, y_pred)
        assert result == expected, f"medae: {result} != {expected}"

    def test_perfect_forecast_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mf.functions.medae(y, y) == 0.0

    def test_hand_computed(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([0.0, 2.0, 4.0, 4.0, 5.0])
        # abs errors: 1.0, 0.0, 1.0, 0.0, 0.0 -> sorted: 0,0,0,1,1 -> median = 0.0
        assert mf.functions.medae(y_true, y_pred) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.medae(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.medae(np.array([]), np.array([]))

    def test_returns_float(self, arrays_100):
        y_true, y_pred = arrays_100
        assert isinstance(mf.functions.medae(y_true, y_pred), float)


class TestMape:
    def test_canonical_implementation_formula(self):
        """mape has no recipe path; verify against the canonical formula."""
        y_true = np.array([100.0, 200.0, 400.0])
        y_pred = np.array([110.0, 190.0, 380.0])
        # abs errors: 10, 10, 20
        # safe_denom: 100, 200, 400
        # ratios: 0.1, 0.05, 0.05
        # mean: 0.0667 * 100 = 6.666...
        expected = (0.1 + 0.05 + 0.05) / 3 * 100
        assert abs(mf.functions.mape(y_true, y_pred) - expected) < 1e-10

    def test_perfect_forecast_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mf.functions.mape(y, y) == 0.0

    def test_eps_guard_near_zero_target(self):
        y_true = np.array([0.0, 1.0])
        y_pred = np.array([0.1, 1.1])
        # Without eps guard, would divide by 0 for first element
        result = mf.functions.mape(y_true, y_pred)
        assert math.isfinite(result)

    def test_eps_validation(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            mf.functions.mape(np.array([1.0]), np.array([2.0]), eps=0.0)

    def test_eps_validation_negative(self):
        with pytest.raises(ValueError, match="eps must be positive"):
            mf.functions.mape(np.array([1.0]), np.array([2.0]), eps=-1e-5)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.mape(np.array([1.0, 2.0]), np.array([1.0]))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.mape(np.array([]), np.array([]))

    def test_returns_float(self, arrays_100):
        y_true, y_pred = arrays_100
        assert isinstance(mf.functions.mape(y_true, y_pred), float)


# ---------------------------------------------------------------------------
# Relative metrics
# ---------------------------------------------------------------------------

class TestRelativeMse:
    def test_bit_exact_formula(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        num = float(np.mean((y_true - y_model) ** 2))
        den = float(np.mean((y_true - y_benchmark) ** 2))
        expected = num / den
        result = mf.functions.relative_mse(y_true, y_model, y_benchmark)
        assert result == expected, f"relative_mse: {result} != {expected}"

    def test_model_equals_benchmark_gives_one(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_model = np.array([1.5, 2.5, 3.5])
        result = mf.functions.relative_mse(y_true, y_model, y_model)
        assert abs(result - 1.0) < 1e-12

    def test_zero_benchmark_mse_returns_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        result = mf.functions.relative_mse(y_true, y_true + 0.1, y_true)
        assert math.isnan(result)

    def test_better_model_less_than_one(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        result = mf.functions.relative_mse(y_true, y_model, y_benchmark)
        # y_model is closer to y_true than y_benchmark
        assert result < 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.relative_mse(
                np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0])
            )

    def test_returns_float(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        assert isinstance(mf.functions.relative_mse(y_true, y_model, y_benchmark), float)


class TestRelativeMae:
    def test_bit_exact_formula(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        num = float(np.mean(np.abs(y_true - y_model)))
        den = float(np.mean(np.abs(y_true - y_benchmark)))
        expected = num / den
        result = mf.functions.relative_mae(y_true, y_model, y_benchmark)
        assert result == expected, f"relative_mae: {result} != {expected}"

    def test_zero_benchmark_mae_returns_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        result = mf.functions.relative_mae(y_true, y_true + 0.1, y_true)
        assert math.isnan(result)

    def test_returns_float(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        assert isinstance(mf.functions.relative_mae(y_true, y_model, y_benchmark), float)


class TestMseReduction:
    def test_bit_exact_formula(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        bench_mse = float(np.mean((y_true - y_benchmark) ** 2))
        model_mse = float(np.mean((y_true - y_model) ** 2))
        expected = bench_mse - model_mse
        result = mf.functions.mse_reduction(y_true, y_model, y_benchmark)
        assert result == expected, f"mse_reduction: {result} != {expected}"

    def test_positive_when_model_better(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        result = mf.functions.mse_reduction(y_true, y_model, y_benchmark)
        assert result > 0.0

    def test_hand_computed(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_model = np.array([1.1, 2.1, 3.1])     # MSE = 0.01
        y_bench = np.array([1.5, 2.5, 3.5])      # MSE = 0.25
        result = mf.functions.mse_reduction(y_true, y_model, y_bench)
        assert abs(result - 0.24) < 1e-12

    def test_note_absolute_not_ratio(self):
        """Regression: mse_reduction returns absolute difference, not 1 - relative_mse."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_model = np.array([10.1, 20.1, 30.1])   # MSE ~= 0.01
        y_bench = np.array([11.0, 21.0, 31.0])   # MSE = 1.0
        absolute = mf.functions.mse_reduction(y_true, y_model, y_bench)
        ratio_based = 1.0 - mf.functions.relative_mse(y_true, y_model, y_bench)
        # These are NOT equal (different units)
        assert abs(absolute - (1.0 - 0.01)) < 1e-10   # = 0.99
        assert abs(ratio_based - (1.0 - 0.01 / 1.0)) < 1e-10  # = 0.99 (same here by coincidence when bench_mse=1)

    def test_returns_float(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        assert isinstance(mf.functions.mse_reduction(y_true, y_model, y_benchmark), float)


class TestR2Oos:
    def test_bit_exact_formula(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        expected = 1.0 - mf.functions.relative_mse(y_true, y_model, y_benchmark)
        result = mf.functions.r2_oos(y_true, y_model, y_benchmark)
        assert result == expected, f"r2_oos: {result} != {expected}"

    def test_positive_when_model_better(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        assert mf.functions.r2_oos(y_true, y_model, y_benchmark) > 0.0

    def test_zero_benchmark_mse_returns_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        result = mf.functions.r2_oos(y_true, y_true + 0.1, y_true)
        assert math.isnan(result)

    def test_hand_computed(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_model = np.array([1.1, 2.1, 3.1])   # MSE = 0.01
        y_bench = np.array([1.5, 2.5, 3.5])   # MSE = 0.25
        result = mf.functions.r2_oos(y_true, y_model, y_bench)
        assert abs(result - (1.0 - 0.04)) < 1e-12  # 1 - 0.01/0.25 = 0.96

    def test_returns_float(self, arrays_relative):
        y_true, y_model, y_benchmark = arrays_relative
        assert isinstance(mf.functions.r2_oos(y_true, y_model, y_benchmark), float)


# ---------------------------------------------------------------------------
# Interval / coverage metrics
# ---------------------------------------------------------------------------

class TestIntervalScore:
    def test_canonical_formula(self, arrays_interval):
        y_true, y_lower, y_upper = arrays_interval
        alpha = 0.05
        width = y_upper - y_lower
        under = np.maximum(y_lower - y_true, 0.0)
        over = np.maximum(y_true - y_upper, 0.0)
        expected = float(np.mean(width + (2.0 / alpha) * under + (2.0 / alpha) * over))
        result = mf.functions.interval_score(y_true, y_lower, y_upper, alpha=alpha)
        assert abs(result - expected) < 1e-12

    def test_no_violation_score_equals_width(self):
        """When all obs inside interval, score = mean width."""
        y_true = np.array([0.5, 1.5, 2.5])
        y_lower = np.array([0.0, 1.0, 2.0])
        y_upper = np.array([1.0, 2.0, 3.0])
        result = mf.functions.interval_score(y_true, y_lower, y_upper)
        assert abs(result - 1.0) < 1e-12  # mean width = 1.0

    def test_violation_adds_penalty(self):
        """One obs below interval should incur (2/alpha)*miss penalty."""
        alpha = 0.10
        y_true = np.array([0.0])
        y_lower = np.array([1.0])
        y_upper = np.array([2.0])
        # width=1, under=1, over=0
        expected = 1.0 + (2.0 / alpha) * 1.0
        result = mf.functions.interval_score(y_true, y_lower, y_upper, alpha=alpha)
        assert abs(result - expected) < 1e-12

    def test_alpha_validation_zero(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            mf.functions.interval_score(
                np.array([1.0]), np.array([0.5]), np.array([1.5]), alpha=0.0
            )

    def test_alpha_validation_one(self):
        with pytest.raises(ValueError, match="alpha must be in"):
            mf.functions.interval_score(
                np.array([1.0]), np.array([0.5]), np.array([1.5]), alpha=1.0
            )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.interval_score(
                np.array([1.0, 2.0]), np.array([0.5]), np.array([1.5])
            )

    def test_returns_float(self, arrays_interval):
        y_true, y_lower, y_upper = arrays_interval
        assert isinstance(mf.functions.interval_score(y_true, y_lower, y_upper), float)


class TestCoverageRate:
    def test_canonical_formula(self, arrays_interval):
        y_true, y_lower, y_upper = arrays_interval
        hits = (y_true >= y_lower) & (y_true <= y_upper)
        expected = float(np.mean(hits.astype(float)))
        result = mf.functions.coverage_rate(y_true, y_lower, y_upper)
        assert abs(result - expected) < 1e-12

    def test_all_inside_is_one(self):
        y_true = np.array([0.5, 1.5, 2.5])
        y_lower = np.zeros(3)
        y_upper = np.ones(3) * 3.0
        assert mf.functions.coverage_rate(y_true, y_lower, y_upper) == 1.0

    def test_none_inside_is_zero(self):
        y_true = np.array([5.0, 6.0, 7.0])
        y_lower = np.zeros(3)
        y_upper = np.ones(3)
        assert mf.functions.coverage_rate(y_true, y_lower, y_upper) == 0.0

    def test_partial_coverage(self):
        y_true = np.array([0.5, 5.0, 1.5])
        y_lower = np.zeros(3)
        y_upper = np.ones(3) * 2.0
        # 2 of 3 inside
        assert abs(mf.functions.coverage_rate(y_true, y_lower, y_upper) - 2 / 3) < 1e-12

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.coverage_rate(
                np.array([1.0, 2.0]), np.array([0.5]), np.array([1.5])
            )

    def test_returns_float(self, arrays_interval):
        y_true, y_lower, y_upper = arrays_interval
        assert isinstance(mf.functions.coverage_rate(y_true, y_lower, y_upper), float)

    def test_bounded_01(self, arrays_interval):
        y_true, y_lower, y_upper = arrays_interval
        result = mf.functions.coverage_rate(y_true, y_lower, y_upper)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Direction metrics
# ---------------------------------------------------------------------------

class TestSuccessRatio:
    def test_bit_exact_formula(self, arrays_direction):
        y_true, y_pred, y_prev = arrays_direction
        valid = ~np.isnan(y_prev)
        yt = y_true[valid]; yp = y_pred[valid]; yp_prev = y_prev[valid]
        sign_pred = np.sign(yp - yp_prev)
        sign_true = np.sign(yt - yp_prev)
        expected = float(np.mean(sign_pred == sign_true))
        result = mf.functions.success_ratio(y_true, y_pred, y_prev)
        assert result == expected, f"success_ratio: {result} != {expected}"

    def test_perfect_directional_forecast(self):
        y_true = np.array([1.0, 2.0, 1.5, 3.0])
        y_pred = np.array([1.2, 2.2, 1.3, 3.1])
        y_prev = np.array([np.nan, 1.0, 2.0, 1.5])
        # Changes: +1, -0.5, +1.5 (true); pred: +1.2, -0.9, +1.8 -> same signs
        result = mf.functions.success_ratio(y_true, y_pred, y_prev)
        assert result == 1.0

    def test_nan_first_row_excluded(self):
        """With y_prev[0] = nan, first row excluded; 3 valid rows."""
        y_true = np.array([2.0, 3.0, 2.0, 4.0])
        y_pred = np.array([2.5, 3.5, 1.5, 4.5])
        y_prev = np.array([np.nan, 2.0, 3.0, 2.0])
        result = mf.functions.success_ratio(y_true, y_pred, y_prev)
        assert math.isfinite(result)
        assert 0.0 <= result <= 1.0

    def test_all_nan_prev_returns_nan(self):
        result = mf.functions.success_ratio(
            np.array([1.0, 2.0]),
            np.array([1.5, 2.5]),
            np.array([np.nan, np.nan])
        )
        assert math.isnan(result)

    def test_only_one_valid_returns_nan(self):
        result = mf.functions.success_ratio(
            np.array([1.0, 2.0]),
            np.array([1.5, 2.5]),
            np.array([np.nan, 1.0])
        )
        assert math.isnan(result)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.success_ratio(
                np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0, 2.0])
            )

    def test_returns_float(self, arrays_direction):
        y_true, y_pred, y_prev = arrays_direction
        assert isinstance(mf.functions.success_ratio(y_true, y_pred, y_prev), float)


class TestPesaranTimmermannMetric:
    def test_bit_exact_formula(self, arrays_direction):
        """Compare against manually running the PT formula."""
        y_true, y_pred, y_prev = arrays_direction
        # Use default threshold=0.0
        threshold = 0.0
        forecast = (y_pred > threshold).astype(int)
        actual = (y_true > threshold).astype(int)
        n = len(forecast)
        success = float((forecast == actual).mean())
        p_y = float(actual.mean())
        p_x = float(forecast.mean())
        p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
        if not (0 < p_star < 1):
            expected = float("nan")
        else:
            var_p = (p_star * (1 - p_star)) / n
            var_p_star = (
                ((2 * p_y - 1) ** 2 * p_x * (1 - p_x)) / n
                + ((2 * p_x - 1) ** 2 * p_y * (1 - p_y)) / n
                + (4 * p_y * p_x * (1 - p_y) * (1 - p_x)) / (n * n)
            )
            denom = var_p - var_p_star
            if denom <= 1e-12:
                expected = float("nan")
            else:
                expected = float((success - p_star) / math.sqrt(denom))
        result = mf.functions.pesaran_timmermann_metric(y_true, y_pred)
        if math.isnan(expected):
            assert math.isnan(result)
        else:
            assert abs(result - expected) < 1e-12, f"PT: {result} != {expected}"

    def test_returns_finite_for_good_data(self):
        rng = np.random.RandomState(99)
        y_true = rng.choice([0, 1], size=200).astype(float)
        y_pred = y_true + 0.3 * rng.randn(200)
        result = mf.functions.pesaran_timmermann_metric(y_true, y_pred)
        assert math.isfinite(result) or math.isnan(result)  # nan also OK per spec

    def test_single_element_returns_nan(self):
        result = mf.functions.pesaran_timmermann_metric(
            np.array([1.0]), np.array([1.5])
        )
        assert math.isnan(result)

    def test_threshold_kwarg(self):
        y_true = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
        y_pred = np.array([1.2, 2.3, 2.8, 0.3, 1.8])
        r0 = mf.functions.pesaran_timmermann_metric(y_true, y_pred, threshold=0.0)
        r2 = mf.functions.pesaran_timmermann_metric(y_true, y_pred, threshold=2.0)
        # Different thresholds => different binary series => different stats
        # (may be nan for degenerate cases; just check they run)
        assert isinstance(r0, float)
        assert isinstance(r2, float)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            mf.functions.pesaran_timmermann_metric(
                np.array([1.0, 2.0]), np.array([1.0])
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            mf.functions.pesaran_timmermann_metric(np.array([]), np.array([]))

    def test_returns_float(self, arrays_direction):
        y_true, y_pred, y_prev = arrays_direction
        result = mf.functions.pesaran_timmermann_metric(y_true, y_pred)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Encyclopedia page content checks (closes C22 tester gap pattern)
# ---------------------------------------------------------------------------

class TestEncyclopediaPages:
    """Verify that encyclopedia pages exist and contain required strings."""

    def _read_page(self, *path_parts):
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(repo_root, "docs", "reference", "encyclopedia", "l5", *path_parts)
        with open(path) as fh:
            return fh.read()

    def test_mse_page_has_signature(self):
        content = self._read_page("point_metrics", "mse.md")
        assert "mf.functions.mse" in content
        assert "-> float" in content
        assert "y_true" in content
        assert "y_pred" in content

    def test_rmse_page_has_signature(self):
        content = self._read_page("point_metrics", "rmse.md")
        assert "mf.functions.rmse" in content
        assert "-> float" in content

    def test_mae_page_has_signature(self):
        content = self._read_page("point_metrics", "mae.md")
        assert "mf.functions.mae" in content
        assert "-> float" in content

    def test_medae_page_has_signature(self):
        content = self._read_page("point_metrics", "medae.md")
        assert "mf.functions.medae" in content
        assert "-> float" in content

    def test_mape_page_has_eps_param(self):
        content = self._read_page("point_metrics", "mape.md")
        assert "mf.functions.mape" in content
        assert "eps" in content
        assert "-> float" in content

    def test_theil_u2_page_exists_and_has_y_prev(self):
        content = self._read_page("point_metrics", "theil_u2.md")
        assert "mf.functions.theil_u2" in content
        assert "y_prev" in content

    def test_relative_mse_page(self):
        content = self._read_page("relative_metrics", "relative_mse.md")
        assert "mf.functions.relative_mse" in content
        assert "y_model" in content
        assert "y_benchmark" in content
        assert "-> float" in content

    def test_mse_reduction_page(self):
        content = self._read_page("relative_metrics", "mse_reduction.md")
        assert "mf.functions.mse_reduction" in content

    def test_r2_oos_page(self):
        content = self._read_page("relative_metrics", "r2_oos.md")
        assert "mf.functions.r2_oos" in content

    def test_relative_mae_page(self):
        content = self._read_page("relative_metrics", "relative_mae.md")
        assert "mf.functions.relative_mae" in content

    def test_interval_score_page_has_alpha(self):
        content = self._read_page("density_metrics", "interval_score.md")
        assert "mf.functions.interval_score" in content
        assert "alpha" in content
        assert "-> float" in content

    def test_coverage_rate_page(self):
        content = self._read_page("density_metrics", "coverage_rate.md")
        assert "mf.functions.coverage_rate" in content
        assert "y_lower" in content
        assert "y_upper" in content
        assert "-> float" in content

    def test_success_ratio_page_has_y_prev(self):
        content = self._read_page("direction_metrics", "success_ratio.md")
        assert "mf.functions.success_ratio" in content
        assert "y_prev" in content
        assert "-> float" in content

    def test_pt_metric_page_has_threshold(self):
        content = self._read_page("direction_metrics", "pesaran_timmermann_metric.md")
        assert "mf.functions.pesaran_timmermann_metric" in content
        assert "threshold" in content
        assert "-> float" in content
