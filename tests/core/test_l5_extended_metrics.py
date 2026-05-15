"""F-P1-8 -- L5 extended metrics: medae / theil_u1 / theil_u2 / success_ratio.

Hand-computed reference values verify correctness of the formula implementations
in _add_l5_extended_metrics().
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _add_l5_extended_metrics


def _make_errors(y_true, y_pred, y_prev=None, model_id="m1", target="y", horizon=1):
    n = len(y_true)
    rows = {
        "model_id": [model_id] * n,
        "target": [target] * n,
        "horizon": [horizon] * n,
        "y_true": list(y_true),
        "y_pred": list(y_pred),
    }
    if y_prev is not None:
        rows["y_prev"] = list(y_prev)
    return pd.DataFrame(rows)


def _make_metrics(model_id="m1", target="y", horizon=1):
    return pd.DataFrame([{"model_id": model_id, "target": target, "horizon": horizon,
                           "mse": 0.0, "mae": 0.0, "rmse": 0.0}])


class TestMedae:
    def test_medae_odd_count(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.5, 2.0, 2.5, 4.0, 4.5])
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        # absolute errors: 0.5, 0.0, 0.5, 0.0, 0.5 -> sorted [0,0,0.5,0.5,0.5] -> median = 0.5
        assert result["medae"].iloc[0] == pytest.approx(0.5)

    def test_medae_even_count(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 2.0, 3.0])
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        # absolute errors: 0, 0, 1, 1 -> median = 0.5
        assert result["medae"].iloc[0] == pytest.approx(0.5)

    def test_medae_zero(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert result["medae"].iloc[0] == pytest.approx(0.0)


class TestTheilU1:
    def test_theil_u1_perfect_forecast(self):
        y_true = np.array([2.0, 4.0, 6.0])
        y_pred = np.array([2.0, 4.0, 6.0])  # perfect
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        # RMSE = 0, denom > 0, so theil_u1 = 0
        assert result["theil_u1"].iloc[0] == pytest.approx(0.0)

    def test_theil_u1_hand_computed(self):
        # y_true = [1, 2], y_pred = [2, 1]
        y_true = np.array([1.0, 2.0])
        y_pred = np.array([2.0, 1.0])
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        # errors = [1, 1] -> RMSE = 1.0
        # sqrt(mean(y_true^2)) = sqrt(2.5) = 1.5811...
        # sqrt(mean(y_pred^2)) = sqrt(2.5) = 1.5811...
        # denom = 3.1623...
        # theil_u1 = 1.0 / 3.1623 = 0.31623
        expected = 1.0 / (math.sqrt(2.5) + math.sqrt(2.5))
        assert result["theil_u1"].iloc[0] == pytest.approx(expected, rel=1e-6)

    def test_theil_u1_bounded_below_1_for_reasonable_data(self):
        rng = np.random.default_rng(42)
        y_true = rng.normal(loc=5, scale=1, size=30)
        y_pred = y_true + rng.normal(scale=0.5, size=30)
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        val = result["theil_u1"].iloc[0]
        assert 0 <= val <= 1.0, f"theil_u1={val} out of [0,1]"


class TestTheilU2:
    def test_theil_u2_nan_when_no_y_prev(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        # No y_prev column
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert math.isnan(result["theil_u2"].iloc[0])

    def test_theil_u2_nan_when_y_prev_all_nan(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        y_prev = [float("nan"), float("nan"), float("nan")]
        errors = _make_errors(y_true, y_pred, y_prev)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert math.isnan(result["theil_u2"].iloc[0])

    def test_theil_u2_perfect_naive_gives_1(self):
        # When forecast == naive (y_pred = y_prev), theil_u2 = 1
        y_prev = np.array([1.0, 2.0, 3.0, 4.0])
        y_true = np.array([2.0, 3.0, 4.0, 5.0])
        y_pred = y_prev.copy()  # naive forecast
        errors = _make_errors(y_true, y_pred, y_prev)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert result["theil_u2"].iloc[0] == pytest.approx(1.0, rel=1e-6)

    def test_theil_u2_perfect_forecast_gives_0(self):
        y_prev = np.array([1.0, 2.0, 3.0, 4.0])
        y_true = np.array([2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()  # perfect forecast
        errors = _make_errors(y_true, y_pred, y_prev)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert result["theil_u2"].iloc[0] == pytest.approx(0.0, abs=1e-9)


class TestSuccessRatio:
    def test_success_ratio_perfect_directional(self):
        y_prev = np.array([1.0, 2.0, 3.0])
        y_true = np.array([2.0, 3.0, 4.0])  # all up
        y_pred = np.array([1.5, 2.5, 3.5])  # all predicted up
        errors = _make_errors(y_true, y_pred, y_prev)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert result["success_ratio"].iloc[0] == pytest.approx(1.0)

    def test_success_ratio_zero_on_all_wrong(self):
        y_prev = np.array([2.0, 3.0, 4.0])
        y_true = np.array([3.0, 4.0, 5.0])  # all up
        y_pred = np.array([1.0, 2.0, 3.0])  # all predicted down
        errors = _make_errors(y_true, y_pred, y_prev)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert result["success_ratio"].iloc[0] == pytest.approx(0.0)

    def test_success_ratio_half(self):
        y_prev = np.array([1.0, 1.0, 1.0, 1.0])
        y_true = np.array([2.0, 2.0, 0.0, 0.0])  # 2 up, 2 down
        y_pred = np.array([2.0, 0.0, 2.0, 0.0])  # 1 correct up, 0 correct down
        errors = _make_errors(y_true, y_pred, y_prev)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert result["success_ratio"].iloc[0] == pytest.approx(0.5)

    def test_success_ratio_nan_when_no_y_prev(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        errors = _make_errors(y_true, y_pred)
        metrics = _make_metrics()
        result = _add_l5_extended_metrics(metrics, errors)
        assert math.isnan(result["success_ratio"].iloc[0])


class TestMultipleModels:
    def test_metrics_computed_per_model(self):
        """Extended metrics are computed separately per (model_id, target, horizon)."""
        errors = pd.DataFrame([
            {"model_id": "m1", "target": "y", "horizon": 1, "y_true": 1.0, "y_pred": 2.0},
            {"model_id": "m1", "target": "y", "horizon": 1, "y_true": 2.0, "y_pred": 2.0},
            {"model_id": "m2", "target": "y", "horizon": 1, "y_true": 1.0, "y_pred": 1.0},
            {"model_id": "m2", "target": "y", "horizon": 1, "y_true": 2.0, "y_pred": 2.0},
        ])
        metrics = pd.DataFrame([
            {"model_id": "m1", "target": "y", "horizon": 1, "mse": 0.5, "mae": 0.5, "rmse": 0.0},
            {"model_id": "m2", "target": "y", "horizon": 1, "mse": 0.0, "mae": 0.0, "rmse": 0.0},
        ])
        result = _add_l5_extended_metrics(metrics, errors)
        m1 = result[result["model_id"] == "m1"].iloc[0]
        m2 = result[result["model_id"] == "m2"].iloc[0]
        # m1: errors are 1,0 -> medae = 0.5
        assert m1["medae"] == pytest.approx(0.5)
        # m2: errors are 0,0 -> medae = 0.0
        assert m2["medae"] == pytest.approx(0.0)
