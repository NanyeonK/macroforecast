"""Phase 4 - relative metrics module behavioural tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrocast.execution.evaluation.metrics import (
    compute_metrics_dict,
    compute_relative_metrics,
    compute_relative_metrics_suite,
)


def _series(values, start="2000-01"):
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float)


def test_perfect_model_yields_zero_relative_rmse_and_unit_oos_r2():
    y = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    model_pred = y.copy()
    bench_pred = _series([0.5] * 8)
    out = compute_relative_metrics(model_predictions=model_pred, benchmark_predictions=bench_pred, actuals=y)
    assert abs(out["relative_rmse"]) < 1e-9
    assert abs(out["oos_r2"] - 1.0) < 1e-9


def test_worse_than_benchmark_yields_relative_rmse_above_one_and_negative_oos_r2():
    y = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    model_pred = _series([0.0] * 8)  # large constant error
    bench_pred = y * 0.95            # very close to truth
    out = compute_relative_metrics(model_predictions=model_pred, benchmark_predictions=bench_pred, actuals=y)
    assert out["relative_rmse"] > 1.0
    assert out["oos_r2"] < 0.0


def test_compute_relative_metrics_suite_returns_dict_of_dicts():
    y = _series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    model_pred = y * 0.99
    rows = []
    for t, val in zip(y.index, [0.5] * 6):
        rec = dict(date=t, benchmark_name="bench_low", benchmark_pred=val)
        rows.append(rec)
    for t, val in zip(y.index, [10.0] * 6):
        rec = dict(date=t, benchmark_name="bench_high", benchmark_pred=val)
        rows.append(rec)
    bench_df = pd.DataFrame(rows)
    out = compute_relative_metrics_suite(model_predictions=model_pred, benchmark_forecasts=bench_df, actuals=y)
    assert set(out.keys()) == set(("relative_msfe", "relative_rmse", "relative_mae", "oos_r2"))
    for key, sub in out.items():
        assert set(sub.keys()) == set(("bench_low", "bench_high"))


def test_zero_benchmark_loss_returns_safe_default():
    y = _series([1.0, 2.0, 3.0, 4.0])
    model_pred = _series([0.5, 1.5, 2.5, 3.5])
    bench_pred = y.copy()  # zero loss benchmark
    out = compute_relative_metrics(model_predictions=model_pred, benchmark_predictions=bench_pred, actuals=y)
    # By the v0.6 convention, zero-loss benchmark yields relative_* = 1.0.
    assert out["relative_rmse"] == 1.0
    assert out["relative_mae"] == 1.0


def test_compute_metrics_dict_contains_relative_keys():
    y = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    model_pred = y * 1.01
    bench_pred = y * 0.95
    out = compute_metrics_dict(horizon=1, benchmark_name="hist", model_predictions=model_pred, benchmark_predictions=bench_pred, actuals=y)
    for key in ("msfe", "rmse", "mae", "relative_msfe", "relative_rmse", "relative_mae", "oos_r2", "benchmark_name", "horizon"):
        assert key in out
    assert out["benchmark_name"] == "hist"
    assert out["horizon"] == 1
