"""Relative-metrics utilities extracted from execution/build.py.

Provides compute_relative_metrics for a single (model, benchmark) pair plus
compute_relative_metrics_suite for benchmark_suite consumers, and
compute_metrics_dict as a thin wrapper matching the pre-existing dict shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_ratio(numerator: float, denominator: float, zero_value: float = 1.0) -> float:
    """Return numerator/denominator, falling back to zero_value when denominator is 0.

    Convention chosen for v0.6: when the benchmark loss is exactly 0 (e.g. perfect
    benchmark or zero-variance series), relative_* defaults to 1.0, mirroring the
    existing _compute_metrics behaviour in execution/build.py.
    """
    if denominator > 0:
        return float(numerator) / float(denominator)
    return float(zero_value)


def compute_relative_metrics(
    *,
    model_predictions: pd.Series,
    benchmark_predictions: pd.Series,
    actuals: pd.Series,
):
    """Compute relative_msfe, relative_rmse, relative_mae, oos_r2 for a single benchmark."""
    aligned = pd.concat(
        [model_predictions.rename("model"), benchmark_predictions.rename("bench"), actuals.rename("y")],
        axis=1,
    ).dropna()
    if aligned.empty:
        out = dict(relative_msfe=float("nan"), relative_rmse=float("nan"), relative_mae=float("nan"), oos_r2=float("nan"))
        return out
    err_m = aligned["y"] - aligned["model"]
    err_b = aligned["y"] - aligned["bench"]
    msfe_m = float((err_m ** 2).mean())
    msfe_b = float((err_b ** 2).mean())
    mae_m = float(err_m.abs().mean())
    mae_b = float(err_b.abs().mean())
    rmse_m = float(np.sqrt(msfe_m))
    rmse_b = float(np.sqrt(msfe_b))
    sse_m = float((err_m ** 2).sum())
    sse_b = float((err_b ** 2).sum())
    rel_msfe = _safe_ratio(msfe_m, msfe_b)
    rel_rmse = _safe_ratio(rmse_m, rmse_b)
    rel_mae = _safe_ratio(mae_m, mae_b)
    if sse_b > 0:
        oos_r2 = 1.0 - sse_m / sse_b
    else:
        oos_r2 = 0.0
    return dict(relative_msfe=rel_msfe, relative_rmse=rel_rmse, relative_mae=rel_mae, oos_r2=oos_r2)


def compute_relative_metrics_suite(
    *,
    model_predictions: pd.Series,
    benchmark_forecasts: pd.DataFrame,
    actuals: pd.Series,
):
    """Per-benchmark dicts when benchmark_suite is active.

    Returns a dict with keys relative_msfe, relative_rmse, relative_mae, oos_r2
    each mapping benchmark_name -> scalar value.
    """
    if "benchmark_name" not in benchmark_forecasts.columns or "benchmark_pred" not in benchmark_forecasts.columns:
        raise ValueError("benchmark_forecasts must have benchmark_name and benchmark_pred columns")
    out = dict(relative_msfe=dict(), relative_rmse=dict(), relative_mae=dict(), oos_r2=dict())
    for name, group in benchmark_forecasts.groupby("benchmark_name"):
        bench_series = group.set_index("date")["benchmark_pred"] if "date" in group.columns else group["benchmark_pred"]
        # Align on actuals index when possible
        if isinstance(bench_series.index, pd.DatetimeIndex):
            bench_aligned = bench_series.reindex(actuals.index, method="nearest")
        else:
            bench_aligned = pd.Series(bench_series.values, index=actuals.index[: len(bench_series)])
        per = compute_relative_metrics(
            model_predictions=model_predictions,
            benchmark_predictions=bench_aligned,
            actuals=actuals,
        )
        for key in ("relative_msfe", "relative_rmse", "relative_mae", "oos_r2"):
            out[key][str(name)] = per[key]
    return out


def compute_metrics_dict(
    *,
    horizon: int,
    benchmark_name: str,
    model_predictions: pd.Series,
    benchmark_predictions: pd.Series,
    actuals: pd.Series,
):
    """Backwards-compatible wrapper matching the dict shape produced by the existing _compute_metrics."""
    rel = compute_relative_metrics(
        model_predictions=model_predictions,
        benchmark_predictions=benchmark_predictions,
        actuals=actuals,
    )
    aligned = pd.concat(
        [model_predictions.rename("model"), benchmark_predictions.rename("bench"), actuals.rename("y")],
        axis=1,
    ).dropna()
    err_m = aligned["y"] - aligned["model"]
    err_b = aligned["y"] - aligned["bench"]
    out = dict()
    out["horizon"] = int(horizon)
    out["benchmark_name"] = str(benchmark_name)
    out["n_predictions"] = int(len(aligned))
    out["msfe"] = float((err_m ** 2).mean()) if not aligned.empty else float("nan")
    out["benchmark_msfe"] = float((err_b ** 2).mean()) if not aligned.empty else float("nan")
    out["mae"] = float(err_m.abs().mean()) if not aligned.empty else float("nan")
    out["benchmark_mae"] = float(err_b.abs().mean()) if not aligned.empty else float("nan")
    out["rmse"] = float(np.sqrt(out["msfe"])) if not aligned.empty else float("nan")
    out["benchmark_rmse"] = float(np.sqrt(out["benchmark_msfe"])) if not aligned.empty else float("nan")
    out["relative_msfe"] = rel["relative_msfe"]
    out["relative_rmse"] = rel["relative_rmse"]
    out["relative_mae"] = rel["relative_mae"]
    out["oos_r2"] = rel["oos_r2"]
    return out
