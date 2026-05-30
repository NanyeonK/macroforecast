from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

MetricLike = str | Callable[[Any, Any], float]


def _aligned_values(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    truth = pd.Series(y_true).astype(float)
    pred = pd.Series(y_pred).astype(float)
    joined = pd.concat([truth.rename("truth"), pred.rename("pred")], axis=1).dropna()
    if joined.empty:
        raise ValueError("metric inputs have no aligned non-missing observations")
    return joined["truth"].to_numpy(dtype=float), joined["pred"].to_numpy(dtype=float)


def mse(y_true: Any, y_pred: Any) -> float:
    """Mean squared error."""

    truth, pred = _aligned_values(y_true, y_pred)
    return float(np.mean((truth - pred) ** 2))


def rmse(y_true: Any, y_pred: Any) -> float:
    """Root mean squared error."""

    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error."""

    truth, pred = _aligned_values(y_true, y_pred)
    return float(np.mean(np.abs(truth - pred)))


def pinball_loss(y_true: Any, y_quantile: Any, *, quantile: float) -> float:
    """Mean quantile pinball loss."""

    truth, pred = _aligned_values(y_true, y_quantile)
    q = float(quantile)
    if not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
    error = truth - pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def gaussian_nll(y_true: Any, y_pred: Any, variance: Any) -> float:
    """Gaussian negative log likelihood using supplied predictive variances."""

    truth = pd.Series(y_true).astype(float)
    pred = pd.Series(y_pred).astype(float)
    var = pd.Series(variance).astype(float)
    joined = pd.concat(
        [truth.rename("truth"), pred.rename("pred"), var.rename("variance")],
        axis=1,
    ).dropna()
    if joined.empty:
        raise ValueError("metric inputs have no aligned non-missing observations")
    values = joined["variance"].to_numpy(dtype=float)
    values = np.maximum(values, 1e-12)
    errors = joined["truth"].to_numpy(dtype=float) - joined["pred"].to_numpy(dtype=float)
    return float(np.mean(0.5 * (np.log(2.0 * np.pi * values) + errors**2 / values)))


def coverage_rate(y_true: Any, lower: Any, upper: Any) -> float:
    """Share of observations covered by lower/upper forecasts."""

    truth = pd.Series(y_true).astype(float)
    lo = pd.Series(lower).astype(float)
    hi = pd.Series(upper).astype(float)
    joined = pd.concat(
        [truth.rename("truth"), lo.rename("lower"), hi.rename("upper")], axis=1
    ).dropna()
    if joined.empty:
        raise ValueError("metric inputs have no aligned non-missing observations")
    covered = (joined["truth"] >= joined["lower"]) & (joined["truth"] <= joined["upper"])
    return float(covered.mean())


def interval_width(lower: Any, upper: Any) -> float:
    """Mean forecast interval width."""

    lo = pd.Series(lower).astype(float)
    hi = pd.Series(upper).astype(float)
    joined = pd.concat([lo.rename("lower"), hi.rename("upper")], axis=1).dropna()
    if joined.empty:
        raise ValueError("metric inputs have no aligned non-missing observations")
    return float((joined["upper"] - joined["lower"]).mean())


def evaluate_forecasts(
    forecasts: Any,
    *,
    by: Sequence[str] = ("model", "horizon"),
    metrics: Sequence[str | MetricLike] = ("mse", "rmse", "mae"),
    actual: str = "actual",
    prediction: str = "prediction",
    variance_prediction: str = "variance_prediction",
    quantile_predictions: str = "quantile_predictions",
) -> pd.DataFrame:
    """Evaluate a forecasting runner output or forecast table."""

    frame = _forecast_frame(forecasts)
    if frame.empty:
        return pd.DataFrame()
    group_keys = [key for key in by if key in frame.columns]
    if group_keys:
        iterator = frame.groupby(group_keys, dropna=False, sort=True)
    else:
        iterator = [((), frame)]
    rows: list[dict[str, Any]] = []
    for key, group in iterator:
        row = _group_row(key, group_keys)
        valid = group[[actual, prediction]].dropna()
        row["n"] = int(len(valid))
        if len(valid) > 0:
            for metric in metrics:
                metric_fn = get_metric(metric)
                metric_name = getattr(metric_fn, "__name__", str(metric))
                row[metric_name] = float(metric_fn(valid[actual], valid[prediction]))
        if variance_prediction in group.columns:
            variance_valid = group[[actual, prediction, variance_prediction]].dropna()
            if len(variance_valid) > 0:
                row["variance_n"] = int(len(variance_valid))
                row["gaussian_nll"] = gaussian_nll(
                    variance_valid[actual],
                    variance_valid[prediction],
                    variance_valid[variance_prediction],
                )
        if quantile_predictions in group.columns:
            row.update(
                _quantile_evaluation(
                    group,
                    actual=actual,
                    quantile_predictions=quantile_predictions,
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


_METRICS: dict[str, Callable[[Any, Any], float]] = {
    "mse": mse,
    "validation_mse": mse,
    "rmse": rmse,
    "validation_rmse": rmse,
    "mae": mae,
    "validation_mae": mae,
}


def get_metric(metric: MetricLike) -> Callable[[Any, Any], float]:
    """Return a metric callable by name or pass through a callable metric."""

    if callable(metric):
        return metric
    key = metric.lower()
    if key not in _METRICS:
        allowed = ", ".join(sorted(_METRICS))
        raise ValueError(f"Unknown metric {metric!r}. Available metrics: {allowed}.")
    return _METRICS[key]


def _forecast_frame(forecasts: Any) -> pd.DataFrame:
    if hasattr(forecasts, "to_frame"):
        return forecasts.to_frame()
    return pd.DataFrame(forecasts).copy()


def _group_row(key: Any, group_keys: Sequence[str]) -> dict[str, Any]:
    if not group_keys:
        return {}
    if len(group_keys) == 1:
        values = (key[0],) if isinstance(key, tuple) else (key,)
    else:
        values = tuple(key)
    return {name: value for name, value in zip(group_keys, values)}


def _quantile_evaluation(
    group: pd.DataFrame, *, actual: str, quantile_predictions: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row_id, (idx, value) in enumerate(group[quantile_predictions].dropna().items()):
        if not isinstance(value, dict) or idx not in group.index:
            continue
        observed = group.at[idx, actual]
        if pd.isna(observed):
            continue
        for level, quantile_value in value.items():
            try:
                q = float(level)
                pred = float(quantile_value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "row_id": row_id,
                    "actual": float(observed),
                    "level": q,
                    "prediction": pred,
                }
            )
    if not rows:
        return {}
    table = pd.DataFrame(rows)
    out: dict[str, Any] = {"quantile_n": int(len(table))}
    for level, level_group in table.groupby("level", sort=True):
        label = _level_label(float(level))
        out[f"pinball_loss_{label}"] = pinball_loss(
            level_group["actual"], level_group["prediction"], quantile=float(level)
        )
    out.update(_interval_metrics(table))
    return out


def _interval_metrics(table: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pivot = table.pivot_table(
        index="row_id", columns="level", values=["actual", "prediction"], aggfunc="first"
    )
    levels = sorted(float(level) for level in table["level"].unique())
    for lower in levels:
        upper = 1.0 - lower
        if lower >= upper or upper not in levels:
            continue
        try:
            actual_values = pivot[("actual", lower)]
            lower_values = pivot[("prediction", lower)]
            upper_values = pivot[("prediction", upper)]
        except KeyError:
            continue
        label = f"{_level_label(lower)}_{_level_label(upper)}"
        out[f"coverage_{label}"] = coverage_rate(actual_values, lower_values, upper_values)
        out[f"interval_width_{label}"] = interval_width(lower_values, upper_values)
    return out


def _level_label(level: float) -> str:
    return f"q{level:g}".replace(".", "_")


__all__ = [
    "MetricLike",
    "coverage_rate",
    "evaluate_forecasts",
    "gaussian_nll",
    "get_metric",
    "interval_width",
    "mae",
    "mse",
    "pinball_loss",
    "rmse",
]
