from __future__ import annotations

from collections.abc import Callable
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


__all__ = ["MetricLike", "get_metric", "mae", "mse", "rmse"]
