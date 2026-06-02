from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from macroforecast.metrics import MetricLike, mae, mse, rmse


def anatomy_output_transform(metric: MetricLike = "forecast") -> Callable[..., Any]:
    """Return an output transformer for the optional anatomy backend.

    This is intentionally an interpretation-side adapter, not a public metrics
    function. Standard forecast metrics score a completed forecast table.
    Anatomy uses a backend transformer to explain either raw forecasts or a
    selected loss during Shapley precomputation.
    """

    if callable(metric):
        return _named_callable(metric)
    key = _canonical_anatomy_output(metric)
    if key == "forecast":
        return lambda y_hat: np.asarray(y_hat, dtype=float)
    if key == "squared_error":
        return lambda y_hat, y: (
            (np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)) ** 2
        )
    if key == "absolute_error":
        return lambda y_hat, y: np.abs(
            np.asarray(y, dtype=float) - np.asarray(y_hat, dtype=float)
        )
    if key == "mse":
        return lambda y_hat, y: mse(y, y_hat)
    if key == "rmse":
        return lambda y_hat, y: rmse(y, y_hat)
    if key == "mae":
        return lambda y_hat, y: mae(y, y_hat)
    allowed = "forecast, squared_error, absolute_error, mse, rmse, mae"
    raise ValueError(
        f"anatomy output {metric!r} is not supported. Available outputs: {allowed}."
    )


def _canonical_anatomy_output(metric: str) -> str:
    key = str(metric).lower().replace("-", "_")
    aliases = {
        "raw": "forecast",
        "prediction": "forecast",
        "pred": "forecast",
        "se": "squared_error",
        "point_squared_error": "squared_error",
        "local_squared_error": "squared_error",
        "ae": "absolute_error",
        "point_absolute_error": "absolute_error",
        "local_absolute_error": "absolute_error",
        "msfe": "mse",
        "mean_squared_error": "mse",
        "root_mean_squared_error": "rmse",
        "mean_absolute_error": "mae",
    }
    return aliases.get(key, key)


def _named_callable(metric: Callable[..., Any]) -> Callable[..., Any]:
    def _transform(y_hat: Any, y: Any) -> Any:
        return metric(y, y_hat)

    _transform.__name__ = getattr(metric, "__name__", "custom_anatomy_output")
    return _transform


__all__ = ["anatomy_output_transform"]
