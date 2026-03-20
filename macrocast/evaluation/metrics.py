"""Core forecast evaluation metrics.

All functions operate on aligned arrays of forecasts and realisations.
The benchmark AR model is used to compute relative metrics.

Metrics:
  - MSFE (Mean Squared Forecast Error)
  - Relative MSFE = model MSFE / benchmark MSFE
  - MAE (Mean Absolute Error)
  - CSFE (Cumulative Squared Forecast Error) time series
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def msfe(y_true: NDArray[np.floating], y_hat: NDArray[np.floating]) -> float:
    """Mean Squared Forecast Error.

    Parameters
    ----------
    y_true : array of shape (T,)
    y_hat  : array of shape (T,)

    Returns
    -------
    float
    """
    return float(np.mean((y_true - y_hat) ** 2))


def mae(y_true: NDArray[np.floating], y_hat: NDArray[np.floating]) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_hat)))


def relative_msfe(
    y_true: NDArray[np.floating],
    y_hat_model: NDArray[np.floating],
    y_hat_benchmark: NDArray[np.floating],
) -> float:
    """Relative MSFE = model MSFE / benchmark MSFE.

    Values below 1 indicate improvement over the benchmark (AR by convention).
    """
    msfe_model = msfe(y_true, y_hat_model)
    msfe_benchmark = msfe(y_true, y_hat_benchmark)
    if msfe_benchmark == 0:
        return float("nan")
    return msfe_model / msfe_benchmark


def csfe(
    y_true: NDArray[np.floating],
    y_hat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Cumulative Squared Forecast Error over time.

    Returns
    -------
    array of shape (T,) — CSFE at each evaluation date.
    """
    return np.cumsum((y_true - y_hat) ** 2)


def oos_r2(
    y_true: NDArray[np.floating],
    y_hat_model: NDArray[np.floating],
    y_hat_benchmark: NDArray[np.floating],
) -> float:
    """Out-of-Sample R² (Campbell and Thompson 2008).

    OOS-R² = 1 - MSFE_model / MSFE_benchmark.
    Positive values indicate model beats benchmark.
    """
    return 1.0 - relative_msfe(y_true, y_hat_model, y_hat_benchmark)
