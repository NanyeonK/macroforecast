from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def combine_mean(forecasts: Any) -> pd.Series:
    """Equal-weight average forecast."""

    frame = _forecast_frame(forecasts)
    return frame.mean(axis=1).rename("combined")


def combine_median(forecasts: Any) -> pd.Series:
    """Cross-model median forecast."""

    frame = _forecast_frame(forecasts)
    return frame.median(axis=1).rename("combined")


def combine_trimmed_mean(forecasts: Any, *, trim: float = 0.1) -> pd.Series:
    """Trim extreme model forecasts before averaging."""

    frame = _forecast_frame(forecasts)
    if not 0 <= trim < 0.5:
        raise ValueError("trim must satisfy 0 <= trim < 0.5")
    values = np.sort(frame.to_numpy(dtype=float), axis=1)
    n_models = values.shape[1]
    cut = int(np.floor(trim * n_models))
    if cut:
        values = values[:, cut:-cut]
    return pd.Series(np.nanmean(values, axis=1), index=frame.index, name="combined")


def combine_winsorized_mean(forecasts: Any, *, limits: tuple[float, float] = (0.1, 0.1)) -> pd.Series:
    """Winsorize cross-model forecasts before averaging."""

    frame = _forecast_frame(forecasts)
    lower, upper = limits
    if lower < 0 or upper < 0 or lower + upper >= 1:
        raise ValueError("limits must be non-negative and sum to less than 1")
    q_low = frame.quantile(lower, axis=1)
    q_high = frame.quantile(1 - upper, axis=1)
    clipped = frame.clip(lower=q_low, upper=q_high, axis=0)
    return clipped.mean(axis=1).rename("combined")


def combine_inverse_mspe(
    forecasts: Any,
    y_true: Any,
    *,
    discount: float = 1.0,
    min_weight: float = 1e-12,
) -> pd.Series:
    """Combine forecasts with inverse discounted MSPE weights."""

    frame = _forecast_frame(forecasts)
    target = pd.Series(y_true).reindex(frame.index).astype(float)
    if not 0 < discount <= 1:
        raise ValueError("discount must satisfy 0 < discount <= 1")
    errors = frame.sub(target, axis=0) ** 2
    weights = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    running = pd.Series(0.0, index=frame.columns, dtype=float)
    for step, date in enumerate(frame.index):
        if step == 0 or float(running.sum()) <= 0:
            weights.loc[date, :] = 1.0 / len(frame.columns)
        else:
            inv = 1.0 / running.clip(lower=min_weight)
            weights.loc[date, :] = inv / inv.sum()
        current = errors.loc[date]
        running = discount * running + current.fillna(running.mean() if running.notna().any() else 0.0)
    combined = (frame * weights).sum(axis=1)
    return combined.rename("combined")


combine_dmspe = combine_inverse_mspe


def combine_best_n(forecasts: Any, y_true: Any, *, n: int = 3) -> pd.Series:
    """Average the historically best ``n`` models by MSPE."""

    frame = _forecast_frame(forecasts)
    target = pd.Series(y_true).reindex(frame.index).astype(float)
    n_value = int(n)
    if n_value < 1:
        raise ValueError("n must be at least 1")
    mspe = frame.sub(target, axis=0).pow(2).expanding(min_periods=1).mean()
    output = pd.Series(index=frame.index, dtype=float, name="combined")
    for date in frame.index:
        best = mspe.loc[date].sort_values().index[:n_value]
        output.loc[date] = frame.loc[date, best].mean()
    return output


def _forecast_frame(forecasts: Any) -> pd.DataFrame:
    if isinstance(forecasts, pd.Series):
        frame = forecasts.to_frame()
    else:
        frame = pd.DataFrame(forecasts).copy()
    if frame.empty:
        raise ValueError("forecasts must not be empty")
    return frame.astype(float)


__all__ = [
    "combine_best_n",
    "combine_dmspe",
    "combine_inverse_mspe",
    "combine_mean",
    "combine_median",
    "combine_trimmed_mean",
    "combine_winsorized_mean",
]
