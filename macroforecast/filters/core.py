from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.tsa.filters.hp_filter import hpfilter


@dataclass(frozen=True)
class FilterResult:
    """Result returned by direct one-series filter callables."""

    values: pd.DataFrame
    method: str
    params: dict[str, Any]
    metadata: dict[str, Any]
    source: str | None = None

    def component(self, name: str) -> pd.Series:
        """Return one named component from ``values``."""

        return self.values[str(name)]


def hp_filter(
    y: Any,
    *,
    dates: Any | None = None,
    lamb: float = 129600.0,
    component: str = "both",
    interpolate_missing: bool = True,
    name: str | None = None,
) -> FilterResult:
    """Apply the two-sided Hodrick-Prescott filter to one series."""

    series = _coerce_series(y, dates=dates, name=name)
    component_value = _normalize_component(component)
    filtered = _interpolate_if_requested(series, enabled=interpolate_missing)
    cycle, trend = hpfilter(filtered, lamb=float(lamb))
    values = _select_components(
        {"cycle": cycle.rename("cycle"), "trend": trend.rename("trend")},
        component=component_value,
        index=series.index,
    )
    params = {
        "lambda": float(lamb),
        "component": component_value,
        "interpolate_missing": bool(interpolate_missing),
    }
    metadata = {
        "kind": "filter",
        "method": "hp_filter",
        "source_series": str(series.name),
        "fit_policy": "full_input_two_sided",
        "backend": "statsmodels.tsa.filters.hp_filter.hpfilter",
        "params": params,
    }
    values.attrs["macroforecast_filter_metadata"] = metadata
    return FilterResult(
        values=values,
        method="hp_filter",
        params=params,
        metadata=metadata,
        source=str(series.name),
    )


def hamilton_filter(
    y: Any,
    *,
    dates: Any | None = None,
    h: int = 8,
    p: int = 4,
    component: str = "both",
    fit_policy: str = "expanding",
    min_train_size: int | None = None,
    missing: str = "drop",
    name: str | None = None,
) -> FilterResult:
    """Apply Hamilton's trend-cycle regression filter to one series."""

    series = _coerce_series(y, dates=dates, name=name)
    h_value = int(h)
    p_value = int(p)
    if h_value <= 0:
        raise ValueError("h must be positive")
    if p_value <= 0:
        raise ValueError("p must be positive")
    component_value = _normalize_component(component)
    fit_value = _normalize_fit_policy(fit_policy)
    min_size = _normalize_min_train_size(min_train_size, minimum=p_value + 1)
    missing_value = str(missing).lower()
    if missing_value not in {"drop", "interpolate"}:
        raise ValueError("missing must be 'drop' or 'interpolate'")
    filtered = _interpolate_if_requested(series, enabled=missing_value == "interpolate")
    cycle, trend = _hamilton_filter_series(
        filtered,
        h=h_value,
        p=p_value,
        fit_policy=fit_value,
        min_train_size=min_size,
    )
    values = _select_components(
        {"cycle": cycle.rename("cycle"), "trend": trend.rename("trend")},
        component=component_value,
        index=series.index,
    )
    params = {
        "h": h_value,
        "p": p_value,
        "component": component_value,
        "fit_policy": fit_value,
        "min_train_size": min_size,
        "missing": missing_value,
    }
    metadata = {
        "kind": "filter",
        "method": "hamilton_filter",
        "source_series": str(series.name),
        "fit_policy": fit_value,
        "label_alignment": "components are labeled at t+h",
        "formula": "y[t+h] on constant, y[t], ..., y[t-p+1]",
        "params": params,
    }
    values.attrs["macroforecast_filter_metadata"] = metadata
    return FilterResult(
        values=values,
        method="hamilton_filter",
        params=params,
        metadata=metadata,
        source=str(series.name),
    )


def savitzky_golay(
    y: Any,
    *,
    dates: Any | None = None,
    window_length: int = 5,
    polyorder: int = 2,
    derivative: int = 0,
    interpolate_missing: bool = True,
    name: str | None = None,
) -> FilterResult:
    """Apply the centered Savitzky-Golay filter to one series."""

    series = _coerce_series(y, dates=dates, name=name)
    window = int(window_length)
    order = int(polyorder)
    deriv = int(derivative)
    if window <= 0 or window % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    if order >= window:
        raise ValueError("polyorder must be smaller than window_length")
    filtered = _interpolate_if_requested(series, enabled=interpolate_missing)
    values = savgol_filter(
        filtered.to_numpy(dtype=float),
        window_length=window,
        polyorder=order,
        deriv=deriv,
    )
    frame = pd.DataFrame({"savgol": values}, index=series.index)
    frame.index.name = series.index.name
    params = {
        "window_length": window,
        "polyorder": order,
        "derivative": deriv,
        "interpolate_missing": bool(interpolate_missing),
    }
    metadata = {
        "kind": "filter",
        "method": "savitzky_golay",
        "source_series": str(series.name),
        "fit_policy": "full_input_centered_window",
        "backend": "scipy.signal.savgol_filter",
        "params": params,
    }
    frame.attrs["macroforecast_filter_metadata"] = metadata
    return FilterResult(
        values=frame,
        method="savitzky_golay",
        params=params,
        metadata=metadata,
        source=str(series.name),
    )


def wavelet_filter(
    y: Any,
    *,
    dates: Any | None = None,
    n_levels: int = 3,
    wavelet: str = "db4",
    name: str | None = None,
) -> FilterResult:
    """Create causal rolling approximation/detail components for one series.

    This is the package's existing wavelet-style filter helper. It records the
    requested wavelet name for provenance, but the current implementation is a
    causal rolling multi-resolution approximation rather than a true DWT.
    """

    series = _coerce_series(y, dates=dates, name=name)
    levels = int(n_levels)
    if levels <= 0:
        raise ValueError("n_levels must be positive")
    pieces: dict[str, pd.Series] = {}
    for level in range(1, levels + 1):
        window = 2**level
        approx = series.rolling(window=window, min_periods=1).mean()
        detail = series - approx
        pieces[f"wA{level}"] = approx.rename(f"wA{level}")
        pieces[f"wD{level}"] = detail.rename(f"wD{level}")
    frame = pd.concat(pieces.values(), axis=1)
    frame.index.name = series.index.name
    params = {"n_levels": levels, "wavelet": str(wavelet)}
    metadata = {
        "kind": "filter",
        "method": "wavelet_filter",
        "source_series": str(series.name),
        "fit_policy": "causal_rolling",
        "params": params,
        "note": "Causal rolling multi-resolution approximation; not a true DWT backend.",
    }
    frame.attrs["macroforecast_filter_metadata"] = metadata
    return FilterResult(
        values=frame,
        method="wavelet_filter",
        params=params,
        metadata=metadata,
        source=str(series.name),
    )


def _infer_seasonal_period(series: pd.Series) -> int | None:
    index = series.index
    freq = getattr(index, "freqstr", None)
    if freq is None and len(index) > 2:
        try:
            freq = pd.infer_freq(index)
        except Exception:
            freq = None
    if not freq:
        return None
    code = str(freq).upper()
    if code.startswith(("M", "BM", "ME", "MS")):
        return 12
    if code.startswith("Q"):
        return 4
    if code.startswith(("W",)):
        return 52
    if code.startswith(("D", "B")):
        return 7
    return None


def stl_decompose(
    y: Any,
    *,
    period: int | None = None,
    seasonal: int = 7,
    trend: int | None = None,
    robust: bool = False,
    dates: Any | None = None,
    name: str | None = None,
) -> FilterResult:
    """Seasonal-Trend decomposition using Loess (STL).

    Decomposes a single series into trend, seasonal and remainder components
    (Cleveland et al. 1990; R ``stats::stl`` / statsmodels ``STL``). ``period`` is
    the seasonal period (inferred from a monthly/quarterly/weekly DatetimeIndex
    when omitted). This is a two-sided full-sample decomposition, so the
    components are not real-time safe as forecasting features without a per-origin
    refit (``fit_policy='full_input_two_sided'``).
    """

    from statsmodels.tsa.seasonal import STL

    series = _coerce_series(y, dates=dates, name=name)
    series = _interpolate_if_requested(series, enabled=True).astype(float)
    resolved_period = int(period) if period is not None else _infer_seasonal_period(series)
    if resolved_period is None or resolved_period < 2:
        raise ValueError(
            "stl_decompose needs a seasonal period >= 2; pass period= explicitly "
            "when the index frequency cannot be inferred"
        )
    if len(series.dropna()) < 2 * resolved_period:
        raise ValueError("stl_decompose needs at least two full seasonal periods")
    result = STL(
        series, period=resolved_period, seasonal=int(seasonal), trend=trend, robust=bool(robust)
    ).fit()
    values = pd.DataFrame(
        {
            "trend": np.asarray(result.trend, dtype=float),
            "seasonal": np.asarray(result.seasonal, dtype=float),
            "resid": np.asarray(result.resid, dtype=float),
        },
        index=series.index,
    )
    params = {
        "period": resolved_period,
        "seasonal": int(seasonal),
        "trend": trend,
        "robust": bool(robust),
    }
    metadata = {
        "kind": "filter",
        "method": "stl_decompose",
        "source_series": str(series.name),
        "fit_policy": "full_input_two_sided",
        "backend": "statsmodels.tsa.seasonal.STL",
        "params": params,
    }
    values.attrs["macroforecast_filter_metadata"] = metadata
    return FilterResult(
        values=values,
        method="stl_decompose",
        params=params,
        metadata=metadata,
        source=str(series.name),
    )


def _coerce_series(y: Any, *, dates: Any | None, name: str | None) -> pd.Series:
    if isinstance(y, pd.Series):
        series = y.astype(float).copy()
        if dates is not None:
            series.index = pd.Index(dates)
    else:
        values = np.asarray(y, dtype=float)
        if values.ndim != 1:
            raise ValueError("y must be one-dimensional")
        index = pd.Index(dates) if dates is not None else pd.RangeIndex(len(values))
        series = pd.Series(values, index=index)
    if len(series.index) != len(series):
        raise ValueError("dates must have the same length as y")
    if name is not None:
        series.name = str(name)
    elif series.name is None:
        series.name = "series"
    return series


def _interpolate_if_requested(series: pd.Series, *, enabled: bool) -> pd.Series:
    if enabled and series.isna().any():
        return series.interpolate(limit_direction="both")
    return series


def _normalize_component(component: str) -> str:
    value = str(component).lower()
    if value not in {"cycle", "trend", "both"}:
        raise ValueError("component must be 'cycle', 'trend', or 'both'")
    return value


def _select_components(
    components: dict[str, pd.Series],
    *,
    component: str,
    index: pd.Index,
) -> pd.DataFrame:
    names = ("cycle", "trend") if component == "both" else (component,)
    frame = pd.concat([components[name] for name in names], axis=1)
    frame.index = index
    frame.index.name = index.name
    return frame


def _normalize_fit_policy(fit_policy: str) -> str:
    value = str(fit_policy).lower()
    if value not in {"expanding", "full_sample"}:
        raise ValueError("fit_policy must be 'expanding' or 'full_sample'")
    return value


def _normalize_min_train_size(value: int | None, *, minimum: int) -> int:
    if value is None:
        return int(minimum)
    size = int(value)
    if size < int(minimum):
        raise ValueError(f"min_train_size must be at least {minimum}")
    return size


def _hamilton_filter_series(
    series: pd.Series,
    *,
    h: int,
    p: int,
    fit_policy: str,
    min_train_size: int,
) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(series, index=series.index, dtype=float)
    cycle = pd.Series(np.nan, index=values.index, name="cycle", dtype=float)
    trend = pd.Series(np.nan, index=values.index, name="trend", dtype=float)
    if len(values) <= h + p:
        return cycle, trend

    rows: list[tuple[int, np.ndarray, float]] = []
    arr = values.to_numpy(dtype=float)
    for anchor_pos in range(p - 1, len(arr) - h):
        target_pos = anchor_pos + h
        regressors = np.array([arr[anchor_pos - lag] for lag in range(p)], dtype=float)
        target = float(arr[target_pos])
        if not np.isfinite(target) or not np.isfinite(regressors).all():
            continue
        rows.append((target_pos, np.r_[1.0, regressors], target))
    if len(rows) < min_train_size:
        return cycle, trend

    target_positions = np.array([row[0] for row in rows], dtype=int)
    x_matrix = np.vstack([row[1] for row in rows]).astype(float)
    y_vector = np.array([row[2] for row in rows], dtype=float)

    if fit_policy == "full_sample":
        fitted = _ols_predict(x_matrix, y_vector, x_matrix)
        trend.iloc[target_positions] = fitted
        cycle.iloc[target_positions] = y_vector - fitted
        return cycle, trend

    for row_idx, target_pos in enumerate(target_positions):
        train_mask = target_positions < target_pos
        if int(train_mask.sum()) < min_train_size:
            continue
        fitted_value = _ols_predict(
            x_matrix[train_mask],
            y_vector[train_mask],
            x_matrix[row_idx : row_idx + 1],
        )[0]
        trend.iloc[target_pos] = fitted_value
        cycle.iloc[target_pos] = y_vector[row_idx] - fitted_value
    return cycle, trend


def _ols_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return np.asarray(x_pred @ beta, dtype=float).reshape(-1)


__all__ = [
    "FilterResult",
    "hamilton_filter",
    "hp_filter",
    "savitzky_golay",
    "wavelet_filter",
]
