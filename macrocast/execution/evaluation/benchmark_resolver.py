"""benchmark_resolver - per-origin benchmark forecast computation.

Provides BenchmarkSpec dataclass and resolve_benchmark_forecasts /
resolve_benchmark_suite functions used by the relative-metrics pipeline.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg


class BenchmarkResolverError(RuntimeError):
    """Raised when benchmark resolution cannot produce a forecast."""


_OPERATIONAL_FAMILIES = (
    "historical_mean",
    "rolling_mean",
    "random_walk",
    "ar_bic",
    "ar_fixed_p",
    "ardi",
    "factor_model",
    "expert_benchmark",
)
_STUB_FAMILIES = ("survey_forecast", "paper_specific_benchmark")


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_model: str
    estimation_window: str = "expanding"
    scope: str = "same_for_all"
    window_len: int = 60
    fixed_p: int = 1
    n_factors: int = 3
    max_p: int = 6
    expert_callable: Callable | None = None
    target_overrides: Mapping[str, "BenchmarkSpec"] | None = None
    horizon_overrides: Mapping[int, "BenchmarkSpec"] | None = None


def _select_training_window(
    series: pd.Series,
    origin: pd.Timestamp,
    estimation_window: str,
    window_len: int,
) -> pd.Series:
    """Slice the target series for benchmark fitting at this origin."""
    full = series.loc[:origin]
    if estimation_window == "expanding":
        return full
    if estimation_window == "rolling":
        if window_len <= 0:
            raise BenchmarkResolverError("rolling estimation_window requires window_len>0")
        return full.iloc[-window_len:]
    if estimation_window == "fixed":
        if window_len <= 0:
            raise BenchmarkResolverError("fixed estimation_window requires window_len>0")
        return series.iloc[:window_len]
    if estimation_window == "paper_exact_window":
        raise NotImplementedError(
            "paper_exact_window estimation requires explicit (start, end) per paper "
            "replication - not implemented in v0.6"
        )
    raise BenchmarkResolverError(f"unknown estimation_window: {estimation_window!r}")


def _historical_mean(train: pd.Series) -> float:
    return float(train.mean())


def _rolling_mean(train: pd.Series, window_len: int) -> float:
    if window_len <= 0:
        return float(train.mean())
    return float(train.iloc[-window_len:].mean())


def _random_walk(train: pd.Series) -> float:
    return float(train.iloc[-1])


def _ar_bic_forecast(train: pd.Series, horizon: int, max_p: int) -> float:
    max_candidate = min(max_p, len(train) - 2)
    if max_candidate < 1:
        raise BenchmarkResolverError(
            f"training window too small ({len(train)}) to fit any AR lag candidate"
        )
    best_bic = np.inf
    best_lag = 1
    best_fit = None
    for lag in range(1, max_candidate + 1):
        try:
            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                warnings.simplefilter("ignore")
                fit = AutoReg(train, lags=lag, trend="c", old_names=False).fit()
        except Exception:
            continue
        bic = float(fit.bic)
        if bic < best_bic:
            best_bic = bic
            best_lag = lag
            best_fit = fit
    if best_fit is None:
        raise BenchmarkResolverError("no AR candidate could be fit")
    forecast = best_fit.forecast(steps=horizon)
    return float(np.asarray(forecast)[-1])


def _ar_fixed_p_forecast(train: pd.Series, horizon: int, fixed_p: int) -> float:
    if fixed_p < 1:
        raise BenchmarkResolverError("ar_fixed_p requires fixed_p>=1")
    if len(train) <= fixed_p + 1:
        raise BenchmarkResolverError("training window too small for ar_fixed_p")
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        fit = AutoReg(train, lags=fixed_p, trend="c", old_names=False).fit()
    forecast = fit.forecast(steps=horizon)
    return float(np.asarray(forecast)[-1])


def _factor_panel_at(
    auxiliary_panel,
    origin: pd.Timestamp,
    n_factors: int,
):
    """Return (factors_history, latest_factor_row) up to and including origin."""
    if auxiliary_panel is None:
        return None, None
    panel = auxiliary_panel.loc[:origin].dropna(how="any")
    if panel.empty or panel.shape[1] < 1:
        return None, None
    n_components = min(n_factors, panel.shape[1], panel.shape[0])
    if n_components < 1:
        return None, None
    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(panel.values)
    return factors, factors[-1]


def _ardi_forecast(
    train: pd.Series,
    horizon: int,
    n_factors: int,
    auxiliary_panel,
    origin: pd.Timestamp,
) -> float:
    """ARDI: AR(1) augmented with first PCA factor; falls back to AR(1)."""
    if len(train) < 4:
        raise BenchmarkResolverError("ARDI requires at least 4 training observations")
    factors, _latest = _factor_panel_at(auxiliary_panel, origin, n_factors)
    if factors is None or factors.shape[0] < len(train):
        return _ar_fixed_p_forecast(train, horizon, fixed_p=1)
    factor_series = pd.Series(factors[-len(train):, 0], index=train.index)
    y = train.values[1:]
    X = np.column_stack([train.values[:-1], factor_series.values[:-1]])
    reg = LinearRegression().fit(X, y)
    last_y = float(train.values[-1])
    last_f = float(factor_series.values[-1])
    pred = float(reg.predict(np.array([[last_y, last_f]]))[0])
    return pred


def _factor_model_forecast(
    train: pd.Series,
    horizon: int,
    n_factors: int,
    auxiliary_panel,
    origin: pd.Timestamp,
) -> float:
    """Pure factor regression: y_{t+h} ~ factors_t (no own lag)."""
    if len(train) < 4:
        raise BenchmarkResolverError("factor_model requires at least 4 training observations")
    factors, _ = _factor_panel_at(auxiliary_panel, origin, n_factors)
    if factors is None or factors.shape[0] < len(train):
        raise BenchmarkResolverError(
            "factor_model requires auxiliary_panel with sufficient observations"
        )
    F = factors[-len(train):, :]
    reg = LinearRegression().fit(F[:-1], train.values[1:])
    pred = float(reg.predict(F[-1:].reshape(1, -1))[0])
    return pred


def _expert_forecast(train: pd.Series, horizon: int, spec: BenchmarkSpec) -> float:
    if spec.expert_callable is None:
        raise BenchmarkResolverError(
            "expert_benchmark requires spec.expert_callable to be supplied"
        )
    value = spec.expert_callable(train.copy(), int(horizon))
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkResolverError(
            "expert_benchmark callable must return a numeric forecast"
        ) from exc


def _dispatch_single(
    train: pd.Series,
    horizon: int,
    spec: BenchmarkSpec,
    origin: pd.Timestamp,
    auxiliary_panel,
) -> float:
    family = spec.benchmark_model
    if family == "historical_mean":
        return _historical_mean(train)
    if family == "rolling_mean":
        return _rolling_mean(train, spec.window_len)
    if family == "random_walk":
        return _random_walk(train)
    if family == "ar_bic":
        return _ar_bic_forecast(train, horizon, spec.max_p)
    if family == "ar_fixed_p":
        return _ar_fixed_p_forecast(train, horizon, spec.fixed_p)
    if family == "ardi":
        return _ardi_forecast(train, horizon, spec.n_factors, auxiliary_panel, origin)
    if family == "factor_model":
        return _factor_model_forecast(train, horizon, spec.n_factors, auxiliary_panel, origin)
    if family == "expert_benchmark":
        return _expert_forecast(train, horizon, spec)
    if family in _STUB_FAMILIES:
        raise NotImplementedError(
            f"benchmark_family {family!r} is registered as a stub"
        )
    if family == "var":
        raise NotImplementedError(
            "var benchmark is registered as future status and not implemented in v0.6"
        )
    if family == "multi_benchmark_suite":
        raise BenchmarkResolverError(
            "multi_benchmark_suite must be dispatched via resolve_benchmark_suite"
        )
    raise BenchmarkResolverError(f"unknown benchmark_model: {family!r}")


def _origin_plus_h(series, origin_eff, horizon):
    try:
        origin_pos = series.index.get_loc(origin_eff)
    except KeyError:
        origin_pos = None
    if origin_pos is not None and origin_pos + horizon < len(series.index):
        return series.index[origin_pos + horizon]
    return origin_eff



def _safe_dispatch(train, horizon, spec, origin_eff, auxiliary_panel):
    try:
        return _dispatch_single(train, horizon, spec, origin_eff, auxiliary_panel)
    except (BenchmarkResolverError, NotImplementedError):
        raise
    except Exception as exc:
        raise BenchmarkResolverError(f"benchmark dispatch failed at origin {origin_eff!s}: {exc}") from exc


def resolve_benchmark_forecasts(*, target_series, horizon, spec, train_origins, auxiliary_panel=None):
    """Compute per-origin benchmark forecast for a single (target, horizon)."""
    if not isinstance(target_series, pd.Series):
        raise BenchmarkResolverError("target_series must be a pandas Series")
    if horizon < 1:
        raise BenchmarkResolverError("horizon must be >= 1")
    series = target_series.dropna().astype(float)
    rows = []
    iter_origins = pd.DatetimeIndex(train_origins)
    for origin in iter_origins:
        if origin not in series.index:
            valid_idx = series.index[series.index <= origin]
            if len(valid_idx) == 0:
                continue
            origin_eff = valid_idx[-1]
        else:
            origin_eff = origin
        train = _select_training_window(series, origin_eff, spec.estimation_window, spec.window_len)
        if len(train) < 2:
            continue
        pred = _safe_dispatch(train, horizon, spec, origin_eff, auxiliary_panel)
        target_date = _origin_plus_h(series, origin_eff, horizon)
        rows.append(dict(date=origin_eff, forecast_target_date=target_date, benchmark_name=spec.benchmark_model, benchmark_pred=pred))
    return pd.DataFrame(rows, columns=["date", "forecast_target_date", "benchmark_name", "benchmark_pred"])


def resolve_benchmark_suite(*, target_series, horizon, suite, train_origins, auxiliary_panel=None):
    """Stack resolve_benchmark_forecasts results across the suite list."""
    if not suite:
        raise BenchmarkResolverError("suite must contain at least one BenchmarkSpec")
    frames = []
    for spec in suite:
        frames.append(
            resolve_benchmark_forecasts(
                target_series=target_series,
                horizon=horizon,
                spec=spec,
                train_origins=train_origins,
                auxiliary_panel=auxiliary_panel,
            )
        )
    if not frames:
        return pd.DataFrame(columns=["date", "forecast_target_date", "benchmark_name", "benchmark_pred"])
    return pd.concat(frames, ignore_index=True)
