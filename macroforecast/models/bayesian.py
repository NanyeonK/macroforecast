from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import as_series

_KSC_PROB = np.asarray(
    [0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750],
    dtype=float,
)
_KSC_MEAN = np.asarray(
    [-10.12999, -3.97281, -8.56686, -2.77786, -0.61961, 1.79518, -1.08819],
    dtype=float,
)
_KSC_VAR = np.asarray(
    [5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261],
    dtype=float,
)
_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_VOL_CLIP = (-20.0, 20.0)


class _UCSVForecaster:
    def __init__(
        self,
        *,
        n_draws: int = 5000,
        burn: int = 1000,
        gamma: float = 0.2,
        random_state: int | None = 1071,
    ) -> None:
        if int(n_draws) < 1:
            raise ValueError("n_draws must be at least 1")
        if int(burn) < 0:
            raise ValueError("burn must be non-negative")
        if int(burn) >= int(n_draws):
            raise ValueError("burn must be smaller than n_draws")
        if float(gamma) <= 0.0:
            raise ValueError("gamma must be positive")
        self.n_draws = int(n_draws)
        self.burn = int(burn)
        self.gamma = float(gamma)
        self.random_state = random_state
        self.forecast_: float = 0.0
        self.trend_: np.ndarray | None = None
        self.obs_log_vol_: np.ndarray | None = None
        self.level_log_vol_: np.ndarray | None = None
        self.n_kept_: int = 0

    def fit(self, y: pd.Series) -> "_UCSVForecaster":
        series = pd.Series(y).dropna().astype(float)
        if series.empty:
            raise ValueError("ucsv requires at least one non-missing target observation")
        values = series.to_numpy(dtype=float)
        if len(values) == 1:
            self.forecast_ = float(values[0])
            self.trend_ = values.copy()
            self.obs_log_vol_ = np.asarray([0.0], dtype=float)
            self.level_log_vol_ = np.asarray([], dtype=float)
            self.n_kept_ = 1
            return self

        rng = np.random.default_rng(self.random_state)
        tau = _initial_trend(values)
        obs_log_vol, level_log_vol = _initial_log_vols(values, tau)
        trend_sum: np.ndarray = np.zeros(len(values), dtype=float)
        obs_vol_sum: np.ndarray = np.zeros(len(values), dtype=float)
        level_vol_sum: np.ndarray = np.zeros(len(values) - 1, dtype=float)
        kept = 0
        for draw in range(self.n_draws):
            obs_log_vol = _sample_log_volatility(
                values - tau,
                obs_log_vol,
                state_variance=self.gamma,
                rng=rng,
            )
            level_log_vol = _sample_log_volatility(
                np.diff(tau),
                level_log_vol,
                state_variance=self.gamma,
                rng=rng,
            )
            tau = _sample_ucsv_trend(
                values,
                obs_variance=np.exp(obs_log_vol),
                level_variance=np.exp(level_log_vol),
                rng=rng,
            )
            if draw >= self.burn:
                trend_sum += tau
                obs_vol_sum += obs_log_vol
                level_vol_sum += level_log_vol
                kept += 1
        if kept == 0:
            raise RuntimeError("ucsv sampler kept zero posterior draws")
        self.n_kept_ = kept
        self.trend_ = trend_sum / float(kept)
        self.obs_log_vol_ = obs_vol_sum / float(kept)
        self.level_log_vol_ = level_vol_sum / float(kept)
        self.forecast_ = float(self.trend_[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.forecast_, dtype=float)


def _initial_trend(values: np.ndarray) -> np.ndarray:
    window = min(9, max(3, len(values) // 10 * 2 + 1))
    if window % 2 == 0:
        window += 1
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )


def _initial_log_vols(values: np.ndarray, trend: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs_var = _safe_variance(values - trend)
    level_var = _safe_variance(np.diff(trend))
    return (
        np.full(len(values), np.log(obs_var), dtype=float),
        np.full(max(len(values) - 1, 0), np.log(level_var), dtype=float),
    )


def _safe_variance(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size <= 1:
        return 1e-4
    var = float(np.var(finite, ddof=1))
    return max(var, 1e-8)


def _sample_log_volatility(
    shocks: np.ndarray,
    current_log_vol: np.ndarray,
    *,
    state_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(shocks) == 0:
        return np.asarray([], dtype=float)
    log_square = np.log(np.asarray(shocks, dtype=float) ** 2 + 1e-12)
    indicators = _sample_ksc_indicators(log_square, current_log_vol, rng)
    adjusted = log_square - _KSC_MEAN[indicators]
    sampled = _sample_random_walk_state(
        adjusted,
        obs_variance=_KSC_VAR[indicators],
        state_variance=float(state_variance),
        rng=rng,
        initial_mean=float(np.median(adjusted)),
        initial_variance=10.0,
    )
    return np.clip(sampled, *_LOG_VOL_CLIP)


def _sample_ksc_indicators(
    log_square: np.ndarray,
    log_vol: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    centered = log_square[:, None] - log_vol[:, None] - _KSC_MEAN[None, :]
    log_probs = (
        np.log(_KSC_PROB)[None, :]
        - 0.5 * (_LOG_2PI + np.log(_KSC_VAR)[None, :])
        - 0.5 * centered**2 / _KSC_VAR[None, :]
    )
    log_probs = log_probs - np.max(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs = probs / probs.sum(axis=1, keepdims=True)
    cumulative = np.cumsum(probs, axis=1)
    draws = rng.random(len(log_square))[:, None]
    return np.sum(draws > cumulative, axis=1).astype(int)


def _sample_ucsv_trend(
    values: np.ndarray,
    *,
    obs_variance: np.ndarray,
    level_variance: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    state_variance = np.r_[0.0, level_variance]
    return _sample_random_walk_state(
        values,
        obs_variance=obs_variance,
        state_variance=state_variance,
        rng=rng,
        initial_mean=float(values[0]),
        initial_variance=max(_safe_variance(values) * 10.0, 1.0),
    )


def _sample_random_walk_state(
    observations: np.ndarray,
    *,
    obs_variance: np.ndarray,
    state_variance: float | np.ndarray,
    rng: np.random.Generator,
    initial_mean: float,
    initial_variance: float,
) -> np.ndarray:
    y = np.asarray(observations, dtype=float)
    r = np.maximum(np.asarray(obs_variance, dtype=float), 1e-10)
    q = np.asarray(state_variance, dtype=float)
    if q.ndim == 0:
        q = np.full(len(y), float(q), dtype=float)
        q[0] = 0.0
    q = np.maximum(q, 0.0)
    n = len(y)
    pred_mean: np.ndarray = np.zeros(n, dtype=float)
    pred_var: np.ndarray = np.zeros(n, dtype=float)
    filt_mean: np.ndarray = np.zeros(n, dtype=float)
    filt_var: np.ndarray = np.zeros(n, dtype=float)
    prev_mean = float(initial_mean)
    prev_var = max(float(initial_variance), 1e-8)
    for pos in range(n):
        a = prev_mean
        p = prev_var + float(q[pos])
        forecast_var = p + float(r[pos])
        gain = p / forecast_var
        m = a + gain * (float(y[pos]) - a)
        c = max((1.0 - gain) * p, 1e-10)
        pred_mean[pos] = a
        pred_var[pos] = max(p, 1e-10)
        filt_mean[pos] = m
        filt_var[pos] = c
        prev_mean = m
        prev_var = c
    state: np.ndarray = np.zeros(n, dtype=float)
    state[-1] = rng.normal(filt_mean[-1], np.sqrt(filt_var[-1]))
    for pos in range(n - 2, -1, -1):
        denom = pred_var[pos + 1]
        smoother_gain = filt_var[pos] / denom
        mean = filt_mean[pos] + smoother_gain * (state[pos + 1] - pred_mean[pos + 1])
        var = max(filt_var[pos] - smoother_gain * smoother_gain * denom, 1e-10)
        state[pos] = rng.normal(mean, np.sqrt(var))
    return state


def ucsv(
    y: Any,
    *,
    n_draws: int = 5000,
    burn: int = 1000,
    gamma: float = 0.2,
    random_state: int | None = 1071,
) -> ModelFit:
    """Fit Stock-Watson unobserved-components stochastic-volatility model.

    The UCSV model decomposes a target series as `y_t = tau_t + epsilon_t` and
    `tau_t = tau_{t-1} + eta_t`, with independent stochastic volatilities for
    the observation and trend innovations. This implementation follows the
    Stock and Watson (2007) Gibbs-sampling setup with the Kim, Shephard, and
    Chib seven-component normal mixture approximation for log-chi-square
    stochastic-volatility states.

    The point forecast at every horizon is the posterior mean of the final trend
    state `tau_T`, so UCSV forecasts are horizon-invariant by construction.
    `gamma` is the random-walk innovation variance for both log-volatility
    states; paper-specific settings such as Medeiros et al.'s volatility
    smoothing values can be supplied through this parameter.
    """

    target = as_series(y).dropna().astype(float)
    estimator = _UCSVForecaster(
        n_draws=int(n_draws),
        burn=int(burn),
        gamma=float(gamma),
        random_state=random_state,
    ).fit(target)
    trend = (
        pd.Series(estimator.trend_, index=target.index, name="trend")
        if estimator.trend_ is not None
        else pd.Series(dtype=float, name="trend")
    )
    dummy_features = ("__origin__",)
    return ModelFit(
        estimator=estimator,
        model="ucsv",
        feature_names=dummy_features,
        target_name=str(target.name) if target.name is not None else None,
        metadata={
            "n_obs": int(len(target)),
            "n_draws": int(n_draws),
            "burn": int(burn),
            "gamma": float(gamma),
            "random_state": random_state,
            "n_kept": int(estimator.n_kept_),
            "forecast_is_final_trend": True,
        },
        diagnostics={
            "trend": trend,
            "forecast": float(estimator.forecast_),
            "obs_log_vol": (
                pd.Series(estimator.obs_log_vol_, index=target.index, name="obs_log_vol")
                if estimator.obs_log_vol_ is not None
                else pd.Series(dtype=float, name="obs_log_vol")
            ),
        },
    )


__all__ = ["ucsv"]
