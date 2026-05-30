from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import as_frame, as_series, fit_estimator, resolve_xy


class _AR:
    def __init__(self, *, n_lag: int = 1) -> None:
        self.n_lag = max(1, int(n_lag))
        self._coef: np.ndarray | None = None
        self._history: np.ndarray | None = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_AR":
        series = pd.Series(y).astype(float).dropna()
        self._fallback = float(series.mean()) if not series.empty else 0.0
        if len(series) <= self.n_lag:
            self._history = series.to_numpy(dtype=float)
            return self
        rows = []
        target = []
        values = series.to_numpy(dtype=float)
        for i in range(self.n_lag, len(values)):
            rows.append([1.0, *values[i - self.n_lag : i][::-1]])
            target.append(values[i])
        design = np.asarray(rows, dtype=float)
        response = np.asarray(target, dtype=float)
        self._coef = np.linalg.lstsq(design, response, rcond=None)[0]
        self._history = values[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._coef is None or self._history is None or len(self._history) == 0:
            return np.full(len(X), self._fallback, dtype=float)
        history = list(np.asarray(self._history, dtype=float))
        preds: list[float] = []
        for _ in range(len(X)):
            row = np.asarray([1.0, *history[-self.n_lag :][::-1]], dtype=float)
            pred = float(row @ self._coef)
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)


def ar(y: Any, *, n_lag: int = 1) -> ModelFit:
    """Fit an autoregression on a single target series."""

    target = as_series(y)
    dummy = pd.DataFrame({"__origin__": np.arange(len(target), dtype=float)}, index=target.index)
    return fit_estimator(_AR(n_lag=n_lag), dummy, target, model="ar", metadata={"n_lag": int(n_lag)})


class _VAR:
    def __init__(self, *, n_lag: int = 1, target: str | None = None) -> None:
        self.n_lag = max(1, int(n_lag))
        self.target = target
        self._results = None
        self._target_name: str | None = None
        self._last_values: np.ndarray | None = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "_VAR":
        if y is None:
            data = X.dropna()
            target_name = self.target or str(data.columns[0])
        else:
            data = pd.concat([pd.Series(y).rename("__target__"), X], axis=1).dropna()
            target_name = "__target__"
        if data.empty:
            return self
        self._target_name = target_name
        self._fallback = float(pd.to_numeric(data[target_name], errors="coerce").mean())
        if data.shape[0] <= self.n_lag + 1 or data.shape[1] < 2:
            self._last_values = data.to_numpy(dtype=float)
            return self
        from statsmodels.tsa.api import VAR

        try:
            self._results = VAR(data).fit(self.n_lag)
            self._last_values = self._results.endog[-self.n_lag :]
        except Exception:
            self._results = None
            self._last_values = data.to_numpy(dtype=float)[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._results is None or self._target_name is None:
            return np.full(len(X), self._fallback, dtype=float)
        forecast = self._results.forecast(self._results.endog[-self.n_lag :], steps=max(1, len(X)))
        target_index = self._results.names.index(self._target_name)
        return np.asarray(forecast[:, target_index], dtype=float)[: len(X)]


def var(panel: Any, *, target: str | None = None, n_lag: int = 1) -> ModelFit:
    """Fit a vector autoregression on a multivariate panel."""

    frame = as_frame(panel)
    estimator = _VAR(n_lag=n_lag, target=target)
    estimator.fit(frame)
    return ModelFit(
        estimator=estimator,
        model="var",
        feature_names=tuple(str(c) for c in frame.columns),
        target_name=target or str(frame.columns[0]),
        metadata={"n_obs": len(frame.dropna()), "n_lag": int(n_lag)},
    )


class _FAR:
    def __init__(self, *, n_factors: int = 3, n_lag: int = 1, random_state: int = 0) -> None:
        self.n_factors = max(1, int(n_factors))
        self.n_lag = max(1, int(n_lag))
        self.random_state = int(random_state)
        self._pca = None
        self._regression = None
        self._x_mean: pd.Series | None = None
        self._y_history: np.ndarray | None = None
        self._fallback: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_FAR":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression

        joined = pd.concat([X, y.rename("__target__")], axis=1).dropna()
        if joined.empty:
            return self
        X_clean = joined.drop(columns="__target__")
        y_clean = joined["__target__"]
        self._fallback = float(y_clean.mean())
        self._x_mean = X_clean.mean(axis=0)
        n_factors = min(self.n_factors, X_clean.shape[1], max(1, X_clean.shape[0] - 1))
        self._pca = PCA(n_components=n_factors, random_state=self.random_state)
        factors = self._pca.fit_transform((X_clean - self._x_mean).fillna(0.0))
        values = y_clean.to_numpy(dtype=float)
        rows = []
        target = []
        for i in range(self.n_lag, len(values)):
            rows.append([*factors[i], *values[i - self.n_lag : i][::-1]])
            target.append(values[i])
        if not rows:
            self._y_history = values[-self.n_lag :]
            return self
        self._regression = LinearRegression().fit(np.asarray(rows), np.asarray(target))
        self._y_history = values[-self.n_lag :]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pca is None or self._regression is None or self._x_mean is None or self._y_history is None:
            return np.full(len(X), self._fallback, dtype=float)
        frame = X.reindex(columns=self._x_mean.index, fill_value=0.0)
        factors = self._pca.transform((frame - self._x_mean).fillna(0.0))
        history = list(np.asarray(self._y_history, dtype=float))
        preds = []
        for i in range(len(X)):
            row = np.asarray([*factors[i], *history[-self.n_lag :][::-1]], dtype=float)
            pred = float(self._regression.predict(row.reshape(1, -1))[0])
            preds.append(pred)
            history.append(pred)
        return np.asarray(preds, dtype=float)


def far(
    X: Any,
    y: Any | None = None,
    *,
    n_factors: int = 3,
    n_lag: int = 1,
    random_state: int = 0,
) -> ModelFit:
    """Fit factor-augmented autoregression."""

    return fit_estimator(
        _FAR(n_factors=n_factors, n_lag=n_lag, random_state=random_state),
        X,
        y,
        model="far",
        metadata={"n_factors": int(n_factors), "n_lag": int(n_lag), "random_state": int(random_state)},
    )


def favar(
    X: Any,
    y: Any | None = None,
    *,
    n_factors: int = 3,
    n_lag: int = 1,
    random_state: int = 0,
) -> ModelFit:
    """Fit PCA factors and a VAR on the target plus factors."""

    from sklearn.decomposition import PCA

    frame, target = resolve_xy(X, y)
    n_factors_resolved = min(max(1, int(n_factors)), frame.shape[1], max(1, frame.shape[0] - 1))
    pca = PCA(n_components=n_factors_resolved, random_state=random_state)
    factors = pca.fit_transform(frame.fillna(0.0))
    factor_frame = pd.DataFrame(
        factors,
        index=frame.index,
        columns=[f"factor_{i + 1}" for i in range(n_factors_resolved)],
    )
    estimator = _VAR(n_lag=n_lag, target="__target__")
    estimator.fit(factor_frame, target)
    fit = ModelFit(
        estimator=estimator,
        model="favar",
        feature_names=tuple(factor_frame.columns),
        target_name=str(target.name) if target.name is not None else None,
        metadata={"n_obs": len(factor_frame), "n_factors": n_factors_resolved, "n_lag": int(n_lag)},
    )
    fit.metadata["pca"] = pca
    return fit


__all__ = ["ar", "far", "favar", "var"]
