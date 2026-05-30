from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator


def ols(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit ordinary least squares."""

    from sklearn.linear_model import LinearRegression

    return fit_estimator(LinearRegression(**kwargs), X, y, model="ols", metadata=dict(kwargs))


def ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, **kwargs: Any) -> ModelFit:
    """Fit ridge regression."""

    from sklearn.linear_model import Ridge

    params = {"alpha": float(alpha), **kwargs}
    return fit_estimator(Ridge(**params), X, y, model="ridge", metadata=params)


def lasso(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    max_iter: int = 20000,
    **kwargs: Any,
) -> ModelFit:
    """Fit lasso regression with a user-supplied alpha."""

    from sklearn.linear_model import Lasso

    params = {"alpha": float(alpha), "max_iter": int(max_iter), **kwargs}
    return fit_estimator(
        Lasso(**params),
        X,
        y,
        model="lasso",
        metadata=params,
    )


def elastic_net(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 20000,
    **kwargs: Any,
) -> ModelFit:
    """Fit elastic net regression."""

    from sklearn.linear_model import ElasticNet

    params = {"alpha": float(alpha), "l1_ratio": float(l1_ratio), "max_iter": int(max_iter), **kwargs}
    return fit_estimator(
        ElasticNet(**params),
        X,
        y,
        model="elastic_net",
        metadata=params,
    )


def bayesian_ridge(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit empirical-Bayes Bayesian ridge regression."""

    from sklearn.linear_model import BayesianRidge

    return fit_estimator(BayesianRidge(**kwargs), X, y, model="bayesian_ridge", metadata=dict(kwargs))


def huber(
    X: Any,
    y: Any | None = None,
    *,
    epsilon: float = 1.35,
    max_iter: int = 1000,
    **kwargs: Any,
) -> ModelFit:
    """Fit robust Huber regression."""

    from sklearn.linear_model import HuberRegressor

    params = {"epsilon": float(epsilon), "max_iter": int(max_iter), **kwargs}
    return fit_estimator(
        HuberRegressor(**params),
        X,
        y,
        model="huber",
        metadata=params,
    )


class _GLMBoost:
    """Componentwise L2 boosting with linear base learners."""

    def __init__(self, *, n_iter: int = 100, learning_rate: float = 0.1) -> None:
        self.n_iter = max(1, int(n_iter))
        self.learning_rate = float(learning_rate)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.feature_names_in_: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_GLMBoost":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        x = X.fillna(0.0).to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        self.intercept_ = float(np.mean(target)) if target.size else 0.0
        residual = target - self.intercept_
        self.coef_ = np.zeros(x.shape[1], dtype=float)
        for _ in range(self.n_iter):
            covariances = x.T @ residual
            best = int(np.argmax(np.abs(covariances)))
            denom = float(x[:, best] @ x[:, best])
            if denom <= 1e-12:
                break
            step = self.learning_rate * float(covariances[best]) / denom
            self.coef_[best] += step
            residual = residual - step * x[:, best]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None:
            return np.zeros(len(X), dtype=float)
        return X.fillna(0.0).to_numpy(dtype=float) @ self.coef_ + self.intercept_


def glmboost(
    X: Any,
    y: Any | None = None,
    *,
    n_iter: int = 100,
    learning_rate: float = 0.1,
) -> ModelFit:
    """Fit componentwise linear boosting."""

    return fit_estimator(
        _GLMBoost(n_iter=n_iter, learning_rate=learning_rate),
        X,
        y,
        model="glmboost",
        metadata={"n_iter": int(n_iter), "learning_rate": float(learning_rate)},
    )


def pls(
    X: Any,
    y: Any | None = None,
    *,
    n_components: int = 3,
    scale: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
    **kwargs: Any,
) -> ModelFit:
    """Fit partial least squares regression."""

    from sklearn.cross_decomposition import PLSRegression

    params = {
        "n_components": int(n_components),
        "scale": bool(scale),
        "max_iter": int(max_iter),
        "tol": float(tol),
        **kwargs,
    }
    return fit_estimator(
        PLSRegression(**params),
        X,
        y,
        model="pls",
        metadata=params,
    )


class SupervisedPCARegressor:
    """Correlation-screened PCA followed by a linear or ridge regression."""

    def __init__(
        self,
        *,
        n_components: int = 3,
        n_selected: int | None = 50,
        min_abs_corr: float = 0.0,
        scale: bool = True,
        alpha: float = 0.0,
        random_state: int = 0,
    ) -> None:
        self.n_components = max(1, int(n_components))
        self.n_selected = None if n_selected is None else max(1, int(n_selected))
        self.min_abs_corr = float(max(0.0, min_abs_corr))
        self.scale = bool(scale)
        self.alpha = float(max(0.0, alpha))
        self.random_state = int(random_state)
        self.selected_features_: tuple[str, ...] = ()
        self.screening_scores_: dict[str, float] = {}
        self.n_components_: int = 0
        self._scaler: Any = None
        self._pca: Any = None
        self._regressor: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupervisedPCARegressor":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.preprocessing import StandardScaler

        frame = X.astype(float).copy()
        target = pd.Series(y, index=frame.index).astype(float)
        if frame.shape[1] == 0:
            raise ValueError("supervised_pca requires at least one predictor")
        values = frame.to_numpy(dtype=float)
        y_values = target.to_numpy(dtype=float)
        scores = _absolute_correlations(values, y_values)
        order = np.argsort(-scores)
        ranked = [int(i) for i in order if scores[i] >= self.min_abs_corr]
        if self.n_selected is not None:
            ranked = ranked[: self.n_selected]
        min_needed = min(self.n_components, values.shape[1])
        if len(ranked) < min_needed:
            ranked_set = set(ranked)
            ranked.extend(int(i) for i in order if int(i) not in ranked_set)
            ranked = ranked[:min_needed]
        if not ranked:
            raise ValueError("supervised_pca selected no predictors")

        selected = frame.iloc[:, ranked]
        self.selected_features_ = tuple(str(column) for column in selected.columns)
        self.screening_scores_ = {
            str(column): float(scores[int(frame.columns.get_loc(column))])
            for column in selected.columns
        }
        selected_values = selected.to_numpy(dtype=float)
        if self.scale:
            self._scaler = StandardScaler()
            selected_values = self._scaler.fit_transform(selected_values)
        else:
            self._scaler = None
        self.n_components_ = min(self.n_components, selected_values.shape[1], selected_values.shape[0])
        self._pca = PCA(n_components=self.n_components_, random_state=self.random_state)
        components = self._pca.fit_transform(selected_values)
        self._regressor = Ridge(alpha=self.alpha) if self.alpha > 0 else LinearRegression()
        self._regressor.fit(components, target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._pca is None or self._regressor is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.selected_features_), fill_value=0.0).astype(float)
        values = frame.to_numpy(dtype=float)
        if self._scaler is not None:
            values = self._scaler.transform(values)
        components = self._pca.transform(values)
        return np.asarray(self._regressor.predict(components), dtype=float)


def _absolute_correlations(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    x_centered = values - np.nanmean(values, axis=0)
    y_centered = target - float(np.nanmean(target))
    x_scale = np.sqrt(np.nansum(x_centered * x_centered, axis=0))
    y_scale = float(np.sqrt(np.nansum(y_centered * y_centered)))
    denom = x_scale * y_scale
    numer = np.nansum(x_centered * y_centered[:, None], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.abs(numer / denom)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def supervised_pca(
    X: Any,
    y: Any | None = None,
    *,
    n_components: int = 3,
    n_selected: int | None = 50,
    min_abs_corr: float = 0.0,
    scale: bool = True,
    alpha: float = 0.0,
    random_state: int = 0,
) -> ModelFit:
    """Fit correlation-screened supervised PCA regression."""

    params = {
        "n_components": int(n_components),
        "n_selected": None if n_selected is None else int(n_selected),
        "min_abs_corr": float(min_abs_corr),
        "scale": bool(scale),
        "alpha": float(alpha),
        "random_state": int(random_state),
    }
    return fit_estimator(
        SupervisedPCARegressor(
            n_components=int(n_components),
            n_selected=None if n_selected is None else int(n_selected),
            min_abs_corr=float(min_abs_corr),
            scale=bool(scale),
            alpha=float(alpha),
            random_state=int(random_state),
        ),
        X,
        y,
        model="supervised_pca",
        metadata=params,
    )


__all__ = [
    "SupervisedPCARegressor",
    "bayesian_ridge",
    "elastic_net",
    "glmboost",
    "huber",
    "lasso",
    "ols",
    "pls",
    "ridge",
    "supervised_pca",
]
