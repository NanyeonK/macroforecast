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


__all__ = [
    "bayesian_ridge",
    "elastic_net",
    "glmboost",
    "huber",
    "lasso",
    "ols",
    "pls",
    "ridge",
]
