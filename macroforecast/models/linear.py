from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, resolve_xy


def ols(X: Any, y: Any | None = None) -> ModelFit:
    """Fit ordinary least squares."""

    from sklearn.linear_model import LinearRegression

    return fit_estimator(LinearRegression(), X, y, model="ols")


def ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0) -> ModelFit:
    """Fit ridge regression."""

    from sklearn.linear_model import Ridge

    return fit_estimator(Ridge(alpha=float(alpha)), X, y, model="ridge", metadata={"alpha": float(alpha)})


def lasso(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    max_iter: int = 20000,
) -> ModelFit:
    """Fit lasso regression with a user-supplied alpha."""

    from sklearn.linear_model import Lasso

    return fit_estimator(
        Lasso(alpha=float(alpha), max_iter=int(max_iter)),
        X,
        y,
        model="lasso",
        metadata={"alpha": float(alpha), "max_iter": int(max_iter)},
    )


def elastic_net(
    X: Any,
    y: Any | None = None,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 20000,
) -> ModelFit:
    """Fit elastic net regression."""

    from sklearn.linear_model import ElasticNet

    return fit_estimator(
        ElasticNet(alpha=float(alpha), l1_ratio=float(l1_ratio), max_iter=int(max_iter)),
        X,
        y,
        model="elastic_net",
        metadata={"alpha": float(alpha), "l1_ratio": float(l1_ratio), "max_iter": int(max_iter)},
    )


def bayesian_ridge(X: Any, y: Any | None = None) -> ModelFit:
    """Fit empirical-Bayes Bayesian ridge regression."""

    from sklearn.linear_model import BayesianRidge

    return fit_estimator(BayesianRidge(), X, y, model="bayesian_ridge")


def huber(
    X: Any,
    y: Any | None = None,
    *,
    epsilon: float = 1.35,
    max_iter: int = 1000,
) -> ModelFit:
    """Fit robust Huber regression."""

    from sklearn.linear_model import HuberRegressor

    return fit_estimator(
        HuberRegressor(epsilon=float(epsilon), max_iter=int(max_iter)),
        X,
        y,
        model="huber",
        metadata={"epsilon": float(epsilon), "max_iter": int(max_iter)},
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


class _PCR:
    def __init__(self, *, n_components: int = 3, random_state: int = 0) -> None:
        self.n_components = max(1, int(n_components))
        self.random_state = int(random_state)
        self._mean: pd.Series | None = None
        self._pca = None
        self._regression = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_PCR":
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression

        n_components = min(self.n_components, X.shape[1], max(1, X.shape[0] - 1))
        self._mean = X.mean(axis=0)
        centered = (X - self._mean).fillna(0.0)
        self._pca = PCA(n_components=n_components, random_state=self.random_state)
        scores = self._pca.fit_transform(centered)
        self._regression = LinearRegression().fit(scores, np.asarray(y, dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._mean is None or self._pca is None or self._regression is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=self._mean.index, fill_value=0.0)
        scores = self._pca.transform((frame - self._mean).fillna(0.0))
        return np.asarray(self._regression.predict(scores), dtype=float)


def pcr(
    X: Any,
    y: Any | None = None,
    *,
    n_components: int = 3,
    random_state: int = 0,
) -> ModelFit:
    """Fit principal component regression."""

    return fit_estimator(
        _PCR(n_components=n_components, random_state=random_state),
        X,
        y,
        model="pcr",
        metadata={"n_components": int(n_components), "random_state": int(random_state)},
    )


__all__ = [
    "bayesian_ridge",
    "elastic_net",
    "glmboost",
    "huber",
    "lasso",
    "ols",
    "pcr",
    "ridge",
]
