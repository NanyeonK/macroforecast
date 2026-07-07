from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator


class _CompleteSubsetRegressor:
    def __init__(
        self,
        *,
        k: int = 4,
        max_subsets: int = 5000,
        random_state: int | None = 1071,
    ) -> None:
        if int(k) < 1:
            raise ValueError("k must be at least 1")
        if int(max_subsets) < 1:
            raise ValueError("max_subsets must be at least 1")
        self.k = int(k)
        self.max_subsets = int(max_subsets)
        self.random_state = random_state
        self.feature_names_in_: np.ndarray | None = None
        self.subsets_: tuple[tuple[int, ...], ...] = ()
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_CompleteSubsetRegressor":
        frame = X.astype(float)
        target = pd.Series(y, index=frame.index).astype(float)
        x_values = frame.to_numpy(dtype=float)
        y_values = target.to_numpy(dtype=float)
        n_features = x_values.shape[1]
        if n_features < self.k:
            raise ValueError(
                f"csr requires at least k={self.k} predictors; got p={n_features}"
            )
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        subsets = _subset_indices(
            n_features,
            self.k,
            max_subsets=self.max_subsets,
            random_state=self.random_state,
        )
        coef_sum = np.zeros(n_features, dtype=float)
        intercept_sum = 0.0
        for subset in subsets:
            design = np.column_stack(
                [np.ones(len(x_values), dtype=float), x_values[:, subset]]
            )
            params = np.linalg.lstsq(design, y_values, rcond=None)[0]
            intercept_sum += float(params[0])
            coef_sum[np.asarray(subset, dtype=int)] += params[1:]
        scale = float(len(subsets))
        self.subsets_ = tuple(subsets)
        self.coef_ = coef_sum / scale
        self.intercept_ = intercept_sum / scale
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(
            float
        )
        return frame.to_numpy(dtype=float) @ self.coef_ + self.intercept_


class _JackknifeModelAveragingRegressor:
    def __init__(
        self,
        *,
        candidates: Literal["nested"] = "nested",
        max_iter: int = 1000,
        tol: float = 1e-9,
    ) -> None:
        if candidates != "nested":
            raise ValueError("jma currently supports candidates='nested' only")
        if int(max_iter) < 1:
            raise ValueError("max_iter must be at least 1")
        if float(tol) <= 0.0:
            raise ValueError("tol must be positive")
        self.candidates = candidates
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.feature_names_in_: np.ndarray | None = None
        self.candidate_indices_: tuple[tuple[int, ...], ...] = ()
        self.candidate_intercepts_: np.ndarray | None = None
        self.candidate_coefs_: tuple[np.ndarray, ...] = ()
        self.loo_predictions_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_JackknifeModelAveragingRegressor":
        frame = X.astype(float)
        target = pd.Series(y, index=frame.index).astype(float)
        x_values = frame.to_numpy(dtype=float)
        y_values = target.to_numpy(dtype=float)
        if x_values.shape[1] < 1:
            raise ValueError("jma requires at least one predictor")
        candidate_indices = _nested_candidate_indices(
            n_features=x_values.shape[1],
            n_obs=len(x_values),
        )
        if not candidate_indices:
            raise ValueError(
                "jma requires at least three observations for leave-one-out OLS"
            )
        loo_columns: list[np.ndarray] = []
        intercepts: list[float] = []
        coefs: list[np.ndarray] = []
        for candidate in candidate_indices:
            intercept, coef, loo = _fit_ols_candidate_with_loo(
                x_values[:, candidate],
                y_values,
                tol=self.tol,
            )
            full_coef = np.zeros(x_values.shape[1], dtype=float)
            full_coef[np.asarray(candidate, dtype=int)] = coef
            intercepts.append(intercept)
            coefs.append(full_coef)
            loo_columns.append(loo)
        loo_predictions = np.column_stack(loo_columns)
        weights = _solve_simplex_least_squares(
            loo_predictions,
            y_values,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        coef_matrix = np.vstack(coefs)
        intercept_arr = np.asarray(intercepts, dtype=float)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.candidate_indices_ = tuple(candidate_indices)
        self.candidate_intercepts_ = intercept_arr
        self.candidate_coefs_ = tuple(coefs)
        self.loo_predictions_ = loo_predictions
        self.weights_ = weights
        self.coef_ = weights @ coef_matrix
        self.intercept_ = float(weights @ intercept_arr)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_names_in_ is None:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).astype(
            float
        )
        return frame.to_numpy(dtype=float) @ self.coef_ + self.intercept_


def _subset_indices(
    n_features: int,
    k: int,
    *,
    max_subsets: int,
    random_state: int | None,
) -> tuple[tuple[int, ...], ...]:
    total = comb(n_features, k)
    if total <= max_subsets:
        return tuple(combinations(range(n_features), k))
    rng = np.random.default_rng(random_state)
    selected: set[tuple[int, ...]] = set()
    while len(selected) < max_subsets:
        subset = tuple(
            sorted(int(value) for value in rng.choice(n_features, size=k, replace=False))
        )
        selected.add(subset)
    return tuple(sorted(selected))


def _nested_candidate_indices(
    *, n_features: int, n_obs: int
) -> tuple[tuple[int, ...], ...]:
    max_size = min(n_features, max(0, n_obs - 2))
    return tuple(tuple(range(size)) for size in range(1, max_size + 1))


def _fit_ols_candidate_with_loo(
    X: np.ndarray,
    y: np.ndarray,
    *,
    tol: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(X), dtype=float), X])
    params = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ params
    residual = y - fitted
    pinv = np.linalg.pinv(design)
    leverage = np.sum(design * pinv.T, axis=1)
    denom = 1.0 - leverage
    if np.any(np.abs(denom) <= tol):
        raise ValueError(
            "jma leave-one-out residual is undefined because an OLS leverage "
            "value is too close to one"
        )
    loo = y - residual / denom
    return float(params[0]), np.asarray(params[1:], dtype=float), loo


def _solve_simplex_least_squares(
    predictions: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int,
    tol: float,
) -> np.ndarray:
    n_candidates = predictions.shape[1]
    if n_candidates == 1:
        return np.ones(1, dtype=float)
    from scipy.optimize import minimize

    gram = predictions.T @ predictions
    rhs = predictions.T @ y

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * weights @ gram @ weights - rhs @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return gram @ weights - rhs

    start = np.full(n_candidates, 1.0 / n_candidates, dtype=float)
    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_candidates,
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": int(max_iter), "ftol": float(tol)},
    )
    if not result.success:
        raise RuntimeError(f"jma simplex solver failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("jma simplex solver returned zero total weight")
    return weights / total


def csr(
    X: Any,
    y: Any | None = None,
    *,
    k: int = 4,
    max_subsets: int = 5000,
    random_state: int | None = 1071,
) -> ModelFit:
    """Fit Complete Subset Regression.

    Complete Subset Regression averages ordinary-least-squares forecasts over
    every `k`-predictor subset of the available predictor set, with an intercept
    included in every subset regression. The method follows Elliott, Gargano,
    and Timmermann (2013) as a general supervised forecasting primitive rather
    than a paper-specific wrapper.

    When the number of possible subsets exceeds `max_subsets`, the estimator
    draws a uniform sample of distinct subsets with `random_state`; with the
    same seed the sampled subset set and forecasts are deterministic.
    """

    params = {
        "k": int(k),
        "max_subsets": int(max_subsets),
        "random_state": random_state,
    }
    return fit_estimator(
        _CompleteSubsetRegressor(
            k=int(k), max_subsets=int(max_subsets), random_state=random_state
        ),
        X,
        y,
        model="csr",
        metadata=params,
    )


def jma(
    X: Any,
    y: Any | None = None,
    *,
    candidates: Literal["nested"] = "nested",
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> ModelFit:
    """Fit Jackknife Model Averaging.

    Jackknife Model Averaging chooses non-negative candidate-model weights that
    sum to one and minimize Hansen and Racine's (2012) leave-one-out
    cross-validation criterion. The current candidate generator is the standard
    nested ordered-predictor sequence: model 1 uses the first predictor, model 2
    uses the first two predictors, and so on, stopping before saturated designs
    where OLS leave-one-out residuals are undefined.

    Candidate leave-one-out predictions use the OLS hat-matrix shortcut, so the
    estimator does not refit each candidate `n` times. Final forecasts average
    full-sample OLS candidate forecasts using the optimized simplex weights.
    """

    params = {
        "candidates": candidates,
        "max_iter": int(max_iter),
        "tol": float(tol),
    }
    return fit_estimator(
        _JackknifeModelAveragingRegressor(
            candidates=candidates,
            max_iter=int(max_iter),
            tol=float(tol),
        ),
        X,
        y,
        model="jma",
        metadata=params,
    )


__all__ = ["csr", "jma"]
