from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any

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
        random_state: int | None = None,
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


def csr(
    X: Any,
    y: Any | None = None,
    *,
    k: int = 4,
    max_subsets: int = 5000,
    random_state: int | None = None,
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


__all__ = ["csr"]
