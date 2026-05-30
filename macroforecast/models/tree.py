from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import fit_estimator, optional_import


def decision_tree(
    X: Any,
    y: Any | None = None,
    *,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit a CART regression tree."""

    from sklearn.tree import DecisionTreeRegressor

    params = {
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        **kwargs,
    }
    return fit_estimator(
        DecisionTreeRegressor(**params),
        X,
        y,
        model="decision_tree",
        metadata=params,
    )


def random_forest(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    n_jobs: int | None = 1,
    **kwargs: Any,
) -> ModelFit:
    """Fit a random forest regressor."""

    from sklearn.ensemble import RandomForestRegressor

    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "n_jobs": n_jobs,
        **kwargs,
    }
    return fit_estimator(
        RandomForestRegressor(**params),
        X,
        y,
        model="random_forest",
        metadata=params,
    )


def extra_trees(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    n_jobs: int | None = 1,
    **kwargs: Any,
) -> ModelFit:
    """Fit an extremely randomized trees regressor."""

    from sklearn.ensemble import ExtraTreesRegressor

    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "n_jobs": n_jobs,
        **kwargs,
    }
    return fit_estimator(
        ExtraTreesRegressor(**params),
        X,
        y,
        model="extra_trees",
        metadata=params,
    )


def gradient_boosting(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit sklearn gradient-boosted regression trees."""

    from sklearn.ensemble import GradientBoostingRegressor

    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        **kwargs,
    }
    return fit_estimator(
        GradientBoostingRegressor(**params),
        X,
        y,
        model="gradient_boosting",
        metadata=params,
    )


def xgboost(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 1.0,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit an XGBoost regressor. Requires the `xgboost` extra."""

    xgb = optional_import("xgboost", extra="xgboost")
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "random_state": int(random_state),
        "objective": "reg:squarederror",
        **kwargs,
    }
    estimator = xgb.XGBRegressor(
        **params,
    )
    return fit_estimator(estimator, X, y, model="xgboost", metadata=params)


def lightgbm(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    num_leaves: int = 31,
    random_state: int = 0,
    **kwargs: Any,
) -> ModelFit:
    """Fit a LightGBM regressor. Requires the `lightgbm` extra."""

    lgb = optional_import("lightgbm", extra="lightgbm")
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "random_state": int(random_state),
        **kwargs,
    }
    estimator = lgb.LGBMRegressor(
        **params,
    )
    return fit_estimator(estimator, X, y, model="lightgbm", metadata=params)


def catboost(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    random_state: int = 0,
    verbose: bool = False,
    **kwargs: Any,
) -> ModelFit:
    """Fit a CatBoost regressor. Requires the `catboost` extra."""

    cb = optional_import("catboost", extra="catboost")
    params = {
        "iterations": int(n_estimators),
        "learning_rate": float(learning_rate),
        "depth": int(max_depth),
        "random_seed": int(random_state),
        "verbose": verbose,
        **kwargs,
    }
    metadata = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        "verbose": verbose,
        **kwargs,
    }
    estimator = cb.CatBoostRegressor(
        **params,
    )
    return fit_estimator(estimator, X, y, model="catboost", metadata=metadata)


class SlowGrowingTreeRegressor:
    """Slow-Growing Tree with soft split propagation."""

    # Package-native compact implementation; no exact paper-replication claim.
    def __init__(
        self,
        *,
        eta: float = 0.1,
        herfindahl_threshold: float = 0.25,
        eta_depth_step: float = 0.01,
        eta_max_plateau: float = 0.5,
        mtry_frac: float = 1.0,
        max_depth: int | None = 10,
        random_state: int = 0,
        min_leaf_size: int = 5,
    ) -> None:
        self.eta = float(np.clip(eta, 1e-6, 1.0))
        self.herfindahl_threshold = float(np.clip(herfindahl_threshold, 1e-6, 1.0))
        self.eta_depth_step = float(eta_depth_step)
        self.eta_max_plateau = float(np.clip(eta_max_plateau, 1e-6, 1.0))
        self.mtry_frac = float(np.clip(mtry_frac, 1e-6, 1.0))
        self.max_depth = None if max_depth is None else int(max_depth)
        self.random_state = int(random_state)
        self.min_leaf_size = max(2, int(min_leaf_size))
        self.feature_names_in_: tuple[str, ...] = ()
        self._nodes: list[tuple] = []
        self._fallback: float = 0.0

    @staticmethod
    def _herfindahl(weights: np.ndarray) -> float:
        total = float(weights.sum())
        if total <= 1e-12:
            return 1.0
        return float((weights**2).sum()) / (total * total)

    @staticmethod
    def _weighted_mean(y: np.ndarray, weights: np.ndarray) -> float:
        total = float(weights.sum())
        if total <= 1e-12:
            return 0.0
        return float((weights * y).sum() / total)

    def _best_split(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray, rng: np.random.Generator):
        n, k_total = X.shape
        if n < 2 * self.min_leaf_size:
            return None
        if self.mtry_frac < 1.0 and k_total > 1:
            keep = max(1, int(round(self.mtry_frac * k_total)))
            features = rng.choice(k_total, keep, replace=False)
        else:
            features = range(k_total)
        total_w = float(weights.sum())
        sum_y = float((weights * y).sum())
        sum_y2 = float((weights * y * y).sum())
        full_sse = sum_y2 - sum_y * sum_y / max(total_w, 1e-12)
        best: tuple[int, float, float] | None = None
        for k in features:
            order = np.argsort(X[:, k], kind="stable")
            x_sorted = X[order, k]
            y_sorted = y[order]
            w_sorted = weights[order]
            cum_w = np.cumsum(w_sorted)
            cum_y = np.cumsum(w_sorted * y_sorted)
            cum_y2 = np.cumsum(w_sorted * y_sorted * y_sorted)
            for j in range(self.min_leaf_size, n - self.min_leaf_size + 1):
                if j >= n or x_sorted[j] == x_sorted[j - 1]:
                    continue
                left_w = float(cum_w[j - 1])
                right_w = total_w - left_w
                if left_w <= 1e-12 or right_w <= 1e-12:
                    continue
                left_y = float(cum_y[j - 1])
                right_y = sum_y - left_y
                left_y2 = float(cum_y2[j - 1])
                right_y2 = sum_y2 - left_y2
                sse = left_y2 - left_y * left_y / left_w + right_y2 - right_y * right_y / right_w
                if best is None or sse < best[2]:
                    best = (int(k), float(0.5 * (x_sorted[j - 1] + x_sorted[j])), float(sse))
        if best is None or best[2] >= full_sse - 1e-12:
            return None
        return best

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SlowGrowingTreeRegressor":
        self.feature_names_in_ = tuple(str(c) for c in X.columns)
        x = X.fillna(0.0).to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        self._fallback = float(target.mean()) if target.size else 0.0
        self._nodes = []
        if x.size == 0 or target.size == 0:
            self._nodes.append(("leaf", self._fallback))
            return self
        self._nodes.append(("placeholder",))
        queue: list[tuple[int, np.ndarray, int]] = [(0, np.ones(len(target), dtype=float), 0)]
        rng = np.random.default_rng(self.random_state)
        while queue:
            node_idx, weights, depth = queue.pop(0)
            mu = self._weighted_mean(target, weights)
            can_split = (
                (self.max_depth is None or depth < self.max_depth)
                and self._herfindahl(weights) < self.herfindahl_threshold
                and float(weights.sum()) > 2 * self.min_leaf_size
            )
            split = self._best_split(x, target, weights, rng) if can_split else None
            if split is None:
                self._nodes[node_idx] = ("leaf", mu)
                continue
            k, cut, _ = split
            plateau = max(self.eta, self.eta_max_plateau)
            eta_depth = float(np.clip(self.eta + self.eta_depth_step * depth, 1e-6, plateau))
            left_mask = x[:, k] <= cut
            left_w = weights * (left_mask.astype(float) + (~left_mask).astype(float) * (1.0 - eta_depth))
            right_w = weights * ((~left_mask).astype(float) + left_mask.astype(float) * (1.0 - eta_depth))
            if left_w.sum() <= 1e-12 or right_w.sum() <= 1e-12:
                self._nodes[node_idx] = ("leaf", mu)
                continue
            left_idx = len(self._nodes)
            right_idx = left_idx + 1
            self._nodes.extend([("placeholder",), ("placeholder",)])
            self._nodes[node_idx] = ("split", k, cut, left_idx, right_idx, eta_depth)
            queue.append((left_idx, left_w, depth + 1))
            queue.append((right_idx, right_w, depth + 1))
        return self

    def _predict_one(self, row: np.ndarray) -> float:
        if not self._nodes:
            return self._fallback
        total = 0.0
        weight_total = 0.0
        stack = [(0, 1.0)]
        while stack:
            idx, weight = stack.pop()
            node = self._nodes[idx]
            if node[0] == "leaf":
                total += weight * float(node[1])
                weight_total += weight
                continue
            if node[0] == "split":
                _, k, cut, left_idx, right_idx, eta_depth = node
                if row[k] <= cut:
                    stack.append((left_idx, weight))
                    stack.append((right_idx, weight * (1.0 - eta_depth)))
                else:
                    stack.append((right_idx, weight))
                    stack.append((left_idx, weight * (1.0 - eta_depth)))
        return total / weight_total if weight_total > 1e-12 else self._fallback

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0)
        values = frame.fillna(0.0).to_numpy(dtype=float)
        return np.asarray([self._predict_one(row) for row in values], dtype=float)


def slow_growing_tree(
    X: Any,
    y: Any | None = None,
    *,
    eta: float = 0.1,
    herfindahl_threshold: float = 0.25,
    eta_depth_step: float = 0.01,
    eta_max_plateau: float = 0.5,
    mtry_frac: float = 1.0,
    max_depth: int | None = 10,
    random_state: int = 0,
    min_leaf_size: int = 5,
) -> ModelFit:
    """Fit a Slow-Growing Tree."""

    params = {
        "eta": float(eta),
        "herfindahl_threshold": float(herfindahl_threshold),
        "eta_depth_step": float(eta_depth_step),
        "eta_max_plateau": float(eta_max_plateau),
        "mtry_frac": float(mtry_frac),
        "max_depth": max_depth,
        "random_state": int(random_state),
        "min_leaf_size": int(min_leaf_size),
    }
    return fit_estimator(
        SlowGrowingTreeRegressor(
            eta=float(eta),
            herfindahl_threshold=float(herfindahl_threshold),
            eta_depth_step=float(eta_depth_step),
            eta_max_plateau=float(eta_max_plateau),
            mtry_frac=float(mtry_frac),
            max_depth=max_depth,
            random_state=int(random_state),
            min_leaf_size=int(min_leaf_size),
        ),
        X,
        y,
        model="slow_growing_tree",
        metadata=params,
    )


class QuantileRegressionForestRegressor:
    """Random-forest point forecasts plus empirical leaf quantiles."""

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 0,
        quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.random_state = int(random_state)
        self.quantile_levels = tuple(float(q) for q in quantile_levels)
        self._forest: Any = None
        self._leaf_targets: list[dict[int, np.ndarray]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileRegressionForestRegressor":
        from sklearn.ensemble import RandomForestRegressor

        self._forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=1,
        )
        frame = X.fillna(0.0)
        target = np.asarray(y, dtype=float)
        self._forest.fit(frame, target)
        leaves = self._forest.apply(frame)
        self._leaf_targets = []
        for tree_idx in range(leaves.shape[1]):
            tree_leaves = leaves[:, tree_idx]
            self._leaf_targets.append({
                int(leaf): target[tree_leaves == leaf] for leaf in np.unique(tree_leaves)
            })
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._forest is None:
            return np.zeros(len(X), dtype=float)
        return np.asarray(self._forest.predict(X.fillna(0.0)), dtype=float)

    def predict_quantiles(self, X: pd.DataFrame, levels: tuple[float, ...] | None = None) -> dict[float, np.ndarray]:
        levels = self.quantile_levels if levels is None else tuple(float(q) for q in levels)
        if self._forest is None or not self._leaf_targets:
            return {q: np.zeros(len(X), dtype=float) for q in levels}
        leaves = self._forest.apply(X.fillna(0.0))
        out: dict[float, np.ndarray] = {q: np.empty(len(X), dtype=float) for q in levels}
        for i in range(len(X)):
            samples: list[float] = []
            for tree_idx in range(leaves.shape[1]):
                values = self._leaf_targets[tree_idx].get(int(leaves[i, tree_idx]), np.array([]))
                samples.extend(np.asarray(values, dtype=float).tolist())
            arr = np.asarray(samples if samples else [0.0], dtype=float)
            for q in levels:
                out[q][i] = float(np.quantile(arr, q))
        return out


def quantile_regression_forest(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> ModelFit:
    """Fit a quantile regression forest."""

    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "quantile_levels": tuple(float(q) for q in quantile_levels),
    }
    return fit_estimator(
        QuantileRegressionForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            quantile_levels=tuple(float(q) for q in quantile_levels),
        ),
        X,
        y,
        model="quantile_regression_forest",
        metadata=params,
    )


def _base_estimator(name: str, params: dict[str, Any], random_state: int):
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    common = {"random_state": random_state}
    if name == "ridge":
        return Ridge(**params)
    if name == "lasso":
        return Lasso(max_iter=20000, **params)
    if name == "elastic_net":
        return ElasticNet(max_iter=20000, **params)
    if name == "decision_tree":
        return DecisionTreeRegressor(**{**common, **params})
    if name == "random_forest":
        return RandomForestRegressor(**{**common, **params})
    if name == "extra_trees":
        return ExtraTreesRegressor(**{**common, **params})
    if name == "gradient_boosting":
        return GradientBoostingRegressor(**{**common, **params})
    if name == "knn":
        return KNeighborsRegressor(**params)
    if name == "svr":
        return SVR(**params)
    raise ValueError(f"unknown bagging base estimator: {name!r}")


class BaggingRegressor:
    def __init__(
        self,
        *,
        base: str = "ridge",
        n_estimators: int = 50,
        max_samples: float = 0.8,
        random_state: int = 0,
        base_params: dict[str, Any] | None = None,
        strategy: str = "standard",
        block_length: int = 4,
    ) -> None:
        self.base = str(base)
        self.n_estimators = max(1, int(n_estimators))
        self.max_samples = float(np.clip(max_samples, 0.05, 1.0))
        self.random_state = int(random_state)
        self.base_params = dict(base_params or {})
        self.strategy = str(strategy)
        self.block_length = max(1, int(block_length))
        self._models: list[Any] = []

    def _draw_indices(self, rng: np.random.Generator, n: int, size: int) -> np.ndarray:
        if self.strategy == "block":
            n_blocks = (size + self.block_length - 1) // self.block_length
            starts = rng.integers(0, n, size=n_blocks)
            idx = ((starts[:, None] + np.arange(self.block_length)[None, :]) % n).reshape(-1)
            return idx[:size]
        return rng.choice(n, size=size, replace=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaggingRegressor":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        size = max(2, int(round(self.max_samples * n)))
        self._models = []
        for i in range(self.n_estimators):
            idx = self._draw_indices(rng, n, size)
            model = _base_estimator(self.base, dict(self.base_params), self.random_state + i)
            model.fit(X.iloc[idx].fillna(0.0), y.iloc[idx])
            self._models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            return np.zeros(len(X), dtype=float)
        preds = np.column_stack([m.predict(X.fillna(0.0)) for m in self._models])
        return preds.mean(axis=1)

    def predict_quantiles(self, X: pd.DataFrame, levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> dict[float, np.ndarray]:
        if not self._models:
            return {q: np.zeros(len(X), dtype=float) for q in levels}
        preds = np.column_stack([m.predict(X.fillna(0.0)) for m in self._models])
        return {float(q): np.quantile(preds, float(q), axis=1) for q in levels}


def bagging(
    X: Any,
    y: Any | None = None,
    *,
    base: str = "ridge",
    n_estimators: int = 50,
    max_samples: float = 0.8,
    random_state: int = 0,
    base_params: dict[str, Any] | None = None,
    strategy: str = "standard",
    block_length: int = 4,
) -> ModelFit:
    """Fit a bootstrap-aggregated ensemble."""

    params = {
        "base": str(base),
        "n_estimators": int(n_estimators),
        "max_samples": float(max_samples),
        "random_state": int(random_state),
        "base_params": dict(base_params or {}),
        "strategy": str(strategy),
        "block_length": int(block_length),
    }
    return fit_estimator(
        BaggingRegressor(
            base=str(base),
            n_estimators=int(n_estimators),
            max_samples=float(max_samples),
            random_state=int(random_state),
            base_params=dict(base_params or {}),
            strategy=str(strategy),
            block_length=int(block_length),
        ),
        X,
        y,
        model="bagging",
        metadata=params,
    )


class BoogingRegressor:
    """Bagging of intentionally overfit stochastic gradient boosting models."""

    # Package-native compact implementation of the Booging idea.
    def __init__(
        self,
        *,
        B: int = 100,
        sample_frac: float = 0.75,
        inner_n_estimators: int = 1500,
        inner_learning_rate: float = 0.1,
        inner_max_depth: int = 3,
        inner_subsample: float = 0.5,
        da_noise_frac: float = 1.0 / 3.0,
        da_drop_rate: float = 0.2,
        random_state: int = 0,
    ) -> None:
        self.B = max(1, int(B))
        self.sample_frac = float(np.clip(sample_frac, 0.1, 1.0))
        self.inner_n_estimators = int(inner_n_estimators)
        self.inner_learning_rate = float(inner_learning_rate)
        self.inner_max_depth = int(inner_max_depth)
        self.inner_subsample = float(np.clip(inner_subsample, 0.1, 1.0))
        self.da_noise_frac = float(da_noise_frac)
        self.da_drop_rate = float(np.clip(da_drop_rate, 0.0, 0.95))
        self.random_state = int(random_state)
        self.feature_names_in_: tuple[str, ...] = ()
        self._sigma: np.ndarray | None = None
        self._models: list[tuple[Any, np.ndarray]] = []

    def _augment(self, values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if self._sigma is None:
            return values
        noise = rng.standard_normal(values.shape) * (self._sigma * self.da_noise_frac)
        return np.hstack([values, values + noise])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BoogingRegressor":
        from sklearn.ensemble import GradientBoostingRegressor

        self.feature_names_in_ = tuple(str(c) for c in X.columns)
        values = X.fillna(0.0).to_numpy(dtype=float)
        target = np.asarray(y, dtype=float)
        n, k = values.shape
        if n == 0 or k == 0:
            return self
        self._sigma = np.std(values, axis=0, ddof=0).clip(min=1e-12)
        rng = np.random.default_rng(self.random_state)
        augmented = self._augment(values, rng)
        sample_size = min(n, max(k + 1, int(round(self.sample_frac * n))))
        keep_cols = max(1, int(round((1.0 - self.da_drop_rate) * augmented.shape[1])))
        self._models = []
        for i in range(self.B):
            row_idx = rng.choice(n, sample_size, replace=False)
            col_idx = rng.choice(augmented.shape[1], keep_cols, replace=False)
            model = GradientBoostingRegressor(
                n_estimators=self.inner_n_estimators,
                learning_rate=self.inner_learning_rate,
                max_depth=self.inner_max_depth,
                subsample=self.inner_subsample,
                random_state=self.random_state + i,
            )
            model.fit(augmented[np.ix_(row_idx, col_idx)], target[row_idx])
            self._models.append((model, col_idx))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0)
        rng = np.random.default_rng(self.random_state + 9999)
        augmented = self._augment(frame.fillna(0.0).to_numpy(dtype=float), rng)
        preds: np.ndarray = np.zeros(len(X), dtype=float)
        for model, col_idx in self._models:
            preds += model.predict(augmented[:, col_idx])
        return preds / len(self._models)


def booging(
    X: Any,
    y: Any | None = None,
    *,
    B: int = 100,
    sample_frac: float = 0.75,
    inner_n_estimators: int = 1500,
    inner_learning_rate: float = 0.1,
    inner_max_depth: int = 3,
    inner_subsample: float = 0.5,
    da_noise_frac: float = 1.0 / 3.0,
    da_drop_rate: float = 0.2,
    random_state: int = 0,
) -> ModelFit:
    """Fit Booging: bagged overfit stochastic gradient boosting with augmentation."""

    params = {
        "B": int(B),
        "sample_frac": float(sample_frac),
        "inner_n_estimators": int(inner_n_estimators),
        "inner_learning_rate": float(inner_learning_rate),
        "inner_max_depth": int(inner_max_depth),
        "inner_subsample": float(inner_subsample),
        "da_noise_frac": float(da_noise_frac),
        "da_drop_rate": float(da_drop_rate),
        "random_state": int(random_state),
    }
    return fit_estimator(
        BoogingRegressor(
            B=int(B),
            sample_frac=float(sample_frac),
            inner_n_estimators=int(inner_n_estimators),
            inner_learning_rate=float(inner_learning_rate),
            inner_max_depth=int(inner_max_depth),
            inner_subsample=float(inner_subsample),
            da_noise_frac=float(da_noise_frac),
            da_drop_rate=float(da_drop_rate),
            random_state=int(random_state),
        ),
        X,
        y,
        model="booging",
        metadata=params,
    )


class MacroRandomForestRegressor:
    """Adapter for the vendored MacroRandomForest reference implementation."""

    # Reference backend is vendored from MacroRandomForest 1.0.6 and smoke-tested.
    def __init__(
        self,
        *,
        x_columns: Sequence[str] | None = None,
        S_columns: Sequence[str] | None = None,
        x_pos: Sequence[int] | None = None,
        S_pos: Sequence[int] | None = None,
        y_pos: int = 0,
        B: int = 50,
        minsize: int = 10,
        mtry_frac: float = 1.0 / 3.0,
        min_leaf_frac_of_x: float = 1.0,
        VI: bool = False,
        ERT: bool = False,
        quantile_rate: float | None = None,
        S_priority_vec: Sequence[float] | None = None,
        random_x: bool = False,
        trend_push: int = 1,
        howmany_random_x: int = 1,
        howmany_keep_best_VI: int = 20,
        cheap_look_at_GTVPs: bool = True,
        prior_var: Sequence[float] | None = None,
        prior_mean: Sequence[float] | None = None,
        subsampling_rate: float = 0.75,
        rw_regul: float = 0.75,
        keep_forest: bool = False,
        block_size: int = 12,
        fast_rw: bool = True,
        ridge_lambda: float = 0.1,
        HRW: int = 0,
        resampling_opt: int = 2,
        print_b: bool = False,
        parallelise: bool = False,
        n_cores: int = 1,
        **kwargs: Any,
    ) -> None:
        self.x_columns = None if x_columns is None else tuple(str(column) for column in x_columns)
        self.S_columns = None if S_columns is None else tuple(str(column) for column in S_columns)
        self.x_pos = None if x_pos is None else tuple(int(pos) for pos in x_pos)
        self.S_pos = None if S_pos is None else tuple(int(pos) for pos in S_pos)
        self.y_pos = int(y_pos)
        self.params: dict[str, Any] = {
            "B": int(B),
            "minsize": int(minsize),
            "mtry_frac": float(mtry_frac),
            "min_leaf_frac_of_x": float(min_leaf_frac_of_x),
            "VI": bool(VI),
            "ERT": bool(ERT),
            "quantile_rate": quantile_rate,
            "S_priority_vec": None if S_priority_vec is None else list(S_priority_vec),
            "random_x": bool(random_x),
            "trend_push": int(trend_push),
            "howmany_random_x": int(howmany_random_x),
            "howmany_keep_best_VI": int(howmany_keep_best_VI),
            "cheap_look_at_GTVPs": bool(cheap_look_at_GTVPs),
            "prior_var": [] if prior_var is None else list(prior_var),
            "prior_mean": [] if prior_mean is None else list(prior_mean),
            "subsampling_rate": float(subsampling_rate),
            "rw_regul": float(rw_regul),
            "keep_forest": bool(keep_forest),
            "block_size": int(block_size),
            "fast_rw": bool(fast_rw),
            "ridge_lambda": float(ridge_lambda),
            "HRW": int(HRW),
            "resampling_opt": int(resampling_opt),
            "print_b": bool(print_b),
            "parallelise": bool(parallelise),
            "n_cores": int(n_cores),
            **kwargs,
        }
        self._train_X: pd.DataFrame | None = None
        self._train_y: pd.Series | None = None
        self._feature_names: tuple[str, ...] = ()
        self.output_: dict[str, Any] | None = None
        self.model_: Any = None
        self._prediction_cache_key: tuple[Any, ...] | None = None
        self._prediction_cache_values: np.ndarray | None = None

    @staticmethod
    def _import_external():
        optional_import("joblib", extra="macro_random_forest")
        optional_import("matplotlib", extra="macro_random_forest")
        from macroforecast.models._mrf_reference import MacroRandomForest

        return MacroRandomForest

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MacroRandomForestRegressor":
        self._import_external()
        self._train_X = X.copy()
        self._train_y = y.copy()
        self._feature_names = tuple(str(column) for column in X.columns)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._train_X is None or self._train_y is None:
            return np.zeros(len(X), dtype=float)
        cache_key = self._cache_key(X)
        if (
            self._prediction_cache_key == cache_key
            and self._prediction_cache_values is not None
        ):
            return self._prediction_cache_values.copy()
        MacroRandomForest = self._import_external()
        train_X = self._train_X.copy()
        test_X = X.reindex(columns=list(self._feature_names), fill_value=0.0)
        train_y = self._train_y.rename("__target__")
        test_y = pd.Series(0.0, index=test_X.index, name="__target__")
        data = pd.concat(
            [
                pd.concat([train_y, train_X], axis=1),
                pd.concat([test_y, test_X], axis=1),
            ],
            axis=0,
        ).reset_index(drop=True)
        oos_pos = np.arange(len(train_X), len(train_X) + len(test_X))
        x_pos = self._resolve_positions(self.x_columns, self.x_pos, train_X.columns)
        S_pos = self._resolve_positions(self.S_columns, self.S_pos, train_X.columns)
        with self._reference_output_context():
            model = MacroRandomForest(
                data=data,
                y_pos=self.y_pos,
                x_pos=np.asarray(x_pos, dtype=int),
                S_pos=np.asarray(S_pos, dtype=int),
                oos_pos=oos_pos,
                **self.params,
            )
        self.model_ = model
        try:
            with self._reference_output_context(), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=PendingDeprecationWarning,
                    module=r"macroforecast\.models\._mrf_reference",
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"invalid value encountered in divide",
                    category=RuntimeWarning,
                    module=r"macroforecast\.models\._mrf_reference",
                )
                self.output_ = model._ensemble_loop()
        except Exception as exc:  # noqa: BLE001 - external backend errors need package context.
            raise RuntimeError(
                "MacroRandomForest backend failed while running _ensemble_loop(). "
                "Check x_columns/S_columns and sample size."
            ) from exc
        values = self._prediction_values(self.output_, len(test_X))
        self._prediction_cache_key = cache_key
        self._prediction_cache_values = values.copy()
        return values

    def _reference_output_context(self):
        if self.params.get("print_b"):
            return contextlib.nullcontext()
        return contextlib.redirect_stdout(io.StringIO())

    @staticmethod
    def _resolve_positions(
        columns: Sequence[str] | None,
        positions: Sequence[int] | None,
        feature_index: pd.Index,
    ) -> list[int]:
        if positions is not None:
            return [int(pos) for pos in positions]
        if columns is None:
            return list(range(1, len(feature_index) + 1))
        missing = [column for column in columns if column not in feature_index]
        if missing:
            raise ValueError(f"macro_random_forest columns not found in X: {missing}")
        return [int(feature_index.get_loc(column)) + 1 for column in columns]

    @staticmethod
    def _prediction_values(output: dict[str, Any], n: int) -> np.ndarray:
        values = output.get("pred_ensemble")
        if values is None:
            values = output.get("pred")
        if isinstance(values, (pd.Series, pd.DataFrame)):
            arr = values.to_numpy(dtype=float)
        else:
            arr = np.asarray(values, dtype=float)
        if arr.ndim == 2 and arr.shape[0] != n and arr.shape[1] == n:
            arr = arr.mean(axis=0)
        return np.asarray(arr, dtype=float).reshape(-1)[-n:]

    @staticmethod
    def _cache_key(X: pd.DataFrame) -> tuple[Any, ...]:
        try:
            hashed = pd.util.hash_pandas_object(X, index=True).to_numpy(dtype=np.uint64)
            value_hash = int(np.bitwise_xor.reduce(hashed)) if len(hashed) else 0
        except Exception:  # noqa: BLE001 - cache keys must never block prediction.
            value_hash = id(X)
        return (tuple(X.index), tuple(str(column) for column in X.columns), X.shape, value_hash)


def macro_random_forest(
    X: Any,
    y: Any | None = None,
    *,
    x_columns: Sequence[str] | None = None,
    S_columns: Sequence[str] | None = None,
    x_pos: Sequence[int] | None = None,
    S_pos: Sequence[int] | None = None,
    y_pos: int = 0,
    B: int = 50,
    minsize: int = 10,
    mtry_frac: float = 1.0 / 3.0,
    min_leaf_frac_of_x: float = 1.0,
    VI: bool = False,
    ERT: bool = False,
    quantile_rate: float | None = None,
    S_priority_vec: Sequence[float] | None = None,
    random_x: bool = False,
    trend_push: int = 1,
    howmany_random_x: int = 1,
    howmany_keep_best_VI: int = 20,
    cheap_look_at_GTVPs: bool = True,
    prior_var: Sequence[float] | None = None,
    prior_mean: Sequence[float] | None = None,
    subsampling_rate: float = 0.75,
    rw_regul: float = 0.75,
    keep_forest: bool = False,
    block_size: int = 12,
    fast_rw: bool = True,
    ridge_lambda: float = 0.1,
    HRW: int = 0,
    resampling_opt: int = 2,
    print_b: bool = False,
    parallelise: bool = False,
    n_cores: int = 1,
    **kwargs: Any,
) -> ModelFit:
    """Fit Macroeconomic Random Forest with the vendored reference backend."""

    params = {
        "x_columns": x_columns,
        "S_columns": S_columns,
        "x_pos": x_pos,
        "S_pos": S_pos,
        "y_pos": int(y_pos),
        "B": int(B),
        "minsize": int(minsize),
        "mtry_frac": float(mtry_frac),
        "min_leaf_frac_of_x": float(min_leaf_frac_of_x),
        "VI": bool(VI),
        "ERT": bool(ERT),
        "quantile_rate": quantile_rate,
        "S_priority_vec": S_priority_vec,
        "random_x": bool(random_x),
        "trend_push": int(trend_push),
        "howmany_random_x": int(howmany_random_x),
        "howmany_keep_best_VI": int(howmany_keep_best_VI),
        "cheap_look_at_GTVPs": bool(cheap_look_at_GTVPs),
        "prior_var": prior_var,
        "prior_mean": prior_mean,
        "subsampling_rate": float(subsampling_rate),
        "rw_regul": float(rw_regul),
        "keep_forest": bool(keep_forest),
        "block_size": int(block_size),
        "fast_rw": bool(fast_rw),
        "ridge_lambda": float(ridge_lambda),
        "HRW": int(HRW),
        "resampling_opt": int(resampling_opt),
        "print_b": bool(print_b),
        "parallelise": bool(parallelise),
        "n_cores": int(n_cores),
        **kwargs,
    }
    estimator = MacroRandomForestRegressor(**params)
    metadata = {
        "x_columns": estimator.x_columns,
        "S_columns": estimator.S_columns,
        "x_pos": estimator.x_pos,
        "S_pos": estimator.S_pos,
        "y_pos": estimator.y_pos,
        **estimator.params,
    }
    return fit_estimator(
        estimator,
        X,
        y,
        model="macro_random_forest",
        metadata=metadata,
        collect_diagnostics=False,
    )


__all__ = [
    "BaggingRegressor",
    "BoogingRegressor",
    "MacroRandomForestRegressor",
    "QuantileRegressionForestRegressor",
    "SlowGrowingTreeRegressor",
    "bagging",
    "booging",
    "catboost",
    "decision_tree",
    "extra_trees",
    "gradient_boosting",
    "lightgbm",
    "macro_random_forest",
    "quantile_regression_forest",
    "random_forest",
    "slow_growing_tree",
    "xgboost",
]
