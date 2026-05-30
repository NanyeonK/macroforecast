from __future__ import annotations

from typing import Any

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
) -> ModelFit:
    """Fit a CART regression tree."""

    from sklearn.tree import DecisionTreeRegressor

    return fit_estimator(
        DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
        ),
        X,
        y,
        model="decision_tree",
        metadata={"max_depth": max_depth, "min_samples_leaf": int(min_samples_leaf), "random_state": int(random_state)},
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
) -> ModelFit:
    """Fit a random forest regressor."""

    from sklearn.ensemble import RandomForestRegressor

    return fit_estimator(
        RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=n_jobs,
        ),
        X,
        y,
        model="random_forest",
        metadata={"n_estimators": int(n_estimators), "max_depth": max_depth, "random_state": int(random_state)},
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
) -> ModelFit:
    """Fit an extremely randomized trees regressor."""

    from sklearn.ensemble import ExtraTreesRegressor

    return fit_estimator(
        ExtraTreesRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=n_jobs,
        ),
        X,
        y,
        model="extra_trees",
        metadata={"n_estimators": int(n_estimators), "max_depth": max_depth, "random_state": int(random_state)},
    )


def gradient_boosting(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    random_state: int = 0,
) -> ModelFit:
    """Fit sklearn gradient-boosted regression trees."""

    from sklearn.ensemble import GradientBoostingRegressor

    return fit_estimator(
        GradientBoostingRegressor(
            n_estimators=int(n_estimators),
            learning_rate=float(learning_rate),
            max_depth=int(max_depth),
            random_state=int(random_state),
        ),
        X,
        y,
        model="gradient_boosting",
        metadata={"n_estimators": int(n_estimators), "learning_rate": float(learning_rate), "max_depth": int(max_depth)},
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
    estimator = xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        subsample=float(subsample),
        random_state=int(random_state),
        objective="reg:squarederror",
        **kwargs,
    )
    return fit_estimator(estimator, X, y, model="xgboost")


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
    estimator = lgb.LGBMRegressor(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        num_leaves=int(num_leaves),
        random_state=int(random_state),
        **kwargs,
    )
    return fit_estimator(estimator, X, y, model="lightgbm")


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
    estimator = cb.CatBoostRegressor(
        iterations=int(n_estimators),
        learning_rate=float(learning_rate),
        depth=int(max_depth),
        random_seed=int(random_state),
        verbose=verbose,
        **kwargs,
    )
    return fit_estimator(estimator, X, y, model="catboost")


def mars(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit multivariate adaptive regression splines. Requires pyearth."""

    pyearth = optional_import("pyearth", extra="mars", package="sklearn-contrib-py-earth")
    return fit_estimator(pyearth.Earth(**kwargs), X, y, model="mars")


class SlowGrowingTreeRegressor:
    """Slow-Growing Tree with soft split propagation."""

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


def slow_growing_tree(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit a Slow-Growing Tree."""

    return fit_estimator(SlowGrowingTreeRegressor(**kwargs), X, y, model="slow_growing_tree", metadata=kwargs)


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
        self._forest = None
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
        out = {q: np.empty(len(X), dtype=float) for q in levels}
        for i in range(len(X)):
            samples: list[float] = []
            for tree_idx in range(leaves.shape[1]):
                values = self._leaf_targets[tree_idx].get(int(leaves[i, tree_idx]), np.array([]))
                samples.extend(np.asarray(values, dtype=float).tolist())
            arr = np.asarray(samples if samples else [0.0], dtype=float)
            for q in levels:
                out[q][i] = float(np.quantile(arr, q))
        return out


def quantile_regression_forest(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit a quantile regression forest."""

    return fit_estimator(
        QuantileRegressionForestRegressor(**kwargs),
        X,
        y,
        model="quantile_regression_forest",
        metadata=kwargs,
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


def bagging(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit a bootstrap-aggregated ensemble."""

    return fit_estimator(BaggingRegressor(**kwargs), X, y, model="bagging", metadata=kwargs)


class BoogingRegressor:
    """Bagging of intentionally overfit stochastic gradient boosting models."""

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
        preds = np.zeros(len(X), dtype=float)
        for model, col_idx in self._models:
            preds += model.predict(augmented[:, col_idx])
        return preds / len(self._models)


def booging(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit Booging: bagged overfit stochastic gradient boosting with augmentation."""

    return fit_estimator(BoogingRegressor(**kwargs), X, y, model="booging", metadata=kwargs)


class MacroRandomForestRegressor:
    """Adapter for the external MacroRandomForest reference implementation."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self._train_X: pd.DataFrame | None = None
        self._train_y: pd.Series | None = None

    @staticmethod
    def _import_external():
        try:
            from macroforecast._vendor.macro_random_forest import MacroRandomForest

            return MacroRandomForest
        except ImportError as exc:
            raise ImportError(
                "macro_random_forest reference code is not bundled in this clean package yet. "
                "`mf.models.macro_random_forest` is reserved for the Goulet Coulombe "
                "Macroeconomic Random Forest implementation and will run once that "
                "reference backend is added."
            ) from exc

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MacroRandomForestRegressor":
        self._import_external()
        self._train_X = X.copy()
        self._train_y = y.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._import_external()
        raise NotImplementedError("macro_random_forest backend adapter is not wired in this clean package yet")


def macro_random_forest(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit:
    """Fit Macroeconomic Random Forest when the reference backend is available."""

    return fit_estimator(MacroRandomForestRegressor(**kwargs), X, y, model="macro_random_forest", metadata=kwargs)


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
    "mars",
    "quantile_regression_forest",
    "random_forest",
    "slow_growing_tree",
    "xgboost",
]
