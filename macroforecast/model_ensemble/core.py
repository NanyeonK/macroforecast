from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.models.types import ModelFit
from macroforecast.models.utils import (
    binary_feature_mask,
    fit_estimator,
    resolve_feature_count,
)

MODEL_ENSEMBLE_BASE_ESTIMATORS: dict[str, str] = {
    "ols": "sklearn.linear_model.LinearRegression",
    "ridge": "sklearn.linear_model.Ridge",
    "lasso": "sklearn.linear_model.Lasso",
    "elastic_net": "sklearn.linear_model.ElasticNet",
    "decision_tree": "sklearn.tree.DecisionTreeRegressor",
    "random_forest": "sklearn.ensemble.RandomForestRegressor",
    "extra_trees": "sklearn.ensemble.ExtraTreesRegressor",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingRegressor",
    "knn": "sklearn.neighbors.KNeighborsRegressor",
    "svr": "sklearn.svm.SVR",
}


def _validate_quantile_levels(levels: Sequence[float]) -> tuple[float, ...]:
    quantiles = tuple(float(level) for level in levels)
    if not quantiles or any(level <= 0.0 or level >= 1.0 for level in quantiles):
        raise ValueError("quantile levels must contain values in (0, 1)")
    return quantiles


def _base_estimator(name: str, params: dict[str, Any], random_state: int):
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor

    common = {"random_state": random_state}
    if name == "ols":
        return LinearRegression(**params)
    if name == "ridge":
        return Ridge(**params)
    if name == "lasso":
        return Lasso(**{"max_iter": 20000, **params})
    if name == "elastic_net":
        return ElasticNet(**{"max_iter": 20000, **params})
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
    raise ValueError(f"unknown model_ensemble base estimator: {name!r}")


def list_model_ensemble_bases() -> pd.DataFrame:
    """Return supported inner estimators for fit-time model ensembles."""

    return pd.DataFrame(
        [
            {
                "name": name,
                "backend": backend,
                "notes": "Available through the `base`, `models`, or `meta_model` options.",
            }
            for name, backend in MODEL_ENSEMBLE_BASE_ESTIMATORS.items()
        ]
    )


def _base_predictions(models: Sequence[Any], X: pd.DataFrame) -> np.ndarray:
    if not models:
        return np.zeros((len(X), 0), dtype=float)
    return np.column_stack(
        [
            np.asarray(model.predict(X.fillna(0.0)), dtype=float).reshape(-1)
            for model in models
        ]
    )


def _member_predictions(
    models: Sequence[tuple[Any, tuple[str, ...]]], X: pd.DataFrame
) -> np.ndarray:
    if not models:
        return np.zeros((len(X), 0), dtype=float)
    return np.column_stack(
        [
            np.asarray(
                model.predict(X.loc[:, list(columns)].fillna(0.0)), dtype=float
            ).reshape(-1)
            for model, columns in models
        ]
    )


def _folds(n: int, n_splits: int, splitter: str) -> list[tuple[np.ndarray, np.ndarray]]:
    if n < 3:
        raise ValueError("stacking/super_learner requires at least 3 observations")
    n_splits = max(2, min(int(n_splits), n - 1))
    index = np.arange(n)
    key = str(splitter)
    if key == "forward":
        chunks = [
            chunk for chunk in np.array_split(index, n_splits + 1)[1:] if len(chunk)
        ]
        folds = []
        for val_idx in chunks:
            train_idx = index[index < int(val_idx[0])]
            if len(train_idx):
                folds.append((train_idx, val_idx))
        if folds:
            return folds
        key = "blocked"
    if key in {"blocked", "kfold"}:
        folds = []
        for val_idx in np.array_split(index, n_splits):
            if len(val_idx) == 0:
                continue
            train_idx = np.setdiff1d(index, val_idx, assume_unique=True)
            if len(train_idx):
                folds.append((train_idx, val_idx))
        if folds:
            return folds
    raise ValueError("splitter must be 'forward', 'blocked', or 'kfold'")


def _fold_metadata(
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    index: pd.Index,
) -> pd.DataFrame:
    rows = []
    for fold_id, (train_idx, validation_idx) in enumerate(folds):
        rows.append(
            {
                "fold": fold_id,
                "n_train": int(len(train_idx)),
                "n_validation": int(len(validation_idx)),
                "train_start": index[int(train_idx[0])] if len(train_idx) else None,
                "train_end": index[int(train_idx[-1])] if len(train_idx) else None,
                "validation_start": (
                    index[int(validation_idx[0])] if len(validation_idx) else None
                ),
                "validation_end": (
                    index[int(validation_idx[-1])] if len(validation_idx) else None
                ),
            }
        )
    return pd.DataFrame(rows)


class BaggingRegressor:
    """Bootstrap or block-bootstrap ensemble over supported base estimators."""

    # R alignment, checked against CRAN ipred::bagging / ipredbagg docs:
    # https://search.r-project.org/CRAN/refmans/ipred/html/bagging.html
    # - ipred::bagging/ipredbagg draws nbagg bootstrap samples and stores one
    #   fitted tree per draw; prediction averages/votes across member trees.
    # - ipredbagg(ns < n) switches to subagging, i.e. sampling without
    #   replacement. This class exposes the same separation through replace=...
    #   and subagging() below.
    # - macroforecast generalizes the base learner beyond rpart-style trees,
    #   because this module owns fit-time model composition, not a tree backend.
    #   Row resampling and member aggregation are package-native; inner learners
    #   are sklearn-compatible estimators.

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
        replace: bool = True,
        max_features: float | int | str | None = None,
    ) -> None:
        if str(strategy) not in {"standard", "block"}:
            raise ValueError("strategy must be 'standard' or 'block'")
        self.base = str(base)
        self.n_estimators = max(1, int(n_estimators))
        self.max_samples = float(np.clip(max_samples, 0.05, 1.0))
        self.random_state = int(random_state)
        self.base_params = dict(base_params or {})
        self.strategy = str(strategy)
        self.block_length = max(1, int(block_length))
        self.replace = bool(replace)
        self.max_features = max_features
        self.feature_names_in_: tuple[str, ...] = ()
        self._models: list[tuple[Any, tuple[str, ...]]] = []
        self.member_indices_: list[np.ndarray] = []
        self.member_samples_: pd.DataFrame | None = None
        self.member_features_: pd.DataFrame | None = None
        self.oob_predictions_: pd.Series | None = None
        self.oob_residuals_: pd.Series | None = None
        self.oob_metrics_: dict[str, float | int] | None = None

    def _draw_indices(self, rng: np.random.Generator, n: int, size: int) -> np.ndarray:
        if self.strategy == "block":
            n_blocks = (size + self.block_length - 1) // self.block_length
            starts = rng.integers(0, n, size=n_blocks)
            idx = (
                (starts[:, None] + np.arange(self.block_length)[None, :]) % n
            ).reshape(-1)
            return idx[:size]
        return rng.choice(n, size=size, replace=self.replace)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaggingRegressor":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        n_features = resolve_feature_count(
            self.max_features, len(self.feature_names_in_)
        )
        size = max(2, int(round(self.max_samples * n)))
        if not self.replace:
            size = min(size, n)
        self._models = []
        self.member_indices_ = []
        member_pred = np.full((n, self.n_estimators), np.nan, dtype=float)
        member_rows = []
        feature_rows = []
        for i in range(self.n_estimators):
            idx = self._draw_indices(rng, n, size)
            columns = tuple(
                rng.choice(self.feature_names_in_, size=n_features, replace=False)
            )
            model = _base_estimator(
                self.base, dict(self.base_params), self.random_state + i
            )
            model.fit(X.loc[:, list(columns)].iloc[idx].fillna(0.0), y.iloc[idx])
            self._models.append((model, columns))
            self.member_indices_.append(idx.copy())
            used = np.zeros(n, dtype=bool)
            used[np.unique(idx)] = True
            oob_idx = np.flatnonzero(~used)
            member_rows.append(
                {
                    "member": i,
                    "n_rows": int(len(idx)),
                    "n_unique_rows": int(len(np.unique(idx))),
                    "n_oob_rows": int(len(oob_idx)),
                    "strategy": self.strategy,
                    "replace": self.replace,
                    "block_length": self.block_length,
                }
            )
            feature_rows.append(
                {
                    "member": i,
                    "n_features": int(len(columns)),
                    "features": columns,
                    "max_features": self.max_features
                    if self.max_features is not None
                    else "all",
                }
            )
            if len(oob_idx):
                member_pred[oob_idx, i] = np.asarray(
                    model.predict(X.loc[:, list(columns)].iloc[oob_idx].fillna(0.0)),
                    dtype=float,
                ).reshape(-1)
        self.member_samples_ = pd.DataFrame(member_rows)
        self.member_features_ = pd.DataFrame(feature_rows)
        self._set_oob_diagnostics(member_pred, y)
        return self

    def _set_oob_diagnostics(self, member_pred: np.ndarray, y: pd.Series) -> None:
        counts = np.sum(~np.isnan(member_pred), axis=1)
        has_oob = counts > 0
        if not bool(has_oob.any()):
            self.oob_predictions_ = None
            self.oob_residuals_ = None
            self.oob_metrics_ = None
            return
        sums = np.nansum(member_pred, axis=1)
        pred = np.full(len(y), np.nan, dtype=float)
        pred[has_oob] = sums[has_oob] / counts[has_oob]
        self.oob_predictions_ = pd.Series(pred, index=y.index, name="oob_prediction")
        residual = y.astype(float) - self.oob_predictions_
        self.oob_residuals_ = residual.rename("oob_residual")
        valid_resid = residual.dropna().to_numpy(dtype=float)
        self.oob_metrics_ = {
            "n": int(len(valid_resid)),
            "coverage": float(has_oob.mean()),
            "mae": float(np.mean(np.abs(valid_resid))) if len(valid_resid) else np.nan,
            "mse": float(np.mean(valid_resid**2)) if len(valid_resid) else np.nan,
            "rmse": float(np.sqrt(np.mean(valid_resid**2)))
            if len(valid_resid)
            else np.nan,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            return np.zeros(len(X), dtype=float)
        return _member_predictions(self._models, X).mean(axis=1)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> dict[float, np.ndarray]:
        levels = _validate_quantile_levels(levels)
        if not self._models:
            return {q: np.zeros(len(X), dtype=float) for q in levels}
        preds = _member_predictions(self._models, X)
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
    replace: bool = True,
    max_features: float | int | str | None = None,
) -> ModelFit:
    """Fit a bootstrap-aggregated fit-time model ensemble."""

    params = {
        "base": str(base),
        "n_estimators": int(n_estimators),
        "max_samples": float(max_samples),
        "random_state": int(random_state),
        "base_params": dict(base_params or {}),
        "strategy": str(strategy),
        "block_length": int(block_length),
        "replace": bool(replace),
        "max_features": max_features if max_features is not None else "all",
        "implementation_note": (
            "ipred::bagging/ipredbagg-style member resampling generalized to "
            "sklearn-compatible base estimators with optional member-level "
            "feature-subspace perturbation."
        ),
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
            replace=bool(replace),
            max_features=max_features,
        ),
        X,
        y,
        model="bagging",
        metadata=params,
    )


def subagging(
    X: Any,
    y: Any | None = None,
    *,
    base: str = "ridge",
    n_estimators: int = 50,
    max_samples: float = 0.632,
    random_state: int = 0,
    base_params: dict[str, Any] | None = None,
    max_features: float | int | str | None = None,
) -> ModelFit:
    """Fit subagging: sampling without replacement before member fits."""

    params = {
        "base": str(base),
        "n_estimators": int(n_estimators),
        "max_samples": float(max_samples),
        "random_state": int(random_state),
        "base_params": dict(base_params or {}),
        "strategy": "standard",
        "replace": False,
        "max_features": max_features if max_features is not None else "all",
        "implementation_note": (
            "ipredbagg(ns < n)-style sampling without replacement generalized to "
            "sklearn-compatible base estimators with optional member-level "
            "feature-subspace perturbation."
        ),
    }
    return fit_estimator(
        BaggingRegressor(
            base=str(base),
            n_estimators=int(n_estimators),
            max_samples=float(max_samples),
            random_state=int(random_state),
            base_params=dict(base_params or {}),
            strategy="standard",
            replace=False,
            max_features=max_features,
        ),
        X,
        y,
        model="subagging",
        metadata=params,
    )


class RandomSubspaceRegressor:
    """Fit base models on randomly selected feature subsets and average them."""

    # R alignment, checked against CRAN regRSM::regRSM docs:
    # https://search.r-project.org/CRAN/refmans/regRSM/html/regRSM.html
    # - regRSM::regRSM repeatedly draws random predictor subspaces of size m and
    #   uses those draws to build regression/importance summaries.
    # - randomForest/ranger expose related random-subspace behavior through mtry,
    #   but only inside tree split search. This class implements Ho-style
    #   member-level feature bagging for any supported base estimator.

    def __init__(
        self,
        *,
        base: str = "ridge",
        n_estimators: int = 100,
        max_features: float | int | str = 0.5,
        max_samples: float = 1.0,
        random_state: int = 0,
        base_params: dict[str, Any] | None = None,
    ) -> None:
        self.base = str(base)
        self.n_estimators = max(1, int(n_estimators))
        self.max_features = max_features
        self.max_samples = float(np.clip(max_samples, 0.05, 1.0))
        self.random_state = int(random_state)
        self.base_params = dict(base_params or {})
        self.feature_names_in_: tuple[str, ...] = ()
        self._models: list[tuple[Any, tuple[str, ...]]] = []
        self.member_features_: pd.DataFrame | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomSubspaceRegressor":
        rng = np.random.default_rng(self.random_state)
        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        p = len(self.feature_names_in_)
        n_features = resolve_feature_count(self.max_features, p)
        n = len(X)
        n_rows = max(2, min(n, int(round(self.max_samples * n))))
        self._models = []
        member_rows = []
        for i in range(self.n_estimators):
            columns = tuple(
                rng.choice(self.feature_names_in_, size=n_features, replace=False)
            )
            rows = rng.choice(n, size=n_rows, replace=False)
            model = _base_estimator(
                self.base, dict(self.base_params), self.random_state + i
            )
            model.fit(X.loc[:, list(columns)].iloc[rows].fillna(0.0), y.iloc[rows])
            self._models.append((model, columns))
            member_rows.append(
                {
                    "member": i,
                    "n_features": len(columns),
                    "features": columns,
                    "n_rows": len(rows),
                }
            )
        self.member_features_ = pd.DataFrame(member_rows)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0)
        preds = [
            np.asarray(
                model.predict(frame.loc[:, list(columns)].fillna(0.0)), dtype=float
            )
            for model, columns in self._models
        ]
        return np.column_stack(preds).mean(axis=1)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> dict[float, np.ndarray]:
        levels = _validate_quantile_levels(levels)
        if not self._models:
            return {q: np.zeros(len(X), dtype=float) for q in levels}
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0)
        preds = [
            np.asarray(
                model.predict(frame.loc[:, list(columns)].fillna(0.0)), dtype=float
            )
            for model, columns in self._models
        ]
        matrix = np.column_stack(preds)
        return {float(q): np.quantile(matrix, float(q), axis=1) for q in levels}


def random_subspace(
    X: Any,
    y: Any | None = None,
    *,
    base: str = "ridge",
    n_estimators: int = 100,
    max_features: float | int | str = 0.5,
    max_samples: float = 1.0,
    random_state: int = 0,
    base_params: dict[str, Any] | None = None,
) -> ModelFit:
    """Fit a random-subspace model ensemble."""

    params = {
        "base": str(base),
        "n_estimators": int(n_estimators),
        "max_features": max_features,
        "max_samples": float(max_samples),
        "random_state": int(random_state),
        "base_params": dict(base_params or {}),
        "implementation_note": (
            "regRSM/random-subspace-style member feature draws generalized to "
            "supported base estimators."
        ),
    }
    return fit_estimator(
        RandomSubspaceRegressor(
            base=str(base),
            n_estimators=int(n_estimators),
            max_features=max_features,
            max_samples=float(max_samples),
            random_state=int(random_state),
            base_params=dict(base_params or {}),
        ),
        X,
        y,
        model="random_subspace",
        metadata=params,
    )


class StackingRegressor:
    """Out-of-fold stacking over supported base estimators."""

    # R alignment, checked against CRAN caretEnsemble::caretStack docs:
    # https://search.r-project.org/CRAN/refmans/caretEnsemble/html/caretStack.html
    # - caretEnsemble::caretStack fits a meta model on out-of-fold predictions
    #   from a caretList. SuperLearner uses the same core idea with a constrained
    #   weight optimizer.
    # - This class owns the stack construction directly for pandas X/y. It keeps
    #   macroforecast's temporal option explicit through splitter="forward",
    #   while still supporting blocked/kfold splits for R-style V-fold behavior.

    def __init__(
        self,
        *,
        models: Sequence[str] = ("ridge", "lasso", "random_forest"),
        meta_model: str = "ridge",
        n_splits: int = 5,
        splitter: str = "forward",
        random_state: int = 0,
        model_params: dict[str, dict[str, Any]] | None = None,
        meta_params: dict[str, Any] | None = None,
        passthrough: bool = False,
    ) -> None:
        if len(models) < 1:
            raise ValueError("models must contain at least one base model")
        if len(set(str(model) for model in models)) != len(models):
            raise ValueError("models must contain unique names")
        self.models = tuple(str(model) for model in models)
        self.meta_model = str(meta_model)
        self.n_splits = max(2, int(n_splits))
        self.splitter = str(splitter)
        self.random_state = int(random_state)
        self.model_params = {str(k): dict(v) for k, v in (model_params or {}).items()}
        self.meta_params = dict(meta_params or {})
        self.passthrough = bool(passthrough)
        self.feature_names_in_: tuple[str, ...] = ()
        self._models: list[Any] = []
        self._meta: Any = None
        self.oof_predictions_: pd.DataFrame | None = None
        self.folds_: pd.DataFrame | None = None
        self.meta_training_index_: tuple[Any, ...] = ()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StackingRegressor":
        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        frame = X.fillna(0.0)
        oof = np.full((len(frame), len(self.models)), np.nan, dtype=float)
        folds = _folds(len(frame), self.n_splits, self.splitter)
        self.folds_ = _fold_metadata(folds, frame.index)
        for train_idx, val_idx in folds:
            for j, name in enumerate(self.models):
                params = dict(self.model_params.get(name, {}))
                model = _base_estimator(name, params, self.random_state + 1000 * j)
                model.fit(frame.iloc[train_idx], y.iloc[train_idx])
                oof[val_idx, j] = np.asarray(
                    model.predict(frame.iloc[val_idx]), dtype=float
                )
        valid = ~np.isnan(oof).any(axis=1)
        if not valid.any():
            raise ValueError("stacking could not produce any out-of-fold predictions")
        meta_X = pd.DataFrame(oof[valid], index=frame.index[valid], columns=self.models)
        if self.passthrough:
            meta_X = pd.concat([meta_X, frame.loc[meta_X.index]], axis=1)
        self._meta = _base_estimator(
            self.meta_model, dict(self.meta_params), self.random_state
        )
        self._meta.fit(meta_X, y.loc[meta_X.index])
        self.oof_predictions_ = pd.DataFrame(
            oof, index=frame.index, columns=self.models
        )
        self.meta_training_index_ = tuple(meta_X.index)
        self._models = []
        for j, name in enumerate(self.models):
            params = dict(self.model_params.get(name, {}))
            model = _base_estimator(name, params, self.random_state + 2000 * j)
            model.fit(frame, y)
            self._models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._meta is None or not self._models:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).fillna(
            0.0
        )
        meta_X = pd.DataFrame(
            _base_predictions(self._models, frame),
            index=frame.index,
            columns=self.models,
        )
        if self.passthrough:
            meta_X = pd.concat([meta_X, frame], axis=1)
        return np.asarray(self._meta.predict(meta_X), dtype=float).reshape(-1)


def stacking(
    X: Any,
    y: Any | None = None,
    *,
    models: Sequence[str] = ("ridge", "lasso", "random_forest"),
    meta_model: str = "ridge",
    n_splits: int = 5,
    splitter: str = "forward",
    random_state: int = 0,
    model_params: dict[str, dict[str, Any]] | None = None,
    meta_params: dict[str, Any] | None = None,
    passthrough: bool = False,
) -> ModelFit:
    """Fit an out-of-fold stacked model ensemble."""

    params = {
        "models": tuple(str(model) for model in models),
        "meta_model": str(meta_model),
        "n_splits": int(n_splits),
        "splitter": str(splitter),
        "random_state": int(random_state),
        "model_params": {str(k): dict(v) for k, v in (model_params or {}).items()},
        "meta_params": dict(meta_params or {}),
        "passthrough": bool(passthrough),
        "implementation_note": (
            "caretEnsemble::caretStack-style OOF prediction stack with "
            "macroforecast temporal split options."
        ),
    }
    return fit_estimator(
        StackingRegressor(
            models=models,
            meta_model=str(meta_model),
            n_splits=int(n_splits),
            splitter=str(splitter),
            random_state=int(random_state),
            model_params=model_params,
            meta_params=meta_params,
            passthrough=bool(passthrough),
        ),
        X,
        y,
        model="stacking",
        metadata=params,
    )


class SuperLearnerRegressor:
    """Convex-weight Super Learner over supported base estimators."""

    # R alignment, checked against CRAN SuperLearner::SuperLearner docs:
    # https://search.r-project.org/CRAN/refmans/SuperLearner/html/SuperLearner.html
    # - SuperLearner::SuperLearner estimates cross-validated risk for a library
    #   of learners, then forms a nonnegative weighted average that sums to one;
    #   it can also report the discrete best learner.
    # - This implementation matches that regression skeleton with OOF
    #   predictions and NNLS/equal/best weights. It is not an R wrapper and does
    #   not reproduce every SuperLearner family/loss/plugin option.

    def __init__(
        self,
        *,
        models: Sequence[str] = ("ridge", "lasso", "random_forest"),
        n_splits: int = 5,
        splitter: str = "forward",
        weight_method: str = "nnls",
        random_state: int = 0,
        model_params: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if len(models) < 1:
            raise ValueError("models must contain at least one base model")
        if len(set(str(model) for model in models)) != len(models):
            raise ValueError("models must contain unique names")
        if str(weight_method) not in {"nnls", "equal", "best"}:
            raise ValueError("weight_method must be 'nnls', 'equal', or 'best'")
        self.models = tuple(str(model) for model in models)
        self.n_splits = max(2, int(n_splits))
        self.splitter = str(splitter)
        self.weight_method = str(weight_method)
        self.random_state = int(random_state)
        self.model_params = {str(k): dict(v) for k, v in (model_params or {}).items()}
        self.feature_names_in_: tuple[str, ...] = ()
        self.weights_: pd.Series | None = None
        self.oof_predictions_: pd.DataFrame | None = None
        self.oof_risk_: pd.Series | None = None
        self.folds_: pd.DataFrame | None = None
        self._models: list[Any] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SuperLearnerRegressor":
        from scipy.optimize import nnls

        self.feature_names_in_ = tuple(str(column) for column in X.columns)
        frame = X.fillna(0.0)
        oof = np.full((len(frame), len(self.models)), np.nan, dtype=float)
        folds = _folds(len(frame), self.n_splits, self.splitter)
        self.folds_ = _fold_metadata(folds, frame.index)
        for train_idx, val_idx in folds:
            for j, name in enumerate(self.models):
                params = dict(self.model_params.get(name, {}))
                model = _base_estimator(name, params, self.random_state + 1000 * j)
                model.fit(frame.iloc[train_idx], y.iloc[train_idx])
                oof[val_idx, j] = np.asarray(
                    model.predict(frame.iloc[val_idx]), dtype=float
                )
        valid = ~np.isnan(oof).any(axis=1)
        if not valid.any():
            raise ValueError(
                "super_learner could not produce any out-of-fold predictions"
            )
        library = oof[valid]
        target = y.iloc[np.flatnonzero(valid)].to_numpy(dtype=float)
        losses = np.mean((library - target[:, None]) ** 2, axis=0)
        if self.weight_method == "equal":
            weights = np.full(len(self.models), 1.0 / len(self.models), dtype=float)
        elif self.weight_method == "best":
            weights = np.zeros(len(self.models), dtype=float)
            weights[int(np.argmin(losses))] = 1.0
        else:
            weights, _ = nnls(library, target)
            total = float(weights.sum())
            if total <= 0.0:
                weights = np.full(len(self.models), 1.0 / len(self.models), dtype=float)
            else:
                weights = weights / total
        self.weights_ = pd.Series(weights, index=self.models, name="weight")
        self.oof_risk_ = pd.Series(losses, index=self.models, name="oof_mse")
        self.oof_predictions_ = pd.DataFrame(
            oof, index=frame.index, columns=self.models
        )
        self._models = []
        for j, name in enumerate(self.models):
            params = dict(self.model_params.get(name, {}))
            model = _base_estimator(name, params, self.random_state + 2000 * j)
            model.fit(frame, y)
            self._models.append(model)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.weights_ is None or not self._models:
            return np.zeros(len(X), dtype=float)
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0).fillna(
            0.0
        )
        preds = _base_predictions(self._models, frame)
        return preds @ self.weights_.to_numpy(dtype=float)


def super_learner(
    X: Any,
    y: Any | None = None,
    *,
    models: Sequence[str] = ("ridge", "lasso", "random_forest"),
    n_splits: int = 5,
    splitter: str = "forward",
    weight_method: str = "nnls",
    random_state: int = 0,
    model_params: dict[str, dict[str, Any]] | None = None,
) -> ModelFit:
    """Fit a SuperLearner-style convex-weight model ensemble."""

    params = {
        "models": tuple(str(model) for model in models),
        "n_splits": int(n_splits),
        "splitter": str(splitter),
        "weight_method": str(weight_method),
        "random_state": int(random_state),
        "model_params": {str(k): dict(v) for k, v in (model_params or {}).items()},
        "implementation_note": (
            "SuperLearner::SuperLearner-style OOF library risk and convex "
            "weighted average for regression."
        ),
    }
    return fit_estimator(
        SuperLearnerRegressor(
            models=models,
            n_splits=int(n_splits),
            splitter=str(splitter),
            weight_method=str(weight_method),
            random_state=int(random_state),
            model_params=model_params,
        ),
        X,
        y,
        model="super_learner",
        metadata=params,
    )


class BoogingRegressor:
    """Bagging of intentionally overfit stochastic gradient boosting models."""

    # Source comparison, checked against Goulet Coulombe, To Bag is to Prune:
    # https://arxiv.org/abs/2008.07063
    # Local R source:
    # bagofprunes/R/PGC_Bag_of_Prunes_v200829.R, Booging(y, X, X.new, ...)
    # - The R function samples rows with sampling.rate, samples columns with
    #   mtry, fits an intentionally overfit gbm::gbm member, then averages B
    #   member predictions. The macroforecast class keeps that randomized
    #   greedy-ensemble structure while using sklearn GradientBoostingRegressor
    #   as the boosting backend.
    # - R defaults are B=100, mtry=0.8, sampling.rate=.75, data.aug=FALSE,
    #   noise.level=0.3, shuffle.rate=0.2, bf=.5, n.trees=1000, tree.depth=3,
    #   nu=.3. The public booging() aliases expose those names.
    # - When data_aug=True, the R code appends two fake feature copies. Continuous
    #   variables get Gaussian perturbations; binary variables are perturbed by
    #   shuffling a subset of rows. Prediction uses deterministic copies of X.new,
    #   so forecast calls do not draw new noise.
    # - R scales continuous X and X.new jointly before fitting. A normal Python
    #   estimator cannot see X.new at fit time, so macroforecast uses train-only
    #   scaling for leakage-safe fit/predict semantics. The perturbation logic and
    #   member aggregation remain aligned with the R algorithm.
    # - This belongs in model_ensemble rather than models because the public
    #   operation is fit-time composition of many inner boosting models.

    def __init__(
        self,
        *,
        B: int = 100,
        sample_frac: float = 0.75,
        inner_n_estimators: int = 1000,
        inner_learning_rate: float = 0.3,
        inner_max_depth: int = 3,
        inner_subsample: float = 0.5,
        mtry: float | int | str | None = None,
        data_aug: bool = False,
        noise_level: float = 0.3,
        shuffle_rate: float = 0.2,
        n_augmented_copies: int = 2,
        scale_continuous: bool = True,
        fix_seeds: bool = True,
        random_state: int = 0,
        sampling_rate: float | None = None,
        n_trees: int | None = None,
        nu: float | None = None,
        tree_depth: int | None = None,
        bf: float | None = None,
        max_features: float | int | str | None = None,
        da_noise_frac: float | None = None,
        da_drop_rate: float | None = 0.2,
    ) -> None:
        if sampling_rate is not None:
            sample_frac = float(sampling_rate)
        if n_trees is not None:
            inner_n_estimators = int(n_trees)
        if nu is not None:
            inner_learning_rate = float(nu)
        if tree_depth is not None:
            inner_max_depth = int(tree_depth)
        if bf is not None:
            inner_subsample = float(bf)
        if da_noise_frac is not None:
            noise_level = float(da_noise_frac)
        if max_features is not None:
            mtry = max_features
        if mtry is None:
            mtry = 1.0 - float(da_drop_rate) if da_drop_rate is not None else 0.8

        if int(inner_n_estimators) <= 0:
            raise ValueError("inner_n_estimators must be positive")
        if float(inner_learning_rate) <= 0.0:
            raise ValueError("inner_learning_rate must be positive")
        if int(inner_max_depth) <= 0:
            raise ValueError("inner_max_depth must be positive")
        if float(noise_level) < 0.0:
            raise ValueError("noise_level must be non-negative")
        if float(shuffle_rate) < 0.0:
            raise ValueError("shuffle_rate must be non-negative")
        self.B = max(1, int(B))
        self.sample_frac = float(np.clip(sample_frac, 0.1, 1.0))
        self.inner_n_estimators = int(inner_n_estimators)
        self.inner_learning_rate = float(inner_learning_rate)
        self.inner_max_depth = int(inner_max_depth)
        self.inner_subsample = float(np.clip(inner_subsample, 0.1, 1.0))
        self.mtry = mtry
        self.data_aug = bool(data_aug)
        self.noise_level = float(noise_level)
        self.shuffle_rate = float(np.clip(shuffle_rate, 0.0, 1.0))
        self.n_augmented_copies = max(0, int(n_augmented_copies))
        self.scale_continuous = bool(scale_continuous)
        self.fix_seeds = bool(fix_seeds)
        self.random_state = int(random_state)
        self.feature_names_in_: tuple[str, ...] = ()
        self.augmented_feature_names_: tuple[str, ...] = ()
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._binary: np.ndarray | None = None
        self._continuous: np.ndarray | None = None
        self._models: list[tuple[Any, np.ndarray]] = []
        self.member_features_: pd.DataFrame | None = None
        self.member_samples_: pd.DataFrame | None = None
        self.augmentation_summary_: dict[str, Any] | None = None

    def _prepare_fit_values(self, X: pd.DataFrame) -> np.ndarray:
        raw = X.fillna(0.0).to_numpy(dtype=float)
        self._binary = binary_feature_mask(raw)
        self._continuous = ~self._binary
        self._mean = np.zeros(raw.shape[1], dtype=float)
        self._scale = np.ones(raw.shape[1], dtype=float)
        if self.scale_continuous and bool(self._continuous.any()):
            cont = self._continuous
            self._mean[cont] = raw[:, cont].mean(axis=0)
            self._scale[cont] = raw[:, cont].std(axis=0, ddof=0).clip(min=1e-12)
            raw = raw.copy()
            raw[:, cont] = (raw[:, cont] - self._mean[cont]) / self._scale[cont]
        return raw

    def _prepare_predict_values(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.reindex(columns=list(self.feature_names_in_), fill_value=0.0)
        raw = frame.fillna(0.0).to_numpy(dtype=float)
        if (
            self.scale_continuous
            and self._continuous is not None
            and self._mean is not None
            and self._scale is not None
            and bool(self._continuous.any())
        ):
            cont = self._continuous
            raw = raw.copy()
            raw[:, cont] = (raw[:, cont] - self._mean[cont]) / self._scale[cont]
        return raw

    def _augment_fit_values(
        self, values: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        names = list(self.feature_names_in_)
        if not self.data_aug or self.n_augmented_copies == 0:
            self.augmented_feature_names_ = tuple(names)
            return values

        if self._continuous is None or self._binary is None:
            raise RuntimeError(
                "BoogingRegressor must prepare values before augmentation"
            )
        parts = [values]
        binary_idx = np.flatnonzero(self._binary)
        continuous_idx = np.flatnonzero(self._continuous)
        for copy_id in range(1, self.n_augmented_copies + 1):
            fake = values.copy()
            if len(continuous_idx):
                fake[:, continuous_idx] = values[:, continuous_idx] + rng.normal(
                    scale=self.noise_level,
                    size=(values.shape[0], len(continuous_idx)),
                )
            if len(binary_idx) and self.shuffle_rate > 0.0 and values.shape[0] > 1:
                n_shuffle = max(1, int(round(self.shuffle_rate * values.shape[0])))
                n_shuffle = min(values.shape[0], n_shuffle)
                pack = rng.choice(values.shape[0], size=n_shuffle, replace=False)
                order = rng.permutation(pack)
                for col in binary_idx:
                    fake[pack, col] = values[order, col]
            parts.append(fake)
            names.extend(f"fake{copy_id}_{name}" for name in self.feature_names_in_)
        self.augmented_feature_names_ = tuple(names)
        return np.hstack(parts)

    def _augment_predict_values(self, values: np.ndarray) -> np.ndarray:
        if not self.data_aug or self.n_augmented_copies == 0:
            return values
        return np.hstack(
            [values] + [values.copy() for _ in range(self.n_augmented_copies)]
        )

    def _member_seed(self, i: int, rng: np.random.Generator) -> int:
        if self.fix_seeds:
            return self.random_state + 2020 + i + 1
        return int(rng.integers(0, np.iinfo(np.int32).max))

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "BoogingRegressor":
        from sklearn.ensemble import GradientBoostingRegressor

        self.feature_names_in_ = tuple(str(c) for c in X.columns)
        values = self._prepare_fit_values(X)
        target = np.asarray(y, dtype=float)
        n, k = values.shape
        if n == 0 or k == 0:
            return self
        rng = np.random.default_rng(self.random_state)
        aug_rng = np.random.default_rng(self.random_state + 1)
        augmented = self._augment_fit_values(values, aug_rng)
        sample_size = min(n, max(2, int(round(self.sample_frac * n))))
        keep_cols = resolve_feature_count(self.mtry, augmented.shape[1])
        inner_subsample = self.inner_subsample
        if n < 100:
            inner_subsample = max(0.4, inner_subsample)
        self._models = []
        feature_rows = []
        sample_rows = []
        for i in range(self.B):
            seed = self._member_seed(i, rng)
            member_rng = np.random.default_rng(seed)
            row_idx = member_rng.choice(n, sample_size, replace=False)
            col_idx = member_rng.choice(augmented.shape[1], keep_cols, replace=False)
            model = GradientBoostingRegressor(
                n_estimators=self.inner_n_estimators,
                learning_rate=self.inner_learning_rate,
                max_depth=self.inner_max_depth,
                subsample=inner_subsample,
                random_state=seed,
            )
            model.fit(augmented[np.ix_(row_idx, col_idx)], target[row_idx])
            self._models.append((model, col_idx))
            feature_rows.append(
                {
                    "member": i,
                    "n_features": int(len(col_idx)),
                    "features": tuple(
                        str(self.augmented_feature_names_[int(j)]) for j in col_idx
                    ),
                    "mtry": self.mtry,
                }
            )
            sample_rows.append(
                {
                    "member": i,
                    "n_rows": int(len(row_idx)),
                    "n_unique_rows": int(len(np.unique(row_idx))),
                    "sample_frac": self.sample_frac,
                    "sampling_rate": self.sample_frac,
                    "inner_subsample": inner_subsample,
                    "bf": inner_subsample,
                }
            )
        self.member_features_ = pd.DataFrame(feature_rows)
        self.member_samples_ = pd.DataFrame(sample_rows)
        self.augmentation_summary_ = {
            "data_aug": self.data_aug,
            "n_original_features": int(k),
            "n_augmented_features": int(augmented.shape[1]),
            "n_augmented_copies": int(self.n_augmented_copies if self.data_aug else 0),
            "n_binary_features": int(self._binary.sum())
            if self._binary is not None
            else 0,
            "n_continuous_features": (
                int(self._continuous.sum()) if self._continuous is not None else 0
            ),
            "noise_level": float(self.noise_level),
            "shuffle_rate": float(self.shuffle_rate),
            "scale_continuous": bool(self.scale_continuous),
            "scaling_note": "train-only continuous scaling; R code scales train and X.new jointly",
        }
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._models:
            return np.zeros(len(X), dtype=float)
        preds = self._prediction_matrix(X)
        return preds.mean(axis=1)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> dict[float, np.ndarray]:
        levels = _validate_quantile_levels(levels)
        if not self._models:
            return {q: np.zeros(len(X), dtype=float) for q in levels}
        preds = self._prediction_matrix(X)
        return {float(q): np.quantile(preds, float(q), axis=1) for q in levels}

    def _prediction_matrix(self, X: pd.DataFrame) -> np.ndarray:
        augmented = self._augment_predict_values(self._prepare_predict_values(X))
        preds = []
        for model, col_idx in self._models:
            preds.append(np.asarray(model.predict(augmented[:, col_idx]), dtype=float))
        return np.column_stack(preds) if preds else np.zeros((len(X), 0), dtype=float)


def booging(
    X: Any,
    y: Any | None = None,
    *,
    B: int = 100,
    sample_frac: float = 0.75,
    inner_n_estimators: int = 1000,
    inner_learning_rate: float = 0.3,
    inner_max_depth: int = 3,
    inner_subsample: float = 0.5,
    mtry: float | int | str | None = None,
    data_aug: bool = False,
    noise_level: float = 0.3,
    shuffle_rate: float = 0.2,
    n_augmented_copies: int = 2,
    scale_continuous: bool = True,
    fix_seeds: bool = True,
    random_state: int = 0,
    sampling_rate: float | None = None,
    n_trees: int | None = None,
    nu: float | None = None,
    tree_depth: int | None = None,
    bf: float | None = None,
    max_features: float | int | str | None = None,
    da_noise_frac: float | None = None,
    da_drop_rate: float | None = 0.2,
) -> ModelFit:
    """Fit Booging: bagged overfit stochastic gradient boosting with augmentation."""

    resolved_sample_frac = (
        float(sampling_rate) if sampling_rate is not None else float(sample_frac)
    )
    resolved_n_estimators = (
        int(n_trees) if n_trees is not None else int(inner_n_estimators)
    )
    resolved_learning_rate = float(nu) if nu is not None else float(inner_learning_rate)
    resolved_max_depth = (
        int(tree_depth) if tree_depth is not None else int(inner_max_depth)
    )
    resolved_subsample = float(bf) if bf is not None else float(inner_subsample)
    resolved_noise = (
        float(da_noise_frac) if da_noise_frac is not None else float(noise_level)
    )
    if max_features is not None:
        resolved_mtry = max_features
    elif mtry is not None:
        resolved_mtry = mtry
    elif da_drop_rate is not None:
        resolved_mtry = 1.0 - float(da_drop_rate)
    else:
        resolved_mtry = 0.8

    params = {
        "B": int(B),
        "sample_frac": resolved_sample_frac,
        "sampling_rate": resolved_sample_frac,
        "inner_n_estimators": resolved_n_estimators,
        "n_trees": resolved_n_estimators,
        "inner_learning_rate": resolved_learning_rate,
        "nu": resolved_learning_rate,
        "inner_max_depth": resolved_max_depth,
        "tree_depth": resolved_max_depth,
        "inner_subsample": resolved_subsample,
        "bf": resolved_subsample,
        "mtry": resolved_mtry,
        "max_features": resolved_mtry,
        "data_aug": bool(data_aug),
        "noise_level": resolved_noise,
        "shuffle_rate": float(shuffle_rate),
        "n_augmented_copies": int(n_augmented_copies),
        "scale_continuous": bool(scale_continuous),
        "fix_seeds": bool(fix_seeds),
        "legacy_da_noise_frac": da_noise_frac,
        "legacy_da_drop_rate": da_drop_rate,
        "random_state": int(random_state),
        "implementation_note": (
            "Goulet Coulombe Booging-style bagged/perturbed overfit stochastic "
            "gradient boosting with sklearn GradientBoostingRegressor members. "
            "R-style aliases are accepted: sampling_rate, mtry, data_aug, "
            "noise_level, shuffle_rate, bf, n_trees, tree_depth, and nu."
        ),
    }
    return fit_estimator(
        BoogingRegressor(
            B=int(B),
            sample_frac=resolved_sample_frac,
            inner_n_estimators=resolved_n_estimators,
            inner_learning_rate=resolved_learning_rate,
            inner_max_depth=resolved_max_depth,
            inner_subsample=resolved_subsample,
            mtry=resolved_mtry,
            data_aug=bool(data_aug),
            noise_level=resolved_noise,
            shuffle_rate=float(shuffle_rate),
            n_augmented_copies=int(n_augmented_copies),
            scale_continuous=bool(scale_continuous),
            fix_seeds=bool(fix_seeds),
            random_state=int(random_state),
        ),
        X,
        y,
        model="booging",
        metadata=params,
    )


__all__ = [
    "BaggingRegressor",
    "BoogingRegressor",
    "MODEL_ENSEMBLE_BASE_ESTIMATORS",
    "RandomSubspaceRegressor",
    "StackingRegressor",
    "SuperLearnerRegressor",
    "bagging",
    "booging",
    "list_model_ensemble_bases",
    "random_subspace",
    "stacking",
    "subagging",
    "super_learner",
]
