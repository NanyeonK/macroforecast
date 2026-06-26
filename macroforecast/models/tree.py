from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from typing import Any, Literal, cast
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

    # Backend wrapper only: sklearn owns CART fitting; macroforecast owns the
    # callable X/y contract, metadata, diagnostics, and persistence.
    params = {
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "implementation_note": (
            "Thin sklearn.tree.DecisionTreeRegressor wrapper; macroforecast owns "
            "the pandas X/y contract, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    return fit_estimator(
        DecisionTreeRegressor(**estimator_params),
        X,
        y,
        model="decision_tree",
        metadata=params,
    )


def _resolved_n_jobs(n_jobs: int | None) -> int:
    """Resolve an unset n_jobs to the package-wide meta default."""
    if n_jobs is not None:
        return int(n_jobs)
    from macroforecast.meta import resolve_n_jobs

    return resolve_n_jobs()


def random_forest(
    X: Any,
    y: Any | None = None,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    min_samples_leaf: int = 1,
    random_state: int = 0,
    n_jobs: int | None = None,
    **kwargs: Any,
) -> ModelFit:
    """Fit a random forest regressor."""

    from sklearn.ensemble import RandomForestRegressor

    # Backend wrapper only: sklearn owns the forest algorithm; macroforecast
    # passes through tree parameters and records the fit metadata.
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "n_jobs": _resolved_n_jobs(n_jobs),
        "implementation_note": (
            "Thin sklearn.ensemble.RandomForestRegressor wrapper; macroforecast "
            "owns the pandas X/y contract, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    return fit_estimator(
        RandomForestRegressor(**estimator_params),
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
    n_jobs: int | None = None,
    **kwargs: Any,
) -> ModelFit:
    """Fit an extremely randomized trees regressor."""

    from sklearn.ensemble import ExtraTreesRegressor

    # Backend wrapper only: sklearn owns ExtraTrees; no package-native tree
    # logic is implemented here.
    params = {
        "n_estimators": int(n_estimators),
        "max_depth": max_depth,
        "min_samples_leaf": int(min_samples_leaf),
        "random_state": int(random_state),
        "n_jobs": _resolved_n_jobs(n_jobs),
        "implementation_note": (
            "Thin sklearn.ensemble.ExtraTreesRegressor wrapper; macroforecast owns "
            "the pandas X/y contract, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    return fit_estimator(
        ExtraTreesRegressor(**estimator_params),
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

    # Backend wrapper only: sklearn owns stage-wise tree boosting; fit-time
    # boosting ensembles such as Booging live in macroforecast.model_ensemble.
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        "implementation_note": (
            "Thin sklearn.ensemble.GradientBoostingRegressor wrapper; fit-time "
            "boosting ensembles such as Booging live in macroforecast.model_ensemble."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    return fit_estimator(
        GradientBoostingRegressor(**estimator_params),
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
    # Optional backend wrapper: xgboost owns the estimator; macroforecast only
    # normalizes parameter names and lazy dependency errors.
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "subsample": float(subsample),
        "random_state": int(random_state),
        "objective": "reg:squarederror",
        "implementation_note": (
            "Thin xgboost.XGBRegressor wrapper; macroforecast owns lazy extra "
            "loading, pandas X/y IO, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    estimator = xgb.XGBRegressor(
        **estimator_params,
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
    # Optional backend wrapper: LightGBM owns the estimator; macroforecast only
    # normalizes parameter names and lazy dependency errors.
    params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(num_leaves),
        "random_state": int(random_state),
        "implementation_note": (
            "Thin lightgbm.LGBMRegressor wrapper; macroforecast owns lazy extra "
            "loading, pandas X/y IO, ModelFit metadata, and diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    estimator = lgb.LGBMRegressor(
        **estimator_params,
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
    # Optional backend wrapper: CatBoost owns the estimator; macroforecast maps
    # public names such as n_estimators/max_depth onto CatBoost's API.
    params = {
        "iterations": int(n_estimators),
        "learning_rate": float(learning_rate),
        "depth": int(max_depth),
        "random_seed": int(random_state),
        "verbose": verbose,
        "implementation_note": (
            "Thin catboost.CatBoostRegressor wrapper; macroforecast maps public "
            "n_estimators/max_depth names to CatBoost iterations/depth."
        ),
        **kwargs,
    }
    metadata = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "random_state": int(random_state),
        "verbose": verbose,
        "implementation_note": params["implementation_note"],
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    estimator = cb.CatBoostRegressor(
        **estimator_params,
    )
    return fit_estimator(estimator, X, y, model="catboost", metadata=metadata)


def _coerce_lgbplus_matrix(
    X: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        if feature_names is not None:
            frame = frame.reindex(
                columns=[str(name) for name in feature_names], fill_value=0.0
            )
        names = tuple(str(column) for column in frame.columns)
        arr = frame.to_numpy(dtype=np.float64)
    else:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {arr.shape!r}")
        names = (
            tuple(str(name) for name in feature_names)
            if feature_names is not None
            else tuple(f"x{i}" for i in range(arr.shape[1]))
        )
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("X must contain at least one feature column")
    if len(names) != arr.shape[1]:
        raise ValueError("feature_names length must match X column count")
    return arr, names


def _standardize_for_lgbplus_correlation(X: np.ndarray) -> np.ndarray:
    centered = X - X.mean(axis=0, keepdims=True)
    scale = X.std(axis=0, keepdims=True)
    scale[scale < 1e-10] = 1.0
    return centered / scale


def _lgbplus_channel_frame(
    *,
    feature_names: Sequence[str],
    tree_gain: np.ndarray,
    linear_counts: np.ndarray,
    linear_abs_contribution: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tree_gain_importance": np.asarray(tree_gain, dtype=float),
            "linear_selection_count": np.asarray(linear_counts, dtype=int),
            "linear_abs_update": np.asarray(linear_abs_contribution, dtype=float),
        },
        index=pd.Index([str(name) for name in feature_names], name="feature"),
    ).sort_values(
        ["linear_selection_count", "tree_gain_importance", "linear_abs_update"],
        ascending=False,
    )


class LGBPlusRegressor:
    """Competition-based LGB+ estimator.

    Source alignment:
    - Reference repository: https://github.com/philgoucou/lgbplus
    - Python file: `python/lgb_plus.py`; R file: `R/lgb_plus.R`.
    - Both versions fit one LightGBM residual tree and one greedy univariate
      linear residual update at every boosting step, then accept the lower-loss
      candidate under `oob`, `validation`, or `training` selection.
    - The R implementation adds `linear_candidate_fraction`, randomly sampling
      feature candidates before choosing the best residual correlation. The
      Python implementation omits that public argument. macroforecast keeps the
      R candidate-subsampling logic and the Python in-class ensemble structure.
    - The competition linear update intentionally has no intercept:
      `coef = sum(x_j * residual) / sum(x_j^2)`, matching both reference files.
    """

    def __init__(
        self,
        *,
        n_ensemble: int = 10,
        n_steps: int = 200,
        learning_rate: float = 0.05,
        subsample: float = 0.7,
        num_leaves: int = 5,
        min_data_in_leaf: int = 20,
        lambda_l2: float = 0.1,
        linear_candidate_fraction: float = 0.5,
        selection_method: Literal["oob", "validation", "training"] = "oob",
        val_fraction: float = 0.2,
        early_stop_patience: int | None = 50,
        aggregation: Literal["mean", "median"] = "mean",
        random_state: int | None = None,
        verbose: bool = False,
        **lgb_params: Any,
    ) -> None:
        self.n_ensemble = int(n_ensemble)
        self.n_steps = int(n_steps)
        self.learning_rate = float(learning_rate)
        self.subsample = float(subsample)
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.lambda_l2 = float(lambda_l2)
        self.linear_candidate_fraction = float(linear_candidate_fraction)
        self.selection_method = selection_method
        self.val_fraction = float(val_fraction)
        self.early_stop_patience = (
            None if early_stop_patience is None else int(early_stop_patience)
        )
        self.aggregation = aggregation
        self.random_state = None if random_state is None else int(random_state)
        self.verbose = bool(verbose)
        self.lgb_params_extra = dict(lgb_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Sequence[str] | None = None,
    ) -> "LGBPlusRegressor":
        lgb = optional_import("lightgbm", extra="lightgbm")
        X_arr, names = _coerce_lgbplus_matrix(X, feature_names)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self._validate_fit_options(X_arr.shape[0], X_arr.shape[1])
        self.feature_names_ = names
        self.n_features_ = int(X_arr.shape[1])

        X_standardized = _standardize_for_lgbplus_correlation(X_arr)
        train_pool_idx, X_val, y_val = self._validation_split(X_arr, y_arr)
        lgb_params = {
            "objective": "regression",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "lambda_l2": self.lambda_l2,
            "learning_rate": 1.0,
            "verbosity": -1,
            "force_col_wise": True,
            "num_threads": 1,
            **self.lgb_params_extra,
        }

        self.ensemble_: list[dict[str, Any]] = []
        self.step_type_counts_: np.ndarray = np.zeros((self.n_ensemble, 2), dtype=int)
        self.linear_feature_counts_: np.ndarray = np.zeros(self.n_features_, dtype=int)
        self.feature_importances_: np.ndarray = np.zeros(self.n_features_, dtype=float)
        self.linear_abs_contribution_: np.ndarray = np.zeros(self.n_features_, dtype=float)
        self.training_history_: dict[str, Any] = {
            "source_reference": "philgoucou/lgbplus python/lgb_plus.py and R/lgb_plus.R",
            "selection_method": self.selection_method,
            "member_loss_history": [],
        }

        for member_idx in range(self.n_ensemble):
            member = self._fit_one_member(
                lgb=lgb,
                X=X_arr,
                y=y_arr,
                X_standardized=X_standardized,
                X_val=X_val,
                y_val=y_val,
                train_pool_idx=train_pool_idx,
                lgb_params=lgb_params,
                member_idx=member_idx,
            )
            self.ensemble_.append(member)
            self.step_type_counts_[member_idx] = member["step_type_counts"]
            self.linear_feature_counts_ += member["linear_feature_counts"]
            self.feature_importances_ += member["feature_importances"]
            self.linear_abs_contribution_ += member["linear_abs_contribution"]
            self.training_history_["member_loss_history"].append(member["loss_history"])

        self.channel_summary_ = self.get_step_type_summary()
        self.channel_importance_ = self.channel_importance()
        return self

    def _validate_fit_options(self, n_samples: int, n_features: int) -> None:
        if self.n_ensemble < 1:
            raise ValueError("n_ensemble must be at least 1")
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.selection_method not in {"oob", "validation", "training"}:
            raise ValueError(
                "selection_method must be 'oob', 'validation', or 'training'"
            )
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be 'mean' or 'median'")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if self.selection_method == "oob" and self.subsample >= 1.0:
            raise ValueError("selection_method='oob' requires subsample < 1")
        if not (0.0 < self.linear_candidate_fraction <= 1.0):
            raise ValueError("linear_candidate_fraction must be in (0, 1]")
        if n_samples < 2:
            raise ValueError("LGB+ requires at least two observations")
        if n_features < 1:
            raise ValueError("LGB+ requires at least one feature")

    def _validation_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        n_samples = X.shape[0]
        if self.selection_method == "validation" and self.val_fraction > 0:
            rng_val = np.random.default_rng(self.random_state)
            n_val = int(np.floor(n_samples * self.val_fraction))
            if n_val <= 0 or n_val >= n_samples:
                raise ValueError(
                    "val_fraction must leave non-empty train and validation sets"
                )
            val_idx = rng_val.choice(n_samples, size=n_val, replace=False)
            train_pool_idx = np.setdiff1d(np.arange(n_samples), val_idx)
            return train_pool_idx, X[val_idx], y[val_idx]
        return np.arange(n_samples), None, None

    def _fit_one_member(
        self,
        *,
        lgb: Any,
        X: np.ndarray,
        y: np.ndarray,
        X_standardized: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        train_pool_idx: np.ndarray,
        lgb_params: dict[str, Any],
        member_idx: int,
    ) -> dict[str, Any]:
        n_samples, n_features = X.shape
        n_train_pool = int(len(train_pool_idx))
        subsample_size = max(1, int(np.floor(n_train_pool * self.subsample)))
        if self.selection_method == "oob" and subsample_size >= n_train_pool:
            raise ValueError("OOB LGB+ needs at least one out-of-bag row per step")
        seed = None if self.random_state is None else self.random_state + member_idx
        rng = np.random.default_rng(seed)
        params = dict(lgb_params)
        if seed is not None:
            params.setdefault("seed", seed)

        init_pred = float(np.mean(y))
        pred = np.full(n_samples, init_pred, dtype=float)
        pred_val: np.ndarray | None = (
            None
            if X_val is None
            else np.full(len(cast("np.ndarray", y_val)), init_pred, dtype=float)
        )
        steps: list[dict[str, Any]] = []
        step_type_counts: np.ndarray = np.zeros(2, dtype=int)
        linear_feature_counts = np.zeros(n_features, dtype=int)
        linear_abs_contribution = np.zeros(n_features, dtype=float)
        feature_importances = np.zeros(n_features, dtype=float)
        loss_history: list[dict[str, Any]] = []
        best_loss = np.inf
        steps_without_improvement = 0
        n_candidates = max(
            1, int(np.floor(n_features * self.linear_candidate_fraction))
        )

        for step_idx in range(self.n_steps):
            sample_idx = rng.choice(train_pool_idx, size=subsample_size, replace=False)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]
            resid_sample = y_sample - pred[sample_idx]

            dtrain = lgb.Dataset(X_sample, label=resid_sample, free_raw_data=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = lgb.train(params, dtrain, num_boost_round=1)
            tree_update_all = np.asarray(booster.predict(X), dtype=float)
            tree_candidate = pred + self.learning_rate * tree_update_all

            candidate_idx = rng.choice(n_features, size=n_candidates, replace=False)
            resid_std = resid_sample - np.mean(resid_sample)
            resid_scale = float(np.std(resid_sample))
            if resid_scale < 1e-10:
                resid_scale = 1.0
            resid_standardized = resid_std / resid_scale
            correlations = (
                X_standardized[sample_idx][:, candidate_idx].T @ resid_standardized
            ) / len(sample_idx)
            correlations = np.nan_to_num(correlations, nan=0.0)
            feature_idx = int(candidate_idx[int(np.argmax(np.abs(correlations)))])
            x_feature = X_sample[:, feature_idx]
            coef = float((x_feature @ resid_sample) / ((x_feature**2).sum() + 1e-10))
            linear_candidate = pred + self.learning_rate * coef * X[:, feature_idx]

            tree_loss, linear_loss, tree_val, linear_val = self._candidate_losses(
                booster=booster,
                X=X,
                y=y,
                X_val=X_val,
                y_val=y_val,
                pred=pred,
                pred_val=pred_val,
                sample_idx=sample_idx,
                train_pool_idx=train_pool_idx,
                y_sample=y_sample,
                tree_candidate=tree_candidate,
                linear_candidate=linear_candidate,
                feature_idx=feature_idx,
                coef=coef,
            )
            if tree_loss <= linear_loss:
                pred = tree_candidate
                if pred_val is not None:
                    pred_val = tree_val
                steps.append({"type": "tree", "booster": booster})
                step_type_counts[0] += 1
                try:
                    importance = np.asarray(
                        booster.feature_importance(importance_type="gain"), dtype=float
                    )
                    if len(importance) == n_features:
                        feature_importances += importance
                except Exception:  # noqa: BLE001 - diagnostics must not break fitting.
                    pass
                chosen = "tree"
                current_loss = float(tree_loss)
            else:
                pred = linear_candidate
                if pred_val is not None:
                    pred_val = linear_val
                steps.append(
                    {"type": "linear", "feature_idx": feature_idx, "coef": coef}
                )
                step_type_counts[1] += 1
                linear_feature_counts[feature_idx] += 1
                linear_abs_contribution[feature_idx] += abs(self.learning_rate * coef)
                chosen = "linear"
                current_loss = float(linear_loss)

            if (
                self.selection_method == "validation"
                and y_val is not None
                and pred_val is not None
            ):
                current_loss = float(np.mean((y_val - pred_val) ** 2))
            train_loss = float(np.mean((y - pred) ** 2))
            loss_history.append(
                {
                    "step": int(step_idx + 1),
                    "chosen": chosen,
                    "tree_loss": float(tree_loss),
                    "linear_loss": float(linear_loss),
                    "selection_loss": current_loss,
                    "train_loss": train_loss,
                    "linear_feature": self.feature_names_[feature_idx],
                }
            )

            if self.early_stop_patience is not None:
                if current_loss < best_loss - 1e-10:
                    best_loss = current_loss
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                if steps_without_improvement >= self.early_stop_patience:
                    break

        return {
            "init": init_pred,
            "steps": steps,
            "step_type_counts": step_type_counts,
            "linear_feature_counts": linear_feature_counts,
            "linear_abs_contribution": linear_abs_contribution,
            "feature_importances": feature_importances,
            "loss_history": loss_history,
        }

    def _candidate_losses(
        self,
        *,
        booster: Any,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        pred: np.ndarray,
        pred_val: np.ndarray | None,
        sample_idx: np.ndarray,
        train_pool_idx: np.ndarray,
        y_sample: np.ndarray,
        tree_candidate: np.ndarray,
        linear_candidate: np.ndarray,
        feature_idx: int,
        coef: float,
    ) -> tuple[float, float, np.ndarray | None, np.ndarray | None]:
        if self.selection_method == "oob":
            oob_idx = np.setdiff1d(train_pool_idx, sample_idx)
            tree_oob = pred[oob_idx] + self.learning_rate * np.asarray(
                booster.predict(X[oob_idx]), dtype=float
            )
            linear_oob = (
                pred[oob_idx] + self.learning_rate * coef * X[oob_idx, feature_idx]
            )
            return (
                float(np.mean((y[oob_idx] - tree_oob) ** 2)),
                float(np.mean((y[oob_idx] - linear_oob) ** 2)),
                None,
                None,
            )
        if (
            self.selection_method == "validation"
            and X_val is not None
            and y_val is not None
        ):
            base_val = (
                np.zeros(len(y_val), dtype=float) if pred_val is None else pred_val
            )
            tree_val = base_val + self.learning_rate * np.asarray(
                booster.predict(X_val), dtype=float
            )
            linear_val = base_val + self.learning_rate * coef * X_val[:, feature_idx]
            return (
                float(np.mean((y_val - tree_val) ** 2)),
                float(np.mean((y_val - linear_val) ** 2)),
                tree_val,
                linear_val,
            )
        return (
            float(np.mean((y_sample - tree_candidate[sample_idx]) ** 2)),
            float(np.mean((y_sample - linear_candidate[sample_idx]) ** 2)),
            None,
            None,
        )

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_components(X)["prediction_total"].to_numpy(dtype=float)

    def predict_individual(self, X: Any) -> np.ndarray:
        X_arr, _ = _coerce_lgbplus_matrix(X, self.feature_names_)
        totals = []
        for member in self.ensemble_:
            components = self._member_components(member, X_arr)
            totals.append(
                components["init"] + components["tree"] + components["linear"]
            )
        return np.vstack(totals)

    def predict_components(self, X: Any) -> pd.DataFrame:
        X_arr, _ = _coerce_lgbplus_matrix(X, self.feature_names_)
        init_parts = []
        tree_parts = []
        linear_parts = []
        total_parts = []
        for member in self.ensemble_:
            components = self._member_components(member, X_arr)
            init_parts.append(components["init"])
            tree_parts.append(components["tree"])
            linear_parts.append(components["linear"])
            total_parts.append(
                components["init"] + components["tree"] + components["linear"]
            )
        aggregate = np.median if self.aggregation == "median" else np.mean
        frame = pd.DataFrame(
            {
                "prediction_total": aggregate(np.vstack(total_parts), axis=0),
                "prediction_init": aggregate(np.vstack(init_parts), axis=0),
                "prediction_tree": aggregate(np.vstack(tree_parts), axis=0),
                "prediction_linear": aggregate(np.vstack(linear_parts), axis=0),
            },
            index=X.index
            if isinstance(X, pd.DataFrame)
            else pd.RangeIndex(X_arr.shape[0]),
        )
        return frame

    def _member_components(
        self, member: dict[str, Any], X: np.ndarray
    ) -> dict[str, np.ndarray]:
        init = np.full(X.shape[0], float(member["init"]), dtype=float)
        tree = np.zeros(X.shape[0], dtype=float)
        linear = np.zeros(X.shape[0], dtype=float)
        for step in member["steps"]:
            if step["type"] == "tree":
                tree += self.learning_rate * np.asarray(
                    step["booster"].predict(X), dtype=float
                )
            else:
                feature_idx = int(step["feature_idx"])
                linear += self.learning_rate * float(step["coef"]) * X[:, feature_idx]
        return {"init": init, "tree": tree, "linear": linear}

    def get_step_type_summary(self) -> dict[str, Any]:
        total_tree = int(self.step_type_counts_[:, 0].sum())
        total_linear = int(self.step_type_counts_[:, 1].sum())
        total = total_tree + total_linear
        return {
            "total_tree": total_tree,
            "total_linear": total_linear,
            "tree_fraction": float(total_tree / total) if total else 0.0,
            "per_member_tree": self.step_type_counts_[:, 0].copy(),
            "per_member_linear": self.step_type_counts_[:, 1].copy(),
        }

    def get_linear_feature_summary(self) -> dict[str, int]:
        order = np.argsort(self.linear_feature_counts_)[::-1]
        return {
            self.feature_names_[idx]: int(self.linear_feature_counts_[idx])
            for idx in order
            if self.linear_feature_counts_[idx] > 0
        }

    def channel_importance(
        self,
        X: Any | None = None,
        y: Any | None = None,
    ) -> pd.DataFrame:
        return _lgbplus_channel_frame(
            feature_names=self.feature_names_,
            tree_gain=self.feature_importances_,
            linear_counts=self.linear_feature_counts_,
            linear_abs_contribution=self.linear_abs_contribution_,
        )

    def summary(self) -> str:
        if not hasattr(self, "ensemble_"):
            return "Model not fitted yet."
        step_summary = self.get_step_type_summary()
        return "\n".join(
            [
                "LGB+ Competition Summary",
                f"Ensemble members: {self.n_ensemble}",
                f"Steps per member: {self.n_steps}",
                f"Selection method: {self.selection_method}",
                f"Tree steps: {step_summary['total_tree']}",
                f"Linear steps: {step_summary['total_linear']}",
            ]
        )


class LGBAPlusRegressor:
    """Alternating LGB^A+ estimator.

    Source alignment:
    - Reference repository: https://github.com/philgoucou/lgbplus
    - Python file: `python/lgb_plus_A.py`; R file: `R/lgb_plus_A.R`.
    - Each cycle fits a LightGBM residual tree block, applies `lr_tree`, then
      selects the feature with the largest absolute residual correlation and
      applies one univariate OLS linear update with intercept and `lr_linear`.
    - R also provides `lgb_plus_A_ensemble`; macroforecast exposes the same idea
      as `n_runs` on this estimator rather than creating a separate public
      helper, keeping model selection and forecasting runner integration simple.
    - The linear slope is computed by centered dot products, exactly equivalent
      to R's `cov(x, resid) / var(x)` and mathematically cleaner than the Python
      file's mixed `np.cov(..., ddof=1) / x.var(ddof=0)` expression.
    """

    def __init__(
        self,
        *,
        n_runs: int = 1,
        n_cycles: int = 25,
        trees_per_cycle: int = 10,
        lr_tree: float = 0.02,
        lr_linear: float = 0.1,
        num_leaves: int = 15,
        min_data_in_leaf: int = 20,
        subsample: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
        **lgb_params: Any,
    ) -> None:
        self.n_runs = int(n_runs)
        self.n_cycles = int(n_cycles)
        self.trees_per_cycle = int(trees_per_cycle)
        self.lr_tree = float(lr_tree)
        self.lr_linear = float(lr_linear)
        self.num_leaves = int(num_leaves)
        self.min_data_in_leaf = int(min_data_in_leaf)
        self.subsample = float(subsample)
        self.random_state = None if random_state is None else int(random_state)
        self.verbose = bool(verbose)
        self.lgb_params_extra = dict(lgb_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Sequence[str] | None = None,
    ) -> "LGBAPlusRegressor":
        lgb = optional_import("lightgbm", extra="lightgbm")
        X_arr, names = _coerce_lgbplus_matrix(X, feature_names)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if len(y_arr) != X_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self._validate_fit_options(X_arr.shape[1])
        self.feature_names_ = names
        self.n_features_ = int(X_arr.shape[1])
        self.runs_: list[dict[str, Any]] = []
        self.feature_importances_: np.ndarray = np.zeros(self.n_features_, dtype=float)
        self.linear_feature_counts_: np.ndarray = np.zeros(self.n_features_, dtype=int)
        self.linear_abs_contribution_: np.ndarray = np.zeros(
            self.n_features_, dtype=float
        )
        self.training_history_: dict[str, Any] = {
            "source_reference": "philgoucou/lgbplus python/lgb_plus_A.py and R/lgb_plus_A.R",
            "n_runs": self.n_runs,
            "run_loss_history": [],
        }

        for run_idx in range(self.n_runs):
            run = self._fit_one_run(lgb, X_arr, y_arr, run_idx)
            self.runs_.append(run)
            self.feature_importances_ += run["feature_importances"]
            self.linear_feature_counts_ += run["linear_feature_counts"]
            self.linear_abs_contribution_ += run["linear_abs_contribution"]
            self.training_history_["run_loss_history"].append(run["loss_history"])

        first = self.runs_[0]
        self.init_ = first["init"]
        self.trees_ = first["trees"]
        self.linear_steps_ = first["linear_steps"]
        self.channel_summary_ = {
            "total_tree_blocks": int(self.n_runs * self.n_cycles),
            "total_linear_steps": int(self.n_runs * self.n_cycles),
            "n_runs": int(self.n_runs),
            "n_cycles": int(self.n_cycles),
            "trees_per_cycle": int(self.trees_per_cycle),
        }
        self.channel_importance_ = self.channel_importance()
        return self

    def _validate_fit_options(self, n_features: int) -> None:
        if self.n_runs < 1:
            raise ValueError("n_runs must be at least 1")
        if self.n_cycles < 1:
            raise ValueError("n_cycles must be at least 1")
        if self.trees_per_cycle < 1:
            raise ValueError("trees_per_cycle must be at least 1")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if n_features < 1:
            raise ValueError("LGB^A+ requires at least one feature")

    def _fit_one_run(
        self,
        lgb: Any,
        X: np.ndarray,
        y: np.ndarray,
        run_idx: int,
    ) -> dict[str, Any]:
        n_features = X.shape[1]
        run_seed = None if self.random_state is None else self.random_state + run_idx
        params = {
            "objective": "regression",
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "learning_rate": 1.0,
            "verbosity": -1,
            "force_col_wise": True,
            "num_threads": 1,
            **self.lgb_params_extra,
        }
        if self.subsample < 1.0:
            params["bagging_fraction"] = self.subsample
            params["bagging_freq"] = 1
        if run_seed is not None:
            params.setdefault("seed", run_seed)
            params.setdefault("bagging_seed", run_seed)

        init = float(np.mean(y))
        pred: np.ndarray = np.full(len(y), init, dtype=float)
        trees: list[Any] = []
        linear_steps: list[dict[str, Any]] = []
        feature_importances = np.zeros(n_features, dtype=float)
        linear_feature_counts = np.zeros(n_features, dtype=int)
        linear_abs_contribution = np.zeros(n_features, dtype=float)
        loss_history: list[dict[str, Any]] = []

        for cycle in range(self.n_cycles):
            resid = y - pred
            dtrain = lgb.Dataset(X, label=resid, free_raw_data=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = lgb.train(
                    params, dtrain, num_boost_round=self.trees_per_cycle
                )
            tree_update = np.asarray(booster.predict(X), dtype=float)
            pred = pred + self.lr_tree * tree_update
            trees.append(booster)
            try:
                importance = np.asarray(
                    booster.feature_importance(importance_type="gain"), dtype=float
                )
                if len(importance) == n_features:
                    feature_importances += importance
            except Exception:  # noqa: BLE001 - diagnostics must not break fitting.
                pass

            resid = y - pred
            correlations = np.zeros(n_features, dtype=float)
            for feature_idx in range(n_features):
                x = X[:, feature_idx]
                if np.std(x) < 1e-10 or np.std(resid) < 1e-10:
                    correlations[feature_idx] = 0.0
                else:
                    correlations[feature_idx] = float(np.corrcoef(x, resid)[0, 1])
            correlations = np.nan_to_num(correlations, nan=0.0)
            best_idx = int(np.argmax(np.abs(correlations)))
            x_best = X[:, best_idx]
            x_centered = x_best - np.mean(x_best)
            resid_centered = resid - np.mean(resid)
            denom = float(np.sum(x_centered**2))
            if denom > 1e-10:
                coef = float((x_centered @ resid_centered) / denom)
                intercept = float(np.mean(resid) - coef * np.mean(x_best))
            else:
                coef = 0.0
                intercept = float(np.mean(resid))
            linear_update = coef * x_best + intercept
            pred = pred + self.lr_linear * linear_update
            linear_steps.append(
                {
                    "feature_idx": best_idx,
                    "coef": coef,
                    "intercept": intercept,
                    "feature": self.feature_names_[best_idx],
                }
            )
            linear_feature_counts[best_idx] += 1
            linear_abs_contribution[best_idx] += abs(self.lr_linear * coef)
            loss_history.append(
                {
                    "cycle": int(cycle + 1),
                    "linear_feature": self.feature_names_[best_idx],
                    "train_loss": float(np.mean((y - pred) ** 2)),
                }
            )

        return {
            "init": init,
            "trees": trees,
            "linear_steps": linear_steps,
            "feature_importances": feature_importances,
            "linear_feature_counts": linear_feature_counts,
            "linear_abs_contribution": linear_abs_contribution,
            "loss_history": loss_history,
        }

    def predict(self, X: Any) -> np.ndarray:
        return self.predict_components(X)["prediction_total"].to_numpy(dtype=float)

    def predict_components(self, X: Any) -> pd.DataFrame:
        X_arr, _ = _coerce_lgbplus_matrix(X, self.feature_names_)
        init_parts = []
        tree_parts = []
        linear_parts = []
        for run in self.runs_:
            components = self._run_components(run, X_arr)
            init_parts.append(components["init"])
            tree_parts.append(components["tree"])
            linear_parts.append(components["linear"])
        init = np.mean(np.vstack(init_parts), axis=0)
        tree = np.mean(np.vstack(tree_parts), axis=0)
        linear = np.mean(np.vstack(linear_parts), axis=0)
        return pd.DataFrame(
            {
                "prediction_total": init + tree + linear,
                "prediction_init": init,
                "prediction_tree": tree,
                "prediction_linear": linear,
            },
            index=X.index
            if isinstance(X, pd.DataFrame)
            else pd.RangeIndex(X_arr.shape[0]),
        )

    def _run_components(
        self, run: dict[str, Any], X: np.ndarray
    ) -> dict[str, np.ndarray]:
        init = np.full(X.shape[0], float(run["init"]), dtype=float)
        tree = np.zeros(X.shape[0], dtype=float)
        linear = np.zeros(X.shape[0], dtype=float)
        for booster, step in zip(run["trees"], run["linear_steps"], strict=True):
            tree += self.lr_tree * np.asarray(booster.predict(X), dtype=float)
            feature_idx = int(step["feature_idx"])
            linear += self.lr_linear * (
                float(step["coef"]) * X[:, feature_idx] + float(step["intercept"])
            )
        return {"init": init, "tree": tree, "linear": linear}

    def get_total_trees(self) -> int:
        return int(self.n_runs * self.n_cycles * self.trees_per_cycle)

    def get_linear_feature_summary(self) -> dict[str, int]:
        order = np.argsort(self.linear_feature_counts_)[::-1]
        return {
            self.feature_names_[idx]: int(self.linear_feature_counts_[idx])
            for idx in order
            if self.linear_feature_counts_[idx] > 0
        }

    def channel_importance(
        self,
        X: Any | None = None,
        y: Any | None = None,
    ) -> pd.DataFrame:
        return _lgbplus_channel_frame(
            feature_names=self.feature_names_,
            tree_gain=self.feature_importances_,
            linear_counts=self.linear_feature_counts_,
            linear_abs_contribution=self.linear_abs_contribution_,
        )

    def summary(self) -> str:
        if not hasattr(self, "runs_"):
            return "Model not fitted yet."
        return "\n".join(
            [
                "LGB^A+ Alternating Summary",
                f"Runs: {self.n_runs}",
                f"Cycles: {self.n_cycles}",
                f"Trees per cycle: {self.trees_per_cycle}",
                f"Total trees: {self.get_total_trees()}",
            ]
        )


def lgb_plus(
    X: Any,
    y: Any | None = None,
    *,
    n_ensemble: int = 10,
    n_steps: int = 200,
    learning_rate: float = 0.05,
    subsample: float = 0.7,
    num_leaves: int = 5,
    min_data_in_leaf: int = 20,
    lambda_l2: float = 0.1,
    linear_candidate_fraction: float = 0.5,
    selection_method: Literal["oob", "validation", "training"] = "oob",
    val_fraction: float = 0.2,
    early_stop_patience: int | None = 50,
    aggregation: Literal["mean", "median"] = "mean",
    random_state: int | None = 0,
    verbose: bool = False,
    **kwargs: Any,
) -> ModelFit:
    """Fit competition-based LGB+ hybrid tree/linear boosting."""

    params = {
        "n_ensemble": int(n_ensemble),
        "n_steps": int(n_steps),
        "learning_rate": float(learning_rate),
        "subsample": float(subsample),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "lambda_l2": float(lambda_l2),
        "linear_candidate_fraction": float(linear_candidate_fraction),
        "selection_method": selection_method,
        "val_fraction": float(val_fraction),
        "early_stop_patience": early_stop_patience,
        "aggregation": aggregation,
        "random_state": random_state,
        "verbose": bool(verbose),
        "implementation_note": (
            "Package-native implementation aligned to philgoucou/lgbplus "
            "python/lgb_plus.py and R/lgb_plus.R; LightGBM supplies residual "
            "tree boosters, macroforecast owns pandas IO, metadata, and "
            "component diagnostics."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    return fit_estimator(
        LGBPlusRegressor(**estimator_params),
        X,
        y,
        model="lgb_plus",
        metadata=params,
    )


def lgba_plus(
    X: Any,
    y: Any | None = None,
    *,
    n_runs: int = 1,
    n_cycles: int = 25,
    trees_per_cycle: int = 10,
    lr_tree: float = 0.02,
    lr_linear: float = 0.1,
    num_leaves: int = 15,
    min_data_in_leaf: int = 20,
    subsample: float = 1.0,
    random_state: int | None = 0,
    verbose: bool = False,
    **kwargs: Any,
) -> ModelFit:
    """Fit alternating LGB^A+ hybrid tree/linear boosting."""

    params = {
        "n_runs": int(n_runs),
        "n_cycles": int(n_cycles),
        "trees_per_cycle": int(trees_per_cycle),
        "lr_tree": float(lr_tree),
        "lr_linear": float(lr_linear),
        "num_leaves": int(num_leaves),
        "min_data_in_leaf": int(min_data_in_leaf),
        "subsample": float(subsample),
        "random_state": random_state,
        "verbose": bool(verbose),
        "implementation_note": (
            "Package-native implementation aligned to philgoucou/lgbplus "
            "python/lgb_plus_A.py and R/lgb_plus_A.R; n_runs absorbs the R "
            "ensemble helper into the estimator API."
        ),
        **kwargs,
    }
    estimator_params = {
        key: value for key, value in params.items() if key != "implementation_note"
    }
    return fit_estimator(
        LGBAPlusRegressor(**estimator_params),
        X,
        y,
        model="lgba_plus",
        metadata=params,
    )


def _validate_quantile_levels(levels: Sequence[float]) -> tuple[float, ...]:
    quantiles = tuple(float(level) for level in levels)
    if not quantiles or any(level <= 0.0 or level >= 1.0 for level in quantiles):
        raise ValueError("quantile levels must contain values in (0, 1)")
    return quantiles


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    total = float(sorted_weights.sum())
    if total <= 0.0:
        return float(sorted_values[-1])
    cumulative = np.cumsum(sorted_weights) / total
    index = min(
        int(np.searchsorted(cumulative, float(quantile), side="left")),
        len(sorted_values) - 1,
    )
    return float(sorted_values[index])


class QuantileRegressionForestRegressor:
    """Random-forest point forecasts plus empirical leaf quantiles."""

    # Source comparison:
    # - Meinshausen/quantregForest-style QRF estimates a conditional response
    #   distribution by assigning training responses forest weights from shared
    #   terminal leaves. R quantregForest stores in-leaf responses directly;
    #   ranger/grf expose related quantile forest APIs with different forest
    #   builders and inference surfaces.
    # - This class follows the QRF leaf-weight skeleton on top of sklearn's
    #   RandomForestRegressor. For a test row, each tree contributes every
    #   training target in the matching leaf with inverse leaf-size weight.
    #   The common 1 / n_estimators multiplier cancels in weighted quantiles.
    # - This is not a full quantregForest/ranger/grf clone: forest fitting,
    #   splitting, OOB behavior, uncertainty tooling, and optional honesty are
    #   inherited from or absent in sklearn. macroforecast owns the callable
    #   X/y contract, leaf target store, quantile extraction, and metadata.

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 0,
        quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95),
    ) -> None:
        self.n_estimators = max(1, int(n_estimators))
        self.max_depth = max_depth
        self.min_samples_leaf = max(1, int(min_samples_leaf))
        self.random_state = int(random_state)
        self.quantile_levels = _validate_quantile_levels(quantile_levels)
        self._forest: Any = None
        self._leaf_targets: list[dict[int, np.ndarray]] = []
        self._fallback: float = 0.0

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
        self._fallback = float(np.mean(target)) if target.size else 0.0
        self._forest.fit(frame, target)
        leaves = self._forest.apply(frame)
        self._leaf_targets = []
        for tree_idx in range(leaves.shape[1]):
            tree_leaves = leaves[:, tree_idx]
            self._leaf_targets.append(
                {
                    int(leaf): target[tree_leaves == leaf]
                    for leaf in np.unique(tree_leaves)
                }
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._forest is None:
            return np.zeros(len(X), dtype=float)
        return np.asarray(self._forest.predict(X.fillna(0.0)), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        if self._forest is None:
            return None
        return np.asarray(self._forest.feature_importances_, dtype=float)

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        levels: tuple[float, ...] | None = None,
    ) -> dict[float, np.ndarray]:
        levels = (
            self.quantile_levels
            if levels is None
            else _validate_quantile_levels(levels)
        )
        if self._forest is None or not self._leaf_targets:
            return {q: np.full(len(X), self._fallback, dtype=float) for q in levels}
        leaves = self._forest.apply(X.fillna(0.0))
        out: dict[float, np.ndarray] = {
            q: np.empty(len(X), dtype=float) for q in levels
        }
        for i in range(len(X)):
            samples: list[float] = []
            weights: list[float] = []
            for tree_idx in range(leaves.shape[1]):
                values = self._leaf_targets[tree_idx].get(
                    int(leaves[i, tree_idx]), np.array([])
                )
                leaf_values = np.asarray(values, dtype=float)
                if leaf_values.size == 0:
                    continue
                samples.extend(leaf_values.tolist())
                # QRF forest weights: within each matching terminal leaf, every
                # training response receives equal mass, scaled by leaf size.
                weights.extend([1.0 / float(leaf_values.size)] * int(leaf_values.size))
            arr = np.asarray(samples if samples else [self._fallback], dtype=float)
            weight_arr = np.asarray(weights if weights else [1.0], dtype=float)
            for q in levels:
                out[q][i] = float(_weighted_quantile(arr, weight_arr, q))
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
        "implementation_note": (
            "Meinshausen/quantregForest-style empirical leaf-weight quantiles "
            "over sklearn RandomForestRegressor terminal leaves."
        ),
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


class MacroRandomForestRegressor:
    """Adapter for the vendored MacroRandomForest reference implementation."""

    # Upstream comparison, checked against:
    # - Python MacroRandomForest 1.0.6 (RyanLucas3/MacroRandomForest, MIT).
    # - R prototype philgoucou/macrorf::MRF (GPL-3).
    #
    # Both upstream APIs expect one numeric matrix/data frame that already
    # contains the target, local-linear variables, state/split variables, and
    # out-of-sample rows. R exposes 1-based `y.pos`, `x.pos`, `S.pos`,
    # `oos.pos`; Python exposes `y_pos`, `x_pos`, `S_pos`, `oos_pos` and uses
    # the same data-matrix contract. Both implementations then translate names
    # internally: user `x_pos` becomes `z_pos` for local-linear coefficient
    # variables, and user `S_pos` becomes the split/state variable set.
    #
    # macroforecast intentionally does not reimplement the MRF tree, ridge, or
    # random-walk-regularization math. It vendors the Python reference backend
    # and adapts only the package-facing contract: callable `X, y` inputs,
    # optional column-name selection, metadata, output normalization, cache, and
    # contextual error messages.
    #
    # Main differences from upstream:
    # - `fit(X, y)` stores aligned pandas data; the expensive reference
    #   `_ensemble_loop()` is run lazily from `predict(X_test)`.
    # - `predict()` builds the upstream single matrix by prepending the stored
    #   target and adding dummy zero targets for OOS rows, then derives
    #   `oos_pos` from the train/test boundary.
    # - `x_columns` and `S_columns` are mapped to 1..k feature positions because
    #   the constructed backend matrix always has the target at column 0.
    # - The separated `X, y` API fixes the target location at column 0; a
    #   nonzero upstream-style `y_pos` would point at a feature after matrix
    #   construction, so it is rejected below instead of silently changing the
    #   dependent variable.
    # - R's `pred.given.mrf()` keep-forest path is not called here. The adapter
    #   follows the direct Python/R MRF path by rebuilding the reference object
    #   for each distinct test matrix, while caching repeated predictions.
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
        if int(y_pos) != 0:
            raise ValueError(
                "macro_random_forest receives target values through y, so y_pos "
                "must be 0. Use x_columns/S_columns or x_pos/S_pos to select "
                "feature roles."
            )
        if x_columns is not None and x_pos is not None:
            raise ValueError("Use either x_columns or x_pos, not both.")
        if S_columns is not None and S_pos is not None:
            raise ValueError("Use either S_columns or S_pos, not both.")
        self.x_columns = (
            None if x_columns is None else tuple(str(column) for column in x_columns)
        )
        self.S_columns = (
            None if S_columns is None else tuple(str(column) for column in S_columns)
        )
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
            resolved = [int(pos) for pos in positions]
            invalid = [
                pos for pos in resolved if pos < 1 or pos > int(len(feature_index))
            ]
            if invalid:
                raise ValueError(
                    "macro_random_forest positions must refer to feature columns "
                    f"1..{len(feature_index)}; got {invalid}"
                )
            return resolved
        if columns is None:
            return list(range(1, len(feature_index) + 1))
        missing = [column for column in columns if column not in feature_index]
        if missing:
            raise ValueError(f"macro_random_forest columns not found in X: {missing}")
        return [int(feature_index.get_loc(column)) + 1 for column in columns]

    @staticmethod
    def _prediction_values(output: dict[str, Any], n: int) -> np.ndarray:
        # Python backend output uses `pred_ensemble`/`S_names`; the R prototype
        # reports analogous fields as `pred.ensemble`/`S.names`. Since this
        # adapter calls the Python backend, only Python keys are accepted here.
        if n == 0:
            return np.array([], dtype=float)
        values = output.get("pred_ensemble")
        if values is None:
            values = output.get("pred")
        if values is None:
            raise RuntimeError(
                "MacroRandomForest backend did not return 'pred_ensemble' or 'pred'."
            )
        if isinstance(values, (pd.Series, pd.DataFrame)):
            arr = values.to_numpy(dtype=float)
        else:
            arr = np.asarray(values, dtype=float)
        if arr.ndim == 2:
            if arr.shape[0] == n:
                arr = arr.reshape(-1) if arr.shape[1] == 1 else arr.mean(axis=1)
            elif arr.shape[1] == n:
                arr = arr.mean(axis=0)
            else:
                arr = arr.reshape(-1)
        out = np.asarray(arr, dtype=float).reshape(-1)
        if len(out) < n:
            raise RuntimeError(
                "MacroRandomForest backend returned fewer predictions than requested."
            )
        return out[-n:]

    @staticmethod
    def _cache_key(X: pd.DataFrame) -> tuple[Any, ...]:
        try:
            hashed = pd.util.hash_pandas_object(X, index=True).to_numpy(dtype=np.uint64)
            value_hash = int(np.bitwise_xor.reduce(hashed)) if len(hashed) else 0
        except Exception:  # noqa: BLE001 - cache keys must never block prediction.
            value_hash = id(X)
        return (
            tuple(X.index),
            tuple(str(column) for column in X.columns),
            X.shape,
            value_hash,
        )


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
    "LGBAPlusRegressor",
    "LGBPlusRegressor",
    "MacroRandomForestRegressor",
    "QuantileRegressionForestRegressor",
    "catboost",
    "decision_tree",
    "extra_trees",
    "gradient_boosting",
    "lgba_plus",
    "lgb_plus",
    "lightgbm",
    "macro_random_forest",
    "quantile_regression_forest",
    "random_forest",
    "xgboost",
]
