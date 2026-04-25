from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, QuantileRegressor, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR

try:
    from xgboost import XGBRegressor
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    LGBMRegressor = None

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:  # pragma: no cover - depends on optional extra
    CatBoostRegressor = None

from ..tuning import HPDistribution, TuningSpec, run_tuning
from ..tuning.hp_spaces import MODEL_HP_SPACES
from .seed_policy import current_seed


def _optional_estimator_cls(cls, *, package: str, extra: str):
    if cls is None:
        raise RuntimeError(
            f"{package} is required for this model_family; install macrocast[{extra}] or macrocast[all]."
        )
    return cls


class ComponentwiseBoostingRegressor:
    def __init__(self, n_iterations: int = 50, learning_rate: float = 0.1):
        self.n_iterations = int(n_iterations)
        self.learning_rate = float(learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        residuals = y.copy()
        self.intercept_ = float(np.mean(y))
        residuals -= self.intercept_
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        for _ in range(self.n_iterations):
            best_j, best_coef, best_sse = 0, 0.0, float("inf")
            for j in range(X.shape[1]):
                xj = X[:, j]
                coef = float(np.dot(xj, residuals) / (np.dot(xj, xj) + 1e-10))
                sse = float(np.sum((residuals - coef * xj) ** 2))
                if sse < best_sse:
                    best_j, best_coef, best_sse = j, coef, sse
            self.coef_[best_j] += self.learning_rate * best_coef
            residuals -= self.learning_rate * best_coef * X[:, best_j]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


class BoostingRidgeRegressor:
    def __init__(self, n_iterations: int = 50, learning_rate: float = 0.1, ridge_alpha: float = 1.0):
        self.n_iterations = int(n_iterations)
        self.learning_rate = float(learning_rate)
        self.ridge_alpha = float(ridge_alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        residuals = y.copy()
        self.intercept_ = float(np.mean(y))
        residuals -= self.intercept_
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        for _ in range(self.n_iterations):
            model = Ridge(alpha=self.ridge_alpha).fit(X, residuals)
            self.coef_ += self.learning_rate * model.coef_
            residuals -= self.learning_rate * model.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


class BoostingLassoRegressor:
    def __init__(self, n_iterations: int = 50, learning_rate: float = 0.1, lasso_alpha: float = 1e-4):
        self.n_iterations = int(n_iterations)
        self.learning_rate = float(learning_rate)
        self.lasso_alpha = float(lasso_alpha)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        residuals = y.copy()
        self.intercept_ = float(np.mean(y))
        residuals -= self.intercept_
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        for _ in range(self.n_iterations):
            model = Lasso(alpha=self.lasso_alpha, max_iter=10000).fit(X, residuals)
            self.coef_ += self.learning_rate * model.coef_
            residuals -= self.learning_rate * model.predict(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def make_model_instance(model_family: str, hp: dict[str, Any] | None = None):
    hp = dict(hp or {})
    if model_family == "ols":
        return LinearRegression()
    if model_family == "ridge":
        return Ridge(alpha=float(hp.get("alpha", 1.0)))
    if model_family == "lasso":
        return Lasso(alpha=float(hp.get("alpha", 1e-4)), max_iter=10000)
    if model_family == "elasticnet":
        return ElasticNet(alpha=float(hp.get("alpha", 1e-4)), l1_ratio=float(hp.get("l1_ratio", 0.5)), max_iter=10000)
    if model_family == "bayesianridge":
        return BayesianRidge()
    if model_family == "huber":
        return HuberRegressor(epsilon=float(hp.get("epsilon", 1.35)), alpha=float(hp.get("alpha", 0.0001)))
    if model_family == "quantile_linear":
        return QuantileRegressor(
            quantile=float(hp.get("quantile", 0.5)),
            alpha=float(hp.get("alpha", 1.0)),
            solver="highs",
        )
    if model_family == "svr_linear":
        return LinearSVR(C=float(hp.get("C", 1.0)), epsilon=float(hp.get("epsilon", 0.01)), max_iter=50000, random_state=current_seed(model_family="svr_linear"))
    if model_family == "svr_rbf":
        return SVR(kernel="rbf", C=float(hp.get("C", 1.0)), epsilon=float(hp.get("epsilon", 0.01)), gamma=hp.get("gamma", "scale"))
    if model_family == "componentwise_boosting":
        return ComponentwiseBoostingRegressor(
            n_iterations=int(hp.get("n_iterations", 50)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
        )
    if model_family == "boosting_ridge":
        return BoostingRidgeRegressor(
            n_iterations=int(hp.get("n_iterations", 50)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            ridge_alpha=float(hp.get("ridge_alpha", 1.0)),
        )
    if model_family == "boosting_lasso":
        return BoostingLassoRegressor(
            n_iterations=int(hp.get("n_iterations", 50)),
            learning_rate=float(hp.get("learning_rate", 0.1)),
            lasso_alpha=float(hp.get("lasso_alpha", 1e-4)),
        )
    if model_family == "randomforest":
        return RandomForestRegressor(
            n_estimators=int(hp.get("n_estimators", 200)),
            max_depth=None if hp.get("max_depth") is None else int(hp.get("max_depth")),
            random_state=current_seed(model_family="randomforest"),
        )
    if model_family == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=int(hp.get("n_estimators", 200)),
            max_depth=None if hp.get("max_depth") is None else int(hp.get("max_depth")),
            random_state=current_seed(model_family="extratrees"),
        )
    if model_family == "gbm":
        return GradientBoostingRegressor(
            n_estimators=int(hp.get("n_estimators", 100)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            max_depth=int(hp.get("max_depth", 3)),
            random_state=current_seed(model_family="gbm"),
        )
    if model_family == "xgboost":
        estimator_cls = _optional_estimator_cls(XGBRegressor, package="xgboost", extra="xgboost")
        return estimator_cls(
            n_estimators=int(hp.get("n_estimators", 100)),
            max_depth=int(hp.get("max_depth", 3)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            random_state=current_seed(model_family="xgboost"),
            verbosity=0,
        )
    if model_family == "lightgbm":
        estimator_cls = _optional_estimator_cls(LGBMRegressor, package="lightgbm", extra="lightgbm")
        return estimator_cls(
            n_estimators=int(hp.get("n_estimators", 100)),
            num_leaves=int(hp.get("num_leaves", 31)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            random_state=current_seed(model_family="lightgbm"),
            verbosity=-1,
        )
    if model_family == "catboost":
        estimator_cls = _optional_estimator_cls(CatBoostRegressor, package="catboost", extra="catboost")
        return estimator_cls(
            iterations=int(hp.get("iterations", 100)),
            learning_rate=float(hp.get("learning_rate", 0.05)),
            depth=int(hp.get("depth", 4)),
            verbose=False,
            random_seed=current_seed(model_family="catboost"),
        )
    if model_family == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=hp.get("hidden_layer_sizes", (32,)),
            alpha=float(hp.get("alpha", 1e-4)),
            learning_rate_init=float(hp.get("learning_rate_init", 1e-3)),
            max_iter=500,
            random_state=current_seed(model_family="mlp"),
        )
    raise ValueError(f"unsupported model_family {model_family!r}")


def fit_adaptive_lasso(X: np.ndarray, y: np.ndarray, hp: dict[str, Any] | None = None) -> object:
    hp = dict(hp or {})
    gamma = float(hp.get("gamma", 1.0))
    init_estimator = str(hp.get("init_estimator", "ridge"))
    alpha = float(hp.get("alpha", 1e-3))
    init = Ridge(alpha=1.0).fit(X, y) if init_estimator == "ridge" else LinearRegression().fit(X, y)
    weights = 1.0 / (np.abs(init.coef_) ** gamma + 1e-6)
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X / weights, y)
    model._adaptive_weights = weights
    return model


def predict_adaptive_lasso(model, X: np.ndarray) -> np.ndarray:
    return model.predict(X / model._adaptive_weights)


def _default_hp_space(model_family: str, X_train: np.ndarray) -> dict[str, HPDistribution]:
    hp_space = dict(MODEL_HP_SPACES.get(model_family, {}))
    if model_family in {"pcr", "pls"}:
        max_comp = max(1, min(10, X_train.shape[0], X_train.shape[1]))
        hp_space = {"n_components": HPDistribution("int", 1, max_comp)}
    return hp_space


def _tuning_budget_spec(training_spec: dict[str, Any]) -> dict[str, Any]:
    early_stopping = str(training_spec.get("early_stopping", "none"))
    early_stop_trials = training_spec.get("early_stop_trials")
    if early_stopping == "none":
        early_stop_trials = None
    elif early_stop_trials is None:
        early_stop_trials = 3
    min_improvement = 0.0
    if early_stopping == "loss_plateau":
        min_improvement = float(training_spec.get("early_stop_min_delta", 1e-4))
    return {
        "max_trials": int(training_spec.get("max_trials", 6)),
        "max_time_seconds": float(training_spec.get("max_time_seconds", 15.0)),
        "early_stop_trials": None if early_stop_trials is None else int(early_stop_trials),
        "early_stopping": early_stopping,
        "min_improvement": min_improvement,
    }


def _fit_without_tuning(model_family: str, X_train: np.ndarray, y_train: np.ndarray, training_spec: dict[str, Any]):
    if model_family == "adaptivelasso":
        return fit_adaptive_lasso(X_train, y_train), {}
    if model_family == "quantile_linear":
        quantile = float(training_spec.get("quantile_level", 0.5))
        model = make_model_instance(model_family, {"quantile": quantile, "alpha": 1.0})
        model.fit(X_train, y_train)
        return model, {"quantile": quantile}
    model = make_model_instance(model_family)
    model.fit(X_train, y_train)
    return model, {}


def fit_with_optional_tuning(model_family: str, X_train: np.ndarray, y_train: np.ndarray, training_spec: dict[str, Any]):
    algo = training_spec.get("search_algorithm", "grid_search")
    convergence_handling = training_spec.get("convergence_handling", "mark_fail")
    supported_algorithms = {"grid_search", "random_search", "bayesian_optimization", "genetic_algorithm"}
    if not bool(training_spec.get("enable_tuning", False)) or algo not in supported_algorithms or model_family == "quantile_linear":
        try:
            return _fit_without_tuning(model_family, X_train, y_train, training_spec)
        except Exception:
            if convergence_handling == "fallback_to_safe_hp":
                if model_family == "adaptivelasso":
                    return fit_adaptive_lasso(X_train, y_train, {"gamma": 1.0}), {"fallback": True}
                model = make_model_instance(model_family)
                model.fit(X_train, y_train)
                return model, {"fallback": True}
            raise
    tuning_spec = TuningSpec(
        search_algorithm=algo,
        tuning_objective=training_spec.get("tuning_objective", "validation_mse"),
        tuning_budget=_tuning_budget_spec(training_spec),
        hp_space=_default_hp_space(model_family, X_train),
        validation_size_rule=training_spec.get("validation_size_rule", "ratio"),
        validation_size_config={
            "ratio": float(training_spec.get("validation_ratio", 0.2)),
            "n": int(training_spec.get("validation_n", 5)),
            "years": int(training_spec.get("validation_years", 1)),
            "obs_per_year": int(training_spec.get("obs_per_year", 12)),
        },
        validation_location=training_spec.get("validation_location", "last_block"),
        embargo_gap=training_spec.get("embargo_gap", "none"),
        embargo_gap_size=int(training_spec.get("embargo_gap_size", 0)),
        seed=int(training_spec.get("random_seed", 42)),
    )
    if model_family == "adaptivelasso":
        class _AdaptiveWrap:
            def __init__(self, hp):
                self.hp = hp

            def fit(self, X, y):
                self.model = fit_adaptive_lasso(X, y, self.hp)
                return self

            def predict(self, X):
                return predict_adaptive_lasso(self.model, X)

        try:
            result = run_tuning(model_family, lambda hp: _AdaptiveWrap(hp), X_train, y_train, tuning_spec)
            return fit_adaptive_lasso(X_train, y_train, result.best_hp), {
                "best_hp": result.best_hp,
                "best_score": result.best_score,
                "total_trials": result.total_trials,
            }
        except Exception:
            if convergence_handling == "fallback_to_safe_hp":
                return fit_adaptive_lasso(X_train, y_train, {"gamma": 1.0}), {"fallback": True}
            raise
    try:
        result = run_tuning(model_family, lambda hp: make_model_instance(model_family, hp), X_train, y_train, tuning_spec)
        model = make_model_instance(model_family, result.best_hp)
        model.fit(X_train, y_train)
        return model, {
            "best_hp": result.best_hp,
            "best_score": result.best_score,
            "total_trials": result.total_trials,
        }
    except Exception:
        if convergence_handling == "fallback_to_safe_hp":
            model = make_model_instance(model_family)
            model.fit(X_train, y_train)
            return model, {"fallback": True}
        raise


def resolve_factor_count(X_train: np.ndarray, y_train: np.ndarray, training_spec: dict[str, Any]) -> int:
    mode = training_spec.get("factor_count", "fixed")
    max_k = max(1, min(int(training_spec.get("max_factors", 5)), X_train.shape[0], X_train.shape[1]))
    if mode == "fixed":
        return max(1, min(int(training_spec.get("fixed_factor_count", 3)), max_k))
    if mode == "cv_select":
        best_k, best_sse = 1, math.inf
        for k in range(1, max_k + 1):
            pca = PCA(n_components=k).fit(X_train)
            scores = pca.transform(X_train)
            recon = pca.inverse_transform(scores)
            sse = float(np.mean((X_train - recon) ** 2))
            if sse < best_sse:
                best_k, best_sse = k, sse
        return best_k
    if mode == "BaiNg_rule":
        Xc = X_train - X_train.mean(axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        T, N = X_train.shape[0], X_train.shape[1]
        penalties = []
        for k in range(1, max_k + 1):
            sigma2 = float(np.sum(s[k:] ** 2) / (T * N)) if k < len(s) else 0.0
            ic = math.log(max(sigma2, 1e-12)) + k * ((N + T) / (N * T)) * math.log((N * T) / (N + T))
            penalties.append((ic, k))
        return min(penalties)[1]
    return max(1, min(3, max_k))


def _target_lag_order(training_spec: dict[str, Any]) -> int:
    return int(training_spec.get("target_lag_count", training_spec.get("factor_ar_lags", 1)))


def build_factor_panel(
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_pred_df: pd.DataFrame,
    training_spec: dict[str, Any],
    include_ar_lags: bool = False,
    fit_state_sink: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train_df.to_numpy(dtype=float)
    X_pred = X_pred_df.to_numpy(dtype=float)
    n_components = resolve_factor_count(X_train, y_train, training_spec)
    pca = PCA(n_components=n_components)
    F_train = pca.fit_transform(X_train)
    F_pred = pca.transform(X_pred)
    if fit_state_sink is not None:
        source_names = [str(col) for col in X_train_df.columns]
        feature_names = [f"factor_{idx}" for idx in range(1, int(n_components) + 1)]
        lag_order = _target_lag_order(training_spec)
        target_lag_feature_names = [f"target_lag_{idx}" for idx in range(1, lag_order + 1)] if include_ar_lags else []
        fit_state_sink.append(
            {
                "block": "pca_static_factors",
                "runtime_policy": "factor_model",
                "n_components": int(n_components),
                "factor_count_mode": training_spec.get("factor_count", "fixed"),
                "feature_names": feature_names + target_lag_feature_names,
                "factor_feature_names": feature_names,
                "target_lag_feature_names": target_lag_feature_names,
                "source_feature_names": source_names,
                "train_window_rows": int(X_train_df.shape[0]),
                "train_window_columns": int(X_train_df.shape[1]),
                "center_mean": [float(x) for x in np.asarray(pca.mean_, dtype=float).reshape(-1)],
                "explained_variance_ratio": [float(x) for x in np.asarray(pca.explained_variance_ratio_, dtype=float)],
                "loadings": {
                    feature: {source: float(value) for source, value in zip(source_names, row)}
                    for feature, row in zip(feature_names, np.asarray(pca.components_, dtype=float))
                },
                "include_target_lags": bool(include_ar_lags),
            }
        )
    if not include_ar_lags:
        return F_train, F_pred
    lag_order = _target_lag_order(training_spec)
    if len(y_train) <= lag_order:
        raise ValueError("insufficient target history for factors_plus_AR")
    y_lags = []
    for idx in range(lag_order, len(y_train)):
        y_lags.append(y_train[idx - lag_order : idx][::-1])
    F_train = F_train[lag_order:]
    lag_arr = np.asarray(y_lags, dtype=float)
    X_aug = np.concatenate([F_train, lag_arr], axis=1)
    pred_lags = np.asarray(y_train[-lag_order:][::-1], dtype=float).reshape(1, -1)
    X_pred_aug = np.concatenate([F_pred, pred_lags], axis=1)
    return X_aug, X_pred_aug


def fit_factor_model(
    model_family: str,
    X_train_df: pd.DataFrame,
    y_train: np.ndarray,
    X_pred_df: pd.DataFrame,
    training_spec: dict[str, Any],
    include_ar_lags: bool = False,
) -> tuple[float, dict[str, Any]]:
    lag_order = _target_lag_order(training_spec)
    fit_state: list[dict[str, Any]] = []
    if model_family == "pcr":
        X_train, X_pred = build_factor_panel(
            X_train_df,
            y_train,
            X_pred_df,
            {**training_spec, "factor_count": training_spec.get("factor_count", "fixed")},
            include_ar_lags=False,
            fit_state_sink=fit_state,
        )
        model, tuning = fit_with_optional_tuning("ols", X_train, y_train, training_spec)
        payload: dict[str, Any] = {"tuning": tuning}
        if fit_state:
            payload["feature_representation_fit_state"] = fit_state[-1]
        return float(model.predict(X_pred)[0]), payload
    if model_family == "pls":
        n_components = resolve_factor_count(X_train_df.to_numpy(dtype=float), y_train, training_spec)
        model = PLSRegression(n_components=n_components)
        model.fit(X_train_df.to_numpy(dtype=float), y_train)
        return float(model.predict(X_pred_df.to_numpy(dtype=float))[0]), {"n_components": n_components}
    if model_family == "factor_augmented_linear":
        X_train, X_pred = build_factor_panel(X_train_df, y_train, X_pred_df, training_spec, include_ar_lags=True, fit_state_sink=fit_state)
        model, tuning = fit_with_optional_tuning("ols", X_train, y_train[lag_order:], training_spec)
        payload = {"tuning": tuning}
        if fit_state:
            payload["feature_representation_fit_state"] = fit_state[-1]
        return float(model.predict(X_pred)[0]), payload
    X_train, X_pred = build_factor_panel(X_train_df, y_train, X_pred_df, training_spec, include_ar_lags=include_ar_lags, fit_state_sink=fit_state)
    y_used = y_train[lag_order:] if include_ar_lags else y_train
    base_family = "ols" if model_family == "factor_pca" else model_family
    model, tuning = fit_with_optional_tuning(base_family, X_train, y_used, training_spec)
    pred = model.predict(X_pred)
    payload = {"tuning": tuning}
    if fit_state:
        payload["feature_representation_fit_state"] = fit_state[-1]
    return float(pred[0]), payload
