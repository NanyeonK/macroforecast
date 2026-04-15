from __future__ import annotations

import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import math
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from statsmodels.tsa.ar_model import AutoReg

from .errors import ExecutionError
from .types import ExecutionResult, ExecutionSpec
from .deep_training import fit_factor_model, fit_with_optional_tuning, fit_adaptive_lasso, predict_adaptive_lasso
from ..preprocessing import (
    PreprocessContract,
    is_operational_preprocess_contract,
    preprocess_summary,
    preprocess_to_dict,
)
from ..raw import load_fred_md, load_fred_qd, load_fred_sd
from ..recipes import RecipeSpec, RunSpec, build_run_spec, recipe_summary

_EXECUTION_ARCHITECTURE = "separate_model_and_benchmark_executors"
_DEFAULT_MINIMUM_TRAIN_SIZE = 5
_DEFAULT_MAX_AR_LAG = 3
_LAG_SELECTION = "bic"


def _normal_two_sided_pvalue(statistic: float) -> float:
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def build_execution_spec(
    *,
    recipe: RecipeSpec,
    run: RunSpec,
    preprocess: PreprocessContract,
) -> ExecutionSpec:
    return ExecutionSpec(recipe=recipe, run=run, preprocess=preprocess)


def _load_raw_for_recipe(recipe: RecipeSpec, local_raw_source: str | Path | None, cache_root: Path):
    vintage = recipe.data_vintage
    if recipe.raw_dataset == "fred_md":
        return load_fred_md(vintage=vintage, cache_root=cache_root, local_source=local_raw_source)
    if recipe.raw_dataset == "fred_qd":
        return load_fred_qd(vintage=vintage, cache_root=cache_root, local_source=local_raw_source)
    if recipe.raw_dataset == "fred_sd":
        return load_fred_sd(vintage=vintage, cache_root=cache_root, local_source=local_raw_source)
    raise ExecutionError(f"unsupported raw_dataset={recipe.raw_dataset!r}")


def _benchmark_spec(recipe: RecipeSpec) -> dict[str, object]:
    spec = dict(recipe.benchmark_config)
    spec.setdefault("benchmark_family", recipe.stage0.fixed_design.benchmark)
    spec.setdefault("minimum_train_size", _DEFAULT_MINIMUM_TRAIN_SIZE)
    spec.setdefault("max_ar_lag", _DEFAULT_MAX_AR_LAG)
    return spec


def _minimum_train_size(recipe: RecipeSpec) -> int:
    return int(_benchmark_spec(recipe)["minimum_train_size"])


def _max_ar_lag(recipe: RecipeSpec) -> int:
    return int(_benchmark_spec(recipe)["max_ar_lag"])


def _rolling_window_size(recipe: RecipeSpec) -> int:
    benchmark_spec = _benchmark_spec(recipe)
    return int(benchmark_spec.get("rolling_window_size", benchmark_spec["minimum_train_size"]))


def _benchmark_family(recipe: RecipeSpec) -> str:
    return str(_benchmark_spec(recipe)["benchmark_family"])


def _model_family(recipe: RecipeSpec) -> str:
    return recipe.stage0.varying_design.model_families[0] if recipe.stage0.varying_design.model_families else "unknown"


def _feature_builder(recipe: RecipeSpec) -> str:
    if recipe.stage0.varying_design.feature_recipes:
        return recipe.stage0.varying_design.feature_recipes[0]
    return "autoreg_lagged_target"


def _recipe_targets(recipe: RecipeSpec) -> tuple[str, ...]:
    return recipe.targets if recipe.targets else (recipe.target,)


def _recipe_for_target(recipe: RecipeSpec, target: str) -> RecipeSpec:
    return replace(recipe, target=target, targets=())


def _model_executor_name(model_family: str, feature_builder: str) -> str:
    if feature_builder == "autoreg_lagged_target":
        return {
            "ar": "ar_bic_autoreg_v0",
            "ols": "ols_autoreg_v0",
            "ridge": "ridge_autoreg_v0",
            "lasso": "lasso_autoreg_v0",
            "elasticnet": "elasticnet_autoreg_v0",
            "bayesianridge": "bayesianridge_autoreg_v0",
            "huber": "huber_autoreg_v0",
            "adaptivelasso": "adaptivelasso_autoreg_v0",
            "svr_linear": "svr_linear_autoreg_v0",
            "svr_rbf": "svr_rbf_autoreg_v0",
            "randomforest": "randomforest_autoreg_v0",
            "extratrees": "extratrees_autoreg_v0",
            "gbm": "gbm_autoreg_v0",
            "xgboost": "xgboost_autoreg_v0",
            "lightgbm": "lightgbm_autoreg_v0",
            "catboost": "catboost_autoreg_v0",
            "mlp": "mlp_autoreg_v0",
            "componentwise_boosting": "componentwise_boosting_autoreg_v0",
            "boosting_ridge": "boosting_ridge_autoreg_v0",
            "boosting_lasso": "boosting_lasso_autoreg_v0",
            "pcr": "pcr_autoreg_v0",
            "pls": "pls_autoreg_v0",
            "factor_augmented_linear": "factor_augmented_linear_autoreg_v0",
            "quantile_linear": "quantile_linear_autoreg_v0",
        }[model_family]
    if feature_builder in {"raw_feature_panel", "raw_X_only", "factor_pca", "factors_plus_AR"}:
        return {
            "ols": "ols_raw_feature_panel_v0",
            "ridge": "ridge_raw_feature_panel_v0",
            "lasso": "lasso_raw_feature_panel_v0",
            "elasticnet": "elasticnet_raw_feature_panel_v0",
            "bayesianridge": "bayesianridge_raw_feature_panel_v0",
            "huber": "huber_raw_feature_panel_v0",
            "adaptivelasso": "adaptivelasso_raw_feature_panel_v0",
            "svr_linear": "svr_linear_raw_feature_panel_v0",
            "svr_rbf": "svr_rbf_raw_feature_panel_v0",
            "randomforest": "randomforest_raw_feature_panel_v0",
            "extratrees": "extratrees_raw_feature_panel_v0",
            "gbm": "gbm_raw_feature_panel_v0",
            "xgboost": "xgboost_raw_feature_panel_v0",
            "lightgbm": "lightgbm_raw_feature_panel_v0",
            "catboost": "catboost_raw_feature_panel_v0",
            "mlp": "mlp_raw_feature_panel_v0",
            "componentwise_boosting": "componentwise_boosting_raw_feature_panel_v0",
            "boosting_ridge": "boosting_ridge_raw_feature_panel_v0",
            "boosting_lasso": "boosting_lasso_raw_feature_panel_v0",
            "pcr": "pcr_raw_feature_panel_v0",
            "pls": "pls_raw_feature_panel_v0",
            "factor_augmented_linear": "factor_augmented_linear_raw_feature_panel_v0",
            "quantile_linear": "quantile_linear_raw_feature_panel_v0",
        }[model_family]
    raise ExecutionError(f"feature_builder {feature_builder!r} is not executable in current runtime slice")


def _model_spec(recipe: RecipeSpec) -> dict[str, object]:
    model_family = _model_family(recipe)
    feature_builder = _feature_builder(recipe)
    return {
        "model_family": model_family,
        "feature_builder": feature_builder,
        "executor_name": _model_executor_name(model_family, feature_builder),
        "framework": recipe.stage0.fixed_design.sample_split,
        "lag_selection": _LAG_SELECTION if model_family == "ar" else "fixed_lag_feature_builder",
        "max_ar_lag": _max_ar_lag(recipe),
    }


def _get_model_executor(recipe: RecipeSpec):
    model_family = _model_family(recipe)
    feature_builder = _feature_builder(recipe)
    if feature_builder == "autoreg_lagged_target":
        dispatch = {
            "ar": _run_ar_model_executor,
            "ols": _run_ols_autoreg_executor,
            "ridge": _run_ridge_autoreg_executor,
            "lasso": _run_lasso_autoreg_executor,
            "elasticnet": _run_elasticnet_autoreg_executor,
            "bayesianridge": _run_bayesianridge_autoreg_executor,
            "huber": _run_huber_autoreg_executor,
            "adaptivelasso": _run_adaptivelasso_autoreg_executor,
            "svr_linear": _run_svr_linear_autoreg_executor,
            "svr_rbf": _run_svr_rbf_autoreg_executor,
            "randomforest": _run_randomforest_autoreg_executor,
            "extratrees": _run_extratrees_autoreg_executor,
            "gbm": _run_gbm_autoreg_executor,
            "xgboost": _run_xgboost_autoreg_executor,
            "lightgbm": _run_lightgbm_autoreg_executor,
            "catboost": _run_catboost_autoreg_executor,
            "mlp": _run_mlp_autoreg_executor,
            "componentwise_boosting": _run_componentwise_boosting_autoreg_executor,
            "boosting_ridge": _run_boosting_ridge_autoreg_executor,
            "boosting_lasso": _run_boosting_lasso_autoreg_executor,
            "pcr": _run_pcr_autoreg_executor,
            "pls": _run_pls_autoreg_executor,
            "factor_augmented_linear": _run_factor_augmented_linear_autoreg_executor,
            "quantile_linear": _run_quantile_linear_autoreg_executor,
        }
        if model_family in dispatch:
            return dispatch[model_family]
    if feature_builder in {"raw_feature_panel", "raw_X_only", "factor_pca", "factors_plus_AR"}:
        dispatch = {
            "ols": _run_ols_raw_panel_executor,
            "ridge": _run_ridge_raw_panel_executor,
            "lasso": _run_lasso_raw_panel_executor,
            "elasticnet": _run_elasticnet_raw_panel_executor,
            "bayesianridge": _run_bayesianridge_raw_panel_executor,
            "huber": _run_huber_raw_panel_executor,
            "adaptivelasso": _run_adaptivelasso_raw_panel_executor,
            "svr_linear": _run_svr_linear_raw_panel_executor,
            "svr_rbf": _run_svr_rbf_raw_panel_executor,
            "randomforest": _run_randomforest_raw_panel_executor,
            "extratrees": _run_extratrees_raw_panel_executor,
            "gbm": _run_gbm_raw_panel_executor,
            "xgboost": _run_xgboost_raw_panel_executor,
            "lightgbm": _run_lightgbm_raw_panel_executor,
            "catboost": _run_catboost_raw_panel_executor,
            "mlp": _run_mlp_raw_panel_executor,
            "componentwise_boosting": _run_componentwise_boosting_raw_panel_executor,
            "boosting_ridge": _run_boosting_ridge_raw_panel_executor,
            "boosting_lasso": _run_boosting_lasso_raw_panel_executor,
            "pcr": _run_pcr_raw_panel_executor,
            "pls": _run_pls_raw_panel_executor,
            "factor_augmented_linear": _run_factor_augmented_linear_raw_panel_executor,
            "quantile_linear": _run_quantile_linear_raw_panel_executor,
        }
        if model_family in dispatch:
            return dispatch[model_family]
    raise ExecutionError(
        f"model_family {model_family!r} with feature_builder {feature_builder!r} is not executable in current runtime slice"
    )


def _get_benchmark_executor(recipe: RecipeSpec):
    benchmark_family = _benchmark_family(recipe)
    if benchmark_family in {"historical_mean", "zero_change", "ar_bic", "custom_benchmark"}:
        return _run_benchmark_executor
    raise ExecutionError(f"benchmark_family {benchmark_family!r} is representable but not executable in current runtime slice")


def _get_target_series(frame: pd.DataFrame, target: str, minimum_train_size: int) -> pd.Series:
    if target not in frame.columns:
        raise ExecutionError(f"target {target!r} not found in raw dataset columns")
    series = frame[target].dropna().astype(float).copy()
    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq is not None:
        series.index = pd.DatetimeIndex(series.index, freq=inferred_freq)
    if len(series) < minimum_train_size + 1:
        raise ExecutionError(
            f"target {target!r} has insufficient non-missing observations for benchmark execution"
        )
    return series


def _fit_autoreg(train: pd.Series, lag: int):
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return AutoReg(train, lags=lag, trend="c", old_names=False).fit()


def _select_ar_bic_model(train: pd.Series, max_ar_lag: int) -> tuple[int, float, object]:
    max_candidate_lag = min(max_ar_lag, len(train) - 2)
    if max_candidate_lag < 1:
        raise ExecutionError("training window too small to fit any AR lag candidate")

    candidates: list[tuple[int, float, object]] = []
    for lag in range(1, max_candidate_lag + 1):
        try:
            fitted = _fit_autoreg(train, lag)
        except Exception:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bic = float(fitted.bic)
        if math.isnan(bic) or bic == math.inf:
            continue
        candidates.append((lag, bic, fitted))

    if not candidates:
        raise ExecutionError("no valid AR/BIC candidate could be fit for the current training window")

    return min(candidates, key=lambda item: item[1])


def _lag_order(recipe: RecipeSpec, train: pd.Series) -> int:
    lag_order = min(_max_ar_lag(recipe), len(train) - 1)
    if lag_order < 1:
        raise ExecutionError("training window too small for lagged supervised model")
    return lag_order


def _build_lagged_supervised_matrix(train: pd.Series, lag_order: int) -> tuple[np.ndarray, np.ndarray]:
    values = train.to_numpy(dtype=float)
    X, y = [], []
    for idx in range(lag_order, len(values)):
        X.append(values[idx - lag_order : idx][::-1])
        y.append(values[idx])
    if not X:
        raise ExecutionError("insufficient training data to build lagged supervised matrix")
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)


def _recursive_predict_sklearn(model, train: pd.Series, horizon: int, lag_order: int) -> float:
    history = list(train.to_numpy(dtype=float))
    for _ in range(horizon):
        features = np.asarray(history[-lag_order:][::-1], dtype=float).reshape(1, -1)
        pred = float(model.predict(features)[0])
        history.append(pred)
    return float(history[-1])


def _raw_panel_columns(frame: pd.DataFrame, target: str) -> list[str]:
    cols = [col for col in frame.columns if col != target]
    if not cols:
        raise ExecutionError("raw_feature_panel requires at least one non-target column")
    return cols


def _fill_missing_with_direction(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "ffill":
        return frame.ffill().bfill()
    if method == "interpolate_linear":
        return frame.interpolate(method="linear", axis=0, limit_direction="both")
    return frame


def _apply_missing_policy(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = contract.x_missing_policy
    if policy in {"none", "drop", "drop_rows", "drop_columns", "drop_if_above_threshold", "missing_indicator"}:
        return X_train, X_pred
    if policy in {"em_impute", "mean_impute"}:
        imputer = SimpleImputer(strategy="mean")
        return (
            pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns),
            pd.DataFrame(imputer.transform(X_pred), index=X_pred.index, columns=X_pred.columns),
        )
    if policy == "median_impute":
        imputer = SimpleImputer(strategy="median")
        return (
            pd.DataFrame(imputer.fit_transform(X_train), index=X_train.index, columns=X_train.columns),
            pd.DataFrame(imputer.transform(X_pred), index=X_pred.index, columns=X_pred.columns),
        )
    if policy in {"ffill", "interpolate_linear"}:
        combined = pd.concat([X_train, X_pred], axis=0)
        filled = _fill_missing_with_direction(combined, policy)
        return filled.iloc[: len(X_train)].copy(), filled.iloc[len(X_train) :].copy()
    raise ExecutionError(f"x_missing_policy {policy!r} is not executable in current runtime slice")


def _clip_frame(frame: pd.DataFrame, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
    return frame.clip(lower=lower, upper=upper, axis=1)


def _apply_outlier_policy(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = contract.x_outlier_policy
    if policy == "none":
        return X_train, X_pred
    if policy == "winsorize":
        lower = X_train.quantile(0.01)
        upper = X_train.quantile(0.99)
        return _clip_frame(X_train, lower, upper), _clip_frame(X_pred, lower, upper)
    if policy == "iqr_clip":
        q1 = X_train.quantile(0.25)
        q3 = X_train.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return _clip_frame(X_train, lower, upper), _clip_frame(X_pred, lower, upper)
    if policy == "zscore_clip":
        mean = X_train.mean()
        std = X_train.std(ddof=0).replace(0, 1.0)
        lower = mean - 3.0 * std
        upper = mean + 3.0 * std
        return _clip_frame(X_train, lower, upper), _clip_frame(X_pred, lower, upper)
    raise ExecutionError(f"x_outlier_policy {policy!r} is not executable in current runtime slice")


def _apply_scaling_policy(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = contract.scaling_policy
    if policy == "none":
        return X_train, X_pred
    scaler = None
    if policy == "standard":
        scaler = StandardScaler()
    elif policy == "robust":
        scaler = RobustScaler()
    elif policy == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ExecutionError(f"scaling_policy {policy!r} is not executable in current runtime slice")
    return (
        pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns),
        pd.DataFrame(scaler.transform(X_pred), index=X_pred.index, columns=X_pred.columns),
    )


def _apply_feature_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = contract.feature_selection_policy
    if policy == "none":
        return X_train, X_pred
    if policy == "correlation_filter":
        corrs = X_train.apply(lambda col: abs(pd.Series(col).corr(pd.Series(y_train))), axis=0).fillna(0.0)
        keep = corrs.sort_values(ascending=False).head(max(1, min(10, len(corrs)))).index.tolist()
        return X_train[keep].copy(), X_pred[keep].copy()
    if policy == "lasso_select":
        model = Lasso(alpha=1e-3, max_iter=10000)
        model.fit(X_train.to_numpy(dtype=float), y_train)
        coef = np.abs(model.coef_)
        keep_idx = np.where(coef > 1e-8)[0]
        if len(keep_idx) == 0:
            keep_idx = np.array([int(np.argmax(coef))]) if len(coef) else np.array([], dtype=int)
        keep = [X_train.columns[i] for i in keep_idx]
        return X_train[keep].copy(), X_pred[keep].copy()
    raise ExecutionError(f"feature_selection_policy {policy!r} is not executable in current runtime slice")


def _apply_dimensionality_reduction(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[np.ndarray, np.ndarray]:
    policy = contract.dimensionality_reduction_policy
    if policy == "none":
        return X_train.to_numpy(dtype=float), X_pred.to_numpy(dtype=float)
    n_components = max(1, min(3, X_train.shape[0], X_train.shape[1]))
    if policy == "pca":
        reducer = PCA(n_components=n_components)
        return reducer.fit_transform(X_train), reducer.transform(X_pred)
    if policy == "static_factor":
        centered_train = X_train.to_numpy(dtype=float) - X_train.to_numpy(dtype=float).mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered_train, full_matrices=False)
        components = vt[:n_components]
        train_scores = centered_train @ components.T
        centered_pred = X_pred.to_numpy(dtype=float) - X_train.to_numpy(dtype=float).mean(axis=0, keepdims=True)
        pred_scores = centered_pred @ components.T
        return train_scores, pred_scores
    raise ExecutionError(f"dimensionality_reduction_policy {policy!r} is not executable in current runtime slice")


def _apply_raw_panel_preprocessing(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[np.ndarray, np.ndarray]:
    if contract.dimensionality_reduction_policy != "none" and contract.feature_selection_policy != "none":
        raise ExecutionError("current runtime slice does not support combining dimensionality reduction with feature selection")
    X_train, X_pred = _apply_missing_policy(X_train, X_pred, contract)
    X_train, X_pred = _apply_outlier_policy(X_train, X_pred, contract)
    X_train, X_pred = _apply_scaling_policy(X_train, X_pred, contract)
    X_train, X_pred = _apply_feature_selection(X_train, y_train, X_pred, contract)
    return _apply_dimensionality_reduction(X_train, X_pred, contract)


def _build_raw_panel_training_data(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    start_idx: int,
    origin_idx: int,
    contract: PreprocessContract,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictors = _raw_panel_columns(frame, target)
    if origin_idx - horizon < start_idx:
        raise ExecutionError("insufficient history for raw_feature_panel training data")
    X_train = frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
    y_train = frame[target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = frame[predictors].iloc[[origin_idx]].astype(float).copy()
    if len(X_train) == 0 or len(y_train) == 0:
        raise ExecutionError("raw_feature_panel produced empty training data")
    X_train_arr, X_pred_arr = _apply_raw_panel_preprocessing(X_train, y_train, X_pred, contract)
    return X_train_arr, y_train, X_pred_arr


def _run_ar_model_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    selected_lag, selected_bic, fitted = _select_ar_bic_model(train, _max_ar_lag(recipe))
    prediction = fitted.predict(start=len(train), end=len(train) + horizon - 1)
    return {
        "y_pred": float(prediction.iloc[-1]),
        "selected_lag": selected_lag,
        "selected_bic": selected_bic,
    }


def _fit_autoreg_sklearn(train: pd.Series, recipe: RecipeSpec, model_family: str, model) -> tuple[int, np.ndarray, np.ndarray, object, dict[str, object]]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    fitted, tuning_payload = fit_with_optional_tuning(model_family, X, y, recipe.training_spec)
    return lag_order, X, y, fitted, tuning_payload




def _recursive_predict_adaptive_lasso(model, train: pd.Series, horizon: int, lag_order: int) -> float:
    history = list(train.to_numpy(dtype=float))
    for _ in range(horizon):
        features = np.asarray(history[-lag_order:][::-1], dtype=float).reshape(1, -1)
        pred = float(predict_adaptive_lasso(model, features)[0])
        history.append(pred)
    return float(history[-1])


def _run_ols_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "ols", LinearRegression())
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_ridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "ridge", Ridge(alpha=1.0))
    return {
        "y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
    }


def _run_lasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "lasso", Lasso(alpha=1e-4, max_iter=10000))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_elasticnet_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "elasticnet", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_randomforest_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "randomforest", RandomForestRegressor(n_estimators=200, random_state=42))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_bayesianridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "bayesianridge", BayesianRidge())
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_huber_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "huber", HuberRegressor())
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_adaptivelasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "adaptivelasso", None)
    return {"y_pred": _recursive_predict_adaptive_lasso(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_svr_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "svr_linear", LinearSVR(C=1.0, epsilon=0.01, max_iter=50000, random_state=42))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_svr_rbf_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "svr_rbf", SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_extratrees_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "extratrees", ExtraTreesRegressor(n_estimators=200, random_state=42))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_gbm_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "gbm", GradientBoostingRegressor(random_state=42))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_xgboost_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "xgboost", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, random_state=42, verbosity=0))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_lightgbm_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "lightgbm", LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_catboost_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "catboost", CatBoostRegressor(iterations=100, learning_rate=0.05, depth=4, verbose=False, random_seed=42))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_mlp_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _fit_raw_panel_model(raw_frame: pd.DataFrame, recipe: RecipeSpec, horizon: int, start_idx: int, origin_idx: int, contract: PreprocessContract, model_family: str, model) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, dict[str, object]]:
    X_train, y_train, X_pred = _build_raw_panel_training_data(raw_frame, recipe.target, horizon, start_idx, origin_idx, contract)
    fitted, tuning_payload = fit_with_optional_tuning(model_family, X_train, y_train, recipe.training_spec)
    return X_train, y_train, X_pred, fitted, tuning_payload


def _run_ols_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "ols", LinearRegression())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_ridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "ridge", Ridge(alpha=1.0))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_lasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "lasso", Lasso(alpha=1e-4, max_iter=10000))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_elasticnet_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "elasticnet", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_randomforest_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "randomforest", RandomForestRegressor(n_estimators=200, random_state=42))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_bayesianridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "bayesianridge", BayesianRidge())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_huber_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "huber", HuberRegressor())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_adaptivelasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "adaptivelasso", None)
    return {"y_pred": float(predict_adaptive_lasso(model, X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_svr_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "svr_linear", LinearSVR(C=1.0, epsilon=0.01, max_iter=50000, random_state=42))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_svr_rbf_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "svr_rbf", SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_extratrees_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "extratrees", ExtraTreesRegressor(n_estimators=200, random_state=42))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_gbm_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "gbm", GradientBoostingRegressor(random_state=42))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_xgboost_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "xgboost", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, random_state=42, verbosity=0))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_lightgbm_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "lightgbm", LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_catboost_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "catboost", CatBoostRegressor(iterations=100, learning_rate=0.05, depth=4, verbose=False, random_seed=42))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_mlp_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _linear_boosting_fit(X: np.ndarray, y: np.ndarray, base: str) -> tuple[np.ndarray, float]:
    residuals = y.copy().astype(float)
    intercept = float(np.mean(y))
    residuals -= intercept
    coef = np.zeros(X.shape[1], dtype=float)
    n_iter = 50
    lr = 0.1
    for _ in range(n_iter):
        if base == "componentwise":
            best_j, best_coef, best_sse = 0, 0.0, float("inf")
            for j in range(X.shape[1]):
                xj = X[:, j]
                c = float(np.dot(xj, residuals) / (np.dot(xj, xj) + 1e-10))
                sse = float(np.sum((residuals - c * xj) ** 2))
                if sse < best_sse:
                    best_j, best_coef, best_sse = j, c, sse
            coef[best_j] += lr * best_coef
            residuals -= lr * best_coef * X[:, best_j]
        elif base == "ridge":
            m = Ridge(alpha=1.0).fit(X, residuals)
            coef += lr * m.coef_
            residuals -= lr * m.predict(X)
        else:
            m = Lasso(alpha=1e-4, max_iter=10000).fit(X, residuals)
            coef += lr * m.coef_
            residuals -= lr * m.predict(X)
    return coef, intercept


def _run_componentwise_boosting_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "componentwise_boosting", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_boosting_ridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "boosting_ridge", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_boosting_lasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "boosting_lasso", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_pcr_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    pred, _ = fit_factor_model("pcr", pd.DataFrame(X), y, pd.DataFrame(np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1]).reshape(1, -1)), recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan}


def _run_pls_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    pred, _ = fit_factor_model("pls", pd.DataFrame(X), y, pd.DataFrame(np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1]).reshape(1, -1)), recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan}


def _run_factor_augmented_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    pred, _ = fit_factor_model("factor_augmented_linear", pd.DataFrame(X), y, pd.DataFrame(np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1]).reshape(1, -1)), recipe.training_spec, include_ar_lags=True)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan}


def _run_quantile_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _ = _fit_autoreg_sklearn(train, recipe, "quantile_linear", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan}


def _run_componentwise_boosting_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "componentwise_boosting", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_boosting_ridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "boosting_ridge", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_boosting_lasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "boosting_lasso", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_pcr_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    predictors = _raw_panel_columns(raw_frame, recipe.target)
    X_train = raw_frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
    y_train = raw_frame[recipe.target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = raw_frame[predictors].iloc[[origin_idx]].astype(float).copy()
    pred, _ = fit_factor_model("pcr", X_train, y_train, X_pred, recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan}


def _run_pls_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    predictors = _raw_panel_columns(raw_frame, recipe.target)
    X_train = raw_frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
    y_train = raw_frame[recipe.target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = raw_frame[predictors].iloc[[origin_idx]].astype(float).copy()
    pred, _ = fit_factor_model("pls", X_train, y_train, X_pred, recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan}


def _run_factor_augmented_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    predictors = _raw_panel_columns(raw_frame, recipe.target)
    X_train = raw_frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
    y_train = raw_frame[recipe.target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = raw_frame[predictors].iloc[[origin_idx]].astype(float).copy()
    pred, _ = fit_factor_model("factor_augmented_linear", X_train, y_train, X_pred, recipe.training_spec, include_ar_lags=True)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan}


def _run_quantile_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _ = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "quantile_linear", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _historical_mean_prediction(train: pd.Series) -> float:
    return float(train.mean())


def _load_custom_benchmark_callable(recipe: RecipeSpec):
    benchmark_spec = _benchmark_spec(recipe)
    plugin_path_raw = benchmark_spec.get("plugin_path")
    callable_name = benchmark_spec.get("callable_name")
    if not plugin_path_raw or not callable_name:
        raise ExecutionError("custom_benchmark requires benchmark_config fields 'plugin_path' and 'callable_name'")
    plugin_path = Path(str(plugin_path_raw)).expanduser()
    if not plugin_path.is_absolute():
        plugin_path = Path.cwd() / plugin_path
    if not plugin_path.exists():
        raise ExecutionError(f"custom benchmark plugin file not found: {plugin_path}")
    spec = importlib.util.spec_from_file_location("macrocast_custom_benchmark_plugin", plugin_path)
    if spec is None or spec.loader is None:
        raise ExecutionError(f"could not load custom benchmark plugin module from {plugin_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, str(callable_name), None)
    if func is None or not callable(func):
        raise ExecutionError(f"custom benchmark callable {callable_name!r} not found in {plugin_path}")
    return func


def _run_benchmark_executor(train: pd.Series, horizon: int, recipe: RecipeSpec) -> float:
    benchmark_family = _benchmark_family(recipe)
    if benchmark_family == "historical_mean":
        return _historical_mean_prediction(train)
    if benchmark_family == "zero_change":
        return float(train.iloc[-1])
    if benchmark_family == "ar_bic":
        fitted = _run_ar_model_executor(train, horizon, recipe, contract=_build_noop_contract())
        return float(fitted["y_pred"])
    if benchmark_family == "custom_benchmark":
        func = _load_custom_benchmark_callable(recipe)
        value = func(train.copy(), int(horizon), dict(_benchmark_spec(recipe)))
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ExecutionError("custom benchmark callable must return a numeric forecast") from exc
    raise ExecutionError(f"benchmark_family {benchmark_family!r} is not executable in current runtime slice")


def _build_noop_contract() -> PreprocessContract:
    return PreprocessContract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="raw_only",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="none",
        preprocess_fit_scope="not_applicable",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )


def _stat_test_spec(provenance_payload: dict | None) -> dict[str, object]:
    compiler = (provenance_payload or {}).get("compiler", {}) if provenance_payload else {}
    return dict(compiler.get("stat_test_spec", {"stat_test": "none"}))


def _importance_spec(provenance_payload: dict | None) -> dict[str, object]:
    compiler = (provenance_payload or {}).get("compiler", {}) if provenance_payload else {}
    return dict(compiler.get("importance_spec", {"importance_method": "none"}))


def _failure_policy_spec(provenance_payload: dict | None) -> dict[str, object]:
    compiler = (provenance_payload or {}).get("compiler", {}) if provenance_payload else {}
    return dict(compiler.get("failure_policy_spec", {"failure_policy": "fail_fast"}))


def _reproducibility_spec(provenance_payload: dict | None) -> dict[str, object]:
    compiler = (provenance_payload or {}).get("compiler", {}) if provenance_payload else {}
    return dict(compiler.get("reproducibility_spec", {"reproducibility_mode": "best_effort"}))


def _compute_mode_spec(provenance_payload: dict | None) -> dict[str, object]:
    compiler = (provenance_payload or {}).get("compiler", {}) if provenance_payload else {}
    return dict(compiler.get("compute_mode_spec", {"compute_mode": "serial"}))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _compute_dm_test(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 2:
        raise ExecutionError("dm test requires at least two forecast errors")
    max_lag = max(int(rows["horizon"].max()) - 1, 0)
    centered = loss_diff - loss_diff.mean()
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_var = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_var += 2.0 * cov
    if long_run_var <= 0:
        raise ExecutionError("dm test long-run variance must be positive")
    statistic = float(loss_diff.mean() / math.sqrt(long_run_var / n))
    p_value = float(_normal_two_sided_pvalue(statistic))
    return {
        "stat_test": "dm",
        "n": n,
        "mean_loss_diff": float(loss_diff.mean()),
        "long_run_variance": float(long_run_var),
        "max_lag": int(max_lag),
        "statistic": statistic,
        "p_value": p_value,
    }


def _compute_cw_test(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    benchmark_sq = rows["benchmark_squared_error"].to_numpy(dtype=float)
    model_sq = rows["squared_error"].to_numpy(dtype=float)
    forecast_adjustment = (rows["benchmark_pred"].to_numpy(dtype=float) - rows["y_pred"].to_numpy(dtype=float)) ** 2
    adjusted_loss_diff = benchmark_sq - (model_sq - forecast_adjustment)
    n = int(len(adjusted_loss_diff))
    if n < 2:
        raise ExecutionError("cw test requires at least two forecast errors")
    variance = float(np.var(adjusted_loss_diff, ddof=1))
    if variance <= 0:
        raise ExecutionError("cw test variance must be positive")
    statistic = float(adjusted_loss_diff.mean() / math.sqrt(variance / n))
    p_value = float(_normal_two_sided_pvalue(statistic))
    return {
        "stat_test": "cw",
        "n": n,
        "mean_adjusted_loss_diff": float(adjusted_loss_diff.mean()),
        "forecast_adjustment_mean": float(forecast_adjustment.mean()),
        "variance": variance,
        "statistic": statistic,
        "p_value": p_value,
    }


def _compute_minimal_importance(
    raw_frame: pd.DataFrame,
    target_series: pd.Series,
    recipe: RecipeSpec,
    contract: PreprocessContract,
) -> dict[str, object]:
    model_family = _model_family(recipe)
    feature_builder = _feature_builder(recipe)
    if model_family not in {"ridge", "lasso", "randomforest"}:
        raise ExecutionError(f"minimal_importance not implemented for model_family {model_family!r}")
    if feature_builder != "raw_feature_panel":
        raise ExecutionError(f"minimal_importance currently requires feature_builder='raw_feature_panel', got {feature_builder!r}")

    aligned_frame = raw_frame.loc[target_series.index]
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    origin_idx = len(target_series) - max(recipe.horizons) - 1
    if origin_idx < _minimum_train_size(recipe) - 1:
        raise ExecutionError("minimal_importance requires at least one valid forecast origin")
    start_idx = max(0, origin_idx + 1 - _rolling_window_size(recipe)) if rolling else 0
    horizon = min(recipe.horizons)
    predictors = _raw_panel_columns(aligned_frame, recipe.target)
    X_train, y_train, _ = _build_raw_panel_training_data(aligned_frame, recipe.target, horizon, start_idx, origin_idx, contract)

    if model_family == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        importance_values = np.abs(model.coef_)
    elif model_family == "lasso":
        model = Lasso(alpha=1e-4, max_iter=10000)
        model.fit(X_train, y_train)
        importance_values = np.abs(model.coef_)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        importance_values = model.feature_importances_

    feature_importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(zip(predictors, importance_values), key=lambda item: item[1], reverse=True)
    ]
    return {
        "importance_method": "minimal_importance",
        "model_family": model_family,
        "feature_builder": feature_builder,
        "n_train": int(len(y_train)),
        "feature_importance": feature_importance,
    }


def _build_predictions(
    raw_frame: pd.DataFrame,
    target_series: pd.Series,
    recipe: RecipeSpec,
    contract: PreprocessContract,
    *,
    compute_mode: str = "serial",
) -> pd.DataFrame:
    minimum_train_size = _minimum_train_size(recipe)
    benchmark_family = _benchmark_family(recipe)
    model_spec = _model_spec(recipe)
    model_executor = _get_model_executor(recipe)
    benchmark_executor = _get_benchmark_executor(recipe)

    aligned_frame = raw_frame.loc[target_series.index]
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    rolling_window_size = _rolling_window_size(recipe)
    outer_window = str(recipe.training_spec.get("outer_window", "rolling" if rolling else "expanding"))
    refit_policy = str(recipe.training_spec.get("refit_policy", "refit_every_step"))
    anchored_max_window_size = int(recipe.training_spec.get("anchored_max_window_size", rolling_window_size))
    refit_k_steps = int(recipe.training_spec.get("refit_k_steps", 3))

    def _rows_for_horizon(horizon: int) -> list[dict[str, object]]:
        horizon_rows: list[dict[str, object]] = []
        locked_start_idx = None
        locked_origin_idx = None
        for origin_idx in range(minimum_train_size - 1, len(target_series) - horizon):
            base_start_idx = max(0, origin_idx + 1 - rolling_window_size) if rolling else 0
            if outer_window == "anchored_rolling":
                if origin_idx + 1 > anchored_max_window_size:
                    base_start_idx = max(0, origin_idx + 1 - anchored_max_window_size)
                else:
                    base_start_idx = 0
            if refit_policy == "fit_once_predict_many":
                if locked_start_idx is None:
                    locked_start_idx = base_start_idx
                    locked_origin_idx = origin_idx
                start_idx = locked_start_idx
                effective_origin_idx = locked_origin_idx
            elif refit_policy == "refit_every_k_steps":
                if locked_start_idx is None or (origin_idx - (minimum_train_size - 1)) % refit_k_steps == 0:
                    locked_start_idx = base_start_idx
                    locked_origin_idx = origin_idx
                start_idx = locked_start_idx
                effective_origin_idx = locked_origin_idx
            else:
                start_idx = base_start_idx
                effective_origin_idx = origin_idx
            train = target_series.iloc[start_idx : effective_origin_idx + 1]
            model_output = model_executor(train, horizon, recipe, contract, aligned_frame, effective_origin_idx, start_idx)
            y_pred = float(model_output["y_pred"])
            benchmark_pred = float(benchmark_executor(train, horizon, recipe))
            y_true = float(target_series.iloc[origin_idx + horizon])
            error = y_true - y_pred
            benchmark_error = y_true - benchmark_pred
            horizon_rows.append(
                {
                    "target": target_series.name,
                    "model_name": model_spec["executor_name"],
                    "benchmark_name": benchmark_family,
                    "horizon": horizon,
                    "origin_date": target_series.index[origin_idx].strftime("%Y-%m-%d"),
                    "target_date": target_series.index[origin_idx + horizon].strftime("%Y-%m-%d"),
                    "fit_origin_date": target_series.index[effective_origin_idx].strftime("%Y-%m-%d"),
                    "selected_lag": int(model_output["selected_lag"]),
                    "selected_bic": float(model_output["selected_bic"]),
                    "train_start_date": target_series.index[start_idx].strftime("%Y-%m-%d"),
                    "train_end_date": target_series.index[origin_idx].strftime("%Y-%m-%d"),
                    "training_window_size": int(len(train)),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "benchmark_pred": benchmark_pred,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": error**2,
                    "benchmark_error": benchmark_error,
                    "benchmark_abs_error": abs(benchmark_error),
                    "benchmark_squared_error": benchmark_error**2,
                }
            )
        return horizon_rows

    rows: list[dict[str, object]] = []
    if compute_mode == "parallel_by_horizon" and len(recipe.horizons) > 1:
        with ThreadPoolExecutor(max_workers=min(len(recipe.horizons), 4)) as ex:
            futures = [ex.submit(_rows_for_horizon, horizon) for horizon in recipe.horizons]
            for future in futures:
                rows.extend(future.result())
    else:
        for horizon in recipe.horizons:
            rows.extend(_rows_for_horizon(horizon))

    if not rows:
        raise ExecutionError("no forecast rows were produced for the requested horizons")

    return pd.DataFrame(rows)


def _compute_metrics(predictions: pd.DataFrame, recipe: RecipeSpec) -> dict[str, object]:
    metrics_by_horizon: dict[str, dict[str, object]] = {}
    for horizon, group in predictions.groupby("horizon", sort=True):
        selected_lag_counts = {
            str(int(lag)): int(count)
            for lag, count in group["selected_lag"].value_counts().sort_index().items()
        }
        msfe = float(group["squared_error"].mean())
        benchmark_msfe = float(group["benchmark_squared_error"].mean())
        csfe = float(group["squared_error"].sum())
        relative_msfe = msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0
        oos_r2 = 1.0 - relative_msfe
        metrics_by_horizon[f"h{int(horizon)}"] = {
            "n_predictions": int(len(group)),
            "msfe": msfe,
            "benchmark_msfe": benchmark_msfe,
            "relative_msfe": relative_msfe,
            "oos_r2": oos_r2,
            "csfe": csfe,
            "mae": float(group["abs_error"].mean()),
            "rmse": float(msfe**0.5),
            "selected_lag_counts": selected_lag_counts,
        }

    return {
        "model_name": _model_spec(recipe)["executor_name"],
        "model_spec": _model_spec(recipe),
        "benchmark_name": _benchmark_family(recipe),
        "benchmark_spec": _benchmark_spec(recipe),
        "target": recipe.target,
        "raw_dataset": recipe.raw_dataset,
        "minimum_train_size": _minimum_train_size(recipe),
        "max_lag": _max_ar_lag(recipe),
        "lag_selection": _LAG_SELECTION,
        "metrics_by_horizon": metrics_by_horizon,
    }


def _build_comparison_summary(predictions: pd.DataFrame, recipe: RecipeSpec) -> dict[str, object]:
    comparison_by_horizon: dict[str, dict[str, object]] = {}
    for horizon, group in predictions.groupby("horizon", sort=True):
        model_msfe = float(group["squared_error"].mean())
        benchmark_msfe = float(group["benchmark_squared_error"].mean())
        loss_diff = group["benchmark_squared_error"] - group["squared_error"]
        comparison_by_horizon[f"h{int(horizon)}"] = {
            "n_predictions": int(len(group)),
            "model_msfe": model_msfe,
            "benchmark_msfe": benchmark_msfe,
            "mean_loss_diff": float(loss_diff.mean()),
            "win_rate": float((group["squared_error"] < group["benchmark_squared_error"]).mean()),
            "tie_rate": float((group["squared_error"] == group["benchmark_squared_error"]).mean()),
            "relative_msfe": model_msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0,
            "oos_r2": 1.0 - (model_msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0),
        }

    return {
        "model_name": _model_spec(recipe)["executor_name"],
        "benchmark_name": _benchmark_family(recipe),
        "target": recipe.target,
        "raw_dataset": recipe.raw_dataset,
        "comparison_by_horizon": comparison_by_horizon,
    }


def _compute_multi_target_metrics(predictions: pd.DataFrame, recipe: RecipeSpec) -> dict[str, object]:
    metrics_by_target = {}
    for target, group in predictions.groupby("target", sort=True):
        metrics_by_target[str(target)] = _compute_metrics(group.reset_index(drop=True), _recipe_for_target(recipe, str(target)))
    return {
        "model_name": _model_spec(recipe)["executor_name"],
        "benchmark_name": _benchmark_family(recipe),
        "targets": list(_recipe_targets(recipe)),
        "metrics_by_target": metrics_by_target,
    }


def _tree_context_summary(tree_context: dict[str, object]) -> str:
    fixed_names = ",".join(sorted(tree_context.get("fixed_axes", {}))) or "none"
    sweep_names = ",".join(sorted(tree_context.get("sweep_axes", {}))) or "none"
    conditional_names = ",".join(sorted(tree_context.get("conditional_axes", {}))) or "none"
    return (
        f"tree_context=route_owner={tree_context.get('route_owner', 'unknown')}; "
        f"execution_posture={tree_context.get('execution_posture', 'unknown')}; "
        f"fixed_axes=[{fixed_names}]; "
        f"sweep_axes=[{sweep_names}]; "
        f"conditional_axes=[{conditional_names}]"
    )


def _build_multi_target_comparison_summary(predictions: pd.DataFrame, recipe: RecipeSpec) -> dict[str, object]:
    comparison_by_target = {}
    for target, group in predictions.groupby("target", sort=True):
        comparison_by_target[str(target)] = _build_comparison_summary(group.reset_index(drop=True), _recipe_for_target(recipe, str(target)))
    return {
        "model_name": _model_spec(recipe)["executor_name"],
        "benchmark_name": _benchmark_family(recipe),
        "targets": list(_recipe_targets(recipe)),
        "comparison_by_target": comparison_by_target,
    }


def execute_recipe(
    *,
    recipe: RecipeSpec,
    preprocess: PreprocessContract,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
    provenance_payload: dict | None = None,
) -> ExecutionResult:
    if not is_operational_preprocess_contract(preprocess):
        raise ExecutionError(
            "current execution slice only supports explicit operational preprocessing contracts"
        )

    if _benchmark_family(recipe) not in {"historical_mean", "zero_change", "ar_bic", "custom_benchmark"}:
        raise ExecutionError(
            f"benchmark_family {_benchmark_family(recipe)!r} is representable but not executable in current runtime slice"
        )
    _get_model_executor(recipe)
    _get_benchmark_executor(recipe)

    run = build_run_spec(recipe)
    spec = build_execution_spec(recipe=recipe, run=run, preprocess=preprocess)
    output_root = Path(output_root)
    run_dir = output_root / run.artifact_subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_result = _load_raw_for_recipe(recipe, local_raw_source, output_root / ".raw_cache")
    targets = _recipe_targets(recipe)
    stat_test_spec = _stat_test_spec(provenance_payload)
    importance_spec = _importance_spec(provenance_payload)
    reproducibility_spec = _reproducibility_spec(provenance_payload)
    failure_policy_spec = _failure_policy_spec(provenance_payload)
    compute_mode_spec = _compute_mode_spec(provenance_payload)
    failure_policy = str(failure_policy_spec.get("failure_policy", "fail_fast"))
    compute_mode = str(compute_mode_spec.get("compute_mode", "serial"))
    prediction_frames = []
    failed_components: list[dict[str, object]] = []
    successful_targets: list[str] = []
    target_series = None
    def _target_job(target: str):
        target_recipe = _recipe_for_target(recipe, target)
        target_series_local = _get_target_series(raw_result.data, target, _minimum_train_size(target_recipe))
        frame = _build_predictions(raw_result.data, target_series_local, target_recipe, preprocess, compute_mode=compute_mode)
        return target, target_series_local, frame

    if compute_mode == "parallel_by_model" and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as ex:
            futures = [ex.submit(_target_job, target) for target in targets]
            for future in futures:
                try:
                    target, target_series_local, frame = future.result()
                    target_series = target_series_local
                    prediction_frames.append(frame)
                    successful_targets.append(target)
                except Exception as exc:
                    err = str(exc)
                    target_name = None
                    if "target '" in err:
                        target_name = err.split("target '", 1)[1].split("'", 1)[0]
                    if failure_policy in {"skip_failed_model", "save_partial_results"}:
                        failed_components.append({"stage": "prediction_build", "target": target_name, "error": err})
                        continue
                    raise
    else:
        for target in targets:
            try:
                target, target_series_local, frame = _target_job(target)
                target_series = target_series_local
                prediction_frames.append(frame)
                successful_targets.append(target)
            except Exception as exc:
                if failure_policy in {"skip_failed_model", "save_partial_results"}:
                    failed_components.append({"stage": "prediction_build", "target": target, "error": str(exc)})
                    continue
                raise
    if not prediction_frames:
        raise ExecutionError("all target/model executions failed; no predictions available to save")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if recipe.targets:
        metrics = _compute_multi_target_metrics(predictions, recipe)
        comparison_summary = _build_multi_target_comparison_summary(predictions, recipe)
    else:
        metrics = _compute_metrics(predictions, recipe)
        comparison_summary = _build_comparison_summary(predictions, recipe)
    if recipe.targets and stat_test_spec.get("stat_test") != "none":
        raise ExecutionError("multi-target point-forecast slice does not yet support statistical-test artifacts")
    if recipe.targets and importance_spec.get("importance_method") != "none":
        raise ExecutionError("multi-target point-forecast slice does not yet support importance artifacts")

    manifest = {
        "recipe_id": recipe.recipe_id,
        "run_id": run.run_id,
        "target": recipe.target,
        "targets": list(recipe.targets),
        "horizons": list(recipe.horizons),
        "raw_dataset": recipe.raw_dataset,
        "route_owner": run.route_owner,
        "raw_artifact": raw_result.artifact.local_path,
        "preprocess_summary": preprocess_summary(preprocess),
        "preprocess_contract": preprocess_to_dict(preprocess),
        "execution_architecture": _EXECUTION_ARCHITECTURE,
        "forecast_engine": _model_spec(recipe)["executor_name"],
        "model_spec": _model_spec(recipe),
        "benchmark_name": _benchmark_family(recipe),
        "benchmark_spec": _benchmark_spec(recipe),
        "data_task_spec": dict(recipe.data_task_spec),
        "training_spec": dict(recipe.training_spec),
        "stat_test_spec": stat_test_spec,
        "importance_spec": importance_spec,
        "reproducibility_spec": reproducibility_spec,
        "failure_policy_spec": failure_policy_spec,
        "compute_mode_spec": compute_mode_spec,
        "lag_selection": _LAG_SELECTION,
        "max_lag": _max_ar_lag(recipe),
        "minimum_train_size": _minimum_train_size(recipe),
        "prediction_rows": int(len(predictions)),
        "metrics_file": "metrics.json",
        "comparison_file": "comparison_summary.json",
        "successful_targets": successful_targets,
        "partial_run": bool(failed_components),
    }
    if provenance_payload:
        manifest.update(provenance_payload)
    if failed_components:
        manifest["failure_log_file"] = "failures.json"
    _write_json(run_dir / "manifest.json", manifest)
    if failed_components:
        _write_json(run_dir / "failures.json", failed_components)
    tree_context = manifest.get("tree_context", {})
    summary_lines = [
        recipe_summary(recipe),
        preprocess_summary(preprocess),
        f"forecast_engine={_model_spec(recipe)['executor_name']}; benchmark={_benchmark_family(recipe)}; prediction_rows={len(predictions)}",
    ]
    if tree_context:
        summary_lines.append(_tree_context_summary(tree_context))
    (run_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    raw_result.data.head(20).to_csv(run_dir / "data_preview.csv")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "comparison_summary.json", comparison_summary)
    if stat_test_spec.get("stat_test") == "dm":
        try:
            dm_payload = _compute_dm_test(predictions)
            _write_json(run_dir / "stat_test_dm.json", dm_payload)
            manifest["stat_test_file"] = "stat_test_dm.json"
        except Exception as exc:
            if failure_policy == "save_partial_results":
                failed_components.append({"stage": "stat_test_artifact", "target": None, "error": str(exc)})
            else:
                raise
    if stat_test_spec.get("stat_test") == "cw":
        try:
            cw_payload = _compute_cw_test(predictions)
            _write_json(run_dir / "stat_test_cw.json", cw_payload)
            manifest["stat_test_file"] = "stat_test_cw.json"
        except Exception as exc:
            if failure_policy == "save_partial_results":
                failed_components.append({"stage": "stat_test_artifact", "target": None, "error": str(exc)})
            else:
                raise
    if importance_spec.get("importance_method") == "minimal_importance":
        try:
            importance_payload = _compute_minimal_importance(raw_result.data, target_series, recipe, preprocess)
            _write_json(run_dir / "importance_minimal.json", importance_payload)
            manifest["importance_file"] = "importance_minimal.json"
        except Exception as exc:
            if failure_policy == "save_partial_results":
                failed_components.append({"stage": "importance_artifact", "target": None, "error": str(exc)})
            else:
                raise
    manifest["partial_run"] = bool(failed_components)
    if failed_components:
        manifest["failure_log_file"] = "failures.json"
        _write_json(run_dir / "failures.json", failed_components)
    _write_json(run_dir / "manifest.json", manifest)

    return ExecutionResult(
        spec=spec,
        run=run,
        raw_result=raw_result,
        artifact_dir=str(run_dir),
    )
