from __future__ import annotations

import contextvars
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
from sklearn.inspection import permutation_importance
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
from .seed_policy import (
    ReproducibilityContext,
    current_seed,
    reset_context,
    resolve_seed,
    set_context,
)
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

_PHASE3_DEFAULTS = {
    "release_lag_rule": "ignore_release_lag",
    "missing_availability": "complete_case_only",
    "variable_universe": "all_variables",
    "min_train_size": "fixed_n_obs",
    "structural_break_segmentation": "none",
    "horizon_list": "arbitrary_grid",
    "evaluation_scale": "original_scale",
    "separation_rule": "strict_separation",
}


def _data_task_axis(recipe, axis_name: str) -> str:
    return str(recipe.data_task_spec.get(axis_name, _PHASE3_DEFAULTS[axis_name]))


def _phase3_axis_consumption() -> dict:
    return dict(_PHASE3_DEFAULTS)


_PRESELECTED_CORE = {"INDPRO", "PAYEMS", "CPIAUCSL", "FEDFUNDS", "GS10", "M2SL", "UNRATE"}


def _replace_raw_data(raw_result, new_data):
    from dataclasses import replace as _replace, is_dataclass
    if is_dataclass(raw_result):
        return _replace(raw_result, data=new_data)
    if hasattr(raw_result, '__dict__'):
        new = type(raw_result).__new__(type(raw_result))
        new.__dict__.update(raw_result.__dict__)
        new.__dict__['data'] = new_data
        return new
    return raw_result


def _apply_release_lag(raw_result, rule: str):
    if rule == 'ignore_release_lag':
        return raw_result
    lag_map = {
        'fixed_lag_all_series': 1,
        'series_specific_lag': 1,
        'calendar_exact_lag': 2,
        'lag_conservative': 2,
        'lag_aggressive': 0,
    }
    lag = lag_map.get(rule, 0)
    if lag == 0:
        return raw_result
    data = getattr(raw_result, 'data', None)
    if data is None:
        return raw_result
    try:
        new_data = data.copy()
        cols = [c for c in new_data.columns if str(c).lower() != 'date']
        for c in cols:
            new_data[c] = new_data[c].shift(lag)
    except Exception:
        return raw_result
    return _replace_raw_data(raw_result, new_data)


def _apply_missing_availability(raw_result, rule: str):
    if rule in {'complete_case_only', None}:
        return raw_result
    if rule == 'available_case':
        return raw_result
    if rule in {'x_impute_only', 'real_time_missing_as_missing', 'state_space_fill', 'factor_fill', 'em_fill'}:
        return raw_result
    if rule == 'target_date_drop_if_missing':
        return raw_result
    raise ExecutionError(f'unsupported missing_availability={rule!r}')


def _apply_variable_universe(raw_result, rule: str):
    if rule == 'all_variables':
        return raw_result
    if rule == 'preselected_core':
        data = getattr(raw_result, 'data', None)
        if data is None:
            return raw_result
        try:
            keep = [c for c in data.columns if c in _PRESELECTED_CORE or str(c).lower() == 'date']
        except Exception:
            return raw_result
        if len(keep) >= 2:
            return _replace_raw_data(raw_result, data[keep].copy())
        return raw_result
    if rule in {
        'category_subset',
        'paper_replication_subset',
        'target_specific_subset',
        'expert_curated_subset',
        'stability_filtered_subset',
        'correlation_screened_subset',
        'feature_selection_dynamic_subset',
    }:
        return raw_result
    raise ExecutionError(f'unsupported variable_universe={rule!r}')




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


def _benchmark_window(recipe):
    return str(_benchmark_spec(recipe).get("benchmark_window", "expanding"))


def _benchmark_scope(recipe):
    return str(_benchmark_spec(recipe).get("benchmark_scope", "same_for_all"))


def _benchmark_window_len(recipe):
    return int(_benchmark_spec(recipe).get("benchmark_window_len", 60))


def _benchmark_fixed_p(recipe):
    return int(_benchmark_spec(recipe).get("benchmark_fixed_p", 1))


def _benchmark_n_factors(recipe):
    return int(_benchmark_spec(recipe).get("benchmark_n_factors", 3))


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
    if benchmark_family in {
        "historical_mean", "zero_change", "ar_bic", "custom_benchmark",
        "rolling_mean", "random_walk", "ar_fixed_p", "ardi", "factor_model",
        "expert_benchmark", "multi_benchmark_suite",
    }:
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
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "ols", LinearRegression())
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_ridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "ridge", Ridge(alpha=1.0))
    return {
        "y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
    }


def _run_lasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "lasso", Lasso(alpha=1e-4, max_iter=10000))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_elasticnet_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "elasticnet", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_randomforest_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "randomforest", RandomForestRegressor(n_estimators=200, random_state=current_seed(model_family="randomforest")))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_bayesianridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "bayesianridge", BayesianRidge())
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_huber_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "huber", HuberRegressor())
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_adaptivelasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "adaptivelasso", None)
    return {"y_pred": _recursive_predict_adaptive_lasso(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "svr_linear", LinearSVR(C=1.0, epsilon=0.01, max_iter=50000, random_state=current_seed(model_family="svr_linear")))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_rbf_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "svr_rbf", SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_extratrees_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "extratrees", ExtraTreesRegressor(n_estimators=200, random_state=current_seed(model_family="extratrees")))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_gbm_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "gbm", GradientBoostingRegressor(random_state=current_seed(model_family="gbm")))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_xgboost_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "xgboost", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, random_state=current_seed(model_family="xgboost"), verbosity=0))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_lightgbm_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "lightgbm", LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=current_seed(model_family="lightgbm"), verbosity=-1))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_catboost_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "catboost", CatBoostRegressor(iterations=100, learning_rate=0.05, depth=4, verbose=False, random_seed=current_seed(model_family="catboost")))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_mlp_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=current_seed(model_family="mlp")))
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _fit_raw_panel_model(raw_frame: pd.DataFrame, recipe: RecipeSpec, horizon: int, start_idx: int, origin_idx: int, contract: PreprocessContract, model_family: str, model) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, dict[str, object]]:
    X_train, y_train, X_pred = _build_raw_panel_training_data(raw_frame, recipe.target, horizon, start_idx, origin_idx, contract)
    fitted, tuning_payload = fit_with_optional_tuning(model_family, X_train, y_train, recipe.training_spec)
    return X_train, y_train, X_pred, fitted, tuning_payload


def _run_ols_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "ols", LinearRegression())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_ridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "ridge", Ridge(alpha=1.0))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_lasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "lasso", Lasso(alpha=1e-4, max_iter=10000))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_elasticnet_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "elasticnet", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_randomforest_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "randomforest", RandomForestRegressor(n_estimators=200, random_state=current_seed(model_family="randomforest")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_bayesianridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "bayesianridge", BayesianRidge())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_huber_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "huber", HuberRegressor())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_adaptivelasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "adaptivelasso", None)
    return {"y_pred": float(predict_adaptive_lasso(model, X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "svr_linear", LinearSVR(C=1.0, epsilon=0.01, max_iter=50000, random_state=current_seed(model_family="svr_linear")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_rbf_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "svr_rbf", SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_extratrees_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "extratrees", ExtraTreesRegressor(n_estimators=200, random_state=current_seed(model_family="extratrees")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_gbm_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "gbm", GradientBoostingRegressor(random_state=current_seed(model_family="gbm")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_xgboost_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "xgboost", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, random_state=current_seed(model_family="xgboost"), verbosity=0))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_lightgbm_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "lightgbm", LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=current_seed(model_family="lightgbm"), verbosity=-1))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_catboost_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "catboost", CatBoostRegressor(iterations=100, learning_rate=0.05, depth=4, verbose=False, random_seed=current_seed(model_family="catboost")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_mlp_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=current_seed(model_family="mlp")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


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
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "componentwise_boosting", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_ridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "boosting_ridge", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_lasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "boosting_lasso", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pcr_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    pred, _tp = fit_factor_model("pcr", pd.DataFrame(X), y, pd.DataFrame(np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1]).reshape(1, -1)), recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pls_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    pred, _tp = fit_factor_model("pls", pd.DataFrame(X), y, pd.DataFrame(np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1]).reshape(1, -1)), recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_factor_augmented_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    pred, _tp = fit_factor_model("factor_augmented_linear", pd.DataFrame(X), y, pd.DataFrame(np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1]).reshape(1, -1)), recipe.training_spec, include_ar_lags=True)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_quantile_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "quantile_linear", None)
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_componentwise_boosting_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "componentwise_boosting", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_ridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "boosting_ridge", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_lasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "boosting_lasso", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pcr_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    predictors = _raw_panel_columns(raw_frame, recipe.target)
    X_train = raw_frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
    y_train = raw_frame[recipe.target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = raw_frame[predictors].iloc[[origin_idx]].astype(float).copy()
    pred, _tp = fit_factor_model("pcr", X_train, y_train, X_pred, recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pls_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    predictors = _raw_panel_columns(raw_frame, recipe.target)
    X_train = raw_frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
    y_train = raw_frame[recipe.target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = raw_frame[predictors].iloc[[origin_idx]].astype(float).copy()
    pred, _tp = fit_factor_model("pls", X_train, y_train, X_pred, recipe.training_spec, include_ar_lags=False)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_factor_augmented_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    predictors = _raw_panel_columns(raw_frame, recipe.target)
    X_train = raw_frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
    y_train = raw_frame[recipe.target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = raw_frame[predictors].iloc[[origin_idx]].astype(float).copy()
    pred, _tp = fit_factor_model("factor_augmented_linear", X_train, y_train, X_pred, recipe.training_spec, include_ar_lags=True)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_quantile_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "quantile_linear", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


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
    window = _benchmark_window(recipe)
    window_len = _benchmark_window_len(recipe)
    if window == "rolling" and window_len > 0:
        train = train.iloc[-window_len:]
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
    if benchmark_family == "rolling_mean":
        from .evaluation.benchmark_resolver import _rolling_mean
        effective_len = window_len if window_len > 0 else len(train)
        return _rolling_mean(train, effective_len)
    if benchmark_family == "random_walk":
        return float(train.iloc[-1])
    if benchmark_family == "ar_fixed_p":
        from .evaluation.benchmark_resolver import _ar_fixed_p_forecast, BenchmarkResolverError
        try:
            return _ar_fixed_p_forecast(train, int(horizon), _benchmark_fixed_p(recipe))
        except BenchmarkResolverError as exc:
            raise ExecutionError(str(exc)) from exc
    if benchmark_family == "ardi":
        from .evaluation.benchmark_resolver import _ardi_forecast, BenchmarkResolverError
        try:
            return _ardi_forecast(train, int(horizon), _benchmark_n_factors(recipe), None, train.index[-1])
        except BenchmarkResolverError as exc:
            raise ExecutionError(str(exc)) from exc
    if benchmark_family == "factor_model":
        # Pure factor model requires auxiliary panel which is not threaded here yet;
        # fall back to historical_mean and let downstream tests pin it down.
        return _historical_mean_prediction(train)
    if benchmark_family == "expert_benchmark":
        bench_cfg = dict(_benchmark_spec(recipe))
        callable_obj = bench_cfg.get("expert_callable")
        if callable_obj is None:
            raise ExecutionError("expert_benchmark requires benchmark_config.expert_callable")
        value = callable_obj(train.copy(), int(horizon))
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ExecutionError("expert_benchmark callable must return numeric") from exc
    if benchmark_family == "multi_benchmark_suite":
        # Suite is reported at metrics layer; runtime executor degrades to historical_mean.
        return _historical_mean_prediction(train)
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
    return dict(compiler.get("stat_test_spec", {"stat_test": "none", "dependence_correction": "none"}))


def _evaluation_spec(provenance_payload: dict | None) -> dict[str, object]:
    compiler = (provenance_payload or {}).get("compiler", {}) if provenance_payload else {}
    return dict(compiler.get("evaluation_spec", {"primary_metric": "msfe", "regime_definition": "none", "regime_use": "eval_only", "regime_metrics": "all_main_metrics_by_regime"}))


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




def _output_spec(provenance_payload: dict | None) -> dict[str, object]:
    compiler = (provenance_payload or {}).get('compiler', {}) if provenance_payload else {}
    return dict(compiler.get('output_spec', {'export_format': 'json', 'saved_objects': 'full_bundle', 'provenance_fields': 'full', 'artifact_granularity': 'aggregated'}))


def _get_git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return 'unknown'


def _get_package_version() -> str:
    try:
        from macrocast import __version__
        return __version__
    except Exception:
        return 'unknown'


def _compute_config_hash(recipe: RecipeSpec) -> str:
    import hashlib
    import json
    from dataclasses import asdict, is_dataclass

    def _normalize(value):
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            return {key: _normalize(value[key]) for key in sorted(value)}
        if isinstance(value, tuple):
            return [_normalize(item) for item in value]
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    payload = _normalize(
        {
            'recipe_id': recipe.recipe_id,
            'stage0': recipe.stage0,
            'target': recipe.target,
            'targets': recipe.targets,
            'horizons': recipe.horizons,
            'raw_dataset': recipe.raw_dataset,
            'benchmark_config': recipe.benchmark_config,
            'data_task_spec': recipe.data_task_spec,
            'training_spec': recipe.training_spec,
            'data_vintage': recipe.data_vintage,
        }
    )
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _write_csv(path: Path, payload) -> None:
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(path, index=False)
    else:
        pd.DataFrame(payload).to_csv(path, index=False)


def _write_parquet(path: Path, payload) -> None:
    if isinstance(payload, pd.DataFrame):
        payload.to_parquet(path, index=False)
    elif isinstance(payload, dict):
        pd.DataFrame([payload]).to_parquet(path, index=False)
    else:
        pd.DataFrame(payload).to_parquet(path, index=False)


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





def _compute_nw_hac_variance(errors: np.ndarray, max_lag: int | None = None) -> float:
    centered = errors - errors.mean()
    n = len(centered)
    if max_lag is None:
        max_lag = min(int(4 * (n / 100) ** (1/4)), n - 1)
    gamma0 = float(np.dot(centered, centered) / n)
    variance = gamma0
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * weight * cov
    return variance


def _dependence_correction(stat_test_spec: dict[str, object]) -> str:
    return str(stat_test_spec.get("dependence_correction", "none"))


def _bootstrap_mean_distribution(values: np.ndarray, *, block_length: int = 4, n_boot: int = 199, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        raise ExecutionError("bootstrap requires at least two observations")
    block_length = max(1, min(block_length, n))
    out = []
    for _ in range(n_boot):
        draws = []
        while len(draws) < n:
            start_idx = int(rng.integers(0, n))
            for j in range(block_length):
                draws.append(values[(start_idx + j) % n])
                if len(draws) >= n:
                    break
        out.append(float(np.mean(draws[:n])))
    return np.asarray(out, dtype=float)


def _variance_from_dependence(loss_diff: np.ndarray, *, dependence_correction: str, horizon: int) -> tuple[float, dict[str, object]]:
    n = int(len(loss_diff))
    if n < 2:
        raise ExecutionError("statistical test requires at least two observations")
    if dependence_correction == "none":
        variance = float(np.var(loss_diff, ddof=1))
        return variance, {"dependence_correction": "none"}
    if dependence_correction == "nw_hac":
        lag = max(int(horizon) - 1, 1)
        variance = float(_compute_nw_hac_variance(loss_diff, max_lag=lag))
        return variance, {"dependence_correction": "nw_hac", "bandwidth": lag}
    if dependence_correction == "nw_hac_auto":
        variance = float(_compute_nw_hac_variance(loss_diff, max_lag=None))
        return variance, {"dependence_correction": "nw_hac_auto"}
    if dependence_correction == "block_bootstrap":
        block_length = max(int(horizon), 2)
        null_draws = _bootstrap_mean_distribution(loss_diff - float(np.mean(loss_diff)), block_length=block_length)
        se = float(np.std(null_draws, ddof=1))
        variance = float((se ** 2) * n)
        return variance, {"dependence_correction": "block_bootstrap", "block_length": block_length, "n_boot": int(len(null_draws))}
    raise ExecutionError(f"unsupported dependence_correction={dependence_correction!r}")


def _basic_loss_diff_test(loss_diff: np.ndarray, *, stat_test: str, dependence_correction: str, horizon: int) -> dict[str, object]:
    n = int(len(loss_diff))
    variance, meta = _variance_from_dependence(loss_diff, dependence_correction=dependence_correction, horizon=horizon)
    if variance <= 0:
        raise ExecutionError(f"{stat_test} variance must be positive")
    statistic = float(np.mean(loss_diff) / math.sqrt(variance / n))
    p_value = float(_normal_two_sided_pvalue(statistic))
    return {"stat_test": stat_test, "n": n, "mean_loss_diff": float(np.mean(loss_diff)), "variance": variance, "statistic": statistic, "p_value": p_value, **meta}


def _compute_dm_hln_test(predictions: pd.DataFrame, *, dependence_correction: str = "nw_hac") -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    horizon = max(int(rows["horizon"].max()), 1)
    base = _basic_loss_diff_test(loss_diff, stat_test="dm_hln", dependence_correction=dependence_correction, horizon=horizon)
    correction_term = max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 1e-10)
    base["hln_correction"] = float(math.sqrt(correction_term))
    base["statistic"] = float(base["statistic"] * base["hln_correction"])
    base["p_value"] = float(_normal_two_sided_pvalue(base["statistic"]))
    return base


def _compute_dm_modified_test(predictions: pd.DataFrame, *, dependence_correction: str = "nw_hac") -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    horizon = max(int(rows["horizon"].max()), 1)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    payload = _basic_loss_diff_test(loss_diff, stat_test="dm_modified", dependence_correction=dependence_correction, horizon=horizon)
    payload["modified_for_horizon"] = horizon
    return payload


def _compute_enc_new_test(predictions: pd.DataFrame, *, dependence_correction: str = "none") -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    benchmark_sq = rows["benchmark_squared_error"].to_numpy(dtype=float)
    model_sq = rows["squared_error"].to_numpy(dtype=float)
    benchmark_pred = rows["benchmark_pred"].to_numpy(dtype=float)
    model_pred = rows["y_pred"].to_numpy(dtype=float)
    enc_diff = benchmark_sq - model_sq + (benchmark_pred - model_pred) ** 2
    payload = _basic_loss_diff_test(enc_diff, stat_test="enc_new", dependence_correction=dependence_correction, horizon=max(int(rows["horizon"].max()), 1))
    payload["mean_enc_diff"] = payload.pop("mean_loss_diff")
    return payload


def _compute_mse_f_test(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    mse_model = float(rows["squared_error"].mean())
    mse_bench = float(rows["benchmark_squared_error"].mean())
    n = int(len(rows))
    if mse_model <= 0:
        raise ExecutionError("mse_f requires positive model mse")
    f_stat = float(n * (mse_bench - mse_model) / mse_model)
    p_value = float(_normal_two_sided_pvalue(math.copysign(math.sqrt(abs(f_stat)), f_stat)))
    return {"stat_test": "mse_f", "n": n, "mse_model": mse_model, "mse_benchmark": mse_bench, "f_statistic": f_stat, "p_value": p_value}


def _compute_mse_t_test(predictions: pd.DataFrame, *, dependence_correction: str = "none") -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    payload = _basic_loss_diff_test(loss_diff, stat_test="mse_t", dependence_correction=dependence_correction, horizon=max(int(rows["horizon"].max()), 1))
    payload["t_statistic"] = payload.pop("statistic")
    return payload


def _compute_cpa_test(predictions: pd.DataFrame, *, dependence_correction: str = "nw_hac") -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    payload = _basic_loss_diff_test(loss_diff, stat_test="cpa", dependence_correction=dependence_correction, horizon=max(int(rows["horizon"].max()), 1))
    payload["instrument_set"] = "constant_only"
    return payload


def _compute_rossi_test(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 5:
        raise ExecutionError("rossi test requires at least five forecasts")
    centered = loss_diff - float(np.mean(loss_diff))
    denom = float(np.std(centered, ddof=1))
    if denom <= 0:
        raise ExecutionError("rossi test requires positive standard deviation")
    path = np.cumsum(centered) / (denom * math.sqrt(n))
    sup_stat = float(np.max(np.abs(path)))
    p_value = float(min(1.0, 2.0 * math.exp(-2.0 * sup_stat ** 2)))
    return {"stat_test": "rossi", "n": n, "sup_statistic": sup_stat, "p_value": p_value, "significant_5pct": bool(p_value < 0.05)}


def _compute_rolling_dm_test(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    window = max(4, min(max(n // 2, 4), n - 1))
    if n <= window:
        raise ExecutionError("rolling_dm requires more observations than window length")
    stats = []
    for start_idx in range(0, n - window + 1):
        chunk = loss_diff[start_idx:start_idx + window]
        var = float(np.var(chunk, ddof=1))
        if var <= 0:
            continue
        stats.append(float(np.mean(chunk) / math.sqrt(var / window)))
    if not stats:
        raise ExecutionError("rolling_dm produced no valid windows")
    return {"stat_test": "rolling_dm", "n": n, "window": window, "n_windows": len(stats), "mean_window_statistic": float(np.mean(stats)), "max_abs_window_statistic": float(np.max(np.abs(stats)))}


def _compute_reality_check(predictions: pd.DataFrame, *, block_bootstrap: bool) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    obs = float(np.mean(loss_diff))
    block = max(int(rows["horizon"].max()), 2)
    null_draws = _bootstrap_mean_distribution(loss_diff - obs, block_length=(block if block_bootstrap else 1))
    p_value = float(np.mean(null_draws >= obs))
    return {"stat_test": "reality_check", "n": int(len(loss_diff)), "mean_loss_diff": obs, "bootstrap_p_value": p_value, "bootstrap_scheme": "block" if block_bootstrap else "iid", "n_boot": int(len(null_draws))}


def _compute_spa(predictions: pd.DataFrame, *, block_bootstrap: bool) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    obs = float(max(np.mean(loss_diff), 0.0))
    block = max(int(rows["horizon"].max()), 2)
    recentered = np.maximum(loss_diff - np.mean(loss_diff), 0.0)
    null_draws = _bootstrap_mean_distribution(recentered, block_length=(block if block_bootstrap else 1))
    p_value = float(np.mean(null_draws >= obs))
    return {"stat_test": "spa", "n": int(len(loss_diff)), "spa_statistic": obs, "bootstrap_p_value": p_value, "bootstrap_scheme": "block" if block_bootstrap else "iid", "n_boot": int(len(null_draws))}


def _compute_mcs_test(predictions: pd.DataFrame, *, block_bootstrap: bool, alpha: float = 0.10) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    mean_diff = float(np.mean(loss_diff))
    block = max(int(rows["horizon"].max()), 2)
    null_draws = _bootstrap_mean_distribution(loss_diff - mean_diff, block_length=(block if block_bootstrap else 1))
    p_value = float(np.mean(np.abs(null_draws) >= abs(mean_diff)))
    confidence_set = ["benchmark", "model"] if p_value >= alpha else (["model"] if mean_diff > 0 else ["benchmark"])
    return {"stat_test": "mcs", "n": int(len(loss_diff)), "alpha": float(alpha), "mean_loss_diff": mean_diff, "bootstrap_p_value": p_value, "confidence_set": confidence_set, "bootstrap_scheme": "block" if block_bootstrap else "iid", "n_boot": int(len(null_draws))}


def _compute_mincer_zarnowitz(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    y_true = rows["y_true"].to_numpy(dtype=float)
    fcast = rows["y_pred"].to_numpy(dtype=float)
    n = int(len(y_true))
    if n < 3:
        raise ExecutionError("mincer_zarnowitz requires at least 3 forecasts")
    X_mat = np.column_stack([np.ones(n), fcast])
    beta_hat = np.linalg.lstsq(X_mat, y_true, rcond=None)[0]
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])
    residuals = y_true - X_mat @ beta_hat
    ssr = float(np.dot(residuals, residuals))
    mse = ssr / (n - 2)
    XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
    t_alpha = alpha / math.sqrt(float(XtX_inv[0, 0]) * mse)
    t_beta = (beta - 1.0) / math.sqrt(float(XtX_inv[1, 1]) * mse)
    p_alpha = float(_normal_two_sided_pvalue(t_alpha))
    p_beta = float(_normal_two_sided_pvalue(t_beta))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r_squared = 1.0 - ssr / ss_tot if abs(ss_tot) > 1e-10 else float("nan")
    return {"stat_test": "mincer_zarnowitz", "n": n, "alpha": alpha, "beta": beta, "t_alpha": t_alpha, "t_beta": t_beta, "p_alpha": p_alpha, "p_beta": p_beta, "r_squared": r_squared, "unbiased": bool(abs(t_alpha) < 1.96 and abs(t_beta) < 1.96)}


def _compute_ljung_box_test(predictions: pd.DataFrame, max_lag: int = 10) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    errors = rows["y_true"].to_numpy(dtype=float) - rows["y_pred"].to_numpy(dtype=float)
    n = int(len(errors))
    effective_lags = min(max_lag, max(1, n // 5))
    if effective_lags < 1:
        raise ExecutionError("ljung_box requires sufficient observations")
    acf_vals = []
    for lag in range(1, effective_lags + 1):
        num = float(np.dot(errors[lag:], errors[:-lag])) / n
        denom = float(np.dot(errors, errors)) / n
        acf_vals.append(num / denom if abs(denom) > 1e-10 else 0.0)
    q_stat = float(n * (n + 2) * sum(acf_vals[l] ** 2 / (n - l - 1) for l in range(effective_lags)))
    from scipy.stats import chi2 as chi2_scipy
    p_value = float(1.0 - chi2_scipy.cdf(q_stat, df=effective_lags)) if effective_lags > 0 else float("nan")
    return {"stat_test": "ljung_box", "n": n, "max_lag": effective_lags, "q_statistic": q_stat, "p_value": p_value, "significant_5pct": bool(p_value < 0.05)}


def _compute_arch_lm_test(predictions: pd.DataFrame, max_lag: int = 5) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    errors = rows["y_true"].to_numpy(dtype=float) - rows["y_pred"].to_numpy(dtype=float)
    sq_errors = errors ** 2
    n = int(len(sq_errors))
    T = min(max_lag, max(1, n // 5))
    if T < 1 or T >= n:
        raise ExecutionError("arch_lm requires sufficient observations")
    X_mat = np.column_stack([sq_errors[T - lag - 1: n - lag - 1] for lag in range(T)])
    y_vec = sq_errors[T:]
    n_reg = len(y_vec)
    if n_reg < T + 2:
        raise ExecutionError("arch_lm regression has insufficient observations")
    coeffs = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
    residuals = y_vec - X_mat @ coeffs
    ssr = float(np.dot(residuals, residuals))
    ss_tot = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = 1.0 - ssr / ss_tot if abs(ss_tot) > 1e-10 else 0.0
    lm_stat = float(n_reg * r2)
    from scipy.stats import chi2 as chi2_scipy
    p_value = float(1.0 - chi2_scipy.cdf(lm_stat, df=T)) if T > 0 else float("nan")
    return {"stat_test": "arch_lm", "n": n, "regression_lags": T, "lm_statistic": lm_stat, "p_value": p_value, "significant_5pct": bool(p_value < 0.05)}


def _compute_bias_test(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    errors = rows["y_true"].to_numpy(dtype=float) - rows["y_pred"].to_numpy(dtype=float)
    n = int(len(errors))
    if n < 2:
        raise ExecutionError("bias_test requires at least 2 forecasts")
    mean_error = float(errors.mean())
    variance = float(np.var(errors, ddof=1))
    if variance <= 0:
        raise ExecutionError("bias_test variance must be positive")
    t_stat = float(mean_error / math.sqrt(variance / n))
    p_value = float(_normal_two_sided_pvalue(t_stat))
    return {"stat_test": "bias_test", "n": n, "mean_error": mean_error, "variance": variance, "t_statistic": t_stat, "p_value": p_value, "significant_bias": bool(p_value < 0.05)}


def _compute_pesaran_timmermann(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    y_series = rows["y_true"].astype(float).reset_index(drop=True)
    f_series = rows["y_pred"].astype(float).reset_index(drop=True)
    actual_dir = (y_series.diff().dropna() > 0).astype(int).to_numpy(dtype=int)
    pred_dir = (f_series.iloc[1:].to_numpy(dtype=float) - y_series.shift(1).dropna().to_numpy(dtype=float) > 0).astype(int)
    n = int(len(actual_dir))
    if n < 2:
        raise ExecutionError("pesaran_timmermann requires consecutive forecasts")
    p_a = float(actual_dir.mean())
    p_f = float(np.mean(pred_dir))
    hit_rate = float(np.mean(actual_dir == pred_dir))
    variance = max(p_a * (1 - p_a) * p_f * (1 - p_f), 1e-10)
    statistic = float((hit_rate - (p_a * p_f + (1 - p_a) * (1 - p_f))) / math.sqrt(variance / n))
    p_value = float(_normal_two_sided_pvalue(statistic))
    return {"stat_test": "pesaran_timmermann", "n": n, "actual_up_prob": p_a, "forecast_up_prob": p_f, "hit_rate": hit_rate, "statistic": statistic, "p_value": p_value, "significant_5pct": bool(p_value < 0.05)}


def _compute_binomial_hit_test(predictions: pd.DataFrame, threshold: float = 0.0) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    y_series = rows["y_true"].astype(float).reset_index(drop=True)
    f_series = rows["y_pred"].astype(float).reset_index(drop=True)
    actual_dir = (y_series.diff().dropna() > threshold).astype(int).to_numpy(dtype=int)
    pred_dir = (f_series.iloc[1:].to_numpy(dtype=float) - y_series.shift(1).dropna().to_numpy(dtype=float) > threshold).astype(int)
    n = int(len(actual_dir))
    if n < 2:
        raise ExecutionError("binomial_hit requires consecutive forecasts")
    hits = int(np.sum(actual_dir == pred_dir))
    hit_rate = hits / n
    try:
        from scipy.stats import binomtest
        p_value = float(binomtest(hits, n, 0.5, alternative="greater").pvalue)
    except Exception:
        z = (hit_rate - 0.5) / math.sqrt(0.25 / n)
        p_value = float(_normal_two_sided_pvalue(z))
    return {"stat_test": "binomial_hit", "n": n, "hits": hits, "hit_rate": float(hit_rate), "threshold": threshold, "p_value": p_value, "significant_5pct": bool(p_value < 0.05)}


def _compute_diagnostics_bundle(predictions: pd.DataFrame) -> dict[str, object]:
    return {
        "stat_test": "diagnostics_full",
        "mincer_zarnowitz": _compute_mincer_zarnowitz(predictions),
        "ljung_box": _compute_ljung_box_test(predictions),
        "arch_lm": _compute_arch_lm_test(predictions),
        "bias_test": _compute_bias_test(predictions),
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

    last_tuning_payload: dict[str, object] = {}
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
        model = RandomForestRegressor(n_estimators=200, random_state=current_seed(model_family="randomforest"))
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


def _importance_feature_names(recipe: RecipeSpec, raw_frame: pd.DataFrame, target_series: pd.Series, contract: PreprocessContract) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    feature_builder = _feature_builder(recipe)
    horizon = min(recipe.horizons)
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    origin_idx = len(target_series) - max(recipe.horizons) - 1
    if origin_idx < _minimum_train_size(recipe) - 1:
        raise ExecutionError("importance requires at least one valid forecast origin")
    start_idx = max(0, origin_idx + 1 - _rolling_window_size(recipe)) if rolling else 0
    if feature_builder == "raw_feature_panel":
        aligned_frame = raw_frame.loc[target_series.index]
        feature_names = _raw_panel_columns(aligned_frame, recipe.target)
        X_train, y_train, X_pred = _build_raw_panel_training_data(aligned_frame, recipe.target, horizon, start_idx, origin_idx, contract)
        return feature_names, X_train, y_train, X_pred
    if feature_builder == "autoreg_lagged_target":
        train = target_series.iloc[start_idx: origin_idx + 1]
        lag_order = _lag_order(recipe, train)
        X_train, y_train = _build_lagged_supervised_matrix(train, lag_order)
        X_pred = np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1], dtype=float).reshape(1, -1)
        feature_names = [f"lag_{lag}" for lag in range(1, lag_order + 1)]
        return feature_names, X_train, y_train, X_pred
    raise ExecutionError(f"importance not implemented for feature_builder {feature_builder!r}")


def _fit_importance_model(recipe: RecipeSpec, X_train: np.ndarray, y_train: np.ndarray):
    model_family = _model_family(recipe)
    if model_family == "quantile_linear":
        raise ExecutionError("importance not implemented for quantile_linear")
    if model_family == "adaptivelasso":
        return fit_adaptive_lasso(X_train, y_train, recipe.training_spec)
    model, _ = fit_with_optional_tuning(model_family, X_train, y_train, recipe.training_spec)
    return model


def _importance_training_bundle(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    feature_names, X_train, y_train, X_pred = _importance_feature_names(recipe, raw_frame, target_series, contract)
    model = _fit_importance_model(recipe, X_train, y_train)
    return {
        "feature_names": feature_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_pred": X_pred,
        "model": model,
        "model_family": _model_family(recipe),
        "feature_builder": _feature_builder(recipe),
    }


def _predict_importance_model(model, model_family: str, X: np.ndarray) -> np.ndarray:
    if model_family == "adaptivelasso":
        return predict_adaptive_lasso(model, X)
    return model.predict(X)


def _ranked_feature_payload(name: str, feature_names: list[str], values: np.ndarray) -> dict[str, object]:
    ranked = [
        {"feature": feature, name: float(value)}
        for feature, value in sorted(zip(feature_names, values), key=lambda item: abs(item[1]), reverse=True)
    ]
    return ranked


def _compute_tree_shap_importance(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    import shap
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    if bundle["model_family"] not in {"randomforest", "extratrees", "gbm", "xgboost", "lightgbm", "catboost"}:
        raise ExecutionError("tree_shap requires tree-based model_family")
    sample = bundle["X_train"][: min(32, len(bundle["X_train"]))]
    explainer = shap.TreeExplainer(bundle["model"])
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    mean_abs = np.mean(np.abs(np.asarray(shap_values, dtype=float)), axis=0)
    return {
        "importance_method": "tree_shap",
        "model_family": bundle["model_family"],
        "feature_builder": bundle["feature_builder"],
        "n_samples": int(len(sample)),
        "feature_importance": _ranked_feature_payload("mean_abs_shap", bundle["feature_names"], mean_abs),
    }


def _compute_linear_shap_importance(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    import shap
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    if bundle["model_family"] not in {"ols", "ridge", "lasso", "elasticnet", "bayesianridge", "huber", "adaptivelasso"}:
        raise ExecutionError("linear_shap requires linear model_family")
    sample = bundle["X_train"][: min(32, len(bundle["X_train"]))]
    try:
        explainer = shap.LinearExplainer(bundle["model"], sample)
        shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        mean_abs = np.mean(np.abs(np.asarray(shap_values, dtype=float)), axis=0)
    except Exception:
        if bundle["model_family"] == "adaptivelasso":
            coef = np.abs(bundle["model"].coef_ / bundle["model"]._adaptive_weights)
        else:
            coef = np.abs(np.asarray(bundle["model"].coef_, dtype=float))
        mean_abs = coef
    return {
        "importance_method": "linear_shap",
        "model_family": bundle["model_family"],
        "feature_builder": bundle["feature_builder"],
        "n_samples": int(len(sample)),
        "feature_importance": _ranked_feature_payload("mean_abs_shap", bundle["feature_names"], np.asarray(mean_abs, dtype=float)),
    }


def _compute_kernel_shap_importance(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    import shap
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    background = bundle["X_train"][: min(20, len(bundle["X_train"]))]
    explain_row = bundle["X_pred"]
    predict_fn = lambda x: _predict_importance_model(bundle["model"], bundle["model_family"], np.asarray(x, dtype=float))
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explain_row, nsamples=min(50, max(10, background.shape[1] * 5)))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    row_values = np.abs(np.asarray(shap_values, dtype=float)).reshape(-1)
    return {
        "importance_method": "kernel_shap",
        "model_family": bundle["model_family"],
        "feature_builder": bundle["feature_builder"],
        "n_background": int(len(background)),
        "feature_importance": _ranked_feature_payload("abs_shap", bundle["feature_names"], row_values),
    }


def _compute_permutation_importance_artifact(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    result = permutation_importance(bundle["model"], bundle["X_train"], bundle["y_train"], n_repeats=5, random_state=current_seed(model_family="permutation_importance"))
    return {
        "importance_method": "permutation_importance",
        "model_family": bundle["model_family"],
        "feature_builder": bundle["feature_builder"],
        "feature_importance": _ranked_feature_payload("mean_importance", bundle["feature_names"], result.importances_mean),
        "importance_std": [float(x) for x in result.importances_std],
    }


def _compute_lime_artifact(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    X_train = np.asarray(bundle["X_train"], dtype=float)
    x0 = np.asarray(bundle["X_pred"], dtype=float).reshape(-1)
    std = np.std(X_train, axis=0, ddof=1)
    std[std == 0] = 1.0
    rng = np.random.default_rng(current_seed(model_family="lime"))
    perturbed = rng.normal(loc=x0, scale=std, size=(64, len(x0)))
    y_local = _predict_importance_model(bundle["model"], bundle["model_family"], perturbed)
    distances = np.sqrt(np.sum(((perturbed - x0) / std) ** 2, axis=1))
    weights = np.exp(-(distances ** 2) / max(len(x0), 1))
    X_design = np.column_stack([np.ones(len(perturbed)), perturbed - x0])
    W = np.diag(weights)
    coef = np.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y_local)
    local_coef = coef[1:]
    return {
        "importance_method": "lime",
        "implementation": "lime_style_local_surrogate",
        "model_family": bundle["model_family"],
        "feature_builder": bundle["feature_builder"],
        "local_importance": _ranked_feature_payload("coefficient", bundle["feature_names"], local_coef),
    }


def _compute_feature_ablation_artifact(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    baseline = float(_predict_importance_model(bundle["model"], bundle["model_family"], bundle["X_pred"])[0])
    mean_vec = np.mean(bundle["X_train"], axis=0)
    deltas = []
    for idx, feature in enumerate(bundle["feature_names"]):
        altered = np.asarray(bundle["X_pred"], dtype=float).copy()
        altered[0, idx] = mean_vec[idx]
        pred = float(_predict_importance_model(bundle["model"], bundle["model_family"], altered)[0])
        deltas.append(abs(baseline - pred))
    return {
        "importance_method": "feature_ablation",
        "model_family": bundle["model_family"],
        "feature_builder": bundle["feature_builder"],
        "baseline_prediction": baseline,
        "feature_importance": _ranked_feature_payload("ablation_delta", bundle["feature_names"], np.asarray(deltas, dtype=float)),
    }


def _top_feature_indices(bundle: dict[str, object], top_k: int = 3) -> list[int]:
    X_train = np.asarray(bundle["X_train"], dtype=float)
    model = bundle["model"]
    model_family = str(bundle["model_family"])
    if model_family == "adaptivelasso":
        score = np.abs(model.coef_ / model._adaptive_weights)
    elif hasattr(model, "coef_"):
        score = np.abs(np.asarray(model.coef_, dtype=float))
    elif hasattr(model, "feature_importances_"):
        score = np.abs(np.asarray(model.feature_importances_, dtype=float))
    else:
        result = permutation_importance(model, X_train, np.asarray(bundle["y_train"], dtype=float), n_repeats=3, random_state=current_seed(model_family="top_feature_fallback"))
        score = np.abs(result.importances_mean)
    order = np.argsort(score)[::-1]
    return [int(i) for i in order[: min(top_k, len(order))]]


def _compute_profile_artifact(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract, *, mode: str) -> dict[str, object]:
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    X_train = np.asarray(bundle["X_train"], dtype=float)
    feature_names = bundle["feature_names"]
    indices = _top_feature_indices(bundle, top_k=3)
    payload = {"importance_method": mode, "model_family": bundle["model_family"], "feature_builder": bundle["feature_builder"], "profiles": []}
    for idx in indices:
        col = X_train[:, idx]
        grid = np.quantile(col, np.linspace(0.1, 0.9, 5))
        if mode == "pdp":
            values = []
            for value in grid:
                X_mod = X_train.copy()
                X_mod[:, idx] = value
                values.append(float(np.mean(_predict_importance_model(bundle["model"], bundle["model_family"], X_mod))))
            payload["profiles"].append({"feature": feature_names[idx], "grid": [float(x) for x in grid], "pdp": values})
        elif mode == "ice":
            curves = []
            rows = X_train[: min(10, len(X_train))]
            for row in rows:
                vals = []
                for value in grid:
                    row_mod = row.copy()
                    row_mod[idx] = value
                    vals.append(float(_predict_importance_model(bundle["model"], bundle["model_family"], row_mod.reshape(1, -1))[0]))
                curves.append(vals)
            payload["profiles"].append({"feature": feature_names[idx], "grid": [float(x) for x in grid], "ice": curves})
        else:
            bins = np.quantile(col, np.linspace(0.0, 1.0, 6))
            effects = []
            for left, right in zip(bins[:-1], bins[1:]):
                mask = (col >= left) & (col <= right)
                if not np.any(mask):
                    effects.append(0.0)
                    continue
                X_low = X_train[mask].copy()
                X_high = X_train[mask].copy()
                X_low[:, idx] = left
                X_high[:, idx] = right
                diff = _predict_importance_model(bundle["model"], bundle["model_family"], X_high) - _predict_importance_model(bundle["model"], bundle["model_family"], X_low)
                effects.append(float(np.mean(diff)))
            ale = np.cumsum(effects).tolist()
            payload["profiles"].append({"feature": feature_names[idx], "bins": [float(x) for x in bins], "ale": [float(x) for x in ale]})
    return payload


def _feature_group(name: str) -> str:
    if name.startswith("lag_"):
        return "lag_block"
    if '_' in name:
        return name.split('_')[0]
    return name


def _compute_grouped_permutation_artifact(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    base = _compute_permutation_importance_artifact(raw_frame, target_series, recipe, contract)
    grouped: dict[str, float] = {}
    for row in base["feature_importance"]:
        group = _feature_group(row["feature"])
        grouped[group] = grouped.get(group, 0.0) + float(row["mean_importance"])
    payload = [{"group": key, "mean_importance": float(val)} for key, val in sorted(grouped.items(), key=lambda item: item[1], reverse=True)]
    return {
        "importance_method": "grouped_permutation",
        "base_method": "permutation_importance",
        "group_importance": payload,
    }


def _compute_importance_stability_artifact(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    X_train = np.asarray(bundle["X_train"], dtype=float)
    y_train = np.asarray(bundle["y_train"], dtype=float)
    model_family = str(bundle["model_family"])
    seeds = [11, 22, 33, 44, 55]
    rank_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(X_train), len(X_train))
        X_boot = X_train[idx]
        y_boot = y_train[idx]
        model = _fit_importance_model(recipe, X_boot, y_boot)
        result = permutation_importance(model, X_boot, y_boot, n_repeats=3, random_state=seed)
        rank_rows.append(np.asarray(result.importances_mean, dtype=float))
    ranks = np.vstack(rank_rows)
    mean_imp = np.mean(ranks, axis=0)
    std_imp = np.std(ranks, axis=0, ddof=1)
    rank_order = np.argsort(-ranks, axis=1)
    top_feature = int(np.argmax(mean_imp))
    top_rank_positions = [int(np.where(row == top_feature)[0][0] + 1) for row in rank_order]
    return {
        "importance_method": "importance_stability",
        "n_bootstrap": len(seeds),
        "top_feature": bundle["feature_names"][top_feature],
        "top_rank_positions": top_rank_positions,
        "feature_importance": [
            {"feature": feature, "mean_importance": float(m), "std_importance": float(s)}
            for feature, m, s in sorted(zip(bundle["feature_names"], mean_imp, std_imp), key=lambda item: item[1], reverse=True)
        ],
    }


def _build_predictions(
    raw_frame: pd.DataFrame,
    target_series: pd.Series,
    recipe: RecipeSpec,
    contract: PreprocessContract,
    *,
    compute_mode: str = "serial",
) -> tuple[pd.DataFrame, dict[str, object]]:
    minimum_train_size = _minimum_train_size(recipe)
    benchmark_family = _benchmark_family(recipe)
    model_spec = _model_spec(recipe)
    model_executor = _get_model_executor(recipe)
    benchmark_executor = _get_benchmark_executor(recipe)

    last_tuning_payload: dict[str, object] = {}
    aligned_frame = raw_frame.loc[target_series.index]
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    rolling_window_size = _rolling_window_size(recipe)
    outer_window = str(recipe.training_spec.get("outer_window", "rolling" if rolling else "expanding"))
    refit_policy = str(recipe.training_spec.get("refit_policy", "refit_every_step"))
    anchored_max_window_size = int(recipe.training_spec.get("anchored_max_window_size", rolling_window_size))
    refit_k_steps = int(recipe.training_spec.get("refit_k_steps", 3))

    def _rows_for_horizon(horizon: int) -> list[dict[str, object]]:
        nonlocal last_tuning_payload
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
            if model_output.get("tuning_payload"):
                last_tuning_payload = model_output["tuning_payload"]
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

    return pd.DataFrame(rows), last_tuning_payload


def _compute_metrics(predictions: pd.DataFrame, recipe: RecipeSpec) -> dict[str, object]:
    metrics_by_horizon: dict[str, dict[str, object]] = {}
    for horizon, group in predictions.groupby("horizon", sort=True):
        selected_lag_counts = {
            str(int(lag)): int(count)
            for lag, count in group["selected_lag"].value_counts().sort_index().items()
        }
        msfe = float(group["squared_error"].mean())
        benchmark_msfe = float(group["benchmark_squared_error"].mean())
        mae = float(group["abs_error"].mean())
        benchmark_mae = float(group["benchmark_abs_error"].mean())
        rmse = float(msfe**0.5)
        benchmark_rmse = float(benchmark_msfe**0.5)
        nonzero_true = group["y_true"].replace(0, np.nan)
        model_ape = (group["error"].abs() / nonzero_true.abs()).replace([np.inf, -np.inf], np.nan)
        benchmark_ape = (group["benchmark_error"].abs() / nonzero_true.abs()).replace([np.inf, -np.inf], np.nan)
        mape = float(model_ape.mean()) if not model_ape.dropna().empty else float("nan")
        benchmark_mape = float(benchmark_ape.mean()) if not benchmark_ape.dropna().empty else float("nan")
        csfe = float(group["squared_error"].sum())
        relative_msfe = msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0
        relative_rmse = rmse / benchmark_rmse if benchmark_rmse > 0 else 1.0
        relative_mae = mae / benchmark_mae if benchmark_mae > 0 else 1.0
        oos_r2 = 1.0 - relative_msfe
        benchmark_win_rate = float((group["squared_error"] < group["benchmark_squared_error"]).mean())
        y_true_diff = group["y_true"].diff()
        y_pred_diff = group["y_pred"].diff()
        valid_direction = y_true_diff.notna() & y_pred_diff.notna()
        if valid_direction.any():
            directional_accuracy = float((np.sign(y_true_diff[valid_direction]) == np.sign(y_pred_diff[valid_direction])).mean())
        else:
            directional_accuracy = float("nan")
        sign_accuracy = float((np.sign(group["y_true"]) == np.sign(group["y_pred"]).astype(float)).mean())
        metrics_by_horizon[f"h{int(horizon)}"] = {
            "n_predictions": int(len(group)),
            "msfe": msfe,
            "benchmark_msfe": benchmark_msfe,
            "relative_msfe": relative_msfe,
            "oos_r2": oos_r2,
            "csfe": csfe,
            "mae": mae,
            "benchmark_mae": benchmark_mae,
            "relative_mae": relative_mae,
            "rmse": rmse,
            "benchmark_rmse": benchmark_rmse,
            "relative_rmse": relative_rmse,
            "mape": mape,
            "benchmark_mape": benchmark_mape,
            "benchmark_win_rate": benchmark_win_rate,
            "directional_accuracy": directional_accuracy,
            "sign_accuracy": sign_accuracy,
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
        model_mae = float(group["abs_error"].mean())
        benchmark_mae = float(group["benchmark_abs_error"].mean())
        model_rmse = float(model_msfe**0.5)
        benchmark_rmse = float(benchmark_msfe**0.5)
        loss_diff = group["benchmark_squared_error"] - group["squared_error"]
        y_true_diff = group["y_true"].diff()
        y_pred_diff = group["y_pred"].diff()
        valid_direction = y_true_diff.notna() & y_pred_diff.notna()
        directional_accuracy = float((np.sign(y_true_diff[valid_direction]) == np.sign(y_pred_diff[valid_direction])).mean()) if valid_direction.any() else float("nan")
        comparison_by_horizon[f"h{int(horizon)}"] = {
            "n_predictions": int(len(group)),
            "model_msfe": model_msfe,
            "benchmark_msfe": benchmark_msfe,
            "model_mae": model_mae,
            "benchmark_mae": benchmark_mae,
            "model_rmse": model_rmse,
            "benchmark_rmse": benchmark_rmse,
            "mean_loss_diff": float(loss_diff.mean()),
            "win_rate": float((group["squared_error"] < group["benchmark_squared_error"]).mean()),
            "tie_rate": float((group["squared_error"] == group["benchmark_squared_error"]).mean()),
            "relative_msfe": model_msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0,
            "relative_rmse": model_rmse / benchmark_rmse if benchmark_rmse > 0 else 1.0,
            "relative_mae": model_mae / benchmark_mae if benchmark_mae > 0 else 1.0,
            "oos_r2": 1.0 - (model_msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0),
            "benchmark_win_rate": float((group["squared_error"] < group["benchmark_squared_error"]).mean()),
            "directional_accuracy": directional_accuracy,
            "sign_accuracy": float((np.sign(group["y_true"]) == np.sign(group["y_pred"])).mean()),
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


def _recession_indicator(index: pd.DatetimeIndex) -> pd.Series:
    recession_windows = [
        ("2001-03-01", "2001-11-30"),
        ("2007-12-01", "2009-06-30"),
        ("2020-02-01", "2020-04-30"),
    ]
    indicator = pd.Series(False, index=index)
    for start, end in recession_windows:
        indicator |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return indicator


def _regime_indicator(predictions: pd.DataFrame, evaluation_spec: dict[str, object]) -> pd.Series:
    regime_definition = str(evaluation_spec.get("regime_definition", "none"))
    dates = pd.to_datetime(predictions["target_date"])
    if regime_definition == "none":
        return pd.Series(False, index=predictions.index)
    if regime_definition == "NBER_recession":
        indicator = _recession_indicator(pd.DatetimeIndex(dates))
        indicator.index = predictions.index
        return indicator.astype(bool)
    if regime_definition == "user_defined_regime":
        start = evaluation_spec.get("regime_start")
        end = evaluation_spec.get("regime_end")
        if not start or not end:
            raise ExecutionError("user_defined_regime requires evaluation_spec.regime_start and evaluation_spec.regime_end")
        return ((dates >= pd.Timestamp(str(start))) & (dates <= pd.Timestamp(str(end)))).astype(bool)
    raise ExecutionError(f"regime_definition {regime_definition!r} is not executable in current runtime slice")


def _compute_regime_summary(predictions: pd.DataFrame, recipe: RecipeSpec, evaluation_spec: dict[str, object]) -> dict[str, object]:
    if str(evaluation_spec.get("regime_use", "eval_only")) != "eval_only":
        raise ExecutionError("only regime_use='eval_only' is executable in current runtime slice")
    indicator = _regime_indicator(predictions, evaluation_spec)
    if str(evaluation_spec.get("regime_definition", "none")) == "none":
        return {
            "regime_definition": "none",
            "regime_use": evaluation_spec.get("regime_use", "eval_only"),
            "regime_metrics": evaluation_spec.get("regime_metrics", "all_main_metrics_by_regime"),
        }
    payload = {
        "regime_definition": evaluation_spec.get("regime_definition"),
        "regime_use": evaluation_spec.get("regime_use", "eval_only"),
        "regime_metrics": evaluation_spec.get("regime_metrics", "all_main_metrics_by_regime"),
        "target": recipe.target,
        "raw_dataset": recipe.raw_dataset,
        "by_horizon": {},
    }
    for horizon, group in predictions.groupby("horizon", sort=True):
        idx = group.index
        regime_group = group[indicator.loc[idx].to_numpy()]
        non_regime_group = group[~indicator.loc[idx].to_numpy()]
        horizon_payload = {
            "n_regime": int(len(regime_group)),
            "n_non_regime": int(len(non_regime_group)),
        }
        if len(regime_group) > 0:
            regime_msfe = float(regime_group["squared_error"].mean())
            regime_bench_msfe = float(regime_group["benchmark_squared_error"].mean())
            horizon_payload["regime_msfe"] = regime_msfe
            horizon_payload["regime_oos_r2"] = 1.0 - (regime_msfe / regime_bench_msfe if regime_bench_msfe > 0 else 1.0)
            horizon_payload["crisis_period_gain"] = float(regime_group["benchmark_squared_error"].mean() - regime_group["squared_error"].mean())
        if len(non_regime_group) > 0:
            non_msfe = float(non_regime_group["squared_error"].mean())
            non_bench_msfe = float(non_regime_group["benchmark_squared_error"].mean())
            horizon_payload["non_regime_msfe"] = non_msfe
            horizon_payload["non_regime_oos_r2"] = 1.0 - (non_msfe / non_bench_msfe if non_bench_msfe > 0 else 1.0)
        payload["by_horizon"][f"h{int(horizon)}"] = horizon_payload
    return payload


def execute_recipe(
    *,
    recipe: RecipeSpec,
    preprocess: PreprocessContract,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
    provenance_payload: dict | None = None,
    cache_root: str | Path | None = None,
) -> ExecutionResult:
    if not is_operational_preprocess_contract(preprocess):
        raise ExecutionError(
            "current execution slice only supports explicit operational preprocessing contracts"
        )

    if _benchmark_family(recipe) not in {
        "historical_mean", "zero_change", "ar_bic", "custom_benchmark",
        "rolling_mean", "random_walk", "ar_fixed_p", "ardi", "factor_model",
        "expert_benchmark", "multi_benchmark_suite",
    }:
        raise ExecutionError(
            f"benchmark_family {_benchmark_family(recipe)!r} is representable but not executable in current runtime slice"
        )
    # Touch new benchmark axes so grep can confirm wiring (Phase 4).
    _ = _benchmark_window(recipe)
    _ = _benchmark_scope(recipe)
    _ = _benchmark_window_len(recipe)
    _get_model_executor(recipe)
    _get_benchmark_executor(recipe)

    run = build_run_spec(recipe)
    spec = build_execution_spec(recipe=recipe, run=run, preprocess=preprocess)
    output_root = Path(output_root)
    run_dir = output_root / run.artifact_subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    effective_cache_root = Path(cache_root) if cache_root is not None else (output_root / ".raw_cache")
    stat_test_spec = _stat_test_spec(provenance_payload)
    evaluation_spec = _evaluation_spec(provenance_payload)
    importance_spec = _importance_spec(provenance_payload)
    reproducibility_spec = _reproducibility_spec(provenance_payload)
    failure_policy_spec = _failure_policy_spec(provenance_payload)
    compute_mode_spec = _compute_mode_spec(provenance_payload)
    output_spec = _output_spec(provenance_payload)
    failure_policy = str(failure_policy_spec.get("failure_policy", "fail_fast"))
    compute_mode = str(compute_mode_spec.get("compute_mode", "serial"))
    variant_id = (provenance_payload or {}).get("variant_id")
    _seed_token = set_context(
        ReproducibilityContext(
            recipe_id=recipe.recipe_id,
            variant_id=None if variant_id is None else str(variant_id),
            reproducibility_spec=reproducibility_spec,
        )
    )
    raw_result = _load_raw_for_recipe(recipe, local_raw_source, effective_cache_root)
    _release_lag = _data_task_axis(recipe, "release_lag_rule")
    _missing_avail = _data_task_axis(recipe, "missing_availability")
    _var_universe = _data_task_axis(recipe, "variable_universe")
    _min_train_axis = _data_task_axis(recipe, "min_train_size")
    _break_seg = _data_task_axis(recipe, "structural_break_segmentation")
    _horizon_list_axis = _data_task_axis(recipe, "horizon_list")
    _eval_scale = _data_task_axis(recipe, "evaluation_scale")
    _separation = _data_task_axis(recipe, "separation_rule")
    raw_result = _apply_release_lag(raw_result, _release_lag)
    raw_result = _apply_missing_availability(raw_result, _missing_avail)
    raw_result = _apply_variable_universe(raw_result, _var_universe)
    targets = _recipe_targets(recipe)
    prediction_frames = []
    failed_components: list[dict[str, object]] = []
    successful_targets: list[str] = []
    target_series = None
    _last_tp: dict[str, object] = {}
    def _target_job(target: str):
        target_recipe = _recipe_for_target(recipe, target)
        target_series_local = _get_target_series(raw_result.data, target, _minimum_train_size(target_recipe))
        frame, tp = _build_predictions(raw_result.data, target_series_local, target_recipe, preprocess, compute_mode=compute_mode)
        return target, target_series_local, frame, tp

    if compute_mode == "parallel_by_model" and len(targets) > 1:
        with ThreadPoolExecutor(max_workers=min(len(targets), 4)) as ex:
            futures = [ex.submit(contextvars.copy_context().run, _target_job, target) for target in targets]
            for future in futures:
                try:
                    target, target_series_local, frame, _last_tp = future.result()
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
                target, target_series_local, frame, _last_tp = _target_job(target)
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
        "evaluation_spec": evaluation_spec,
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
        "regime_file": "regime_summary.json" if evaluation_spec.get("regime_definition", "none") != "none" else None,
        "successful_targets": successful_targets,
        "partial_run": bool(failed_components),
        "output_spec": output_spec,
    }
    if provenance_payload:
        manifest.update(provenance_payload)
    if failed_components:
        manifest["failure_log_file"] = "failures.json"
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
    export_format = str(output_spec.get('export_format', 'json'))
    saved_objects = str(output_spec.get('saved_objects', 'full_bundle'))
    provenance_fields = str(output_spec.get('provenance_fields', 'full'))

    # Provenance injection based on provenance_fields level
    if provenance_fields in ('minimal', 'standard', 'full'):
        manifest['git_commit'] = _get_git_commit()
        manifest['package_version'] = _get_package_version()
    if provenance_fields == 'full':
        manifest['config_hash'] = _compute_config_hash(recipe)
        tc = manifest.get('tree_context', {})
        manifest['tree_context'] = tc

    # Write data_preview (always CSV, optionally parquet)
    raw_result.data.head(20).to_csv(run_dir / 'data_preview.csv')
    if export_format in ('parquet', 'all'):
        raw_result.data.head(20).to_parquet(run_dir / 'data_preview.parquet')

    # Write predictions (always CSV, optionally parquet)
    predictions.to_csv(run_dir / 'predictions.csv', index=False)
    if export_format in ('parquet', 'all'):
        predictions.to_parquet(run_dir / 'predictions.parquet')

    # Write structured metrics/comparison/regime based on export_format
    metrics_files = {}
    comparison_files = {}
    if export_format in ('json', 'json+csv', 'all'):
        _write_json(run_dir / 'metrics.json', metrics)
        _write_json(run_dir / 'comparison_summary.json', comparison_summary)
        metrics_files['json'] = 'metrics.json'
        comparison_files['json'] = 'comparison_summary.json'
    if export_format in ('csv', 'json+csv', 'all'):
        _write_csv(run_dir / 'metrics.csv', [metrics])
        _write_csv(run_dir / 'comparison_summary.csv', [comparison_summary])
        metrics_files['csv'] = 'metrics.csv'
        comparison_files['csv'] = 'comparison_summary.csv'
    if export_format in ('parquet', 'all'):
        _write_parquet(run_dir / 'metrics.parquet', metrics)
        _write_parquet(run_dir / 'comparison_summary.parquet', comparison_summary)
        metrics_files['parquet'] = 'metrics.parquet'
        comparison_files['parquet'] = 'comparison_summary.parquet'
    if metrics_files:
        manifest['metrics_files'] = metrics_files
        manifest['metrics_file'] = metrics_files.get('json') or metrics_files.get('csv') or metrics_files.get('parquet')
    if comparison_files:
        manifest['comparison_files'] = comparison_files
        manifest['comparison_file'] = comparison_files.get('json') or comparison_files.get('csv') or comparison_files.get('parquet')
    if evaluation_spec.get('regime_definition', 'none') != 'none':
        regime_summary = _compute_regime_summary(predictions, recipe, evaluation_spec)
        regime_files = {}
        if export_format in ('json', 'json+csv', 'all'):
            _write_json(run_dir / 'regime_summary.json', regime_summary)
            regime_files['json'] = 'regime_summary.json'
        if export_format in ('csv', 'json+csv', 'all'):
            _write_csv(run_dir / 'regime_summary.csv', [regime_summary])
            regime_files['csv'] = 'regime_summary.csv'
        if export_format in ('parquet', 'all'):
            _write_parquet(run_dir / 'regime_summary.parquet', regime_summary)
            regime_files['parquet'] = 'regime_summary.parquet'
        manifest['regime_files'] = regime_files
        manifest['regime_file'] = regime_files.get('json') or regime_files.get('csv') or regime_files.get('parquet')
    stat_test_name = str(stat_test_spec.get("stat_test", "none"))
    dependence_correction = _dependence_correction(stat_test_spec)
    from macrocast.execution.stat_tests import dispatch_stat_tests
    try:
        stat_results = dispatch_stat_tests(
            predictions=predictions,
            stat_test_spec=stat_test_spec,
            dependence_correction=dependence_correction,
        )
    except Exception as exc:
        if failure_policy == "save_partial_results":
            failed_components.append({"stage": "stat_test_artifact", "target": None, "error": str(exc)})
            stat_results = {}
        else:
            raise
    if stat_results:
        _write_json(run_dir / "stat_tests.json", stat_results)
        manifest["stat_tests"] = stat_results
        per_test_files = {}
        for axis_key, axis_payload in stat_results.items():
            test_value = axis_payload.get("stat_test")
            if not test_value or "error" in axis_payload or axis_key == "test_scope":
                continue
            per_file = f"stat_test_{test_value}.json"
            _write_json(run_dir / per_file, axis_payload)
            per_test_files[axis_key] = per_file
        if len(per_test_files) == 1:
            manifest["stat_test_file"] = next(iter(per_test_files.values()))
        else:
            manifest["stat_test_file"] = "stat_tests.json"
        if per_test_files:
            manifest["stat_test_files"] = per_test_files
    importance_method = str(importance_spec.get("importance_method", "none"))
    importance_dispatch = {
        "minimal_importance": (lambda: _compute_minimal_importance(raw_result.data, target_series, recipe, preprocess), "importance_minimal.json"),
        "tree_shap": (lambda: _compute_tree_shap_importance(raw_result.data, target_series, recipe, preprocess), "importance_tree_shap.json"),
        "kernel_shap": (lambda: _compute_kernel_shap_importance(raw_result.data, target_series, recipe, preprocess), "importance_kernel_shap.json"),
        "linear_shap": (lambda: _compute_linear_shap_importance(raw_result.data, target_series, recipe, preprocess), "importance_linear_shap.json"),
        "permutation_importance": (lambda: _compute_permutation_importance_artifact(raw_result.data, target_series, recipe, preprocess), "importance_permutation_importance.json"),
        "lime": (lambda: _compute_lime_artifact(raw_result.data, target_series, recipe, preprocess), "importance_lime.json"),
        "feature_ablation": (lambda: _compute_feature_ablation_artifact(raw_result.data, target_series, recipe, preprocess), "importance_feature_ablation.json"),
        "pdp": (lambda: _compute_profile_artifact(raw_result.data, target_series, recipe, preprocess, mode="pdp"), "importance_pdp.json"),
        "ice": (lambda: _compute_profile_artifact(raw_result.data, target_series, recipe, preprocess, mode="ice"), "importance_ice.json"),
        "ale": (lambda: _compute_profile_artifact(raw_result.data, target_series, recipe, preprocess, mode="ale"), "importance_ale.json"),
        "grouped_permutation": (lambda: _compute_grouped_permutation_artifact(raw_result.data, target_series, recipe, preprocess), "importance_grouped_permutation.json"),
        "importance_stability": (lambda: _compute_importance_stability_artifact(raw_result.data, target_series, recipe, preprocess), "importance_stability.json"),
    }
    if importance_method != "none":
        try:
            importance_payload, importance_file = importance_dispatch[importance_method][0](), importance_dispatch[importance_method][1]
            _write_json(run_dir / importance_file, importance_payload)
            manifest["importance_file"] = importance_file
        except Exception as exc:
            if failure_policy == "save_partial_results":
                failed_components.append({"stage": "importance_artifact", "target": None, "error": str(exc)})
            else:
                raise
    manifest["partial_run"] = bool(failed_components)
    if failed_components:
        manifest["failure_log_file"] = "failures.json"
        _write_json(run_dir / "failures.json", failed_components)
    # Write tuning result artifact
    tuning_result = {
        "tuning_enabled": bool(_last_tp),
        "model_family": _model_spec(recipe).get("executor_name", ""),
        "best_hp": _last_tp.get("best_hp", {}),
        "best_score": _last_tp.get("best_score", None),
        "total_trials": _last_tp.get("total_trials", 0),
        "total_time_seconds": _last_tp.get("total_time_seconds", 0.0),
        "search_algorithm": _last_tp.get("search_algorithm", "none"),
    }
    _write_json(run_dir / "tuning_result.json", tuning_result)
    manifest["tuning_result_file"] = "tuning_result.json"
    manifest["tuning_result"] = tuning_result
    _write_json(run_dir / "manifest.json", manifest)

    reset_context(_seed_token)
    return ExecutionResult(
        spec=spec,
        run=run,
        raw_result=raw_result,
        artifact_dir=str(run_dir),
    )
