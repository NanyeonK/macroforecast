from __future__ import annotations

import importlib.util
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg

from .errors import ExecutionError
from .types import ExecutionResult, ExecutionSpec
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
    if recipe.raw_dataset == "fred_md":
        return load_fred_md(cache_root=cache_root, local_source=local_raw_source)
    if recipe.raw_dataset == "fred_qd":
        return load_fred_qd(cache_root=cache_root, local_source=local_raw_source)
    if recipe.raw_dataset == "fred_sd":
        return load_fred_sd(cache_root=cache_root, local_source=local_raw_source)
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


def _model_executor_name(model_family: str, feature_builder: str) -> str:
    if feature_builder == "autoreg_lagged_target":
        return {
            "ar": "ar_bic_autoreg_v0",
            "ridge": "ridge_autoreg_v0",
            "lasso": "lasso_autoreg_v0",
            "elasticnet": "elasticnet_autoreg_v0",
            "randomforest": "randomforest_autoreg_v0",
        }[model_family]
    if feature_builder == "raw_feature_panel":
        return {
            "ridge": "ridge_raw_feature_panel_v0",
            "lasso": "lasso_raw_feature_panel_v0",
            "elasticnet": "elasticnet_raw_feature_panel_v0",
            "randomforest": "randomforest_raw_feature_panel_v0",
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
            "ridge": _run_ridge_autoreg_executor,
            "lasso": _run_lasso_autoreg_executor,
            "elasticnet": _run_elasticnet_autoreg_executor,
            "randomforest": _run_randomforest_autoreg_executor,
        }
        if model_family in dispatch:
            return dispatch[model_family]
    if feature_builder == "raw_feature_panel":
        dispatch = {
            "ridge": _run_ridge_raw_panel_executor,
            "lasso": _run_lasso_raw_panel_executor,
            "elasticnet": _run_elasticnet_raw_panel_executor,
            "randomforest": _run_randomforest_raw_panel_executor,
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


def _apply_raw_panel_preprocessing(
    X_train: np.ndarray,
    X_pred: np.ndarray,
    contract: PreprocessContract,
) -> tuple[np.ndarray, np.ndarray]:
    if contract.x_missing_policy == "em_impute":
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_pred = imputer.transform(X_pred)
    if contract.scaling_policy == "standard":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_pred = scaler.transform(X_pred)
    return X_train, X_pred


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
    X_train = frame[predictors].iloc[start_idx : origin_idx - horizon + 1].to_numpy(dtype=float)
    y_train = frame[target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
    X_pred = frame[predictors].iloc[origin_idx].to_numpy(dtype=float).reshape(1, -1)
    if len(X_train) == 0 or len(y_train) == 0:
        raise ExecutionError("raw_feature_panel produced empty training data")
    X_train, X_pred = _apply_raw_panel_preprocessing(X_train, X_pred, contract)
    return X_train, y_train, X_pred


def _run_ar_model_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    selected_lag, selected_bic, fitted = _select_ar_bic_model(train, _max_ar_lag(recipe))
    prediction = fitted.predict(start=len(train), end=len(train) + horizon - 1)
    return {
        "y_pred": float(prediction.iloc[-1]),
        "selected_lag": selected_lag,
        "selected_bic": selected_bic,
    }


def _run_ridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return {
        "y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
    }


def _run_lasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    model = Lasso(alpha=1e-4, max_iter=10000)
    model.fit(X, y)
    return {
        "y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
    }


def _run_elasticnet_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000)
    model.fit(X, y)
    return {
        "y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
    }


def _run_randomforest_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    lag_order = _lag_order(recipe, train)
    X, y = _build_lagged_supervised_matrix(train, lag_order)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    return {
        "y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
    }


def _run_ridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    X_train, y_train, X_pred = _build_raw_panel_training_data(raw_frame, recipe.target, horizon, start_idx, origin_idx, contract)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_lasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    X_train, y_train, X_pred = _build_raw_panel_training_data(raw_frame, recipe.target, horizon, start_idx, origin_idx, contract)
    model = Lasso(alpha=1e-4, max_iter=10000)
    model.fit(X_train, y_train)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_elasticnet_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    X_train, y_train, X_pred = _build_raw_panel_training_data(raw_frame, recipe.target, horizon, start_idx, origin_idx, contract)
    model = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train, y_train)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan}


def _run_randomforest_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    X_train, y_train, X_pred = _build_raw_panel_training_data(raw_frame, recipe.target, horizon, start_idx, origin_idx, contract)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
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
    if model_family not in {"ridge", "randomforest"}:
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


def _build_predictions(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    minimum_train_size = _minimum_train_size(recipe)
    benchmark_family = _benchmark_family(recipe)
    model_spec = _model_spec(recipe)
    model_executor = _get_model_executor(recipe)
    benchmark_executor = _get_benchmark_executor(recipe)

    aligned_frame = raw_frame.loc[target_series.index]
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    rolling_window_size = _rolling_window_size(recipe)

    for horizon in recipe.horizons:
        for origin_idx in range(minimum_train_size - 1, len(target_series) - horizon):
            start_idx = max(0, origin_idx + 1 - rolling_window_size) if rolling else 0
            train = target_series.iloc[start_idx : origin_idx + 1]
            model_output = model_executor(train, horizon, recipe, contract, aligned_frame, origin_idx, start_idx)
            y_pred = float(model_output["y_pred"])
            benchmark_pred = float(benchmark_executor(train, horizon, recipe))
            y_true = float(target_series.iloc[origin_idx + horizon])
            error = y_true - y_pred
            benchmark_error = y_true - benchmark_pred
            rows.append(
                {
                    "target": target_series.name,
                    "model_name": model_spec["executor_name"],
                    "benchmark_name": benchmark_family,
                    "horizon": horizon,
                    "origin_date": target_series.index[origin_idx].strftime("%Y-%m-%d"),
                    "target_date": target_series.index[origin_idx + horizon].strftime("%Y-%m-%d"),
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
    target_series = _get_target_series(raw_result.data, recipe.target, _minimum_train_size(recipe))
    predictions = _build_predictions(raw_result.data, target_series, recipe, preprocess)
    metrics = _compute_metrics(predictions, recipe)
    stat_test_spec = _stat_test_spec(provenance_payload)
    importance_spec = _importance_spec(provenance_payload)

    manifest = {
        "recipe_id": recipe.recipe_id,
        "run_id": run.run_id,
        "target": recipe.target,
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
        "stat_test_spec": stat_test_spec,
        "importance_spec": importance_spec,
        "lag_selection": _LAG_SELECTION,
        "max_lag": _max_ar_lag(recipe),
        "minimum_train_size": _minimum_train_size(recipe),
        "prediction_rows": int(len(predictions)),
        "metrics_file": "metrics.json",
    }
    if provenance_payload:
        manifest.update(provenance_payload)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (run_dir / "summary.txt").write_text(
        recipe_summary(recipe)
        + "\n"
        + preprocess_summary(preprocess)
        + "\n"
        + f"forecast_engine={_model_spec(recipe)['executor_name']}; benchmark={_benchmark_family(recipe)}; prediction_rows={len(predictions)}\n",
        encoding="utf-8",
    )
    raw_result.data.head(20).to_csv(run_dir / "data_preview.csv")
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if stat_test_spec.get("stat_test") == "dm":
        dm_payload = _compute_dm_test(predictions)
        (run_dir / "stat_test_dm.json").write_text(json.dumps(dm_payload, indent=2), encoding="utf-8")
        manifest["stat_test_file"] = "stat_test_dm.json"
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if stat_test_spec.get("stat_test") == "cw":
        cw_payload = _compute_cw_test(predictions)
        (run_dir / "stat_test_cw.json").write_text(json.dumps(cw_payload, indent=2), encoding="utf-8")
        manifest["stat_test_file"] = "stat_test_cw.json"
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if importance_spec.get("importance_method") == "minimal_importance":
        importance_payload = _compute_minimal_importance(raw_result.data, target_series, recipe, preprocess)
        (run_dir / "importance_minimal.json").write_text(json.dumps(importance_payload, indent=2), encoding="utf-8")
        manifest["importance_file"] = "importance_minimal.json"
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return ExecutionResult(
        spec=spec,
        run=run,
        raw_result=raw_result,
        artifact_dir=str(run_dir),
    )
