"""Pandas-first macro forecasting workflow tools."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.10.0a0"

_LAZY_EXPORTS = {
    "DEFAULT_RANDOM_SEED": ".meta",
    "configure": ".meta",
    "get_config": ".meta",
    "get_option": ".meta",
    "reset_config": ".meta",
    "use_config": ".meta",
    "DataBundle": ".data",
    "DataSpec": ".data",
    "as_panel": ".data",
    "attach_metadata": ".data",
    "metadata": ".data",
    "panel_info": ".data",
    "spec": ".data",
    "validate_panel": ".data",
    "combine": ".data",
    "list_vintages": ".data",
    "load_custom_csv": ".data",
    "load_custom_parquet": ".data",
    "load_fred_md": ".data",
    "load_fred_qd": ".data",
    "load_fred_sd": ".data",
    "load_fred_md_sd": ".data",
    "load_fred_qd_sd": ".data",
    "PreprocessedData": ".preprocessing",
    "preprocess": ".preprocessing",
    "reprocess": ".preprocessing",
    "FeatureSet": ".feature_engineering",
    "average_target": ".feature_engineering",
    "build_features": ".feature_engineering",
    "compose_features": ".feature_engineering",
    "direct_target": ".feature_engineering",
    "feature_matrix": ".feature_engineering",
    "group_pca": ".feature_engineering",
    "group_pca_step": ".feature_engineering",
    "lag": ".feature_engineering",
    "lag_step": ".feature_engineering",
    "lags_then_pca": ".feature_engineering",
    "maf_features": ".feature_engineering",
    "maf_step": ".feature_engineering",
    "moving_average_ladder": ".feature_engineering",
    "moving_average_pca_lags": ".feature_engineering",
    "moving_average_step": ".feature_engineering",
    "path_targets": ".feature_engineering",
    "pca_features": ".feature_engineering",
    "pca_step": ".feature_engineering",
    "pca_then_lags": ".feature_engineering",
    "rolling_mean": ".feature_engineering",
    "rolling_step": ".feature_engineering",
    "scale_features": ".feature_engineering",
    "scale_step": ".feature_engineering",
    "time_features": ".feature_engineering",
    "DataSummaryReport": ".data_summary",
    "summarize_data": ".data_summary",
    "DataAnalysisReport": ".data_analysis",
    "analyze_data": ".data_analysis",
    "ModelFit": ".models",
    "ModelParameter": ".models",
    "ModelSpec": ".models",
    "VolatilityFit": ".models",
    "ar": ".models",
    "bagging": ".models",
    "bayesian_ridge": ".models",
    "booging": ".models",
    "catboost": ".models",
    "decision_tree": ".models",
    "describe_model": ".models",
    "egarch": ".models",
    "elastic_net": ".models",
    "extra_trees": ".models",
    "far": ".models",
    "favar": ".models",
    "garch11": ".models",
    "get_model": ".models",
    "glmboost": ".models",
    "gradient_boosting": ".models",
    "huber": ".models",
    "lasso": ".models",
    "lightgbm": ".models",
    "list_model_specs": ".models",
    "macro_random_forest": ".models",
    "mars": ".models",
    "model_search_space": ".models",
    "ols": ".models",
    "pcr": ".models",
    "quantile_regression_forest": ".models",
    "random_forest": ".models",
    "realized_garch": ".models",
    "ridge": ".models",
    "slow_growing_tree": ".models",
    "var": ".models",
    "xgboost": ".models",
    "ParamDistribution": ".selection",
    "SearchError": ".selection",
    "SearchResult": ".selection",
    "SearchSpec": ".selection",
    "bayesian_search": ".selection",
    "choice": ".selection",
    "cv_path": ".selection",
    "fixed": ".selection",
    "genetic_search": ".selection",
    "grid": ".selection",
    "log_uniform": ".selection",
    "random_search": ".selection",
    "randint": ".selection",
    "select_params": ".selection",
    "search_spec": ".selection",
    "uniform": ".selection",
    "MetricLike": ".evaluation",
    "get_metric": ".evaluation",
    "mae": ".evaluation",
    "mse": ".evaluation",
    "rmse": ".evaluation",
    "Split": ".window",
    "WindowSpec": ".window",
    "blocked_kfold": ".window",
    "blocked_kfold_split": ".window",
    "expanding": ".window",
    "expanding_split": ".window",
    "last_block": ".window",
    "last_block_split": ".window",
    "make_splitter": ".window",
    "normalize_window_name": ".window",
    "poos": ".window",
    "poos_split": ".window",
    "resolve_window": ".window",
    "rolling_blocks": ".window",
    "rolling_blocks_split": ".window",
    "split_table": ".window",
}

_LAZY_MODULES: tuple[str, ...] = (
    "meta",
    "data",
    "preprocessing",
    "feature_engineering",
    "data_summary",
    "data_analysis",
    "models",
    "selection",
    "evaluation",
    "window",
)

__all__ = sorted(set(_LAZY_EXPORTS) | set(_LAZY_MODULES))


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_LAZY_MODULES))
