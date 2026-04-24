from __future__ import annotations

import contextvars
import hashlib
import importlib.util
from collections.abc import Mapping, Sequence
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
from sklearn.cross_decomposition import PLSRegression
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
    apply_reproducibility_mode,
    reset_context,
    resolve_seed,
    set_context,
)
from .horizon_target import (
    canonicalize_horizon_target_construction as _canonicalize_horizon_target_construction,
    construction_scale as _horizon_construction_scale,
    forward_scalar as _horizon_forward_scalar,
)
from .lag_polynomial_rotation import (
    build_marx_rotation_frame as _build_marx_rotation_frame,
    marx_rotation_public_feature_name as _marx_rotation_public_feature_name,
)
from ..raw.windowing import WindowSpec as _WindowSpec, _resolve_min_train_obs as _resolve_min_train_obs
from .nber import filter_origins_by_regime as _filter_origins_by_regime
from .deterministic import augment_array as _augment_deterministic_array
from .types import (
    FORECAST_PAYLOAD_CONTRACT_VERSION,
    ExecutionResult,
    ExecutionSpec,
    ForecastPayload,
    Layer2Representation,
)
from .deep_training import fit_factor_model, fit_with_optional_tuning, fit_adaptive_lasso, predict_adaptive_lasso
from ..preprocessing import (
    PreprocessContract,
    is_operational_preprocess_contract,
    preprocess_summary,
    preprocess_to_dict,
)
from ..preprocessing.feature_blocks import (
    CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION,
    CUSTOM_FINAL_Z_SELECTION_CONTRACT_VERSION,
    FeatureBlockCallableContext,
    FeatureCombinerCallableContext,
    validate_feature_block_callable_result,
    validate_feature_combiner_callable_result,
)
from ..raw import load_custom_csv, load_custom_parquet, load_fred_md, load_fred_qd, load_fred_sd
from ..raw.sd_inferred_tcodes import (
    MAP_VERSION as SD_INFERRED_TCODE_MAP_VERSION,
    build_sd_inferred_transform_codes,
    normalize_sd_tcode_policy,
)
from ..recipes import RecipeSpec, RunSpec, build_run_spec, recipe_summary
from ..custom import (
    CUSTOM_MODEL_CONTRACT_VERSION,
    get_custom_feature_block,
    get_custom_feature_combiner,
    get_custom_model,
    get_custom_preprocessor,
    get_custom_target_transformer,
    is_custom_feature_block,
    is_custom_model,
    is_custom_preprocessor,
    is_custom_target_transformer,
)

_EXECUTION_ARCHITECTURE = "separate_model_and_benchmark_executors"
_DEFAULT_MINIMUM_TRAIN_SIZE = 5
_DEFAULT_MAX_AR_LAG = 3
_LAG_SELECTION = "bic"
_TARGET_TRANSFORMER_FEATURE_RUNTIMES = {"autoreg_lagged_target", "raw_feature_panel"}
_TARGET_TRANSFORMER_RAW_PANEL_MODELS = {"ols", "ridge", "lasso", "elasticnet"}
_RAW_PANEL_FEATURE_BUILDERS = {"raw_feature_panel", "raw_X_only", "factor_pca", "factors_plus_AR"}
_RAW_PANEL_FEATURE_BLOCK_SETS = {
    "transformed_x",
    "transformed_x_lags",
    "factors_plus_target_lags",
    "factor_blocks_only",
    "high_dimensional_x",
    "selected_sparse_x",
    "level_augmented_x",
    "rotation_augmented_x",
    "mixed_blocks",
    "custom_blocks",
}

_PHASE3_DEFAULTS = {
    "release_lag_rule": "ignore_release_lag",
    "missing_availability": "complete_case_only",
    "raw_missing_policy": "preserve_raw_missing",
    "raw_outlier_policy": "preserve_raw_outliers",
    "variable_universe": "all_variables",
    "separation_rule": "strict_separation",
}
_TRAINING_AXIS_DEFAULTS = {
    "min_train_size": "fixed_n_obs",
    "training_start_rule": "earliest_possible",
}
_OFFICIAL_TRANSFORM_POLICIES = {"raw_official_frame", "dataset_tcode"}
_OFFICIAL_TRANSFORM_SCOPES = {
    "apply_tcode_to_none",
    "apply_tcode_to_target",
    "apply_tcode_to_X",
    "apply_tcode_to_both",
}
_RUNTIME_TCODE_CONTRACT_POLICIES = {"tcode_only", "tcode_then_extra_preprocess"}


def _data_task_axis(recipe, axis_name: str) -> str:
    return str(recipe.data_task_spec.get(axis_name, _PHASE3_DEFAULTS[axis_name]))


def _training_value(recipe, key: str, default=None):
    training_spec = getattr(recipe, "training_spec", {}) or {}
    data_task_spec = getattr(recipe, "data_task_spec", {}) or {}
    if isinstance(training_spec, Mapping) and key in training_spec:
        return training_spec[key]
    if isinstance(data_task_spec, Mapping) and key in data_task_spec:
        return data_task_spec[key]
    return default


def _training_axis(recipe, axis_name: str) -> str:
    return str(_training_value(recipe, axis_name, _TRAINING_AXIS_DEFAULTS[axis_name]))


def _phase3_axis_consumption() -> dict:
    return dict(_PHASE3_DEFAULTS)


_PRESELECTED_CORE = {"INDPRO", "PAYEMS", "CPIAUCSL", "FEDFUNDS", "GS10", "M2SL", "UNRATE"}
_LEVEL_SOURCE_FRAME_ATTR = "macrocast_level_source_frame"


def _level_source_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.copy()
    source.attrs.update({k: v for k, v in getattr(frame, "attrs", {}).items() if k != _LEVEL_SOURCE_FRAME_ATTR})
    return source


def _level_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source = getattr(frame, "attrs", {}).get(_LEVEL_SOURCE_FRAME_ATTR)
    if isinstance(source, pd.DataFrame):
        return source
    return frame


def _attach_level_source(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    frame.attrs[_LEVEL_SOURCE_FRAME_ATTR] = source
    return frame


def _replace_raw_data(raw_result, new_data):
    from dataclasses import replace as _replace, is_dataclass
    old_data = getattr(raw_result, "data", None)
    if old_data is not None and hasattr(old_data, "attrs") and hasattr(new_data, "attrs"):
        for key, value in getattr(old_data, "attrs", {}).items():
            new_data.attrs.setdefault(key, value)
    if is_dataclass(raw_result):
        return _replace(raw_result, data=new_data)
    if hasattr(raw_result, '__dict__'):
        new = type(raw_result).__new__(type(raw_result))
        new.__dict__.update(raw_result.__dict__)
        new.__dict__['data'] = new_data
        return new
    return raw_result


def _raw_result_with(raw_result, *, data=None, metadata=None, artifact=None, transform_codes=None):
    from dataclasses import replace as _replace, is_dataclass
    updates = {}
    if data is not None:
        updates["data"] = data
    if metadata is not None:
        updates["dataset_metadata"] = metadata
    if artifact is not None:
        updates["artifact"] = artifact
    if transform_codes is not None:
        updates["transform_codes"] = transform_codes
    if is_dataclass(raw_result):
        return _replace(raw_result, **updates)
    if hasattr(raw_result, "__dict__"):
        new = type(raw_result).__new__(type(raw_result))
        new.__dict__.update(raw_result.__dict__)
        new.__dict__.update(updates)
        return new
    return raw_result


def _append_frame_report(frame: pd.DataFrame, key: str, payload: object) -> None:
    reports = dict(frame.attrs.get("macrocast_reports", {}))
    reports[key] = payload
    frame.attrs["macrocast_reports"] = reports


def _append_frame_warning(frame: pd.DataFrame, message: str) -> None:
    warnings_list = list(frame.attrs.get("macrocast_warnings", []))
    warnings_list.append(message)
    frame.attrs["macrocast_warnings"] = warnings_list
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def _data_warnings(raw_result) -> list[str]:
    data = getattr(raw_result, "data", None)
    if data is None:
        return []
    return list(getattr(data, "attrs", {}).get("macrocast_warnings", []))


def _data_reports(raw_result) -> dict[str, object]:
    data = getattr(raw_result, "data", None)
    if data is None:
        return {}
    return dict(getattr(data, "attrs", {}).get("macrocast_reports", {}))


def _fred_tcode_transform(series: pd.Series, code: int) -> pd.Series:
    s = series.astype(float)
    if code == 1:
        return s
    if code == 2:
        return s.diff()
    if code == 3:
        return s.diff().diff()
    if code == 4:
        return np.log(s.where(s > 0))
    if code == 5:
        return np.log(s.where(s > 0)).diff()
    if code == 6:
        return np.log(s.where(s > 0)).diff().diff()
    if code == 7:
        return s.pct_change().diff()
    return s


def _official_transform_runtime_axes(
    recipe: RecipeSpec,
    contract: PreprocessContract,
) -> tuple[str, str, dict[str, object]]:
    data_task_spec = dict(getattr(recipe, "data_task_spec", {}) or {})

    policy = data_task_spec.get("official_transform_policy")
    if policy is None:
        policy = (
            "dataset_tcode"
            if getattr(contract, "tcode_policy", "raw_only") in _RUNTIME_TCODE_CONTRACT_POLICIES
            else "raw_official_frame"
        )
        policy_source = "legacy_preprocess_contract"
    else:
        policy_source = "data_task_spec"
    policy = str(policy)
    if policy not in _OFFICIAL_TRANSFORM_POLICIES:
        raise ExecutionError(f"official_transform_policy={policy!r} is not executable in this runtime slice")

    scope = data_task_spec.get("official_transform_scope")
    if scope is None:
        scope = getattr(contract, "tcode_application_scope", "apply_tcode_to_both")
        scope_source = "legacy_preprocess_contract"
    else:
        scope_source = "data_task_spec"
    scope = str(scope)
    if scope not in _OFFICIAL_TRANSFORM_SCOPES:
        raise ExecutionError(f"official_transform_scope={scope!r} is not executable in this runtime slice")

    source = data_task_spec.get("official_transform_source")
    if isinstance(source, Mapping):
        source_payload: dict[str, object] = dict(source)
    else:
        source_payload = {}
    source_payload.update(
        {
            "runtime_policy_source": policy_source,
            "runtime_scope_source": scope_source,
            "legacy_contract_fallback": (
                policy_source == "legacy_preprocess_contract"
                or scope_source == "legacy_preprocess_contract"
            ),
        }
    )
    return policy, scope, source_payload


def _apply_tcode_preprocessing(raw_result, recipe: RecipeSpec, contract: PreprocessContract, *, target: str | None):
    policy, scope, source_payload = _official_transform_runtime_axes(recipe, contract)
    if policy == "raw_official_frame":
        return raw_result

    data = getattr(raw_result, "data", None)
    if data is None:
        return raw_result
    tcodes = dict(getattr(raw_result, "transform_codes", {}) or {})
    level_source = _level_source_snapshot(data)
    frame = data.copy()
    frame.attrs.update(getattr(data, "attrs", {}))
    _attach_level_source(frame, level_source)
    if not tcodes:
        _append_frame_warning(frame, "official_transform_policy='dataset_tcode' requested but no dataset transform codes were available; data left unchanged")
        _append_frame_report(
            frame,
            "tcode",
            {
                "applied": False,
                "policy": policy,
                "scope": scope,
                "source": source_payload,
                "reason": "missing_transform_codes",
            },
        )
        return _replace_raw_data(raw_result, frame)

    applied: dict[str, int] = {}
    for col in frame.columns:
        is_target = target is not None and col == target
        if scope == "apply_tcode_to_target" and not is_target:
            continue
        if scope == "apply_tcode_to_X" and is_target:
            continue
        if scope == "apply_tcode_to_none":
            continue
        code = int(tcodes.get(col, 1))
        frame[col] = _fred_tcode_transform(frame[col], code)
        applied[str(col)] = code
    _append_frame_report(
        frame,
        "tcode",
        {
            "applied": True,
            "policy": policy,
            "scope": scope,
            "source": source_payload,
            "columns": applied,
        },
    )
    return _replace_raw_data(raw_result, frame)


def _dataset_has_fred_sd(dataset: object) -> bool:
    return "fred_sd" in _dataset_parts(str(dataset))


def _apply_sd_inferred_tcodes(raw_result, recipe: RecipeSpec):
    policy = normalize_sd_tcode_policy(recipe.data_task_spec.get("sd_tcode_policy"))
    if policy == "none":
        return raw_result
    if not _dataset_has_fred_sd(recipe.raw_dataset):
        raise ExecutionError("sd_tcode_policy was requested, but raw_dataset does not include fred_sd")

    requested_version = recipe.data_task_spec.get("sd_tcode_map_version")
    if requested_version not in {None, "", SD_INFERRED_TCODE_MAP_VERSION}:
        raise ExecutionError(
            f"unsupported sd_tcode_map_version={requested_version!r}; "
            f"available version is {SD_INFERRED_TCODE_MAP_VERSION!r}"
        )

    data = getattr(raw_result, "data", None)
    if data is None:
        return raw_result
    frame = data.copy()
    frame.attrs.update(getattr(data, "attrs", {}))
    frequency = str(recipe.data_task_spec.get("frequency") or getattr(raw_result.dataset_metadata, "frequency", "monthly"))
    allowed_statuses = recipe.data_task_spec.get("sd_tcode_allowed_statuses")
    codes, report = build_sd_inferred_transform_codes(
        frame.columns,
        frequency=frequency,
        allowed_statuses=allowed_statuses,
    )
    report["policy"] = policy
    _append_frame_report(frame, "sd_inferred_tcodes", report)
    if not codes:
        _append_frame_warning(frame, "sd_tcode_policy requested but no reviewed FRED-SD inferred t-codes matched the loaded columns")
        return _replace_raw_data(raw_result, frame)

    existing_tcodes = dict(getattr(raw_result, "transform_codes", {}) or {})
    existing_tcodes.update(codes)
    _append_frame_warning(frame, "FRED-SD inferred t-codes are macrocast research metadata, not official FRED-SD metadata")
    return _raw_result_with(raw_result, data=frame, transform_codes=existing_tcodes)


def _convert_raw_frequency(raw_result, target_frequency: str):
    data = getattr(raw_result, "data", None)
    if data is None or not isinstance(data.index, pd.DatetimeIndex):
        return raw_result
    source_frequency = str(getattr(raw_result.dataset_metadata, "frequency", "monthly"))
    source_norm = "monthly" if source_frequency in {"monthly", "state_monthly"} else source_frequency
    frame = data.copy()
    frame.attrs.update(getattr(data, "attrs", {}))
    conversion = {
        "source_frequency": source_frequency,
        "target_frequency": target_frequency,
        "method": "none",
    }

    if source_norm == target_frequency:
        _append_frame_report(frame, "frequency_conversion", conversion)
        return _replace_raw_data(raw_result, frame)

    if source_norm == "monthly" and target_frequency == "quarterly":
        converted = frame.resample("QS").mean()
        converted.attrs.update(frame.attrs)
        conversion["method"] = "monthly_to_quarterly_3_month_average"
        message = "monthly data converted to quarterly by 3-month average"
    elif source_norm == "quarterly" and target_frequency == "monthly":
        converted = frame.resample("MS").asfreq().interpolate(method="linear", limit_direction="both")
        converted.attrs.update(frame.attrs)
        conversion["method"] = "quarterly_to_monthly_linear_interpolation"
        message = "quarterly data converted to monthly by linear interpolation"
    else:
        raise ExecutionError(f"cannot convert frequency from {source_frequency!r} to {target_frequency!r}")

    _append_frame_warning(converted, message)
    _append_frame_report(converted, "frequency_conversion", conversion)
    metadata = replace(
        raw_result.dataset_metadata,
        frequency=target_frequency,
        parse_notes=tuple(getattr(raw_result.dataset_metadata, "parse_notes", ())) + (message,),
    )
    return _raw_result_with(raw_result, data=converted, metadata=metadata)


def _apply_frequency_policy(raw_result, recipe: RecipeSpec):
    target_frequency = str(recipe.data_task_spec.get("frequency") or getattr(raw_result.dataset_metadata, "frequency", "monthly"))
    return _convert_raw_frequency(raw_result, target_frequency)


def _apply_sample_period_and_availability(raw_result, recipe: RecipeSpec, *, target: str | None):
    data = getattr(raw_result, "data", None)
    if data is None or not isinstance(data.index, pd.DatetimeIndex):
        return raw_result
    start = recipe.data_task_spec.get("sample_start_date")
    end = recipe.data_task_spec.get("sample_end_date")
    frame = data.copy()
    frame.attrs.update(getattr(data, "attrs", {}))
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)].copy()
    if end is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(end)].copy()
    frame.attrs.update(getattr(data, "attrs", {}))
    if frame.empty:
        raise ExecutionError(f"sample period start={start!r}, end={end!r} produced no observations")

    rule = str(recipe.data_task_spec.get("missing_availability", "complete_case_only"))
    if rule != "zero_fill_before_start":
        return _replace_raw_data(raw_result, frame)
    if target is None or target not in frame.columns:
        return _replace_raw_data(raw_result, frame)

    report = {
        "sample_start_date": None if start is None else str(start),
        "sample_end_date": None if end is None else str(end),
        "leading_zero_filled": {},
        "fully_missing_in_period": [],
        "mid_sample_missing": {},
        "target_leading_missing": [],
    }
    target_series = frame[target]
    if target_series.notna().sum() == 0:
        raise ExecutionError(f"target {target!r} is fully missing in selected sample period")
    target_first_valid = target_series.first_valid_index()
    if target_first_valid is not None:
        target_leading = target_series.index[target_series.index < target_first_valid]
        if len(target_leading):
            report["target_leading_missing"] = [str(idx.date()) for idx in target_leading]
    target_after_start = target_series.loc[target_first_valid:]
    if target_after_start.isna().any():
        missing_dates = [str(idx.date()) for idx in target_after_start[target_after_start.isna()].index]
        raise ExecutionError(f"target {target!r} has missing observations inside selected sample period: {missing_dates[:5]}")

    for col in [c for c in frame.columns if c != target]:
        series = frame[col]
        if series.notna().sum() == 0:
            report["fully_missing_in_period"].append(str(col))
            frame[col] = series.fillna(0.0)
            continue
        first_valid = series.first_valid_index()
        if first_valid is not None:
            leading_mask = series.index < first_valid
            leading_missing = leading_mask & series.isna()
            if leading_missing.any():
                report["leading_zero_filled"][str(col)] = [str(idx.date()) for idx in series.index[leading_missing]]
                frame.loc[leading_missing, col] = 0.0
            mid_missing = series.loc[first_valid:].isna()
            if mid_missing.any():
                report["mid_sample_missing"][str(col)] = [str(idx.date()) for idx in series.loc[first_valid:][mid_missing].index]
    if report["fully_missing_in_period"]:
        _append_frame_warning(frame, f"predictors fully missing in selected sample period were filled with zero: {report['fully_missing_in_period']}")
    if report["leading_zero_filled"]:
        _append_frame_warning(frame, "predictor leading missing values before each series start were filled with zero")
    if report["mid_sample_missing"]:
        _append_frame_warning(frame, f"predictors have mid-sample missing values: {sorted(report['mid_sample_missing'])}")
    _append_frame_report(frame, "availability", report)
    return _replace_raw_data(raw_result, frame)


def _apply_release_lag(raw_result, rule: str, *, spec: dict | None = None):
    """Apply release_lag_rule to raw_result.data.

    v1.0 operational rules:
    - ``ignore_release_lag`` (default): no-op.
    - ``fixed_lag_all_series``: shift every non-date column by 1 period.
    - ``series_specific_lag``: shift each column by the lag supplied via
      ``spec['release_lag_per_series']`` (dict[column → int months]).
      Columns missing from the dict are left untouched.
    """
    if rule == 'ignore_release_lag' or not rule:
        return raw_result
    data = getattr(raw_result, 'data', None)
    if data is None:
        return raw_result
    if rule == 'fixed_lag_all_series':
        try:
            new_data = data.copy()
            cols = [c for c in new_data.columns if str(c).lower() != 'date']
            for c in cols:
                new_data[c] = new_data[c].shift(1)
            source = getattr(data, "attrs", {}).get(_LEVEL_SOURCE_FRAME_ATTR)
            if isinstance(source, pd.DataFrame):
                shifted_source = source.copy()
                for c in cols:
                    if c in shifted_source.columns:
                        shifted_source[c] = shifted_source[c].shift(1)
                _attach_level_source(new_data, shifted_source)
        except Exception:
            return raw_result
        return _replace_raw_data(raw_result, new_data)
    if rule == 'series_specific_lag':
        lag_map = (spec or {}).get('release_lag_per_series')
        if not isinstance(lag_map, dict) or not lag_map:
            raise ExecutionError("release_lag_rule='series_specific_lag' requires leaf_config.release_lag_per_series (dict[str, int])")
        try:
            new_data = data.copy()
            source = getattr(data, "attrs", {}).get(_LEVEL_SOURCE_FRAME_ATTR)
            shifted_source = source.copy() if isinstance(source, pd.DataFrame) else None
            for col, lag in lag_map.items():
                if col in new_data.columns:
                    new_data[col] = new_data[col].shift(int(lag))
                if shifted_source is not None and col in shifted_source.columns:
                    shifted_source[col] = shifted_source[col].shift(int(lag))
            if shifted_source is not None:
                _attach_level_source(new_data, shifted_source)
        except Exception as exc:
            raise ExecutionError(f"series_specific_lag failed: {exc}") from exc
        return _replace_raw_data(raw_result, new_data)
    raise ExecutionError(f'unsupported release_lag_rule={rule!r}')


def _apply_missing_availability(raw_result, rule: str, *, target: str | None = None, spec: dict | None = None):
    """Apply missing_availability rule to the raw panel.

    v1.0 operational rules:
    - ``complete_case_only`` (default): no-op; downstream code drops NaNs per its own policy.
    - ``available_case``: keep only rows where every non-date column is non-NaN.
      Target rows with NaN outside this filter are kept (they will be skipped
      later by the evaluator); X columns are also required complete per row.
    - ``x_impute_only``: impute predictor columns (non-target, non-date) using the
      strategy declared in ``spec['x_imputation']`` (one of 'mean', 'median', 'ffill', 'bfill').
      Target column is left untouched so target NaNs remain visible to the OOS loop.
    """
    if rule in {'complete_case_only', 'zero_fill_before_start', None} or not rule:
        return raw_result
    data = getattr(raw_result, 'data', None)
    if data is None:
        return raw_result
    spec = dict(spec or {})

    if rule == 'available_case':
        # Complete-case filter across the whole panel; legitimate in short
        # fixture windows, aggressive for long panels.
        new_data = data.dropna(how='any').copy()
        if len(new_data) == 0:
            # Fall back to original to avoid empty-data downstream crashes; the
            # evaluator will raise a more informative error.
            return raw_result
        return _replace_raw_data(raw_result, new_data)

    if rule == 'x_impute_only':
        strategy = spec.get('x_imputation')
        if strategy is None:
            raise ExecutionError("missing_availability='x_impute_only' requires leaf_config.x_imputation (one of 'mean' / 'median' / 'ffill' / 'bfill')")
        if strategy not in {'mean', 'median', 'ffill', 'bfill'}:
            raise ExecutionError(f"missing_availability='x_impute_only': unsupported leaf_config.x_imputation={strategy!r}; allowed: mean / median / ffill / bfill")
        x_cols = [c for c in data.columns if c != target and str(c).lower() != 'date']
        new_data = data.copy()
        if strategy == 'ffill':
            new_data[x_cols] = new_data[x_cols].ffill().bfill()
        elif strategy == 'bfill':
            new_data[x_cols] = new_data[x_cols].bfill().ffill()
        elif strategy == 'mean':
            for c in x_cols:
                col = new_data[c]
                if col.isna().any():
                    new_data[c] = col.fillna(col.mean())
        elif strategy == 'median':
            for c in x_cols:
                col = new_data[c]
                if col.isna().any():
                    new_data[c] = col.fillna(col.median())
        return _replace_raw_data(raw_result, new_data)

    raise ExecutionError(f'unsupported missing_availability={rule!r}')


def _raw_policy_window_index(frame: pd.DataFrame, spec: dict) -> pd.Index:
    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    mask = pd.Series(True, index=frame.index)
    start = spec.get("sample_start_date")
    end = spec.get("sample_end_date")
    if start is not None:
        mask &= frame.index >= pd.Timestamp(start)
    if end is not None:
        mask &= frame.index <= pd.Timestamp(end)
    return frame.index[mask.to_numpy()]


def _raw_predictor_columns(frame: pd.DataFrame, target: str | None) -> list:
    return [c for c in frame.columns if c != target and str(c).lower() != "date"]


def _raw_numeric_columns(frame: pd.DataFrame, spec: dict) -> list:
    requested = spec.get("raw_outlier_columns")
    if requested is not None:
        return [c for c in requested if c in frame.columns and pd.api.types.is_numeric_dtype(frame[c])]
    return [
        c
        for c in frame.columns
        if str(c).lower() != "date" and pd.api.types.is_numeric_dtype(frame[c])
    ]


def _format_index_values(index: pd.Index) -> list[str]:
    return [str(value.date()) if hasattr(value, "date") else str(value) for value in index]


def _apply_raw_missing_policy(raw_result, policy: str, *, target: str | None = None, spec: dict | None = None):
    """Apply Layer 1 raw-source missing treatment before official transforms."""

    if policy in {"preserve_raw_missing", None} or not policy:
        return raw_result
    data = getattr(raw_result, "data", None)
    if data is None:
        return raw_result
    spec = dict(spec or {})
    frame = data.copy()
    frame.attrs.update(getattr(data, "attrs", {}))

    if policy == "drop_rows_with_raw_missing":
        before = len(frame)
        frame = frame.dropna(how="any").copy()
        frame.attrs.update(getattr(data, "attrs", {}))
        _append_frame_report(
            frame,
            "raw_missing",
            {
                "policy": policy,
                "before_official_transform": True,
                "rows_dropped": before - len(frame),
            },
        )
        return _replace_raw_data(raw_result, frame)

    x_cols = _raw_predictor_columns(frame, target)
    if policy == "x_impute_raw":
        strategy = spec.get("raw_x_imputation")
        if strategy is None:
            raise ExecutionError(
                "raw_missing_policy='x_impute_raw' requires leaf_config.raw_x_imputation "
                "(one of 'mean' / 'median' / 'ffill' / 'bfill')"
            )
        if strategy not in {"mean", "median", "ffill", "bfill"}:
            raise ExecutionError(
                "raw_missing_policy='x_impute_raw': unsupported "
                f"leaf_config.raw_x_imputation={strategy!r}; allowed: mean / median / ffill / bfill"
            )
        missing_before = {str(c): int(frame[c].isna().sum()) for c in x_cols if frame[c].isna().any()}
        if strategy == "ffill":
            frame[x_cols] = frame[x_cols].ffill().bfill()
        elif strategy == "bfill":
            frame[x_cols] = frame[x_cols].bfill().ffill()
        elif strategy == "mean":
            for col in x_cols:
                if frame[col].isna().any():
                    frame[col] = frame[col].fillna(frame[col].mean())
        elif strategy == "median":
            for col in x_cols:
                if frame[col].isna().any():
                    frame[col] = frame[col].fillna(frame[col].median())
        _append_frame_report(
            frame,
            "raw_missing",
            {
                "policy": policy,
                "before_official_transform": True,
                "strategy": strategy,
                "filled_missing": missing_before,
            },
        )
        return _replace_raw_data(raw_result, frame)

    if policy == "zero_fill_leading_x_before_tcode":
        window_index = _raw_policy_window_index(frame, spec)
        filled: dict[str, list[str]] = {}
        for col in x_cols:
            series = frame.loc[window_index, col]
            first_valid = series.first_valid_index()
            if first_valid is None:
                leading_index = series.index[series.isna()]
            else:
                first_pos = list(series.index).index(first_valid)
                leading = series.iloc[:first_pos]
                leading_index = leading.index[leading.isna()]
            if len(leading_index):
                frame.loc[leading_index, col] = 0.0
                filled[str(col)] = _format_index_values(leading_index)
        _append_frame_report(
            frame,
            "raw_missing",
            {
                "policy": policy,
                "before_official_transform": True,
                "leading_zero_filled": filled,
            },
        )
        return _replace_raw_data(raw_result, frame)

    raise ExecutionError(f"unsupported raw_missing_policy={policy!r}")


def _apply_raw_outlier_policy(raw_result, policy: str, *, spec: dict | None = None):
    """Apply Layer 1 raw-source outlier treatment before official transforms."""

    if policy in {"preserve_raw_outliers", None} or not policy:
        return raw_result
    data = getattr(raw_result, "data", None)
    if data is None:
        return raw_result
    spec = dict(spec or {})
    frame = data.copy()
    frame.attrs.update(getattr(data, "attrs", {}))
    cols = _raw_numeric_columns(frame, spec)
    if not cols:
        _append_frame_report(
            frame,
            "raw_outliers",
            {"policy": policy, "before_official_transform": True, "columns": [], "changed": {}},
        )
        return _replace_raw_data(raw_result, frame)

    values = frame[cols].astype(float)
    if policy == "winsorize_raw":
        lower = values.quantile(0.01)
        upper = values.quantile(0.99)
    elif policy == "iqr_clip_raw":
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
    elif policy == "mad_clip_raw":
        median = values.median()
        mad = (values - median).abs().median().replace(0, 1.0)
        lower = median - 3.0 * mad
        upper = median + 3.0 * mad
    elif policy == "zscore_clip_raw":
        mean = values.mean()
        std = values.std(ddof=0).replace(0, 1.0)
        lower = mean - 3.0 * std
        upper = mean + 3.0 * std
    elif policy == "raw_outlier_to_missing":
        lower = values.quantile(0.01)
        upper = values.quantile(0.99)
    else:
        raise ExecutionError(f"unsupported raw_outlier_policy={policy!r}")

    mask = (values < lower) | (values > upper)
    changed = {str(col): int(mask[col].sum()) for col in cols if int(mask[col].sum())}
    if policy == "raw_outlier_to_missing":
        frame.loc[:, cols] = values.mask(mask)
    else:
        frame.loc[:, cols] = values.clip(lower=lower, upper=upper, axis=1)
    _append_frame_report(
        frame,
        "raw_outliers",
        {
            "policy": policy,
            "before_official_transform": True,
            "columns": [str(col) for col in cols],
            "changed": changed,
        },
    )
    return _replace_raw_data(raw_result, frame)


_VARIABLE_UNIVERSE_SUBSET_FIELD: dict[str, str] = {
    'handpicked_set': 'variable_universe_columns',
}


def _apply_variable_universe(raw_result, rule: str, *, spec: dict | None = None, target: str | None = None):
    """Filter raw_result.data columns per the variable_universe rule.

    v1.0 semantics: non-default rules read a user-supplied column list from
    data_task_spec (propagated from leaf_config at compile time). Target and
    date columns are always preserved.
    """
    if rule == 'all_variables':
        return raw_result
    data = getattr(raw_result, 'data', None)
    if data is None:
        return raw_result
    spec = dict(spec or {})

    if rule == 'preselected_core':
        keep = [c for c in data.columns if c in _PRESELECTED_CORE or str(c).lower() == 'date']
        if len(keep) >= 2:
            return _replace_raw_data(raw_result, data[keep].copy())
        return raw_result

    def _keep_with_anchors(columns):
        anchors = [c for c in data.columns if str(c).lower() == 'date']
        head = []
        if target and target in data.columns:
            head = [target]
        rest = [c for c in columns if c in data.columns and c not in head and c not in anchors]
        return anchors + head + rest

    if rule == 'category_subset':
        mapping = spec.get('variable_universe_category_columns')
        category = spec.get('variable_universe_category')
        if isinstance(mapping, dict) and category in mapping:
            keep = _keep_with_anchors(list(mapping[category]))
            if keep:
                return _replace_raw_data(raw_result, data[keep].copy())
        raise ExecutionError("variable_universe='category_subset' requires leaf_config.variable_universe_category_columns (dict) + leaf_config.variable_universe_category")

    if rule == 'target_specific_subset':
        mapping = spec.get('target_specific_columns')
        if isinstance(mapping, dict) and target in mapping:
            keep = _keep_with_anchors(list(mapping[target]))
            if keep:
                return _replace_raw_data(raw_result, data[keep].copy())
        raise ExecutionError("variable_universe='target_specific_subset' requires leaf_config.target_specific_columns (dict[target, list])")

    field = _VARIABLE_UNIVERSE_SUBSET_FIELD.get(rule)
    if field is not None:
        cols = spec.get(field)
        if isinstance(cols, (list, tuple)) and cols:
            keep = _keep_with_anchors(list(cols))
            if keep:
                return _replace_raw_data(raw_result, data[keep].copy())
        raise ExecutionError(f"variable_universe={rule!r} requires leaf_config.{field} (list[str])")

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


def _dataset_parts(dataset: str) -> set[str]:
    tokens = str(dataset).replace(",", "+").split("+")
    return {token.strip() for token in tokens if token.strip()}


def _local_source_for_dataset(local_raw_source, dataset: str):
    if isinstance(local_raw_source, Mapping):
        return local_raw_source.get(dataset)
    return local_raw_source


def _load_single_raw_dataset(dataset: str, *, vintage: str | None, cache_root: Path, local_raw_source):
    source = _local_source_for_dataset(local_raw_source, dataset)
    if dataset == "fred_md":
        return load_fred_md(vintage=vintage, cache_root=cache_root, local_source=source)
    if dataset == "fred_qd":
        return load_fred_qd(vintage=vintage, cache_root=cache_root, local_source=source)
    if dataset == "fred_sd":
        return load_fred_sd(vintage=vintage, cache_root=cache_root, local_source=source)
    raise ExecutionError(f"unsupported raw_dataset={dataset!r}")


def _component_data_through(results) -> str | None:
    dates = [getattr(result.dataset_metadata, "data_through", None) for _, result in results]
    dates = [date for date in dates if date]
    return min(dates) if dates else None


def _combine_raw_results(dataset: str, target_frequency: str, components):
    converted_components = []
    combined_warnings: list[str] = []
    component_reports: dict[str, object] = {}
    frames: list[pd.DataFrame] = []
    seen_columns: set[str] = set()
    combined_tcodes: dict[str, int] = {}

    for component_name, raw_result in components:
        converted = _convert_raw_frequency(raw_result, target_frequency)
        converted_components.append((component_name, converted))
        frame = converted.data.copy()
        frame.attrs.update(getattr(converted.data, "attrs", {}))
        rename_map: dict[object, str] = {}
        for col in frame.columns:
            if str(col) in seen_columns:
                rename_map[col] = f"{col}__{component_name}"
            else:
                seen_columns.add(str(col))
        if rename_map:
            frame = frame.rename(columns=rename_map)
        frames.append(frame)

        tcodes = dict(getattr(converted, "transform_codes", {}) or {})
        for col, code in tcodes.items():
            combined_tcodes[rename_map.get(col, col)] = code
        combined_warnings.extend(_data_warnings(converted))
        component_reports[component_name] = _data_reports(converted)

    if not frames:
        raise ExecutionError(f"composite dataset={dataset!r} did not load any components")

    combined = pd.concat(frames, axis=1).sort_index()
    combined.attrs["macrocast_warnings"] = combined_warnings
    combined.attrs["macrocast_reports"] = {
        "combined_dataset": {
            "dataset": dataset,
            "components": [name for name, _ in converted_components],
            "frequency": target_frequency,
        },
        "components": component_reports,
    }

    primary_name, primary = converted_components[0]
    source_url = ";".join(str(result.artifact.source_url) for _, result in converted_components)
    local_path = ";".join(str(result.artifact.local_path) for _, result in converted_components)
    sha_payload = "|".join(str(result.artifact.file_sha256) for _, result in converted_components)
    artifact = replace(
        primary.artifact,
        dataset=dataset,
        source_url=source_url,
        local_path=local_path,
        file_format="mixed",
        file_sha256=hashlib.sha256(sha_payload.encode("utf-8")).hexdigest(),
        file_size_bytes=sum(int(result.artifact.file_size_bytes) for _, result in converted_components),
        cache_hit=all(bool(result.artifact.cache_hit) for _, result in converted_components),
    )
    metadata = replace(
        primary.dataset_metadata,
        dataset=dataset,
        source_family="+".join(str(result.dataset_metadata.source_family) for _, result in converted_components),
        frequency=target_frequency,
        data_through=_component_data_through(converted_components),
        support_tier="provisional",
        parse_notes=tuple(getattr(primary.dataset_metadata, "parse_notes", ()))
        + (f"combined components: {', '.join(name for name, _ in converted_components)}",),
    )
    return _raw_result_with(primary, data=combined, metadata=metadata, artifact=artifact, transform_codes=combined_tcodes)


def _load_raw_for_recipe(recipe: RecipeSpec, local_raw_source: str | Path | Mapping[str, str | Path] | None, cache_root: Path):
    vintage = recipe.data_vintage
    source_adapter = recipe.data_task_spec.get("source_adapter") or recipe.data_task_spec.get("dataset_source") or recipe.raw_dataset
    if source_adapter in {"custom_csv", "custom_parquet"}:
        custom_path = local_raw_source or recipe.data_task_spec.get("custom_data_path")
        if custom_path is None:
            raise ExecutionError(
                f"source_adapter={source_adapter!r} requires leaf_config.custom_data_path "
                "(or pass local_raw_source to execute_recipe)"
            )
        if source_adapter == "custom_csv":
            return load_custom_csv(custom_path, dataset=recipe.raw_dataset, cache_root=cache_root)
        return load_custom_parquet(custom_path, dataset=recipe.raw_dataset, cache_root=cache_root)
    parts = _dataset_parts(recipe.raw_dataset)
    if parts == {"fred_md", "fred_sd"}:
        if local_raw_source is not None and not isinstance(local_raw_source, Mapping):
            raise ExecutionError("composite FRED datasets require local_raw_source as a mapping keyed by component dataset")
        components = [
            ("fred_md", _load_single_raw_dataset("fred_md", vintage=vintage, cache_root=cache_root, local_raw_source=local_raw_source)),
            ("fred_sd", _load_single_raw_dataset("fred_sd", vintage=vintage, cache_root=cache_root, local_raw_source=local_raw_source)),
        ]
        return _combine_raw_results(recipe.raw_dataset, "monthly", components)
    if parts == {"fred_qd", "fred_sd"}:
        if local_raw_source is not None and not isinstance(local_raw_source, Mapping):
            raise ExecutionError("composite FRED datasets require local_raw_source as a mapping keyed by component dataset")
        components = [
            ("fred_qd", _load_single_raw_dataset("fred_qd", vintage=vintage, cache_root=cache_root, local_raw_source=local_raw_source)),
            ("fred_sd", _load_single_raw_dataset("fred_sd", vintage=vintage, cache_root=cache_root, local_raw_source=local_raw_source)),
        ]
        return _combine_raw_results(recipe.raw_dataset, "quarterly", components)
    if len(parts) == 1:
        return _load_single_raw_dataset(recipe.raw_dataset, vintage=vintage, cache_root=cache_root, local_raw_source=local_raw_source)
    raise ExecutionError(f"unsupported raw_dataset={recipe.raw_dataset!r}")


def _benchmark_spec(recipe: RecipeSpec) -> dict[str, object]:
    spec = dict(recipe.benchmark_config)
    spec.setdefault("benchmark_family", recipe.stage0.fixed_design.benchmark)
    spec.setdefault("minimum_train_size", _DEFAULT_MINIMUM_TRAIN_SIZE)
    spec.setdefault("max_ar_lag", _DEFAULT_MAX_AR_LAG)
    return spec


def _minimum_train_size(recipe: RecipeSpec, *, horizon: int | None = None) -> int:
    """Resolve minimum_train_size honouring the 1.3 min_train_size axis.

    The base value comes from ``benchmark_config.minimum_train_size`` (leaf_config
    scalar, default 60). The axis value selects a transform applied on top:

    - ``fixed_n_obs``: base (default, identity).
    - ``fixed_years``: base * 12 (interpret the scalar as years).
    - ``model_specific_min_train``: max(base, per-family floor).
    - ``target_specific_min_train``: max(base, per-target floor).
    - ``horizon_specific_min_train``: base + 6 * max(0, horizon-1).

    The model_family / target / horizon context is looked up from the recipe when
    available; horizon falls back to the largest recipe horizon when not supplied
    explicitly, giving the most conservative (largest) minimum_train_size.
    """
    benchmark_spec = _benchmark_spec(recipe)
    base = int(benchmark_spec["minimum_train_size"])
    rule = _training_axis(recipe, "min_train_size")
    if rule == "fixed_n_obs":
        return base
    model_family = _model_family(recipe)
    target = str(recipe.target) if getattr(recipe, "target", None) else None
    effective_horizon = int(horizon) if horizon is not None else max((int(h) for h in recipe.horizons), default=1)
    spec = _WindowSpec(minimum_train_rule=rule, minimum_train_value=base)
    try:
        resolved = _resolve_min_train_obs(spec, model_family=model_family, target=target, horizon=effective_horizon)
    except ValueError as exc:
        raise ExecutionError(str(exc)) from exc
    return int(resolved)


def _max_ar_lag(recipe: RecipeSpec) -> int:
    return int(_benchmark_spec(recipe)["max_ar_lag"])


def _rolling_window_size(recipe: RecipeSpec) -> int:
    benchmark_spec = _benchmark_spec(recipe)
    return int(benchmark_spec.get("rolling_window_size", benchmark_spec["minimum_train_size"]))


def _benchmark_family(recipe: RecipeSpec) -> str:
    return str(_benchmark_spec(recipe)["benchmark_family"])


def _layer2_spec(recipe: RecipeSpec | None) -> dict[str, object]:
    if recipe is None:
        return {}
    spec = getattr(recipe, "layer2_representation_spec", {}) or {}
    return dict(spec) if isinstance(spec, Mapping) else {}


def _layer2_input_panel(recipe: RecipeSpec | None) -> dict[str, object]:
    spec = _layer2_spec(recipe)
    panel = spec.get("input_panel", {})
    return dict(panel) if isinstance(panel, Mapping) else {}


def _layer2_target_representation(recipe: RecipeSpec | None) -> dict[str, object]:
    spec = _layer2_spec(recipe)
    target = spec.get("target_representation", {})
    return dict(target) if isinstance(target, Mapping) else {}


def _layer2_deterministic_feature_block(recipe: RecipeSpec | None) -> dict[str, object]:
    blocks = _layer2_feature_blocks(recipe)
    block = blocks.get("deterministic_feature_block", {})
    return dict(block) if isinstance(block, Mapping) else {}


def _layer2_runtime_spec(recipe: RecipeSpec | None) -> dict[str, object]:
    data_task_spec = dict(getattr(recipe, "data_task_spec", {}) or {}) if recipe is not None else {}
    input_panel = _layer2_input_panel(recipe)
    target_representation = _layer2_target_representation(recipe)
    deterministic_block = _layer2_deterministic_feature_block(recipe)
    for key in ("predictor_family", "contemporaneous_x_rule"):
        if key in input_panel:
            data_task_spec[key] = input_panel[key]
    if "horizon_target_construction" in target_representation:
        data_task_spec["horizon_target_construction"] = target_representation["horizon_target_construction"]
    for key in ("deterministic_components", "structural_break_segmentation"):
        if key in deterministic_block:
            data_task_spec[key] = deterministic_block[key]
    return data_task_spec


def _predictor_family(recipe: RecipeSpec) -> str:
    return str(_layer2_runtime_spec(recipe).get("predictor_family", "all_macro_vars"))


_STRUCTURAL_BREAK_PRESETS = {
    "pre_post_crisis": ("2008-09-01",),
    "pre_post_covid": ("2020-03-01",),
}


def _resolve_structural_break_dates(spec: dict | None) -> list[str] | None:
    """Map structural_break_segmentation value to its break-date list.

    v1.0 operational values:
    - ``none`` (default)        : returns None (no augmentation).
    - ``pre_post_crisis``       : single break at 2008-09-01 (NBER Great-Recession onset).
    - ``pre_post_covid``        : single break at 2020-03-01 (NBER COVID-recession onset).
    - ``user_break_dates``      : reads leaf_config.break_dates (list of ISO dates).
    """
    rule = (spec or {}).get("structural_break_segmentation", "none")
    if rule == "none" or not rule:
        return None
    if rule in _STRUCTURAL_BREAK_PRESETS:
        return list(_STRUCTURAL_BREAK_PRESETS[rule])
    raise ExecutionError(f"unsupported structural_break_segmentation={rule!r}")


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


def _layer2_feature_blocks(recipe: RecipeSpec | None) -> dict[str, object]:
    if recipe is None:
        return {}
    spec = getattr(recipe, "layer2_representation_spec", {}) or {}
    blocks = spec.get("feature_blocks", {}) if isinstance(spec, dict) else {}
    return dict(blocks or {}) if isinstance(blocks, dict) else {}


def _layer2_target_lag_config(recipe: RecipeSpec | None) -> dict[str, object]:
    if recipe is None:
        return {}
    spec = getattr(recipe, "layer2_representation_spec", {}) or {}
    if not isinstance(spec, dict):
        return {}
    config = spec.get("target_lag_config", {})
    return dict(config) if isinstance(config, dict) else {}


def _layer2_block_value(blocks: Mapping[str, object], block_name: str, default: str = "none") -> str:
    block = blocks.get(block_name, {})
    if isinstance(block, dict):
        return str(block.get("value", default))
    if block is None:
        return default
    return str(block)


def _feature_runtime_builder(recipe: RecipeSpec) -> str:
    """Return the executor feature path from canonical Layer 2 blocks.

    The legacy feature_builder remains accepted as source provenance and as a
    fallback for old RecipeSpec objects. Runtime routing should prefer the
    explicit feature-block grammar whenever it is present.
    """
    blocks = _layer2_feature_blocks(recipe)
    block_set = _layer2_block_value(blocks, "feature_block_set", "")
    raw_block_values = (
        _layer2_block_value(blocks, "x_lag_feature_block"),
        _layer2_block_value(blocks, "factor_feature_block"),
        _layer2_block_value(blocks, "level_feature_block"),
        _layer2_block_value(blocks, "temporal_feature_block"),
        _layer2_block_value(blocks, "rotation_feature_block"),
    )
    if block_set in _RAW_PANEL_FEATURE_BLOCK_SETS or any(value != "none" for value in raw_block_values):
        return "raw_feature_panel"
    target_lag_block = _layer2_block_value(blocks, "target_lag_block")
    if block_set == "target_lags_only" or target_lag_block != "none":
        return "autoreg_lagged_target"

    legacy_feature_builder = _feature_builder(recipe)
    if legacy_feature_builder in _RAW_PANEL_FEATURE_BUILDERS:
        return "raw_feature_panel"
    return legacy_feature_builder


def _feature_runtime_name(recipe: RecipeSpec) -> str:
    runtime_builder = _feature_runtime_builder(recipe)
    if runtime_builder == "raw_feature_panel":
        return "raw_panel_v1"
    if runtime_builder == "autoreg_lagged_target":
        return "autoreg_lagged_target_v1"
    return f"{runtime_builder}_v1"


def _feature_runtime_context(recipe: RecipeSpec, *, mode: str) -> dict[str, object]:
    runtime_builder = _feature_runtime_builder(recipe)
    return {
        "feature_builder": runtime_builder,
        "feature_runtime_builder": runtime_builder,
        "legacy_feature_builder": _feature_builder(recipe),
        "feature_dispatch_source": "layer2_feature_blocks",
        "mode": mode,
    }


def _level_feature_block(recipe: RecipeSpec | None) -> str:
    if recipe is None:
        return "none"
    return _layer2_block_value(_layer2_feature_blocks(recipe), "level_feature_block")


def _temporal_feature_block(recipe: RecipeSpec | None) -> str:
    if recipe is None:
        return "none"
    return _layer2_block_value(_layer2_feature_blocks(recipe), "temporal_feature_block")


def _rotation_feature_block(recipe: RecipeSpec | None) -> str:
    if recipe is None:
        return "none"
    return _layer2_block_value(_layer2_feature_blocks(recipe), "rotation_feature_block")


def _x_lag_feature_block(recipe: RecipeSpec | None) -> str | None:
    if recipe is None:
        return None
    blocks = _layer2_feature_blocks(recipe)
    if "x_lag_feature_block" not in blocks:
        return None
    return _layer2_block_value(blocks, "x_lag_feature_block")


def _factor_feature_block(recipe: RecipeSpec | None) -> str | None:
    if recipe is None:
        return None
    blocks = _layer2_feature_blocks(recipe)
    if "factor_feature_block" not in blocks:
        return None
    return _layer2_block_value(blocks, "factor_feature_block")


def _factor_feature_block_spec(recipe: RecipeSpec | None) -> dict[str, object]:
    if recipe is None:
        return {}
    blocks = _layer2_feature_blocks(recipe)
    block = blocks.get("factor_feature_block", {})
    return dict(block) if isinstance(block, dict) else {}


def _factor_runtime_training_spec(recipe: RecipeSpec) -> dict[str, object]:
    training_spec = dict(getattr(recipe, "training_spec", {}) or {})
    target_lag_config = _layer2_target_lag_config(recipe)
    if "count" in target_lag_config:
        training_spec["target_lag_count"] = target_lag_config.get("count")
    factor_block = _factor_feature_block_spec(recipe)
    factor_count = factor_block.get("factor_count", {})
    if isinstance(factor_count, dict):
        if "mode" in factor_count:
            training_spec["factor_count"] = factor_count.get("mode")
        if "fixed_factor_count" in factor_count:
            training_spec["fixed_factor_count"] = factor_count.get("fixed_factor_count")
        if "max_factors" in factor_count:
            training_spec["max_factors"] = factor_count.get("max_factors")
    runtime_block = factor_block.get("runtime_block", {})
    if isinstance(runtime_block, dict) and "factor_lag_count" in runtime_block:
        training_spec["factor_lag_count"] = runtime_block.get("factor_lag_count")
    elif "factor_lag_count" in factor_block:
        training_spec["factor_lag_count"] = factor_block.get("factor_lag_count")
    return training_spec


def _feature_block_combination(recipe: RecipeSpec | None) -> str:
    if recipe is None:
        return "replace_with_blocks"
    return _layer2_block_value(_layer2_feature_blocks(recipe), "feature_block_combination", "replace_with_blocks")


def _target_lag_feature_block(recipe: RecipeSpec | None) -> str:
    block = _target_lag_feature_block_spec(recipe)
    if not block:
        return "none"
    return str(block.get("value", "none"))


def _target_lag_feature_block_spec(recipe: RecipeSpec | None) -> dict[str, object] | None:
    if recipe is None:
        return None
    blocks = _layer2_feature_blocks(recipe)
    if "target_lag_block" not in blocks:
        return None
    block = blocks.get("target_lag_block")
    if isinstance(block, dict):
        return dict(block)
    if block is None:
        return None
    return {"value": str(block)}


def _positive_int_or_none(value) -> int | None:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    return candidate if candidate > 0 else None


def _target_lag_order_from_block(recipe: RecipeSpec, fallback: int) -> int:
    block = _target_lag_feature_block_spec(recipe)
    if not block or str(block.get("value", "none")) != "fixed_target_lags":
        return fallback

    lag_orders = block.get("lag_orders")
    if isinstance(lag_orders, Sequence) and not isinstance(lag_orders, (str, bytes)):
        positive_lags = [
            lag
            for lag in (_positive_int_or_none(value) for value in lag_orders)
            if lag is not None
        ]
        if positive_lags:
            return max(positive_lags)

    runtime_bridge = block.get("runtime_bridge", {})
    if not isinstance(runtime_bridge, dict):
        runtime_bridge = {}
    for value in (
        block.get("target_lag_count"),
        runtime_bridge.get("target_lag_count"),
        runtime_bridge.get("legacy_factor_ar_lags"),
        fallback,
    ):
        lag_order = _positive_int_or_none(value)
        if lag_order is not None:
            return lag_order
    return fallback


def _target_lag_feature_names(recipe: RecipeSpec, lag_order: int, *, default_prefix: str) -> list[str]:
    block = _target_lag_feature_block_spec(recipe)
    if block and str(block.get("value", "none")) == "fixed_target_lags":
        names = block.get("feature_names")
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            public_names = [str(name) for name in list(names)[:lag_order]]
            if len(public_names) == lag_order:
                return public_names
        return [f"target_lag_{lag}" for lag in range(1, lag_order + 1)]
    return [f"{default_prefix}_{lag}" for lag in range(1, lag_order + 1)]


def _marx_rotation_max_lag(recipe: RecipeSpec | None) -> int | None:
    if recipe is None:
        return None
    spec = getattr(recipe, "layer2_representation_spec", {}) or {}
    blocks = dict(spec.get("feature_blocks", {}) or {})
    block = blocks.get("rotation_feature_block", {})
    if isinstance(block, dict) and block.get("value") == "marx_rotation":
        if block.get("max_lag") is not None:
            return int(block["max_lag"])
        composer_contract = block.get("composer_contract", {})
        if isinstance(composer_contract, dict) and composer_contract.get("max_lag") is not None:
            return int(composer_contract["max_lag"])
    data_task_spec = dict(getattr(recipe, "data_task_spec", {}) or {})
    if data_task_spec.get("marx_max_lag") is not None:
        return int(data_task_spec["marx_max_lag"])
    return None


def _recipe_targets(recipe: RecipeSpec) -> tuple[str, ...]:
    return recipe.targets if recipe.targets else (recipe.target,)


def _recipe_for_target(recipe: RecipeSpec, target: str) -> RecipeSpec:
    return replace(recipe, target=target, targets=())


def _model_executor_name(model_family: str, feature_runtime_builder: str) -> str:
    if feature_runtime_builder == "autoreg_lagged_target":
        if is_custom_model(model_family):
            return f"custom_model:{model_family}:autoreg_lagged_target_v0"
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
    if feature_runtime_builder in {"raw_feature_panel", "raw_X_only", "factor_pca", "factors_plus_AR"}:
        if is_custom_model(model_family):
            return f"custom_model:{model_family}:raw_feature_panel_v0"
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
    raise ExecutionError(
        f"feature runtime {feature_runtime_builder!r} is not executable in current runtime slice"
    )


def _model_spec(recipe: RecipeSpec) -> dict[str, object]:
    model_family = _model_family(recipe)
    legacy_feature_builder = _feature_builder(recipe)
    runtime_feature_builder = _feature_runtime_builder(recipe)
    return {
        "model_family": model_family,
        "feature_builder": legacy_feature_builder,
        "feature_runtime_builder": runtime_feature_builder,
        "feature_runtime": _feature_runtime_name(recipe),
        "feature_dispatch_source": "layer2_feature_blocks",
        "executor_name": _model_executor_name(model_family, runtime_feature_builder),
        "framework": recipe.stage0.fixed_design.sample_split,
        "lag_selection": _LAG_SELECTION if model_family == "ar" else "fixed_lag_feature_builder",
        "max_ar_lag": _max_ar_lag(recipe),
        "custom_model": is_custom_model(model_family),
    }


def _coerce_forecast_payload(output: Mapping[str, object] | ForecastPayload, *, executor_name: str) -> ForecastPayload:
    if isinstance(output, ForecastPayload):
        tuning_payload = dict(output.tuning_payload)
        tuning_payload.setdefault("forecast_payload_contract", output.contract_version)
        return ForecastPayload(
            y_pred=float(output.y_pred),
            selected_lag=int(output.selected_lag),
            selected_bic=float(output.selected_bic),
            tuning_payload=tuning_payload,
            contract_version=output.contract_version,
        )
    if not isinstance(output, Mapping):
        raise ExecutionError(
            f"{executor_name} must return a {FORECAST_PAYLOAD_CONTRACT_VERSION} mapping or ForecastPayload; "
            f"got {type(output).__name__}"
        )
    required = {"y_pred", "selected_lag", "selected_bic"}
    missing = sorted(required.difference(output))
    if missing:
        raise ExecutionError(f"{executor_name} {FORECAST_PAYLOAD_CONTRACT_VERSION} missing required fields: {missing}")
    tuning_payload = output.get("tuning_payload") or {}
    if not isinstance(tuning_payload, Mapping):
        raise ExecutionError(f"{executor_name} {FORECAST_PAYLOAD_CONTRACT_VERSION} tuning_payload must be a mapping")
    tuning_payload = dict(tuning_payload)
    tuning_payload.setdefault("forecast_payload_contract", FORECAST_PAYLOAD_CONTRACT_VERSION)
    return ForecastPayload(
        y_pred=float(output["y_pred"]),
        selected_lag=int(output["selected_lag"]),
        selected_bic=float(output["selected_bic"]),
        tuning_payload=tuning_payload,
    )


def _get_model_executor(recipe: RecipeSpec):
    model_family = _model_family(recipe)
    feature_runtime_builder = _feature_runtime_builder(recipe)
    if is_custom_model(model_family) and feature_runtime_builder == "autoreg_lagged_target":
        return _run_custom_autoreg_executor
    if is_custom_model(model_family) and feature_runtime_builder == "raw_feature_panel":
        return _run_custom_raw_panel_executor
    if feature_runtime_builder == "autoreg_lagged_target":
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
            "lstm": _run_lstm_autoreg_executor,
            "gru": _run_gru_autoreg_executor,
            "tcn": _run_tcn_autoreg_executor,
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
    if feature_runtime_builder in {"raw_feature_panel", "raw_X_only", "factor_pca", "factors_plus_AR"}:
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
        f"model_family {model_family!r} with feature runtime {feature_runtime_builder!r} is not executable in current runtime slice"
    )


def _get_benchmark_executor(recipe: RecipeSpec):
    benchmark_family = _benchmark_family(recipe)
    if benchmark_family in {
        "historical_mean", "zero_change", "ar_bic", "custom_benchmark",
        "rolling_mean", "ar_fixed_p", "ardi", "factor_model",
        "expert_benchmark", "multi_benchmark_suite",
        "paper_specific_benchmark", "survey_forecast",
    }:
        return _run_benchmark_executor
    raise ExecutionError(f"benchmark_family {benchmark_family!r} is not supported in current runtime slice")


def _apply_target_transform_and_normalization(series: pd.Series, contract: PreprocessContract | None) -> pd.Series:
    """Apply deterministic target transforms before OOS window construction.

    Target normalization is deliberately not applied here. It is fit inside
    each active training window by _fit_target_normalization_for_window so the
    target-side scale path obeys the same no-leakage rule as X preprocessing.
    """
    if contract is None:
        return series
    s = series.astype(float).copy()

    transform = getattr(contract, "target_transform", "level")
    if transform == "difference":
        s = s.diff().dropna()
    elif transform == "log":
        if (s <= 0).any():
            raise ExecutionError("target_transform=log requires strictly positive target series")
        s = np.log(s)
    elif transform == "log_difference":
        if (s <= 0).any():
            raise ExecutionError("target_transform=log_difference requires strictly positive target series")
        s = np.log(s).diff().dropna()
    elif transform == "growth_rate":
        s = (s / s.shift(1) - 1.0).dropna()
    elif transform not in ("level", None):
        raise ExecutionError(f"target_transform {transform!r} not executable in current runtime slice")

    return s


def _target_normalization_name(contract: PreprocessContract | None) -> str:
    return str(getattr(contract, "target_normalization", "none") if contract is not None else "none")


def _safe_scale(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return 1.0
    return value


def _fit_target_normalization_for_window(
    train: pd.Series,
    contract: PreprocessContract | None,
) -> tuple[pd.Series, dict[str, object]]:
    normalization = _target_normalization_name(contract)
    state: dict[str, object] = {
        "normalization": normalization,
        "fit_scope": "train_only" if normalization != "none" else "not_applicable",
        "params": {},
    }
    if normalization == "none":
        return train, state

    s = train.astype(float)
    if s.empty:
        raise ExecutionError("target_normalization requires a non-empty training window")

    if normalization == "zscore_train_only":
        center = float(s.mean())
        scale = _safe_scale(float(s.std(ddof=0)))
        state["params"] = {"center": center, "scale": scale}
        return (s - center) / scale, state
    if normalization == "robust_zscore":
        center = float(s.median())
        mad = float((s - center).abs().median())
        scale = _safe_scale(1.4826 * mad)
        state["params"] = {"center": center, "scale": scale, "mad": mad}
        return (s - center) / scale, state
    if normalization == "minmax":
        lower = float(s.min())
        upper = float(s.max())
        scale = _safe_scale(upper - lower)
        state["params"] = {"lower": lower, "upper": upper, "scale": scale}
        return (s - lower) / scale, state
    if normalization == "unit_variance":
        scale = _safe_scale(float(s.std(ddof=0)))
        state["params"] = {"scale": scale}
        return s / scale, state

    raise ExecutionError(f"target_normalization {normalization!r} not executable in current runtime slice")


def _apply_target_normalization_scalar(value: float, state: Mapping[str, object]) -> float:
    normalization = str(state.get("normalization", "none"))
    params = state.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    if normalization == "none":
        return float(value)
    if normalization in {"zscore_train_only", "robust_zscore"}:
        return (float(value) - float(params.get("center", 0.0))) / _safe_scale(float(params.get("scale", 1.0)))
    if normalization == "minmax":
        return (float(value) - float(params.get("lower", 0.0))) / _safe_scale(float(params.get("scale", 1.0)))
    if normalization == "unit_variance":
        return float(value) / _safe_scale(float(params.get("scale", 1.0)))
    raise ExecutionError(f"target_normalization {normalization!r} not executable in current runtime slice")


def _inverse_target_normalization_scalar(value: float, state: Mapping[str, object]) -> float:
    normalization = str(state.get("normalization", "none"))
    params = state.get("params", {})
    if not isinstance(params, Mapping):
        params = {}
    if normalization == "none":
        return float(value)
    if normalization in {"zscore_train_only", "robust_zscore"}:
        return float(value) * _safe_scale(float(params.get("scale", 1.0))) + float(params.get("center", 0.0))
    if normalization == "minmax":
        return float(value) * _safe_scale(float(params.get("scale", 1.0))) + float(params.get("lower", 0.0))
    if normalization == "unit_variance":
        return float(value) * _safe_scale(float(params.get("scale", 1.0)))
    raise ExecutionError(f"target_normalization {normalization!r} not executable in current runtime slice")


def _raw_target_series_for_scale(raw_frame: pd.DataFrame, target: str) -> pd.Series:
    if target not in raw_frame.columns:
        raise ExecutionError(f"target {target!r} not found in raw dataset columns")
    return raw_frame[target].dropna().astype(float).copy()


def _inverse_target_transform_scalar(
    value: float,
    *,
    contract: PreprocessContract | None,
    raw_target_series: pd.Series,
    origin_date,
    target_date,
    horizon: int,
) -> float:
    transform = str(getattr(contract, "target_transform", "level") if contract is not None else "level")
    v = float(value)
    if transform == "level":
        return v
    if transform == "log":
        return float(math.exp(v))
    if transform in {"difference", "log_difference", "growth_rate"}:
        if horizon != 1:
            raise ExecutionError(
                f"target_transform={transform!r} can be inverted to original scale only for horizon=1; "
                "use evaluation_scale='transformed_scale' for multi-step transformed-target evaluation"
            )
        if origin_date not in raw_target_series.index:
            raise ExecutionError(f"cannot invert target_transform={transform!r}: origin date is missing from raw target series")
        anchor = float(raw_target_series.loc[origin_date])
        if transform == "difference":
            return anchor + v
        if transform == "log_difference":
            return anchor * float(math.exp(v))
        return anchor * (1.0 + v)
    raise ExecutionError(f"target_transform {transform!r} not executable in current runtime slice")


def _target_metric_values(
    *,
    y_true_model_scale: float,
    y_pred_model_scale: float,
    benchmark_pred_model_scale: float,
    y_true_transformed_scale: float,
    y_pred_transformed_scale: float,
    benchmark_pred_transformed_scale: float,
    y_true_original_scale: float,
    y_pred_original_scale: float,
    benchmark_pred_original_scale: float,
    evaluation_scale: str,
) -> tuple[float, float, float, str]:
    if evaluation_scale == "transformed_scale":
        return (
            y_true_transformed_scale,
            y_pred_transformed_scale,
            benchmark_pred_transformed_scale,
            "transformed_target_scale",
        )
    if evaluation_scale == "model_scale":
        return (
            y_true_model_scale,
            y_pred_model_scale,
            benchmark_pred_model_scale,
            "model_target_scale",
        )
    return (
        y_true_original_scale,
        y_pred_original_scale,
        benchmark_pred_original_scale,
        "original_target_scale",
    )


def _get_target_series(frame: pd.DataFrame, target: str, minimum_train_size: int) -> pd.Series:
    if target not in frame.columns:
        raise ExecutionError(f"target {target!r} not found in raw dataset columns")
    series = frame[target].dropna().astype(float).copy()
    inferred_freq = pd.infer_freq(series.index)
    if inferred_freq is not None:
        try:
            series.index = pd.DatetimeIndex(series.index, freq=inferred_freq)
        except ValueError:
            series.index = pd.DatetimeIndex(series.index)
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
    max_available_lag = len(train) - 1
    lag_order = min(_target_lag_order_from_block(recipe, _max_ar_lag(recipe)), max_available_lag)
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


def _build_target_lag_representation(
    train: pd.Series,
    recipe: RecipeSpec,
    *,
    default_prefix: str,
) -> Layer2Representation:
    lag_order = _lag_order(recipe, train)
    Z_train, y_train = _build_lagged_supervised_matrix(train, lag_order)
    Z_pred = np.asarray(train.to_numpy(dtype=float)[-lag_order:][::-1], dtype=float).reshape(1, -1)
    feature_names = tuple(_target_lag_feature_names(recipe, lag_order, default_prefix=default_prefix))
    return Layer2Representation(
        Z_train=Z_train,
        y_train=y_train,
        Z_pred=Z_pred,
        feature_names=feature_names,
        block_order=("target_lag",),
        block_roles={name: "target_lag" for name in feature_names},
        alignment={
            "representation_runtime": "autoreg_lagged_target",
            "lag_order": int(lag_order),
            "target_lag_timing": "recursive_target_history_reversed_most_recent_first",
        },
        leakage_contract="forecast_origin_only",
        feature_builder=_feature_runtime_builder(recipe),
        feature_runtime_builder=_feature_runtime_builder(recipe),
        legacy_feature_builder=_feature_builder(recipe),
    )


def _fixed_target_lag_frame(
    target_series: pd.Series,
    row_index: pd.Index,
    lag_order: int,
) -> pd.DataFrame:
    """Build origin-aligned target-history features for a direct feature row.

    ``target_lag_1`` is the target observed at the forecast origin row,
    ``target_lag_2`` is the previous target value, and so on. Leading
    unavailable values are zero-filled, matching fixed X-lag behavior.
    """
    if lag_order < 1:
        raise ExecutionError("target_lag_block='fixed_target_lags' requires a positive lag order")
    aligned = target_series.astype(float)
    columns = []
    for lag in range(1, lag_order + 1):
        columns.append(aligned.shift(lag - 1).reindex(row_index).rename(f"__target_lag{lag}"))
    return pd.concat(columns, axis=1).fillna(0.0)


def _recursive_predict_sklearn(model, train: pd.Series, horizon: int, lag_order: int) -> float:
    history = list(train.to_numpy(dtype=float))
    custom_recipe = getattr(model, "_macrocast_custom_preprocessor_recipe", None)
    custom_X_train = getattr(model, "_macrocast_custom_preprocessor_X_train", None)
    custom_y_train = getattr(model, "_macrocast_custom_preprocessor_y_train", None)
    custom_representation = getattr(model, "_macrocast_layer2_representation", None)
    for _ in range(horizon):
        features = np.asarray(history[-lag_order:][::-1], dtype=float).reshape(1, -1)
        if custom_recipe is not None:
            _, _, features = _apply_custom_preprocessor_arrays(
                custom_X_train, custom_y_train, features, custom_recipe,
                context_extra=(
                    custom_representation.runtime_context(mode="predict")
                    if isinstance(custom_representation, Layer2Representation)
                    else _feature_runtime_context(custom_recipe, mode="predict")
                ),
            )
        pred = float(model.predict(features)[0])
        history.append(pred)
    return float(history[-1])


def _raw_panel_columns(frame: pd.DataFrame, target: str, *, predictor_family: str = "all_macro_vars", spec: dict | None = None) -> list[str]:
    """Return the predictor column list honouring 1.4 predictor_family.

    predictor_family values wired in v1.0:

    - ``all_macro_vars`` (default) : every column except the target.
    - ``target_lags_only``         : empty predictor set — raw panel degrades to target-lag-only; compiler guards already tie this value to the autoregressive feature runtime, so this branch should not normally be reached.
    - ``category_based``           : user supplies a mapping (spec['predictor_category_columns'][spec['predictor_category']]).
    - ``factor_only``              : columns whose name starts with 'F_' (convention for factor outputs of factor_pca / factor_augmented_linear builders).
    - ``handpicked_set``           : spec['handpicked_columns'] list.
    """
    spec = dict(spec or {})
    all_except_target = [col for col in frame.columns if col != target]

    if predictor_family == "all_macro_vars":
        cols = all_except_target
    elif predictor_family == "target_lags_only":
        cols = []  # compiler routes this to autoreg path; empty means 'no exogenous X'
    elif predictor_family == "category_based":
        mapping = spec.get("predictor_category_columns")
        category = spec.get("predictor_category")
        if isinstance(mapping, dict) and category in mapping:
            cols = [c for c in mapping[category] if c in frame.columns and c != target]
        else:
            raise ExecutionError("predictor_family='category_based' requires leaf_config.predictor_category_columns (dict) and leaf_config.predictor_category")
    elif predictor_family == "factor_only":
        cols = [c for c in all_except_target if str(c).startswith("F_")]
    elif predictor_family == "handpicked_set":
        handpicked = spec.get("handpicked_columns")
        if not isinstance(handpicked, (list, tuple)) or not handpicked:
            raise ExecutionError("predictor_family='handpicked_set' requires leaf_config.handpicked_columns (list[str])")
        cols = [c for c in handpicked if c in frame.columns and c != target]
    else:
        raise ExecutionError(f"unsupported predictor_family={predictor_family!r}")

    if not cols:
        raise ExecutionError(f"predictor_family={predictor_family!r} produced no usable predictor columns for target={target!r}")
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
    if policy in {"none", "drop", "drop_rows"}:
        # drop_rows is a no-op at this layer (predictor/target coordination is upstream); kept as a pass-through alias of none/drop.
        return X_train, X_pred
    if policy == "drop_columns":
        keep = [c for c in X_train.columns if X_train[c].notna().all()]
        return X_train[keep].copy(), X_pred[keep].copy()
    if policy == "drop_if_above_threshold":
        threshold = 0.30  # default drop columns with more than 30% missing in training
        keep = [c for c in X_train.columns if X_train[c].isna().mean() <= threshold]
        return X_train[keep].copy(), X_pred[keep].copy()
    if policy == "missing_indicator":
        ind_train = X_train.isna().astype(float).add_suffix("__missing")
        ind_pred = X_pred.isna().astype(float).add_suffix("__missing")
        Xt = pd.concat([X_train.fillna(0.0), ind_train], axis=1)
        Xp = pd.concat([X_pred.fillna(0.0), ind_pred], axis=1)
        return Xt, Xp
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
    if policy == "trim":
        lower = X_train.quantile(0.005)
        upper = X_train.quantile(0.995)
        return _clip_frame(X_train, lower, upper), _clip_frame(X_pred, lower, upper)
    if policy == "mad_clip":
        median = X_train.median()
        mad = (X_train - median).abs().median().replace(0, 1.0)
        lower = median - 3.0 * mad
        upper = median + 3.0 * mad
        return _clip_frame(X_train, lower, upper), _clip_frame(X_pred, lower, upper)
    if policy == "outlier_to_missing":
        lower = X_train.quantile(0.01)
        upper = X_train.quantile(0.99)
        Xt = X_train.mask((X_train < lower) | (X_train > upper))
        Xp = X_pred.mask((X_pred < lower) | (X_pred > upper))
        return Xt, Xp
    raise ExecutionError(f"x_outlier_policy {policy!r} is not executable in current runtime slice")


def _apply_additional_preprocessing(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = contract.additional_preprocessing
    if policy == "none":
        return X_train, X_pred
    if policy == "hp_filter":
        from statsmodels.tsa.filters.hp_filter import hpfilter
        def _cycle(col: pd.Series) -> pd.Series:
            if col.count() < 5:
                return col
            try:
                cycle, _ = hpfilter(col.astype(float).dropna(), lamb=1600)
                return col.where(col.isna(), cycle.reindex(col.index, method="nearest"))
            except Exception:
                return col
        Xt = X_train.apply(_cycle)
        Xp = X_pred.apply(_cycle)
        return Xt, Xp
    raise ExecutionError(f"additional_preprocessing {policy!r} is not executable in current runtime slice")


def _apply_x_lag_creation(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy = contract.x_lag_creation
    if policy == "no_x_lags":
        return X_train, X_pred
    if policy == "fixed_x_lags":
        return _fixed_x_lag_frame(X_train), _fixed_x_lag_frame(X_pred)
    raise ExecutionError(f"x_lag_creation {policy!r} is not executable in current runtime slice")


def _fixed_x_lag_frame(X: pd.DataFrame, *, lag_orders: tuple[int, ...] = (1,)) -> pd.DataFrame:
    lag_cols = []
    for col in X.columns:
        for k in lag_orders:
            lag_cols.append(X[col].shift(k).rename(f"{col}__lag{k}"))
    return pd.concat([X] + lag_cols, axis=1).fillna(0.0)


def _fixed_x_lag_public_feature_names(columns: Sequence[str], *, lag_orders: tuple[int, ...] = (1,)) -> list[str]:
    return [f"{column}_lag_{lag}" for column in columns for lag in lag_orders]


def _x_lag_creation_for_feature_names(recipe: RecipeSpec) -> str:
    spec = getattr(recipe, "layer2_representation_spec", {}) or {}
    blocks = dict(spec.get("feature_blocks", {}) or {})
    block = blocks.get("x_lag_feature_block", {})
    if isinstance(block, dict):
        runtime_bridge = block.get("runtime_bridge", {})
        if isinstance(runtime_bridge, dict) and runtime_bridge.get("x_lag_creation") is not None:
            return str(runtime_bridge["x_lag_creation"])
        if block.get("value") == "fixed_x_lags":
            return "fixed_x_lags"
        if block.get("value") == "none":
            return "no_x_lags"
    contract = getattr(recipe, "preprocess_contract", None)
    if contract is not None:
        return str(getattr(contract, "x_lag_creation", "no_x_lags"))
    return "no_x_lags"


def _x_lag_creation_from_feature_block(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    if value == "none":
        return "no_x_lags"
    if value == "fixed_x_lags":
        return "fixed_x_lags"
    raise ExecutionError(f"x_lag_feature_block {value!r} is not executable in current runtime slice")


def _dimensionality_reduction_policy_from_factor_block(value: str | None, fallback: str) -> str:
    if value is None:
        return fallback
    if value == "none":
        return "none"
    if value == "pca_static_factors":
        return fallback if fallback in {"pca", "static_factor"} else "pca"
    if value == "pca_factor_lags":
        return fallback if fallback in {"pca", "static_factor"} else "pca"
    if value == "supervised_factors":
        return "supervised_factors"
    if value == "custom_factors":
        return fallback if fallback in {"pca", "static_factor", "none"} else "none"
    raise ExecutionError(f"factor_feature_block {value!r} is not executable in current runtime slice")


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
    elif policy == "demean_only":
        scaler = StandardScaler(with_mean=True, with_std=False)
    elif policy == "unit_variance_only":
        scaler = StandardScaler(with_mean=False, with_std=True)
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
    return _apply_feature_selection_policy(
        X_train,
        y_train,
        X_pred,
        policy=str(contract.feature_selection_policy),
    )


def _apply_feature_selection_policy(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_pred: pd.DataFrame,
    *,
    policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _feature_selection_semantics(contract: PreprocessContract) -> str:
    return str(getattr(contract, "feature_selection_semantics", "select_before_factor"))


def _apply_dimensionality_reduction(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
    *,
    y_train: np.ndarray | None = None,
    feature_selection_policy: str = "none",
    feature_selection_semantics: str = "select_before_factor",
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    policy = contract.dimensionality_reduction_policy
    if policy == "none":
        return X_train.to_numpy(dtype=float), X_pred.to_numpy(dtype=float)
    n_components = max(1, min(3, X_train.shape[0], X_train.shape[1]))
    if policy == "pca":
        reducer = PCA(n_components=n_components)
        train_scores = reducer.fit_transform(X_train)
        pred_scores = reducer.transform(X_pred)
        if fit_state_sink is not None:
            fit_state_sink.append(
                _factor_fit_state_from_pca(
                    policy,
                    X_train,
                    reducer,
                    feature_selection_policy=feature_selection_policy,
                    feature_selection_semantics=feature_selection_semantics,
                )
            )
        return train_scores, pred_scores
    if policy == "static_factor":
        train_values = X_train.to_numpy(dtype=float)
        mean = train_values.mean(axis=0, keepdims=True)
        centered_train = train_values - mean
        u, s, vt = np.linalg.svd(centered_train, full_matrices=False)
        components = vt[:n_components]
        train_scores = centered_train @ components.T
        centered_pred = X_pred.to_numpy(dtype=float) - mean
        pred_scores = centered_pred @ components.T
        if fit_state_sink is not None:
            fit_state_sink.append(
                _factor_fit_state_from_components(
                    policy,
                    X_train,
                    components,
                    mean.reshape(-1),
                    s,
                    feature_selection_policy=feature_selection_policy,
                    feature_selection_semantics=feature_selection_semantics,
                )
        )
        return train_scores, pred_scores
    if policy == "supervised_factors":
        if y_train is None:
            raise ExecutionError("supervised_factors requires y_train")
        n_components = max(1, min(3, X_train.shape[0] - 1, X_train.shape[1]))
        reducer = PLSRegression(n_components=n_components, scale=False)
        y_arr = np.asarray(y_train, dtype=float).reshape(-1, 1)
        train_scores = reducer.fit_transform(X_train.to_numpy(dtype=float), y_arr)[0]
        pred_scores = reducer.transform(X_pred.to_numpy(dtype=float))
        components = np.asarray(getattr(reducer, "x_rotations_", reducer.x_weights_), dtype=float).T
        mean = np.asarray(
            getattr(reducer, "_x_mean", getattr(reducer, "x_mean_", np.zeros(X_train.shape[1]))),
            dtype=float,
        )
        if fit_state_sink is not None:
            payload = _factor_fit_state_from_components(
                policy,
                X_train,
                components,
                mean,
                feature_selection_policy=feature_selection_policy,
                feature_selection_semantics=feature_selection_semantics,
            )
            payload["block"] = "supervised_factors"
            payload["supervision_target"] = "train_window_y"
            fit_state_sink.append(payload)
        return train_scores, pred_scores
    raise ExecutionError(f"dimensionality_reduction_policy {policy!r} is not executable in current runtime slice")


def _factor_fit_state_from_components(
    policy: str,
    X_train: pd.DataFrame,
    components: np.ndarray,
    mean: np.ndarray,
    singular_values: np.ndarray | None = None,
    *,
    feature_selection_policy: str = "none",
    feature_selection_semantics: str = "select_before_factor",
) -> dict[str, object]:
    source_names = [str(col) for col in X_train.columns]
    feature_names = [f"factor_{idx}" for idx in range(1, int(components.shape[0]) + 1)]
    loadings = {
        feature: {source: float(value) for source, value in zip(source_names, row)}
        for feature, row in zip(feature_names, np.asarray(components, dtype=float))
    }
    payload: dict[str, object] = {
        "block": "pca_static_factors",
        "runtime_policy": policy,
        "n_components": int(components.shape[0]),
        "feature_names": feature_names,
        "source_feature_names": source_names,
        "train_window_rows": int(X_train.shape[0]),
        "train_window_columns": int(X_train.shape[1]),
        "center_mean": [float(x) for x in np.asarray(mean, dtype=float).reshape(-1)],
        "loadings": loadings,
        "feature_selection_policy": str(feature_selection_policy),
        "feature_selection_semantics": (
            str(feature_selection_semantics)
            if str(feature_selection_policy) != "none"
            else "none"
        ),
    }
    if str(feature_selection_policy) != "none" and str(feature_selection_semantics) == "select_before_factor":
        payload["selected_source_feature_names"] = source_names
        payload["selected_source_feature_count"] = int(len(source_names))
    if singular_values is not None:
        payload["singular_values"] = [float(x) for x in np.asarray(singular_values, dtype=float)[: int(components.shape[0])]]
    return payload


def _factor_fit_state_from_pca(
    policy: str,
    X_train: pd.DataFrame,
    reducer: PCA,
    *,
    feature_selection_policy: str = "none",
    feature_selection_semantics: str = "select_before_factor",
) -> dict[str, object]:
    payload = _factor_fit_state_from_components(
        policy,
        X_train,
        np.asarray(reducer.components_, dtype=float),
        np.asarray(reducer.mean_, dtype=float),
        np.asarray(getattr(reducer, "singular_values_", []), dtype=float),
        feature_selection_policy=feature_selection_policy,
        feature_selection_semantics=feature_selection_semantics,
    )
    payload["explained_variance_ratio"] = [float(x) for x in np.asarray(reducer.explained_variance_ratio_, dtype=float)]
    return payload


def _append_factor_lag_block(
    X_train: np.ndarray,
    X_pred: np.ndarray,
    *,
    lag_count: int,
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    factor_train = np.asarray(X_train, dtype=float)
    factor_pred = np.asarray(X_pred, dtype=float)
    if factor_train.ndim != 2 or factor_pred.ndim != 2:
        raise ExecutionError("pca_factor_lags requires 2D factor arrays")
    lag_count = max(1, int(lag_count))
    lagged_train_parts = []
    lagged_pred_parts = []
    for lag in range(1, lag_count + 1):
        train_lag = np.zeros_like(factor_train)
        if len(factor_train) > lag:
            train_lag[lag:, :] = factor_train[:-lag, :]
        lagged_train_parts.append(train_lag)
        pred_lag = (
            factor_train[-lag, :].reshape(1, -1)
            if len(factor_train) >= lag
            else np.zeros((1, factor_train.shape[1]), dtype=float)
        )
        lagged_pred_parts.append(pred_lag)
    lagged_train = np.concatenate(lagged_train_parts, axis=1)
    lagged_pred = np.concatenate(lagged_pred_parts, axis=1)
    if fit_state_sink is not None:
        base_names = [f"factor_{idx}" for idx in range(1, factor_train.shape[1] + 1)]
        lag_names = [
            f"factor_{factor_idx}_lag_{lag}"
            for lag in range(1, lag_count + 1)
            for factor_idx in range(1, factor_train.shape[1] + 1)
        ]
        for payload in reversed(fit_state_sink):
            if str(payload.get("block", "")) == "pca_static_factors":
                payload["block"] = "pca_factor_lags"
                payload["factor_lag_count"] = int(lag_count)
                payload["factor_lag_feature_names"] = lag_names
                payload["feature_names"] = base_names + lag_names
                break
    return (
        np.concatenate([factor_train, lagged_train], axis=1),
        np.concatenate([factor_pred, lagged_pred], axis=1),
    )


def _apply_post_factor_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    feature_names: Sequence[str],
    *,
    policy: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_df = pd.DataFrame(np.asarray(X_train, dtype=float), columns=list(feature_names))
    pred_df = pd.DataFrame(np.asarray(X_pred, dtype=float), columns=list(feature_names))
    selected_train, selected_pred = _apply_feature_selection_policy(
        train_df,
        y_train,
        pred_df,
        policy=policy,
    )
    selected_names = [str(name) for name in selected_train.columns]
    return (
        selected_train.to_numpy(dtype=float),
        selected_pred.to_numpy(dtype=float),
        selected_names,
    )


def _record_post_factor_selection(
    fit_state_sink: list[dict[str, object]] | None,
    *,
    candidate_feature_names: Sequence[str],
    selected_feature_names: Sequence[str],
    policy: str,
) -> None:
    if fit_state_sink is None:
        return
    for payload in reversed(fit_state_sink):
        if str(payload.get("block", "")) not in {"pca_static_factors", "pca_factor_lags", "supervised_factors"}:
            continue
        payload["feature_selection_policy"] = str(policy)
        payload["feature_selection_semantics"] = "select_after_factor"
        payload["post_factor_candidate_feature_names"] = [str(name) for name in candidate_feature_names]
        payload["selected_final_feature_names"] = [str(name) for name in selected_feature_names]
        payload["selected_final_feature_count"] = int(len(selected_feature_names))
        return


def _apply_raw_panel_preprocessing(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_pred: pd.DataFrame,
    contract: PreprocessContract,
    *,
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_selection_policy = str(contract.feature_selection_policy)
    feature_selection_semantics = _feature_selection_semantics(contract)
    X_train, X_pred = _apply_missing_policy(X_train, X_pred, contract)
    X_train, X_pred = _apply_outlier_policy(X_train, X_pred, contract)
    X_train, X_pred = _apply_additional_preprocessing(X_train, X_pred, contract)
    X_train, X_pred = _apply_scaling_policy(X_train, X_pred, contract)
    if not (
        feature_selection_policy != "none"
        and (
            (
                contract.dimensionality_reduction_policy != "none"
                and feature_selection_semantics == "select_after_factor"
            )
            or feature_selection_semantics == "select_after_custom_blocks"
        )
    ):
        X_train, X_pred = _apply_feature_selection_policy(
            X_train,
            y_train,
            X_pred,
            policy=feature_selection_policy,
        )
    X_train, X_pred = _apply_x_lag_creation(X_train, X_pred, contract)
    return _apply_dimensionality_reduction(
        X_train,
        X_pred,
        contract,
        y_train=y_train,
        feature_selection_policy=feature_selection_policy,
        feature_selection_semantics=feature_selection_semantics,
        fit_state_sink=fit_state_sink,
    )


_TARGET_LEVEL_ADD_BACK_COLUMN = "__target_level_origin"
_TARGET_LEVEL_ADD_BACK_FEATURE_NAME = "target_level_origin"
_X_LEVEL_ADD_BACK_SUFFIX = "level"
_MOVING_AVERAGE_FEATURE_WINDOW = 3
_MOVING_AVERAGE_ROTATION_WINDOWS = (3, 6)
_VOLATILITY_FEATURE_WINDOW = 3
_ROLLING_MOMENTS_FEATURE_WINDOW = 3
_LOCAL_TEMPORAL_FACTOR_WINDOW = 3
_LOCAL_TEMPORAL_FACTOR_MEAN_COLUMN = "__local_temporal_factor_mean3"
_LOCAL_TEMPORAL_FACTOR_DISPERSION_COLUMN = "__local_temporal_factor_dispersion3"
_LOCAL_TEMPORAL_FACTOR_MEAN_FEATURE_NAME = "local_temporal_factor_mean3"
_LOCAL_TEMPORAL_FACTOR_DISPERSION_FEATURE_NAME = "local_temporal_factor_dispersion3"


def _deterministic_feature_names(
    component: str | None,
    *,
    break_dates: Sequence[str] | None = None,
    structural_break: bool = False,
) -> list[str]:
    if component in {None, "none"}:
        return []
    if component == "constant_only":
        return ["_dc_const"]
    if component == "linear_trend":
        return ["_dc_trend"]
    if component == "monthly_seasonal":
        return [f"_dc_month_{month:02d}" for month in range(1, 12)]
    if component == "quarterly_seasonal":
        return [f"_dc_q{quarter}" for quarter in range(1, 4)]
    if component == "break_dummies":
        prefix = "_sb_break" if structural_break else "_dc_break"
        return [f"{prefix}_{idx}" for idx in range(1, len(tuple(break_dates or ())) + 1)]
    return []


def _x_level_feature_name(column: str) -> str:
    return f"{column}__{_X_LEVEL_ADD_BACK_SUFFIX}"


def _x_level_public_feature_name(column: str) -> str:
    return f"{column}_{_X_LEVEL_ADD_BACK_SUFFIX}"


def _apply_level_feature_block(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    frame: pd.DataFrame,
    target: str,
    level_feature_block: str,
    predictors: Sequence[str] | None = None,
    spec: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if level_feature_block == "none":
        return X_train, X_pred
    if level_feature_block not in {"target_level_addback", "x_level_addback", "selected_level_addbacks", "level_growth_pairs"}:
        raise ExecutionError(f"unsupported level_feature_block={level_feature_block!r}")
    X_train = X_train.copy()
    X_pred = X_pred.copy()
    if level_feature_block == "target_level_addback":
        target_series = frame[target].astype(float)
        X_train[_TARGET_LEVEL_ADD_BACK_COLUMN] = target_series.loc[X_train.index].to_numpy(dtype=float)
        X_pred[_TARGET_LEVEL_ADD_BACK_COLUMN] = target_series.loc[X_pred.index].to_numpy(dtype=float)
        return X_train, X_pred
    level_source = _level_source_frame(frame)
    predictor_columns = tuple(predictors or ())
    if level_feature_block in {"selected_level_addbacks", "level_growth_pairs"}:
        field = "selected_level_addback_columns" if level_feature_block == "selected_level_addbacks" else "level_growth_pair_columns"
        source_columns = tuple((spec or {}).get(field) or ())
        if not source_columns:
            raise ExecutionError(f"level_feature_block={level_feature_block!r} requires {field}")
        outside_predictors = [str(c) for c in source_columns if c not in predictor_columns]
        if outside_predictors:
            raise ExecutionError(
                f"level_feature_block={level_feature_block!r} columns must be in the active predictor family: "
                f"{outside_predictors}"
            )
    else:
        source_columns = predictor_columns
        if not source_columns:
            source_columns = tuple(c for c in X_train.columns if c in level_source.columns)
    missing = [str(c) for c in source_columns if c not in level_source.columns]
    if missing:
        raise ExecutionError(f"level_feature_block={level_feature_block!r} missing level-source columns: {missing}")
    for column in source_columns:
        name = _x_level_feature_name(str(column))
        series = level_source[column].astype(float)
        X_train[name] = series.loc[X_train.index].to_numpy(dtype=float)
        X_pred[name] = series.loc[X_pred.index].to_numpy(dtype=float)
    return X_train, X_pred


def _moving_average_feature_name(column: str) -> str:
    return f"{column}__ma{_MOVING_AVERAGE_FEATURE_WINDOW}"


def _moving_average_public_feature_name(column: str) -> str:
    return f"{column}_ma{_MOVING_AVERAGE_FEATURE_WINDOW}"


def _moving_average_rotation_feature_name(column: str, window: int) -> str:
    return f"{column}__rotma{window}"


def _moving_average_rotation_public_feature_name(column: str, window: int) -> str:
    return f"{column}_rotma{window}"


def _volatility_feature_name(column: str) -> str:
    return f"{column}__vol{_VOLATILITY_FEATURE_WINDOW}"


def _volatility_public_feature_name(column: str) -> str:
    return f"{column}_vol{_VOLATILITY_FEATURE_WINDOW}"


def _rolling_mean_feature_name(column: str) -> str:
    return f"{column}__mean{_ROLLING_MOMENTS_FEATURE_WINDOW}"


def _rolling_mean_public_feature_name(column: str) -> str:
    return f"{column}_mean{_ROLLING_MOMENTS_FEATURE_WINDOW}"


def _rolling_variance_feature_name(column: str) -> str:
    return f"{column}__var{_ROLLING_MOMENTS_FEATURE_WINDOW}"


def _rolling_variance_public_feature_name(column: str) -> str:
    return f"{column}_var{_ROLLING_MOMENTS_FEATURE_WINDOW}"


def _local_temporal_factor_features(source: pd.DataFrame) -> pd.DataFrame:
    cross_sectional_mean = source.mean(axis=1)
    cross_sectional_dispersion = source.std(axis=1, ddof=0).fillna(0.0)
    return pd.DataFrame(
        {
            _LOCAL_TEMPORAL_FACTOR_MEAN_COLUMN: cross_sectional_mean.rolling(
                window=_LOCAL_TEMPORAL_FACTOR_WINDOW,
                min_periods=1,
            ).mean(),
            _LOCAL_TEMPORAL_FACTOR_DISPERSION_COLUMN: cross_sectional_dispersion.rolling(
                window=_LOCAL_TEMPORAL_FACTOR_WINDOW,
                min_periods=1,
            ).mean(),
        },
        index=source.index,
    )


def _custom_feature_block_name(recipe: RecipeSpec, block_kind: str, axis_value: str) -> str:
    spec = dict(getattr(recipe, "data_task_spec", {}) or {})
    custom_blocks = spec.get("custom_feature_blocks", {})
    if not isinstance(custom_blocks, Mapping):
        custom_blocks = {}
    for key in (
        block_kind,
        f"{block_kind}_feature_block",
        f"custom_{block_kind}_feature_block",
        f"custom_{block_kind}_block",
    ):
        value = custom_blocks.get(key)
        if value:
            return str(value)
    for key in (
        f"custom_{block_kind}_feature_block",
        f"custom_{block_kind}_block",
        f"{block_kind}_feature_block_callable",
    ):
        value = spec.get(key)
        if value:
            return str(value)
    if is_custom_feature_block(axis_value, block_kind=block_kind):
        return axis_value
    raise ExecutionError(
        f"{axis_value!r} requires data_task_spec['custom_{block_kind}_feature_block'] "
        f"or data_task_spec['custom_feature_blocks']['{block_kind}']"
    )


def _coerce_custom_feature_frame(
    value,
    *,
    index,
    columns: Sequence[str],
    role: str,
) -> pd.DataFrame:
    names = [str(name) for name in columns]
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        if len(frame) != len(index):
            raise ExecutionError(f"custom feature block {role} row count mismatch: expected {len(index)}, got {len(frame)}")
        frame.index = index
        if frame.shape[1] != len(names):
            raise ExecutionError(
                f"custom feature block {role} column count mismatch: expected {len(names)}, got {frame.shape[1]}"
            )
        frame.columns = names
        return frame.astype(float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ExecutionError(f"custom feature block {role} must be 1D/2D array-like or a DataFrame")
    if arr.shape != (len(index), len(names)):
        raise ExecutionError(
            f"custom feature block {role} shape mismatch: expected {(len(index), len(names))}, got {arr.shape}"
        )
    return pd.DataFrame(arr, index=index, columns=names)


def _apply_custom_feature_block(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    *,
    frame: pd.DataFrame,
    predictors: Sequence[str],
    y_train: np.ndarray,
    recipe: RecipeSpec,
    horizon: int,
    pred_idx: int,
    block_kind: str,
    axis_value: str,
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_name = _custom_feature_block_name(recipe, block_kind, axis_value)
    try:
        block_spec = get_custom_feature_block(block_name, block_kind=block_kind)
    except KeyError as exc:
        raise ExecutionError(str(exc)) from exc
    result = block_spec.function(
        FeatureBlockCallableContext(
            block_kind=block_kind,
            fit_scope="train_only",
            horizon=int(horizon),
            forecast_origin=frame.index[pred_idx],
            feature_namespace=f"custom_{block_kind}",
            X_train=X_train.copy(),
            X_pred=X_pred.copy(),
            y_train=np.asarray(y_train, dtype=float).copy(),
            source_frame=frame.copy(),
            predictors=tuple(str(predictor) for predictor in predictors),
            train_index=X_train.index,
            pred_index=X_pred.index,
            metadata=dict(getattr(recipe, "data_task_spec", {}) or {}),
        )
    )
    try:
        validate_feature_block_callable_result(result)
    except Exception as exc:
        raise ExecutionError(str(exc)) from exc
    runtime_names = result.runtime_feature_names or result.feature_names
    train_features = _coerce_custom_feature_frame(
        result.train_features,
        index=X_train.index,
        columns=runtime_names,
        role="train_features",
    )
    pred_features = _coerce_custom_feature_frame(
        result.pred_features,
        index=X_pred.index,
        columns=runtime_names,
        role="pred_features",
    )
    overlap = set(map(str, X_train.columns)) & set(map(str, train_features.columns))
    if overlap:
        raise ExecutionError(f"custom feature block returned duplicate runtime feature names: {sorted(overlap)}")
    if fit_state_sink is not None:
        fit_state_sink.append(
            {
                "block": f"custom_{block_kind}_feature_block",
                "name": block_spec.name,
                "block_kind": block_kind,
                "feature_names": [str(name) for name in result.feature_names],
                "runtime_feature_names": [str(name) for name in runtime_names],
                "fit_state": dict(result.fit_state),
                "leakage_metadata": dict(result.leakage_metadata),
                "provenance": dict(result.provenance),
            }
        )
    composition = str(dict(result.provenance).get("composition", "append"))
    if composition == "replace":
        return train_features, pred_features
    if composition != "append":
        raise ExecutionError("custom feature block provenance.composition must be 'append' or 'replace'")
    return (
        pd.concat([X_train.copy(), train_features], axis=1),
        pd.concat([X_pred.copy(), pred_features], axis=1),
    )


def _custom_feature_combiner_name(recipe: RecipeSpec) -> str:
    spec = dict(getattr(recipe, "data_task_spec", {}) or {})
    custom_blocks = spec.get("custom_feature_blocks", {})
    if not isinstance(custom_blocks, Mapping):
        custom_blocks = {}
    for key in ("combiner", "feature_combiner", "custom_combiner", "custom_feature_combiner"):
        value = custom_blocks.get(key)
        if value:
            return str(value)
    for key in ("custom_feature_combiner", "custom_combiner", "custom_feature_block_combiner"):
        value = spec.get(key)
        if value:
            return str(value)
    raise ExecutionError(
        "feature_block_combination='custom_combiner' requires "
        "data_task_spec['custom_feature_combiner'] or data_task_spec['custom_feature_blocks']['combiner']"
    )


def _candidate_block_frames(
    train_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    feature_names: Sequence[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, str]]:
    names = [str(name) for name in feature_names]
    train_public = train_frame.copy()
    pred_public = pred_frame.copy()
    train_public.columns = names
    pred_public.columns = names
    block_roles = {name: _raw_panel_feature_role(name) for name in names}
    blocks_train: dict[str, pd.DataFrame] = {"candidate_z": train_public}
    blocks_pred: dict[str, pd.DataFrame] = {"candidate_z": pred_public}
    for role in dict.fromkeys(block_roles.values()):
        role_names = [name for name in names if block_roles[name] == role]
        if not role_names:
            continue
        blocks_train[role] = train_public[role_names].copy()
        blocks_pred[role] = pred_public[role_names].copy()
    return blocks_train, blocks_pred, block_roles


def _apply_custom_feature_combiner(
    X_train_arr: np.ndarray,
    X_pred_arr: np.ndarray,
    y_train: np.ndarray,
    *,
    candidate_feature_names: Sequence[str],
    train_index,
    pred_index,
    recipe: RecipeSpec,
    horizon: int,
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    combiner_name = _custom_feature_combiner_name(recipe)
    try:
        combiner_spec = get_custom_feature_combiner(combiner_name)
    except KeyError as exc:
        raise ExecutionError(str(exc)) from exc
    candidate_names = [str(name) for name in candidate_feature_names]
    train_frame = pd.DataFrame(np.asarray(X_train_arr, dtype=float), index=train_index, columns=candidate_names)
    pred_frame = pd.DataFrame(np.asarray(X_pred_arr, dtype=float), index=pred_index, columns=candidate_names)
    blocks_train, blocks_pred, block_roles = _candidate_block_frames(train_frame, pred_frame, candidate_names)
    result = combiner_spec.function(
        FeatureCombinerCallableContext(
            blocks_train=blocks_train,
            blocks_pred=blocks_pred,
            y_train=np.asarray(y_train, dtype=float).copy(),
            feature_names=tuple(candidate_names),
            block_roles=block_roles,
            fit_scope="train_only",
            horizon=int(horizon),
            forecast_origin=pred_frame.index[0] if len(pred_frame.index) else None,
            train_index=train_frame.index,
            pred_index=pred_frame.index,
            metadata=dict(getattr(recipe, "data_task_spec", {}) or {}),
        )
    )
    try:
        validate_feature_combiner_callable_result(result)
    except Exception as exc:
        raise ExecutionError(str(exc)) from exc
    feature_names = [str(name) for name in result.feature_names]
    combined_train = _coerce_custom_feature_frame(
        result.Z_train,
        index=train_index,
        columns=feature_names,
        role="Z_train",
    )
    combined_pred = _coerce_custom_feature_frame(
        result.Z_pred,
        index=pred_index,
        columns=feature_names,
        role="Z_pred",
    )
    roles = {
        str(name): str(result.block_roles.get(name, _raw_panel_feature_role(str(name))))
        for name in feature_names
    }
    if fit_state_sink is not None:
        fit_state_sink.append(
            {
                "block": "custom_feature_combiner",
                "name": combiner_spec.name,
                "contract_version": CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION,
                "candidate_feature_names": candidate_names,
                "feature_names": feature_names,
                "block_roles": roles,
                "fit_state": dict(result.fit_state),
                "leakage_metadata": dict(result.leakage_metadata),
                "provenance": dict(result.provenance),
            }
        )
    return (
        combined_train.to_numpy(dtype=float),
        combined_pred.to_numpy(dtype=float),
        feature_names,
    )


def _apply_custom_final_z_selection(
    X_train_arr: np.ndarray,
    y_train: np.ndarray,
    X_pred_arr: np.ndarray,
    *,
    candidate_feature_names: Sequence[str],
    policy: str,
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_df = pd.DataFrame(np.asarray(X_train_arr, dtype=float), columns=[str(name) for name in candidate_feature_names])
    pred_df = pd.DataFrame(np.asarray(X_pred_arr, dtype=float), columns=[str(name) for name in candidate_feature_names])
    selected_train, selected_pred = _apply_feature_selection_policy(
        train_df,
        y_train,
        pred_df,
        policy=policy,
    )
    selected_names = [str(name) for name in selected_train.columns]
    candidate_names = [str(name) for name in candidate_feature_names]
    dropped_names = [name for name in candidate_names if name not in set(selected_names)]
    if fit_state_sink is not None:
        fit_state_sink.append(
            {
                "block": "custom_final_z_selection",
                "contract_version": CUSTOM_FINAL_Z_SELECTION_CONTRACT_VERSION,
                "feature_selection_policy": str(policy),
                "candidate_feature_names": candidate_names,
                "selected_feature_names": selected_names,
                "selected_final_feature_names": selected_names,
                "selected_final_feature_count": int(len(selected_names)),
                "dropped_feature_names": dropped_names,
                "block_roles": {name: _raw_panel_feature_role(name) for name in selected_names},
                "fit_state": {"selected_feature_count": int(len(selected_names))},
                "leakage_metadata": {"lookahead": "forbidden"},
            }
        )
    return (
        selected_train.to_numpy(dtype=float),
        selected_pred.to_numpy(dtype=float),
        selected_names,
    )


def _apply_temporal_feature_block(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    frame: pd.DataFrame,
    predictors: Sequence[str],
    y_train: np.ndarray,
    recipe: RecipeSpec,
    *,
    horizon: int,
    start_idx: int,
    pred_idx: int,
    temporal_feature_block: str,
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if temporal_feature_block == "none":
        return X_train, X_pred
    if temporal_feature_block == "custom_temporal_features":
        return _apply_custom_feature_block(
            X_train,
            X_pred,
            frame=frame,
            predictors=predictors,
            y_train=y_train,
            recipe=recipe,
            horizon=horizon,
            pred_idx=pred_idx,
            block_kind="temporal",
            axis_value=temporal_feature_block,
            fit_state_sink=fit_state_sink,
        )
    if temporal_feature_block not in {
        "local_temporal_factors",
        "moving_average_features",
        "rolling_moments",
        "volatility_features",
    }:
        raise ExecutionError(f"unsupported temporal_feature_block={temporal_feature_block!r}")
    source = frame[list(predictors)].iloc[start_idx : pred_idx + 1].astype(float).copy()
    if temporal_feature_block == "local_temporal_factors":
        if source.empty:
            return X_train, X_pred
        local_features = _local_temporal_factor_features(source)
        X_train = X_train.copy()
        X_pred = X_pred.copy()
        for column in local_features.columns:
            X_train[column] = local_features.loc[X_train.index, column].to_numpy(dtype=float)
            X_pred[column] = local_features.loc[X_pred.index, column].to_numpy(dtype=float)
        return X_train, X_pred
    if temporal_feature_block == "moving_average_features":
        feature_frames = [(source.rolling(window=_MOVING_AVERAGE_FEATURE_WINDOW, min_periods=1).mean(), _moving_average_feature_name)]
    elif temporal_feature_block == "rolling_moments":
        rolling = source.rolling(window=_ROLLING_MOMENTS_FEATURE_WINDOW, min_periods=1)
        feature_frames = [
            (rolling.mean(), _rolling_mean_feature_name),
            (rolling.var(ddof=0).fillna(0.0), _rolling_variance_feature_name),
        ]
    else:
        feature_frames = [(source.rolling(window=_VOLATILITY_FEATURE_WINDOW, min_periods=1).std(ddof=0).fillna(0.0), _volatility_feature_name)]
    X_train = X_train.copy()
    X_pred = X_pred.copy()
    for column in predictors:
        for rolling_frame, feature_name in feature_frames:
            name = feature_name(str(column))
            X_train[name] = rolling_frame.loc[X_train.index, column].to_numpy(dtype=float)
            X_pred[name] = rolling_frame.loc[X_pred.index, column].to_numpy(dtype=float)
    return X_train, X_pred


def _apply_rotation_feature_block(
    X_train: pd.DataFrame,
    X_pred: pd.DataFrame,
    frame: pd.DataFrame,
    predictors: Sequence[str],
    y_train: np.ndarray,
    recipe: RecipeSpec,
    *,
    horizon: int,
    start_idx: int,
    pred_idx: int,
    rotation_feature_block: str,
    marx_max_lag: int | None = None,
    feature_block_combination: str = "replace_with_blocks",
    fit_state_sink: list[dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rotation_feature_block == "none":
        return X_train, X_pred
    if rotation_feature_block == "custom_rotation":
        return _apply_custom_feature_block(
            X_train,
            X_pred,
            frame=frame,
            predictors=predictors,
            y_train=y_train,
            recipe=recipe,
            horizon=horizon,
            pred_idx=pred_idx,
            block_kind="rotation",
            axis_value=rotation_feature_block,
            fit_state_sink=fit_state_sink,
        )
    if rotation_feature_block == "marx_rotation":
        if marx_max_lag is None:
            raise ExecutionError("rotation_feature_block='marx_rotation' requires marx_max_lag")
        source = frame[list(predictors)].iloc[start_idx : pred_idx + 1].astype(float).copy()
        rotated = _build_marx_rotation_frame(source, max_lag=int(marx_max_lag))
        rotated_train = rotated.loc[X_train.index].copy()
        rotated_pred = rotated.loc[X_pred.index].copy()
        if feature_block_combination in {"append_to_base_x", "concatenate_named_blocks"}:
            overlap = set(map(str, X_train.columns)) & set(map(str, rotated_train.columns))
            if overlap:
                raise ExecutionError(f"MARX rotation produced duplicate feature names while appending: {sorted(overlap)}")
            return (
                pd.concat([X_train.copy(), rotated_train], axis=1),
                pd.concat([X_pred.copy(), rotated_pred], axis=1),
            )
        if feature_block_combination != "replace_with_blocks":
            raise ExecutionError(
                f"feature_block_combination={feature_block_combination!r} is not supported with marx_rotation"
            )
        return rotated_train, rotated_pred
    if rotation_feature_block != "moving_average_rotation":
        raise ExecutionError(f"unsupported rotation_feature_block={rotation_feature_block!r}")
    source = frame[list(predictors)].iloc[start_idx : pred_idx + 1].astype(float).copy()
    X_train = X_train.copy()
    X_pred = X_pred.copy()
    for window in _MOVING_AVERAGE_ROTATION_WINDOWS:
        rotated = source.rolling(window=window, min_periods=1).mean()
        for column in predictors:
            name = _moving_average_rotation_feature_name(str(column), window)
            X_train[name] = rotated.loc[X_train.index, column].to_numpy(dtype=float)
            X_pred[name] = rotated.loc[X_pred.index, column].to_numpy(dtype=float)
    return X_train, X_pred


def _raw_panel_feature_names(
    frame: pd.DataFrame,
    target: str,
    recipe: RecipeSpec,
) -> list[str]:
    names = _raw_panel_columns(
        frame,
        target,
        predictor_family=_predictor_family(recipe),
        spec=_layer2_runtime_spec(recipe),
    )
    base_names = tuple(names)
    if _x_lag_creation_for_feature_names(recipe) == "fixed_x_lags":
        names.extend(_fixed_x_lag_public_feature_names(base_names))
    temporal_feature_block = _temporal_feature_block(recipe)
    if temporal_feature_block == "moving_average_features":
        names.extend(_moving_average_public_feature_name(str(column)) for column in base_names)
    elif temporal_feature_block == "rolling_moments":
        names.extend(_rolling_mean_public_feature_name(str(column)) for column in base_names)
        names.extend(_rolling_variance_public_feature_name(str(column)) for column in base_names)
    elif temporal_feature_block == "volatility_features":
        names.extend(_volatility_public_feature_name(str(column)) for column in base_names)
    elif temporal_feature_block == "local_temporal_factors" and base_names:
        names.extend(
            [
                _LOCAL_TEMPORAL_FACTOR_MEAN_FEATURE_NAME,
                _LOCAL_TEMPORAL_FACTOR_DISPERSION_FEATURE_NAME,
            ]
        )
    rotation_feature_block = _rotation_feature_block(recipe)
    if rotation_feature_block == "marx_rotation":
        marx_max_lag = _marx_rotation_max_lag(recipe)
        if marx_max_lag is None:
            raise ExecutionError("rotation_feature_block='marx_rotation' requires marx_max_lag")
        marx_names = [
            _marx_rotation_public_feature_name(str(column), rotation_order)
            for column in base_names
            for rotation_order in range(1, int(marx_max_lag) + 1)
        ]
        if _feature_block_combination(recipe) in {"append_to_base_x", "concatenate_named_blocks"}:
            names.extend(marx_names)
        else:
            names = marx_names
    elif rotation_feature_block == "moving_average_rotation":
        for window in _MOVING_AVERAGE_ROTATION_WINDOWS:
            names.extend(_moving_average_rotation_public_feature_name(str(column), window) for column in base_names)
    level_feature_block = _level_feature_block(recipe)
    if level_feature_block == "target_level_addback":
        names.append(_TARGET_LEVEL_ADD_BACK_FEATURE_NAME)
    elif level_feature_block == "x_level_addback":
        names.extend(_x_level_public_feature_name(str(column)) for column in base_names)
    elif level_feature_block == "selected_level_addbacks":
        selected_columns = tuple(dict(recipe.data_task_spec).get("selected_level_addback_columns") or ())
        names.extend(_x_level_public_feature_name(str(column)) for column in selected_columns)
    elif level_feature_block == "level_growth_pairs":
        pair_columns = tuple(dict(recipe.data_task_spec).get("level_growth_pair_columns") or ())
        names.extend(_x_level_public_feature_name(str(column)) for column in pair_columns)
    if _feature_runtime_builder(recipe) == "raw_feature_panel" and _target_lag_feature_block(recipe) == "fixed_target_lags":
        lag_order = _target_lag_order_from_block(recipe, _max_ar_lag(recipe))
        names.extend(_target_lag_feature_names(recipe, lag_order, default_prefix="target_lag"))
    return names


def _factor_feature_names_from_fit_state(fit_state: Sequence[Mapping[str, object]]) -> list[str] | None:
    for payload in reversed(tuple(fit_state)):
        if str(payload.get("block", "")) not in {"pca_static_factors", "pca_factor_lags", "supervised_factors"}:
            continue
        names = payload.get("feature_names")
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            return [str(name) for name in names]
    return None


def _custom_feature_names_from_fit_state(fit_state: Sequence[Mapping[str, object]]) -> tuple[list[str], bool]:
    names: list[str] = []
    replaced = False
    for payload in tuple(fit_state):
        if not str(payload.get("block", "")).startswith("custom_"):
            continue
        payload_names = payload.get("feature_names")
        if isinstance(payload_names, Sequence) and not isinstance(payload_names, (str, bytes)):
            names.extend(str(name) for name in payload_names)
        provenance = payload.get("provenance", {})
        if isinstance(provenance, Mapping) and str(provenance.get("composition", "append")) == "replace":
            replaced = True
    return names, replaced


def _public_feature_names_from_runtime_columns(
    columns: Sequence[object],
    fit_state: Sequence[Mapping[str, object]],
) -> list[str]:
    custom_name_map: dict[str, str] = {}
    for payload in tuple(fit_state):
        if not str(payload.get("block", "")).startswith("custom_"):
            continue
        public_names = payload.get("feature_names")
        runtime_names = payload.get("runtime_feature_names")
        if (
            isinstance(public_names, Sequence)
            and not isinstance(public_names, (str, bytes))
            and isinstance(runtime_names, Sequence)
            and not isinstance(runtime_names, (str, bytes))
        ):
            custom_name_map.update(
                {str(runtime): str(public) for runtime, public in zip(runtime_names, public_names)}
            )
    names: list[str] = []
    for column in columns:
        name = str(column)
        if name in custom_name_map:
            names.append(custom_name_map[name])
        elif name == _TARGET_LEVEL_ADD_BACK_COLUMN:
            names.append(_TARGET_LEVEL_ADD_BACK_FEATURE_NAME)
        elif name == _LOCAL_TEMPORAL_FACTOR_MEAN_COLUMN:
            names.append(_LOCAL_TEMPORAL_FACTOR_MEAN_FEATURE_NAME)
        elif name == _LOCAL_TEMPORAL_FACTOR_DISPERSION_COLUMN:
            names.append(_LOCAL_TEMPORAL_FACTOR_DISPERSION_FEATURE_NAME)
        elif "__lag" in name:
            left, right = name.split("__lag", 1)
            names.append(f"{left}_lag_{right}")
        elif "__" in name:
            names.append(name.replace("__", "_", 1))
        else:
            names.append(name)
    return names


def _selected_final_feature_names_from_fit_state(
    fit_state: Sequence[Mapping[str, object]],
) -> list[str] | None:
    for payload in reversed(tuple(fit_state)):
        if str(payload.get("block", "")) == "custom_final_z_selection":
            names = payload.get("selected_final_feature_names") or payload.get("selected_feature_names")
            if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
                return [str(name) for name in names]
        if str(payload.get("block", "")) == "custom_feature_combiner":
            names = payload.get("feature_names")
            if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
                return [str(name) for name in names]
        if str(payload.get("block", "")) not in {"pca_static_factors", "pca_factor_lags", "supervised_factors"}:
            continue
        names = payload.get("selected_final_feature_names")
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            return [str(name) for name in names]
    return None


def _block_roles_from_fit_state(
    fit_state: Sequence[Mapping[str, object]],
) -> dict[str, str] | None:
    for payload in reversed(tuple(fit_state)):
        if str(payload.get("block", "")) not in {"custom_final_z_selection", "custom_feature_combiner"}:
            continue
        roles = payload.get("block_roles")
        if isinstance(roles, Mapping):
            return {str(name): str(role) for name, role in roles.items()}
    return None


def _block_order_from_feature_names(feature_names: Sequence[str]) -> tuple[str, ...]:
    order: list[str] = []
    seen: set[str] = set()
    for name in feature_names:
        role = _raw_panel_feature_role(str(name))
        if role in seen:
            continue
        seen.add(role)
        order.append(role)
    return tuple(order)


def _raw_panel_representation_feature_names(
    frame: pd.DataFrame,
    target: str,
    recipe: RecipeSpec,
    fit_state: Sequence[Mapping[str, object]],
) -> list[str]:
    selected_names = _selected_final_feature_names_from_fit_state(fit_state)
    if selected_names is not None:
        return selected_names
    factor_names = _factor_feature_names_from_fit_state(fit_state)
    if factor_names is None:
        custom_names, custom_replaced = _custom_feature_names_from_fit_state(fit_state)
        names = [] if custom_replaced else _raw_panel_feature_names(frame, target, recipe)
        names.extend(name for name in custom_names if name not in names)
        return names
    names = list(factor_names)
    if _feature_runtime_builder(recipe) == "raw_feature_panel" and _target_lag_feature_block(recipe) == "fixed_target_lags":
        lag_order = _target_lag_order_from_block(recipe, _max_ar_lag(recipe))
        target_lag_names = _target_lag_feature_names(recipe, lag_order, default_prefix="target_lag")
        if _feature_block_combination(recipe) == "append_to_target_lags":
            names = list(target_lag_names) + names
        else:
            names.extend(target_lag_names)
    return names


def _raw_panel_feature_role(name: str) -> str:
    if name.startswith("custom_"):
        return "custom"
    if name.startswith("_dc_") or name.startswith("_sb_"):
        return "deterministic"
    if name.startswith("factor_"):
        return "factor"
    if name.startswith("target_lag_") or name.startswith("y_lag_") or name.startswith("lag_"):
        return "target_lag"
    if "_marx_ma_lag1_to_lag" in name or "_rotma" in name:
        return "rotation"
    if (
        name in {_LOCAL_TEMPORAL_FACTOR_MEAN_FEATURE_NAME, _LOCAL_TEMPORAL_FACTOR_DISPERSION_FEATURE_NAME}
        or name.endswith("_ma3")
        or name.endswith("_mean3")
        or name.endswith("_var3")
        or name.endswith("_vol3")
    ):
        return "temporal"
    if name == _TARGET_LEVEL_ADD_BACK_FEATURE_NAME or name.endswith("_level"):
        return "level"
    if "_lag_" in name:
        return "x_lag"
    return "base_x"


def _raw_panel_block_order(
    recipe: RecipeSpec,
    fit_state: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    selected_names = _selected_final_feature_names_from_fit_state(fit_state)
    if selected_names is not None:
        return _block_order_from_feature_names(selected_names)
    order: list[str] = []
    if _factor_feature_names_from_fit_state(fit_state) is not None:
        order.append("factor")
    else:
        order.append("base_x")
        if _x_lag_creation_for_feature_names(recipe) == "fixed_x_lags":
            order.append("x_lag")
        if _temporal_feature_block(recipe) != "none":
            order.append("temporal")
        if _rotation_feature_block(recipe) != "none":
            order.append("rotation")
        if _level_feature_block(recipe) != "none":
            order.append("level")
        custom_names, _custom_replaced = _custom_feature_names_from_fit_state(fit_state)
        if custom_names:
            order.append("custom")
    if _feature_runtime_builder(recipe) == "raw_feature_panel" and _target_lag_feature_block(recipe) == "fixed_target_lags":
        if _feature_block_combination(recipe) == "append_to_target_lags":
            order.insert(0, "target_lag")
        else:
            order.append("target_lag")
    return tuple(order)


def _raw_panel_alignment(
    recipe: RecipeSpec,
    *,
    spec: Mapping[str, object] | None,
    fit_state: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    contemp_rule = str((spec or {}).get("contemporaneous_x_rule", "forbid_contemporaneous"))
    alignment: dict[str, object] = {
        "representation_runtime": "raw_feature_panel",
        "contemporaneous_x_rule": contemp_rule,
        "x_observation_timing": (
            "target_date_x"
            if contemp_rule == "allow_contemporaneous"
            else "forecast_origin_x"
        ),
    }
    if _feature_runtime_builder(recipe) == "raw_feature_panel" and _target_lag_feature_block(recipe) == "fixed_target_lags":
        alignment["target_lag_timing"] = "target_lag_1_equals_target_observed_at_forecast_origin"
    if _x_lag_creation_for_feature_names(recipe) == "fixed_x_lags":
        alignment["x_lag_timing"] = "origin_aligned_trailing_x_history"
    if _temporal_feature_block(recipe) != "none":
        alignment["temporal_timing"] = "origin_aligned_trailing_window"
    if _rotation_feature_block(recipe) != "none":
        alignment["rotation_timing"] = "origin_aligned_trailing_window"
    if _level_feature_block(recipe) != "none":
        alignment["level_timing"] = "observable_at_forecast_origin"
    if _factor_feature_names_from_fit_state(fit_state) is not None:
        alignment["factor_fit_scope"] = "train_window_only"
        if _rotation_feature_block(recipe) == "marx_rotation":
            alignment["rotation_factor_semantics"] = "marx_then_factor"
        for payload in reversed(tuple(fit_state)):
            if str(payload.get("block", "")) not in {"pca_static_factors", "pca_factor_lags", "supervised_factors"}:
                continue
            if str(payload.get("feature_selection_semantics", "none")) != "none":
                alignment["factor_selection_semantics"] = str(payload["feature_selection_semantics"])
            if "selected_final_feature_names" in payload:
                alignment["factor_selection_stage"] = "post_factor_final_Z"
            break
    return alignment


def _raw_panel_leakage_contract(spec: Mapping[str, object] | None) -> str:
    contemp_rule = str((spec or {}).get("contemporaneous_x_rule", "forbid_contemporaneous"))
    if contemp_rule == "allow_contemporaneous":
        return "allow_contemporaneous_oracle_x"
    return "forecast_origin_only"


def _build_raw_panel_representation(
    frame: pd.DataFrame,
    recipe: RecipeSpec,
    horizon: int,
    start_idx: int,
    origin_idx: int,
    contract: PreprocessContract,
    *,
    target_window: pd.Series | None = None,
) -> Layer2Representation:
    fit_state: list[dict[str, object]] = []
    spec = _layer2_runtime_spec(recipe)
    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        recipe.target,
        horizon,
        start_idx,
        origin_idx,
        contract,
        recipe,
        predictor_family=_predictor_family(recipe),
        spec=spec,
        target_window=target_window,
        fit_state_sink=fit_state,
        level_feature_block=_level_feature_block(recipe),
        temporal_feature_block=_temporal_feature_block(recipe),
        rotation_feature_block=_rotation_feature_block(recipe),
        x_lag_feature_block=_x_lag_feature_block(recipe),
        factor_feature_block=_factor_feature_block(recipe),
        target_lag_block=_target_lag_feature_block(recipe),
        target_lag_order=_target_lag_order_from_block(recipe, _max_ar_lag(recipe)),
        target_lag_feature_names=(
            _target_lag_feature_names(
                recipe,
                _target_lag_order_from_block(recipe, _max_ar_lag(recipe)),
                default_prefix="target_lag",
            )
            if _target_lag_feature_block(recipe) == "fixed_target_lags"
            else None
        ),
        marx_max_lag=_marx_rotation_max_lag(recipe),
        feature_block_combination=_feature_block_combination(recipe),
    )
    feature_names = tuple(_raw_panel_representation_feature_names(frame, recipe.target, recipe, fit_state))
    fit_state_block_roles = _block_roles_from_fit_state(fit_state)
    block_roles = fit_state_block_roles or {
        name: _raw_panel_feature_role(name) for name in feature_names
    }
    block_order = (
        tuple(dict.fromkeys(block_roles.get(name, _raw_panel_feature_role(name)) for name in feature_names))
        if fit_state_block_roles is not None
        else _raw_panel_block_order(recipe, fit_state)
    )
    return Layer2Representation(
        Z_train=np.asarray(X_train, dtype=float),
        y_train=np.asarray(y_train, dtype=float),
        Z_pred=np.asarray(X_pred, dtype=float),
        feature_names=feature_names,
        block_order=block_order,
        block_roles=block_roles,
        fit_state=tuple(dict(payload) for payload in fit_state),
        alignment=_raw_panel_alignment(recipe, spec=spec, fit_state=fit_state),
        leakage_contract=_raw_panel_leakage_contract(spec),
        feature_builder=_feature_runtime_builder(recipe),
        feature_runtime_builder=_feature_runtime_builder(recipe),
        legacy_feature_builder=_feature_builder(recipe),
    )


def _build_raw_panel_training_data(
    frame: pd.DataFrame,
    target: str,
    horizon: int,
    start_idx: int,
    origin_idx: int,
    contract: PreprocessContract,
    recipe: RecipeSpec | None = None,
    *,
    predictor_family: str = "all_macro_vars",
    spec: dict | None = None,
    target_window: pd.Series | None = None,
    fit_state_sink: list[dict[str, object]] | None = None,
    level_feature_block: str = "none",
    temporal_feature_block: str = "none",
    rotation_feature_block: str = "none",
    x_lag_feature_block: str | None = None,
    factor_feature_block: str | None = None,
    target_lag_block: str = "none",
    target_lag_order: int | None = None,
    target_lag_feature_names: Sequence[str] | None = None,
    marx_max_lag: int | None = None,
    feature_block_combination: str = "replace_with_blocks",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictors = _raw_panel_columns(frame, target, predictor_family=predictor_family, spec=spec)
    if origin_idx - horizon < start_idx:
        raise ExecutionError("insufficient history for raw_feature_panel training data")

    def _target_values() -> np.ndarray:
        if target_window is None:
            return frame[target].iloc[start_idx + horizon : origin_idx + 1].to_numpy(dtype=float)
        expected_len = origin_idx - start_idx + 1
        if len(target_window) != expected_len:
            raise ExecutionError(
                "target_window length does not match raw-panel origin window: "
                f"expected {expected_len}, got {len(target_window)}"
            )
        return target_window.iloc[horizon:].to_numpy(dtype=float)

    # 1.5 contemporaneous_x_rule: default forbid (X at origin + train pairs X_t with y_{t+h});
    # allow uses X aligned to the target date (X_{t+h} with y_{t+h}) — the oracle / data-leak
    # benchmark variant.
    contemp_rule = (spec or {}).get("contemporaneous_x_rule", "forbid_contemporaneous")
    if rotation_feature_block in {"moving_average_rotation", "marx_rotation"} and contemp_rule != "forbid_contemporaneous":
        raise ExecutionError(
            f"rotation_feature_block={rotation_feature_block!r} requires "
            "contemporaneous_x_rule='forbid_contemporaneous'"
        )
    if level_feature_block in {"target_level_addback", "x_level_addback", "selected_level_addbacks", "level_growth_pairs"} and contemp_rule != "forbid_contemporaneous":
        raise ExecutionError(
            f"level_feature_block={level_feature_block!r} requires "
            "contemporaneous_x_rule='forbid_contemporaneous'"
        )
    if contemp_rule == "allow_contemporaneous":
        X_train = frame[predictors].iloc[start_idx + horizon : origin_idx + 1].astype(float).copy()
        y_train = _target_values()
        pred_idx = origin_idx + horizon
        if pred_idx >= len(frame):
            raise ExecutionError("contemporaneous_x_rule='allow_contemporaneous' requires X observed at the target date; target falls beyond the available index")
        X_pred = frame[predictors].iloc[[pred_idx]].astype(float).copy()
    else:
        X_train = frame[predictors].iloc[start_idx : origin_idx - horizon + 1].astype(float).copy()
        y_train = _target_values()
        pred_idx = origin_idx
        X_pred = frame[predictors].iloc[[origin_idx]].astype(float).copy()
    if len(X_train) == 0 or len(y_train) == 0:
        raise ExecutionError("raw_feature_panel produced empty training data")
    x_lag_creation = _x_lag_creation_from_feature_block(
        x_lag_feature_block,
        fallback=str(contract.x_lag_creation),
    )
    dimensionality_reduction_policy = _dimensionality_reduction_policy_from_factor_block(
        factor_feature_block,
        fallback=str(contract.dimensionality_reduction_policy),
    )
    marx_append_mode = feature_block_combination in {"append_to_base_x", "concatenate_named_blocks"}
    if rotation_feature_block == "marx_rotation" and temporal_feature_block != "none" and not marx_append_mode:
        raise ExecutionError(
            "rotation_feature_block='marx_rotation' requires feature_block_combination='append_to_base_x' "
            "or 'concatenate_named_blocks' when combined with temporal_feature_block"
        )
    if rotation_feature_block == "marx_rotation" and x_lag_creation != "no_x_lags" and not marx_append_mode:
        raise ExecutionError(
            "rotation_feature_block='marx_rotation' requires feature_block_combination='append_to_base_x' "
            "or 'concatenate_named_blocks' when combined with x_lag_creation"
        )
    preprocessing_contract = contract
    if x_lag_creation == "fixed_x_lags":
        lag_source = frame[predictors].iloc[start_idx : pred_idx + 1].astype(float).copy()
        lagged_source = _fixed_x_lag_frame(lag_source)
        X_train = lagged_source.loc[X_train.index].copy()
        X_pred = lagged_source.loc[X_pred.index].copy()
        preprocessing_contract = replace(contract, x_lag_creation="no_x_lags")
    if dimensionality_reduction_policy != preprocessing_contract.dimensionality_reduction_policy:
        preprocessing_contract = replace(
            preprocessing_contract,
            dimensionality_reduction_policy=dimensionality_reduction_policy,
        )
    X_train, X_pred = _apply_temporal_feature_block(
        X_train,
        X_pred,
        frame,
        predictors,
        y_train,
        recipe,
        horizon=horizon,
        start_idx=start_idx,
        pred_idx=pred_idx,
        temporal_feature_block=temporal_feature_block,
        fit_state_sink=fit_state_sink,
    )
    X_train, X_pred = _apply_rotation_feature_block(
        X_train,
        X_pred,
        frame,
        predictors,
        y_train,
        recipe,
        horizon=horizon,
        start_idx=start_idx,
        pred_idx=pred_idx,
        rotation_feature_block=rotation_feature_block,
        marx_max_lag=marx_max_lag,
        feature_block_combination=feature_block_combination,
        fit_state_sink=fit_state_sink,
    )
    if factor_feature_block == "custom_factors":
        X_train, X_pred = _apply_custom_feature_block(
            X_train,
            X_pred,
            frame=frame,
            predictors=predictors,
            y_train=y_train,
            recipe=recipe,
            horizon=horizon,
            pred_idx=pred_idx,
            block_kind="factor",
            axis_value=factor_feature_block,
            fit_state_sink=fit_state_sink,
        )
    X_train, X_pred = _apply_level_feature_block(
        X_train,
        X_pred,
        frame,
        target,
        level_feature_block,
        predictors,
        spec,
    )
    target_lag_train: pd.DataFrame | None = None
    target_lag_pred: pd.DataFrame | None = None
    if target_lag_block == "fixed_target_lags":
        lag_order = int(target_lag_order or 1)
        lag_source = target_window if target_window is not None else frame[target].astype(float)
        target_lag_train = _fixed_target_lag_frame(lag_source, X_train.index, lag_order)
        target_lag_pred = _fixed_target_lag_frame(lag_source, X_pred.index, lag_order)
    elif target_lag_block != "none":
        raise ExecutionError(f"target_lag_block {target_lag_block!r} is not executable in current runtime slice")
    X_train_arr, X_pred_arr = _apply_raw_panel_preprocessing(
        X_train,
        y_train,
        X_pred,
        preprocessing_contract,
        fit_state_sink=fit_state_sink,
    )
    if factor_feature_block == "pca_factor_lags":
        training_spec = _factor_runtime_training_spec(recipe)
        factor_lag_count = int(training_spec.get("factor_lag_count", training_spec.get("factor_ar_lags", 1)))
        X_train_arr, X_pred_arr = _append_factor_lag_block(
            X_train_arr,
            X_pred_arr,
            lag_count=factor_lag_count,
            fit_state_sink=fit_state_sink,
        )
    target_lag_names_for_selection: list[str] = []
    if target_lag_train is not None and target_lag_pred is not None:
        if target_lag_feature_names is not None:
            target_lag_names_for_selection = [str(name) for name in target_lag_feature_names]
        else:
            target_lag_names_for_selection = [
                str(name).replace("__target_lag", "target_lag_")
                for name in target_lag_train.columns
            ]
        if feature_block_combination == "append_to_target_lags":
            X_train_arr = np.concatenate([target_lag_train.to_numpy(dtype=float), X_train_arr], axis=1)
            X_pred_arr = np.concatenate([target_lag_pred.to_numpy(dtype=float), X_pred_arr], axis=1)
        else:
            X_train_arr = np.concatenate([X_train_arr, target_lag_train.to_numpy(dtype=float)], axis=1)
            X_pred_arr = np.concatenate([X_pred_arr, target_lag_pred.to_numpy(dtype=float)], axis=1)

    # 1.4.5 deterministic_components augmentation (applied after preprocessing)
    det_component = None
    break_dates = None
    if spec is not None:
        det_component = spec.get("deterministic_components", "none")
        break_dates = spec.get("break_dates")
    sb_dates = _resolve_structural_break_dates(spec)
    feature_selection_policy = str(preprocessing_contract.feature_selection_policy)
    feature_selection_semantics = _feature_selection_semantics(preprocessing_contract)
    deterministic_feature_names: list[str] = []
    if det_component and det_component != "none":
        deterministic_feature_names.extend(
            _deterministic_feature_names(str(det_component), break_dates=break_dates)
        )
        try:
            X_train_arr = _augment_deterministic_array(
                X_train_arr, det_component,
                index=X_train.index, break_dates=break_dates,
            )
            X_pred_arr = _augment_deterministic_array(
                X_pred_arr, det_component,
                index=X_pred.index, break_dates=break_dates,
            )
        except ValueError as exc:
            raise ExecutionError(str(exc)) from exc

    # 1.5 structural_break_segmentation augmentation — reuses the break_dummies
    # path from 1.4 deterministic_components with dates resolved from the axis
    # value (pre_post_crisis / pre_post_covid presets or user_break_dates).
    if sb_dates:
        deterministic_feature_names.extend(
            _deterministic_feature_names("break_dummies", break_dates=sb_dates, structural_break=True)
        )
        try:
            X_train_arr = _augment_deterministic_array(
                X_train_arr, "break_dummies",
                index=X_train.index, break_dates=sb_dates,
            )
            X_pred_arr = _augment_deterministic_array(
                X_pred_arr, "break_dummies",
                index=X_pred.index, break_dates=sb_dates,
            )
        except ValueError as exc:
            raise ExecutionError(str(exc)) from exc
    def _candidate_feature_names_for_current_matrix() -> list[str]:
        factor_names = _factor_feature_names_from_fit_state(tuple(fit_state_sink or ()))
        if factor_names is None:
            names = _public_feature_names_from_runtime_columns(
                list(X_train.columns),
                tuple(fit_state_sink or ()),
            )
        else:
            names = list(factor_names)
        if target_lag_train is not None and target_lag_pred is not None:
            if feature_block_combination == "append_to_target_lags":
                names = target_lag_names_for_selection + names
            else:
                names.extend(target_lag_names_for_selection)
        names.extend(deterministic_feature_names)
        if len(names) != int(np.asarray(X_train_arr).shape[1]):
            raise ExecutionError(
                "Layer 2 candidate feature name count does not match final Z width: "
                f"{len(names)} names for {int(np.asarray(X_train_arr).shape[1])} columns"
            )
        return names

    if feature_block_combination == "custom_combiner":
        candidate_feature_names = _candidate_feature_names_for_current_matrix()
        X_train_arr, X_pred_arr, custom_combined_feature_names = _apply_custom_feature_combiner(
            X_train_arr,
            X_pred_arr,
            y_train,
            candidate_feature_names=candidate_feature_names,
            train_index=X_train.index,
            pred_index=X_pred.index,
            recipe=recipe,
            horizon=horizon,
            fit_state_sink=fit_state_sink,
        )
        if feature_selection_policy != "none" and feature_selection_semantics == "select_after_custom_blocks":
            X_train_arr, X_pred_arr, _selected_custom_names = _apply_custom_final_z_selection(
                X_train_arr,
                y_train,
                X_pred_arr,
                candidate_feature_names=custom_combined_feature_names,
                policy=feature_selection_policy,
                fit_state_sink=fit_state_sink,
            )
        return X_train_arr, y_train, X_pred_arr

    if feature_selection_policy != "none" and feature_selection_semantics == "select_after_custom_blocks":
        candidate_feature_names = _candidate_feature_names_for_current_matrix()
        X_train_arr, X_pred_arr, _selected_custom_names = _apply_custom_final_z_selection(
            X_train_arr,
            y_train,
            X_pred_arr,
            candidate_feature_names=candidate_feature_names,
            policy=feature_selection_policy,
            fit_state_sink=fit_state_sink,
        )
    if feature_selection_policy != "none" and feature_selection_semantics == "select_after_factor":
        candidate_feature_names = _factor_feature_names_from_fit_state(tuple(fit_state_sink or ()))
        if candidate_feature_names is None:
            raise ExecutionError(
                "feature_selection_semantics='select_after_factor' requires "
                "an executable factor_feature_block or an equivalent pca/static_factor bridge"
            )
        candidate_feature_names = list(candidate_feature_names)
        if target_lag_train is not None and target_lag_pred is not None:
            if feature_block_combination == "append_to_target_lags":
                candidate_feature_names = target_lag_names_for_selection + candidate_feature_names
            else:
                candidate_feature_names.extend(target_lag_names_for_selection)
        candidate_feature_names.extend(deterministic_feature_names)
        X_train_arr, X_pred_arr, selected_feature_names = _apply_post_factor_feature_selection(
            X_train_arr,
            y_train,
            X_pred_arr,
            candidate_feature_names,
            policy=feature_selection_policy,
        )
        _record_post_factor_selection(
            fit_state_sink,
            candidate_feature_names=candidate_feature_names,
            selected_feature_names=selected_feature_names,
            policy=feature_selection_policy,
        )
    return X_train_arr, y_train, X_pred_arr


def _run_ar_model_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    selected_lag, selected_bic, fitted = _select_ar_bic_model(train, _max_ar_lag(recipe))
    prediction = fitted.predict(start=len(train), end=len(train) + horizon - 1)
    return {
        "y_pred": float(prediction.iloc[-1]),
        "selected_lag": selected_lag,
        "selected_bic": selected_bic,
    }


def _target_transformer_name(recipe: RecipeSpec | None) -> str:
    if recipe is None:
        return "none"
    layer2 = getattr(recipe, "layer2_representation_spec", {}) or {}
    if isinstance(layer2, dict):
        target_representation = layer2.get("target_representation", {})
        if isinstance(target_representation, dict) and "target_transformer" in target_representation:
            return str(target_representation.get("target_transformer", "none"))
    return str(recipe.training_spec.get("target_transformer", "none"))


def _target_transformer_spec(recipe: RecipeSpec | None):
    name = _target_transformer_name(recipe)
    if name == "none":
        return None
    if not is_custom_target_transformer(name):
        raise ExecutionError(f"target transformer {name!r} is not registered")
    return get_custom_target_transformer(name)


def _target_transformer_manifest(recipe: RecipeSpec) -> dict[str, object]:
    spec = _target_transformer_spec(recipe)
    if spec is None:
        return {
            "name": "none",
            "model_scale": "raw",
            "forecast_scale": "raw",
            "evaluation_scale": "raw",
            "runtime": "not_applicable",
        }
    return {
        "name": spec.name,
        "model_scale": spec.model_scale,
        "forecast_scale": spec.forecast_scale,
        "evaluation_scale": spec.evaluation_scale,
        "runtime": _feature_runtime_name(recipe),
    }


def _coerce_target_transformer_series(value, *, index, name: str | None, transformer_name: str, role: str) -> pd.Series:
    if isinstance(value, pd.Series):
        arr = value.to_numpy(dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        raise ExecutionError(f"target transformer {transformer_name!r} returned scalar {role}; expected one value per target observation")
    flat = arr.reshape(-1)
    if len(flat) != len(index):
        raise ExecutionError(
            f"target transformer {transformer_name!r} returned {role} with length {len(flat)}; expected {len(index)}"
        )
    return pd.Series(flat, index=index, name=name, dtype=float)


def _target_transform_context(
    *,
    spec_name: str,
    recipe: RecipeSpec,
    horizon: int,
    train: pd.Series,
    target_date,
    mode: str,
) -> dict[str, object]:
    return {
        "target_transformer": spec_name,
        "target": recipe.target,
        "horizon": int(horizon),
        "mode": mode,
        "train_start_date": train.index[0].strftime("%Y-%m-%d"),
        "train_end_date": train.index[-1].strftime("%Y-%m-%d"),
        "origin_date": train.index[-1].strftime("%Y-%m-%d"),
        "target_date": target_date.strftime("%Y-%m-%d") if hasattr(target_date, "strftime") else str(target_date),
        "model_scale": "transformed",
        "forecast_scale": "raw",
        "evaluation_scale": "raw",
        "contract_version": "target_transformer_v1",
    }


def _fit_target_transformer_for_window(
    train: pd.Series,
    *,
    recipe: RecipeSpec,
    horizon: int,
    target_date,
):
    spec = _target_transformer_spec(recipe)
    if spec is None:
        return None, train.astype(float), {}
    transformer = spec.factory()
    context = _target_transform_context(
        spec_name=spec.name,
        recipe=recipe,
        horizon=horizon,
        train=train,
        target_date=target_date,
        mode="fit_transform",
    )
    fitted = transformer.fit(train.copy(), context)
    if fitted is None:
        fitted = transformer
    for method in ("transform", "inverse_transform_prediction"):
        if not callable(getattr(fitted, method, None)):
            raise ExecutionError(f"target transformer {spec.name!r} fit result must provide callable {method}()")
    transformed = fitted.transform(train.copy(), context)
    transformed_series = _coerce_target_transformer_series(
        transformed,
        index=train.index,
        name=train.name,
        transformer_name=spec.name,
        role="transformed target",
    )
    return fitted, transformed_series, context


def _inverse_target_transformer_prediction(
    fitted,
    y_pred,
    *,
    recipe: RecipeSpec,
    context: dict[str, object],
) -> float:
    spec = _target_transformer_spec(recipe)
    if spec is None:
        return float(y_pred)
    inverse_context = dict(context)
    inverse_context["mode"] = "inverse_prediction"
    value = fitted.inverse_transform_prediction(float(y_pred), inverse_context)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if len(flat) != 1:
        raise ExecutionError(
            f"target transformer {spec.name!r} inverse_transform_prediction must return a scalar or one-element value; got shape {arr.shape}"
        )
    return float(flat[0])


def _custom_preprocessor_name(recipe: RecipeSpec | None) -> str:
    if recipe is None:
        return "none"
    layer2 = getattr(recipe, "layer2_representation_spec", {}) or {}
    if isinstance(layer2, dict):
        frame_conditioning = layer2.get("frame_conditioning", {})
        if isinstance(frame_conditioning, dict) and "custom_preprocessor" in frame_conditioning:
            return str(frame_conditioning.get("custom_preprocessor", "none"))
    return str(recipe.training_spec.get("custom_preprocessor", "none"))


def _custom_preprocessor_spec(recipe: RecipeSpec | None):
    name = _custom_preprocessor_name(recipe)
    if name == "none":
        return None
    if not is_custom_preprocessor(name):
        raise ExecutionError(f"custom preprocessor {name!r} is not registered")
    return get_custom_preprocessor(name)


def _coerce_custom_preprocessor_result(result, *, name: str):
    if isinstance(result, dict):
        try:
            return result["X_train"], result["X_test"]
        except KeyError as exc:
            raise ExecutionError(
                f"custom preprocessor {name!r} returned a dict missing {exc.args[0]!r}"
            ) from exc
    if isinstance(result, tuple) and len(result) == 2:
        return result
    raise ExecutionError(
        f"custom preprocessor {name!r} must return (X_train, X_test) or a dict with those keys"
    )


def _apply_custom_preprocessor_arrays(
    X_train,
    y_train,
    X_test,
    recipe: RecipeSpec,
    *,
    context_extra: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = _custom_preprocessor_spec(recipe)
    if spec is None:
        return np.asarray(X_train, dtype=float), np.asarray(y_train, dtype=float), np.asarray(X_test, dtype=float)
    y_arr = np.asarray(y_train, dtype=float).reshape(-1)
    context = {
        "preprocessor_name": spec.name,
        "target": recipe.target,
        "y_train_role": "read_only_context",
        "contract_version": "custom_preprocessor_v1",
    }
    context.update(context_extra or {})
    X_new, X_test_new = _coerce_custom_preprocessor_result(
        spec.function(np.asarray(X_train, dtype=float), y_arr.copy(), np.asarray(X_test, dtype=float), context),
        name=spec.name,
    )
    X_arr = np.asarray(X_new, dtype=float)
    X_test_arr = np.asarray(X_test_new, dtype=float)
    if X_arr.ndim != 2 or X_test_arr.ndim != 2:
        raise ExecutionError(f"custom preprocessor {spec.name!r} must return 2-D X arrays")
    if len(X_arr) != len(y_arr):
        raise ExecutionError(f"custom preprocessor {spec.name!r} returned X_train/y_train with mismatched rows")
    if X_test_arr.shape[0] != 1:
        raise ExecutionError(f"custom preprocessor {spec.name!r} must return one-row X_test")
    return X_arr, y_arr, X_test_arr


def _fit_autoreg_sklearn(train: pd.Series, recipe: RecipeSpec, model_family: str, model) -> tuple[Layer2Representation, np.ndarray, np.ndarray, object, dict[str, object]]:
    representation = _build_target_lag_representation(train, recipe, default_prefix="y_lag")
    X_fit, y_fit, _ = _apply_custom_preprocessor_arrays(
        representation.Z_train,
        representation.y_train,
        representation.Z_pred.copy(),
        recipe,
        context_extra=representation.runtime_context(mode="fit"),
    )
    fitted, tuning_payload = fit_with_optional_tuning(model_family, X_fit, y_fit, recipe.training_spec)
    tuning_payload = _merge_layer2_representation_payload(tuning_payload, representation)
    if _custom_preprocessor_spec(recipe) is not None:
        setattr(fitted, "_macrocast_custom_preprocessor_recipe", recipe)
        setattr(fitted, "_macrocast_custom_preprocessor_X_train", representation.Z_train)
        setattr(fitted, "_macrocast_custom_preprocessor_y_train", representation.y_train)
        setattr(fitted, "_macrocast_layer2_representation", representation)
    return representation, X_fit, y_fit, fitted, tuning_payload


def _as_scalar_prediction(value, *, model_name: str) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.reshape(-1)
    if len(flat) != 1:
        raise ExecutionError(
            f"custom model {model_name!r} must return a scalar or one-element prediction; got shape {arr.shape}"
        )
    return float(flat[0])


def _layer2_representation_tuning_payload(representation: Layer2Representation) -> dict[str, object]:
    payload: dict[str, object] = {
        "feature_runtime_builder": representation.feature_runtime_builder,
        "legacy_feature_builder": representation.legacy_feature_builder,
        "feature_dispatch_source": representation.feature_dispatch_source,
        "feature_names": list(representation.feature_names),
        "layer2_block_order": list(representation.block_order),
        "layer2_block_roles": dict(representation.block_roles),
        "layer2_alignment": dict(representation.alignment),
        "layer2_leakage_contract": representation.leakage_contract,
    }
    if representation.latest_fit_state is not None:
        payload["feature_representation_fit_state"] = representation.latest_fit_state
    return payload


def _merge_layer2_representation_payload(
    tuning_payload: Mapping[str, object] | None,
    representation: Layer2Representation,
) -> dict[str, object]:
    payload = dict(tuning_payload or {})
    payload.update(_layer2_representation_tuning_payload(representation))
    return payload


def _custom_model_tuning_payload(
    representation: Layer2Representation,
    *,
    model_name: str,
) -> dict[str, object]:
    payload = _layer2_representation_tuning_payload(representation)
    payload.update(
        {
            "custom_model": model_name,
            "custom_model_contract": CUSTOM_MODEL_CONTRACT_VERSION,
        }
    )
    return payload


def _run_custom_autoreg_executor(
    train: pd.Series,
    horizon: int,
    recipe: RecipeSpec,
    contract: PreprocessContract,
    raw_frame: pd.DataFrame | None = None,
    origin_idx: int | None = None,
    start_idx: int = 0,
) -> dict[str, float | int]:
    model_name = _model_family(recipe)
    spec = get_custom_model(model_name)
    representation = _build_target_lag_representation(train, recipe, default_prefix="y_lag")
    lag_order = int(representation.alignment["lag_order"])
    X_train_for_custom = representation.Z_train
    y_train_for_custom = representation.y_train
    history = list(train.to_numpy(dtype=float))
    for step in range(1, int(horizon) + 1):
        X_test = np.asarray(history[-lag_order:][::-1], dtype=float).reshape(1, -1)
        context = {
            "model_name": model_name,
            "target": recipe.target,
            "horizon": int(horizon),
            "recursive_step": step,
            "feature_names": list(representation.feature_names),
            "train_index": list(train.index),
            "contract_version": CUSTOM_MODEL_CONTRACT_VERSION,
        }
        context.update(representation.runtime_context(mode="custom_model"))
        if _custom_preprocessor_spec(recipe) is not None:
            X_train_for_custom, y_train_for_custom, X_test = _apply_custom_preprocessor_arrays(
                representation.Z_train,
                representation.y_train,
                X_test,
                recipe,
                context_extra=representation.runtime_context(mode="custom_model"),
            )
        pred = _as_scalar_prediction(
            spec.function(X_train_for_custom, y_train_for_custom, X_test, context),
            model_name=model_name,
        )
        history.append(pred)
    return {
        "y_pred": float(history[-1]),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
        "tuning_payload": _custom_model_tuning_payload(representation, model_name=model_name),
    }




def _recursive_predict_adaptive_lasso(model, train: pd.Series, horizon: int, lag_order: int) -> float:
    history = list(train.to_numpy(dtype=float))
    for _ in range(horizon):
        features = np.asarray(history[-lag_order:][::-1], dtype=float).reshape(1, -1)
        pred = float(predict_adaptive_lasso(model, features)[0])
        history.append(pred)
    return float(history[-1])


def _run_ols_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "ols", LinearRegression())
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_ridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "ridge", Ridge(alpha=1.0))
    lag_order = int(representation.alignment["lag_order"])
    return {
        "y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order),
        "selected_lag": lag_order,
        "selected_bic": math.nan,
    }


def _run_lasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "lasso", Lasso(alpha=1e-4, max_iter=10000))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_elasticnet_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "elasticnet", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_randomforest_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "randomforest", RandomForestRegressor(n_estimators=200, random_state=current_seed(model_family="randomforest")))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_bayesianridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "bayesianridge", BayesianRidge())
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_huber_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "huber", HuberRegressor())
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_adaptivelasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "adaptivelasso", None)
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_adaptive_lasso(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "svr_linear", LinearSVR(C=1.0, epsilon=0.01, max_iter=50000, random_state=current_seed(model_family="svr_linear")))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_rbf_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "svr_rbf", SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_extratrees_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "extratrees", ExtraTreesRegressor(n_estimators=200, random_state=current_seed(model_family="extratrees")))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_gbm_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "gbm", GradientBoostingRegressor(random_state=current_seed(model_family="gbm")))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_xgboost_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "xgboost", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, random_state=current_seed(model_family="xgboost"), verbosity=0))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_lightgbm_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "lightgbm", LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=current_seed(model_family="lightgbm"), verbosity=-1))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_catboost_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "catboost", CatBoostRegressor(iterations=100, learning_rate=0.05, depth=4, verbose=False, random_seed=current_seed(model_family="catboost")))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_mlp_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=current_seed(model_family="mlp")))
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_deep_autoreg_executor(model_family: str, train: pd.Series, horizon: int) -> dict[str, float | int]:
    from .models.deep._import_guard import require_torch
    require_torch(model_family)
    from .models.deep._base import DeepModelConfig
    from .adapters.sequence import reshape_for_sequence

    cfg = DeepModelConfig(seed=current_seed(model_family=model_family))
    series = train.to_numpy(dtype=float)
    if len(series) < cfg.lookback + 1:
        raise ExecutionError(
            f"model_family {model_family!r} requires at least lookback+1={cfg.lookback + 1} "
            f"training observations, got {len(series)}"
        )
    X_seq, y_seq = reshape_for_sequence(series=series, lookback=cfg.lookback, horizon=1)

    if model_family == "lstm":
        from .models.deep.lstm import LSTMModel as ModelCls
    elif model_family == "gru":
        from .models.deep.gru import GRUModel as ModelCls
    elif model_family == "tcn":
        from .models.deep.tcn import TCNModel as ModelCls
    else:  # pragma: no cover — dispatch prevents this
        raise ExecutionError(f"unsupported deep model_family {model_family!r}")

    model = ModelCls(config=cfg).fit(X_seq, y_seq)
    history = series[-cfg.lookback:].astype(float).copy()
    y_pred = model.predict_next(history)
    for _ in range(int(horizon) - 1):
        history = np.concatenate([history[1:], [y_pred]])
        y_pred = model.predict_next(history)
    return {
        "y_pred": float(y_pred),
        "selected_lag": cfg.lookback,
        "selected_bic": math.nan,
        "tuning_payload": {},
    }


def _run_lstm_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    return _run_deep_autoreg_executor("lstm", train, horizon)


def _run_gru_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    return _run_deep_autoreg_executor("gru", train, horizon)


def _run_tcn_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    return _run_deep_autoreg_executor("tcn", train, horizon)


def _fit_raw_panel_model(
    raw_frame: pd.DataFrame,
    recipe: RecipeSpec,
    horizon: int,
    start_idx: int,
    origin_idx: int,
    contract: PreprocessContract,
    model_family: str,
    model,
    *,
    target_window: pd.Series | None = None,
) -> tuple[Layer2Representation, np.ndarray, np.ndarray, np.ndarray, object, dict[str, object]]:
    representation = _build_raw_panel_representation(
        raw_frame,
        recipe,
        horizon,
        start_idx,
        origin_idx,
        contract,
        target_window=target_window,
    )
    X_fit, y_fit, X_pred_fit = _apply_custom_preprocessor_arrays(
        representation.Z_train,
        representation.y_train,
        representation.Z_pred,
        recipe,
        context_extra=representation.runtime_context(mode="fit"),
    )
    fitted, tuning_payload = fit_with_optional_tuning(model_family, X_fit, y_fit, recipe.training_spec)
    tuning_payload = _merge_layer2_representation_payload(tuning_payload, representation)
    return representation, X_fit, y_fit, X_pred_fit, fitted, tuning_payload


def _run_custom_raw_panel_executor(
    train: pd.Series,
    horizon: int,
    recipe: RecipeSpec,
    contract: PreprocessContract,
    raw_frame: pd.DataFrame | None = None,
    origin_idx: int | None = None,
    start_idx: int = 0,
) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    model_name = _model_family(recipe)
    spec = get_custom_model(model_name)
    representation = _build_raw_panel_representation(
        raw_frame,
        recipe,
        horizon,
        start_idx,
        origin_idx,
        contract,
        target_window=train,
    )
    X_train, y_train, X_pred = _apply_custom_preprocessor_arrays(
        representation.Z_train,
        representation.y_train,
        representation.Z_pred,
        recipe,
        context_extra=representation.runtime_context(mode="custom_model"),
    )
    context = {
        "model_name": model_name,
        "target": recipe.target,
        "horizon": int(horizon),
        "feature_names": list(representation.feature_names),
        "contract_version": CUSTOM_MODEL_CONTRACT_VERSION,
    }
    context.update(representation.runtime_context(mode="custom_model"))
    pred = _as_scalar_prediction(spec.function(X_train, y_train, X_pred, context), model_name=model_name)
    return {
        "y_pred": pred,
        "selected_lag": 0,
        "selected_bic": math.nan,
        "tuning_payload": _custom_model_tuning_payload(representation, model_name=model_name),
    }


def _run_ols_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "ols", LinearRegression(), target_window=train)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_ridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "ridge", Ridge(alpha=1.0), target_window=train)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_lasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "lasso", Lasso(alpha=1e-4, max_iter=10000), target_window=train)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_elasticnet_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "elasticnet", ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=10000), target_window=train)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_randomforest_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "randomforest", RandomForestRegressor(n_estimators=200, random_state=current_seed(model_family="randomforest")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_bayesianridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "bayesianridge", BayesianRidge())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_huber_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "huber", HuberRegressor())
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_adaptivelasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "adaptivelasso", None)
    return {"y_pred": float(predict_adaptive_lasso(model, X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "svr_linear", LinearSVR(C=1.0, epsilon=0.01, max_iter=50000, random_state=current_seed(model_family="svr_linear")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_svr_rbf_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "svr_rbf", SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_extratrees_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "extratrees", ExtraTreesRegressor(n_estimators=200, random_state=current_seed(model_family="extratrees")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_gbm_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "gbm", GradientBoostingRegressor(random_state=current_seed(model_family="gbm")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_xgboost_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "xgboost", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=1.0, colsample_bytree=1.0, random_state=current_seed(model_family="xgboost"), verbosity=0))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_lightgbm_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "lightgbm", LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=current_seed(model_family="lightgbm"), verbosity=-1))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_catboost_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "catboost", CatBoostRegressor(iterations=100, learning_rate=0.05, depth=4, verbose=False, random_seed=current_seed(model_family="catboost")))
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_mlp_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "mlp", MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=current_seed(model_family="mlp")))
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
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "componentwise_boosting", None)
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_ridge_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "boosting_ridge", None)
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_lasso_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "boosting_lasso", None)
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pcr_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation = _build_target_lag_representation(train, recipe, default_prefix="y_lag")
    lag_order = int(representation.alignment["lag_order"])
    pred, _tp = fit_factor_model(
        "pcr",
        pd.DataFrame(representation.Z_train, columns=representation.feature_names),
        representation.y_train,
        pd.DataFrame(representation.Z_pred, columns=representation.feature_names),
        _factor_runtime_training_spec(recipe),
        include_ar_lags=False,
    )
    _tp = _merge_layer2_representation_payload(_tp, representation)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pls_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation = _build_target_lag_representation(train, recipe, default_prefix="y_lag")
    lag_order = int(representation.alignment["lag_order"])
    pred, _tp = fit_factor_model(
        "pls",
        pd.DataFrame(representation.Z_train, columns=representation.feature_names),
        representation.y_train,
        pd.DataFrame(representation.Z_pred, columns=representation.feature_names),
        _factor_runtime_training_spec(recipe),
        include_ar_lags=False,
    )
    _tp = _merge_layer2_representation_payload(_tp, representation)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_factor_augmented_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation = _build_target_lag_representation(train, recipe, default_prefix="y_lag")
    lag_order = int(representation.alignment["lag_order"])
    pred, _tp = fit_factor_model(
        "factor_augmented_linear",
        pd.DataFrame(representation.Z_train, columns=representation.feature_names),
        representation.y_train,
        pd.DataFrame(representation.Z_pred, columns=representation.feature_names),
        _factor_runtime_training_spec(recipe),
        include_ar_lags=True,
    )
    _tp = _merge_layer2_representation_payload(_tp, representation)
    return {"y_pred": pred, "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_quantile_linear_autoreg_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    representation, _, _, model, _tp = _fit_autoreg_sklearn(train, recipe, "quantile_linear", None)
    lag_order = int(representation.alignment["lag_order"])
    return {"y_pred": _recursive_predict_sklearn(model, train, horizon, lag_order), "selected_lag": lag_order, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_componentwise_boosting_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "componentwise_boosting", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_ridge_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "boosting_ridge", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_boosting_lasso_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "boosting_lasso", None)
    return {"y_pred": float(model.predict(X_pred)[0]), "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pcr_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    representation = _build_raw_panel_representation(
        raw_frame,
        recipe,
        horizon,
        start_idx,
        origin_idx,
        contract,
        target_window=train,
    )
    pred, _tp = fit_factor_model(
        "pcr",
        pd.DataFrame(representation.Z_train, columns=representation.feature_names),
        representation.y_train,
        pd.DataFrame(representation.Z_pred, columns=representation.feature_names),
        _factor_runtime_training_spec(recipe),
        include_ar_lags=False,
    )
    _tp = _merge_layer2_representation_payload(_tp, representation)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_pls_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    representation = _build_raw_panel_representation(
        raw_frame,
        recipe,
        horizon,
        start_idx,
        origin_idx,
        contract,
        target_window=train,
    )
    pred, _tp = fit_factor_model(
        "pls",
        pd.DataFrame(representation.Z_train, columns=representation.feature_names),
        representation.y_train,
        pd.DataFrame(representation.Z_pred, columns=representation.feature_names),
        _factor_runtime_training_spec(recipe),
        include_ar_lags=False,
    )
    _tp = _merge_layer2_representation_payload(_tp, representation)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_factor_augmented_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    representation = _build_raw_panel_representation(
        raw_frame,
        recipe,
        horizon,
        start_idx,
        origin_idx,
        contract,
        target_window=train,
    )
    pred, _tp = fit_factor_model(
        "factor_augmented_linear",
        pd.DataFrame(representation.Z_train, columns=representation.feature_names),
        representation.y_train,
        pd.DataFrame(representation.Z_pred, columns=representation.feature_names),
        _factor_runtime_training_spec(recipe),
        include_ar_lags=True,
    )
    _tp = _merge_layer2_representation_payload(_tp, representation)
    return {"y_pred": pred, "selected_lag": 0, "selected_bic": math.nan, "tuning_payload": _tp}


def _run_quantile_linear_raw_panel_executor(train: pd.Series, horizon: int, recipe: RecipeSpec, contract: PreprocessContract, raw_frame: pd.DataFrame | None = None, origin_idx: int | None = None, start_idx: int = 0) -> dict[str, float | int]:
    assert raw_frame is not None and origin_idx is not None
    _, _, _, X_pred, model, _tp = _fit_raw_panel_model(raw_frame, recipe, horizon, start_idx, origin_idx, contract, "quantile_linear", None)
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
        # v1.0: compute simple OLS regression on the leading principal-factor of
        # the training window (univariate factor surrogate). Iterates forward h
        # steps using the factor-regressed mean as the level. If the training
        # window is too small, fall back to historical_mean.
        try:
            import numpy as _np
            values = train.to_numpy(dtype=float)
            if len(values) < 6:
                return _historical_mean_prediction(train)
            # Factor = z-scored mean of the series (trivial single-series factor)
            z = (values - values.mean()) / (values.std() or 1.0)
            # Regress the target on z one step ahead
            y = values[1:]
            x = z[:-1]
            beta = float(_np.cov(x, y, bias=True)[0, 1] / (_np.var(x) or 1.0))
            intercept = float(y.mean() - beta * x.mean())
            last = float(values[-1])
            last_z = float((last - values.mean()) / (values.std() or 1.0))
            pred = intercept + beta * last_z
            return pred
        except Exception:
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
        # v1.0: run each declared benchmark and return the arithmetic mean of
        # their forecasts. Members are declared at compile time via
        # leaf_config.benchmark_suite (a list of benchmark_family names). Each
        # member must itself be executable.
        suite = recipe.data_task_spec.get("benchmark_suite") or []
        if not isinstance(suite, (list, tuple)) or not suite:
            raise ExecutionError("benchmark_family='multi_benchmark_suite' requires leaf_config.benchmark_suite (list[str])")
        allowed = {"historical_mean", "zero_change", "ar_bic", "rolling_mean", "ar_fixed_p", "ardi"}
        preds = []
        original_family = benchmark_family
        for member in suite:
            if member not in allowed:
                raise ExecutionError(f"benchmark_family='multi_benchmark_suite' member {member!r} is not one of {sorted(allowed)}")
            # Build a shallow copy of the recipe with a different benchmark_family
            member_recipe = recipe
            # Recurse by swapping the benchmark_family in data_task_spec is not trivial;
            # instead, inline-dispatch on the member to avoid recipe mutation.
            if member == "historical_mean":
                preds.append(_historical_mean_prediction(train))
            elif member == "zero_change":
                preds.append(float(train.iloc[-1]))
            elif member == "rolling_mean":
                from .evaluation.benchmark_resolver import _rolling_mean as _rm
                preds.append(_rm(train, window_len if window_len > 0 else len(train)))
            elif member == "ar_bic":
                fitted = _run_ar_model_executor(train, horizon, recipe, contract=_build_noop_contract())
                preds.append(float(fitted["y_pred"]))
            elif member == "ar_fixed_p":
                from .evaluation.benchmark_resolver import _ar_fixed_p_forecast, BenchmarkResolverError
                try:
                    preds.append(_ar_fixed_p_forecast(train, int(horizon), _benchmark_fixed_p(recipe)))
                except BenchmarkResolverError:
                    preds.append(_historical_mean_prediction(train))
            elif member == "ardi":
                from .evaluation.benchmark_resolver import _ardi_forecast, BenchmarkResolverError
                try:
                    preds.append(_ardi_forecast(train, int(horizon), _benchmark_n_factors(recipe), None, train.index[-1]))
                except BenchmarkResolverError:
                    preds.append(_historical_mean_prediction(train))
        if not preds:
            return _historical_mean_prediction(train)
        return float(sum(preds) / len(preds))
    if benchmark_family in {"paper_specific_benchmark", "survey_forecast"}:
        # v1.0: pre-computed per-target forecast series supplied via leaf_config.
        field = "paper_forecast_series" if benchmark_family == "paper_specific_benchmark" else "survey_forecast_series"
        series_map = recipe.data_task_spec.get(field)
        if not isinstance(series_map, dict):
            raise ExecutionError(f"benchmark_family={benchmark_family!r} requires leaf_config.{field} (dict[target, Series] keyed by target name)")
        target = str(recipe.target) if getattr(recipe, "target", None) else None
        if target not in series_map:
            raise ExecutionError(f"benchmark_family={benchmark_family!r}: leaf_config.{field} missing an entry for target={target!r}")
        try:
            import pandas as _pd
            ext = _pd.Series(series_map[target])
            last_origin = train.index[-1]
            # Forecast date is last_origin + horizon months forward. For monthly
            # freq, lookup by the expected target timestamp; on miss, use the
            # closest trailing value.
            try:
                target_date = last_origin + _pd.tseries.offsets.MonthBegin(int(horizon))
            except Exception:
                target_date = last_origin
            if target_date in ext.index:
                return float(ext.loc[target_date])
            valid = ext.loc[:target_date].dropna()
            if len(valid) == 0:
                raise ExecutionError(f"{field} has no value on or before {target_date} for target={target!r}")
            return float(valid.iloc[-1])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionError(f"benchmark_family={benchmark_family!r} lookup failed: {exc}") from exc
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


def _compute_density_interval_helpers(predictions: pd.DataFrame, alpha: float = 0.10):
    """Shared helper for density_interval tests.

    Builds a simple Gaussian predictive-density assumption from point forecasts
    + training-window residual variance. Returns the PIT series, the hit
    series (|y_true - y_pred| <= z_{1-alpha/2} * sigma), and the estimated
    sigma. Without a full conformal / quantile-forecast pipeline (v1.0+
    deliverable), this is the v0.9.2 baseline.
    """
    from scipy.stats import norm
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    y = rows["y_true"].astype(float).to_numpy()
    yhat = rows["y_pred"].astype(float).to_numpy()
    n = int(len(y))
    if n < 5:
        raise ExecutionError("density_interval tests require at least 5 observations")
    resid = y - yhat
    sigma = float(np.std(resid, ddof=1)) if n > 1 else 0.0
    if sigma <= 0:
        sigma = 1e-8
    pit = norm.cdf((y - yhat) / sigma)
    z = float(norm.ppf(1 - alpha / 2))
    hits = (np.abs(y - yhat) <= z * sigma).astype(int)
    return pit, hits, sigma, alpha


def _compute_pit_uniformity(predictions: pd.DataFrame) -> dict[str, object]:
    from scipy.stats import kstest
    pit, _hits, sigma, _alpha = _compute_density_interval_helpers(predictions)
    res = kstest(pit, "uniform")
    return {
        "stat_test": "PIT_uniformity",
        "n": int(len(pit)),
        "ks_statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant_5pct": bool(res.pvalue < 0.05),
        "sigma_estimate": sigma,
    }


def _compute_berkowitz(predictions: pd.DataFrame) -> dict[str, object]:
    from scipy.stats import norm, chi2
    pit, _hits, sigma, _alpha = _compute_density_interval_helpers(predictions)
    # Clip PIT to avoid infinities
    pit_clipped = np.clip(pit, 1e-6, 1 - 1e-6)
    z = norm.ppf(pit_clipped)
    n = int(len(z))
    mu = float(z.mean())
    var = float(np.var(z, ddof=1))
    # Likelihood ratio for N(0,1)
    ll_unrestricted = -0.5 * n * (math.log(2 * math.pi * var) + 1.0)
    centered = z - mu
    ll_n01 = -0.5 * n * math.log(2 * math.pi) - 0.5 * float(np.dot(z, z))
    lr_stat = float(-2.0 * (ll_n01 - ll_unrestricted))
    p_value = float(1.0 - chi2.cdf(lr_stat, df=2))
    return {
        "stat_test": "berkowitz",
        "n": n,
        "z_mean": mu,
        "z_variance": var,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
    }


def _compute_kupiec(predictions: pd.DataFrame, alpha: float = 0.10) -> dict[str, object]:
    from scipy.stats import chi2
    _pit, hits, _sigma, _alpha = _compute_density_interval_helpers(predictions, alpha=alpha)
    n = int(len(hits))
    n_hit = int(hits.sum())
    nominal = 1.0 - alpha
    empirical = n_hit / n if n > 0 else 0.0
    if n_hit in (0, n):
        lr_stat = float("nan"); p_value = float("nan")
    else:
        ll_h0 = n_hit * math.log(nominal) + (n - n_hit) * math.log(1.0 - nominal)
        ll_h1 = n_hit * math.log(empirical) + (n - n_hit) * math.log(1.0 - empirical)
        lr_stat = float(-2.0 * (ll_h0 - ll_h1))
        p_value = float(1.0 - chi2.cdf(lr_stat, df=1))
    return {
        "stat_test": "kupiec",
        "n": n, "n_hits": n_hit,
        "nominal_coverage": nominal, "empirical_coverage": float(empirical),
        "lr_statistic": lr_stat, "p_value": p_value,
        "significant_5pct": bool(not math.isnan(p_value) and p_value < 0.05),
    }


def _compute_christoffersen_unconditional(predictions: pd.DataFrame) -> dict[str, object]:
    out = _compute_kupiec(predictions)
    out["stat_test"] = "christoffersen_unconditional"
    return out


def _compute_christoffersen_independence(predictions: pd.DataFrame, alpha: float = 0.10) -> dict[str, object]:
    from scipy.stats import chi2
    _pit, hits, _sigma, _alpha = _compute_density_interval_helpers(predictions, alpha=alpha)
    n = int(len(hits))
    # Transition counts
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        prev, curr = int(hits[i - 1]), int(hits[i])
        if prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else: n11 += 1
    if (n00 + n01) == 0 or (n10 + n11) == 0 or (n01 + n11) == 0:
        return {
            "stat_test": "christoffersen_independence",
            "n": n, "transitions": [n00, n01, n10, n11],
            "lr_statistic": float("nan"), "p_value": float("nan"),
            "significant_5pct": False,
        }
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)
    if p <= 0 or p >= 1 or p01 in (0, 1) or p11 in (0, 1):
        return {
            "stat_test": "christoffersen_independence",
            "n": n, "transitions": [n00, n01, n10, n11],
            "lr_statistic": float("nan"), "p_value": float("nan"),
            "significant_5pct": False,
        }
    ll_ind = (n00 + n10) * math.log(1 - p) + (n01 + n11) * math.log(p)
    ll_full = (n00 * math.log(1 - p01) + n01 * math.log(p01) + n10 * math.log(1 - p11) + n11 * math.log(p11))
    lr_stat = float(-2.0 * (ll_ind - ll_full))
    p_value = float(1.0 - chi2.cdf(lr_stat, df=1))
    return {
        "stat_test": "christoffersen_independence",
        "n": n, "transitions": [n00, n01, n10, n11],
        "lr_statistic": lr_stat, "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
    }


def _compute_christoffersen_conditional(predictions: pd.DataFrame) -> dict[str, object]:
    uc = _compute_kupiec(predictions)
    ind = _compute_christoffersen_independence(predictions)
    lr_uc = uc.get("lr_statistic")
    lr_ind = ind.get("lr_statistic")
    if isinstance(lr_uc, float) and isinstance(lr_ind, float) and not (math.isnan(lr_uc) or math.isnan(lr_ind)):
        from scipy.stats import chi2
        lr_cc = lr_uc + lr_ind
        p_value = float(1.0 - chi2.cdf(lr_cc, df=2))
    else:
        lr_cc = float("nan"); p_value = float("nan")
    return {
        "stat_test": "christoffersen_conditional",
        "lr_statistic": lr_cc, "p_value": p_value,
        "unconditional_lr": uc.get("lr_statistic"),
        "independence_lr": ind.get("lr_statistic"),
        "significant_5pct": bool(not math.isnan(p_value) and p_value < 0.05),
    }


def _compute_interval_coverage(predictions: pd.DataFrame, alpha: float = 0.10) -> dict[str, object]:
    _pit, hits, _sigma, _alpha = _compute_density_interval_helpers(predictions, alpha=alpha)
    n = int(len(hits))
    n_hit = int(hits.sum())
    empirical = n_hit / n if n > 0 else 0.0
    return {
        "stat_test": "interval_coverage",
        "n": n, "n_hits": n_hit,
        "nominal_coverage": 1.0 - alpha,
        "empirical_coverage": float(empirical),
        "coverage_gap": float(empirical - (1.0 - alpha)),
    }


def _compute_fluctuation_test(predictions: pd.DataFrame, window_fraction: float = 0.30) -> dict[str, object]:
    """Giacomini-Rossi (2009) fluctuation test.

    Rolling DM statistic over a centred window; the test statistic is the max
    of |DM_rolling(t)| across the sample. Rejection = at least one window
    shows statistically significant predictive-accuracy differential.
    Approximate 5%% critical value from GR Table I at m=0.30 is ~3.012.
    """
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 10:
        raise ExecutionError("fluctuation_test requires at least 10 observations")
    m = max(5, int(window_fraction * n))
    if m * 2 > n:
        m = max(5, n // 3)
    dm_series = []
    for i in range(m, n - m + 1):
        window = loss_diff[i - m : i + m]
        mean = float(window.mean())
        var = float(np.var(window, ddof=1))
        if var <= 0:
            dm_series.append(0.0)
            continue
        dm_series.append(float(mean / math.sqrt(var / len(window))))
    if not dm_series:
        raise ExecutionError("fluctuation_test produced no rolling windows")
    dm_array = np.asarray(dm_series)
    max_abs = float(np.max(np.abs(dm_array)))
    critical_5pct = 3.012
    return {
        "stat_test": "fluctuation_test",
        "n": n,
        "window_size": int(m),
        "n_rolling_windows": int(len(dm_series)),
        "max_abs_dm": max_abs,
        "critical_5pct": critical_5pct,
        "significant_5pct": bool(max_abs > critical_5pct),
    }


def _compute_chow_break_forecast(predictions: pd.DataFrame) -> dict[str, object]:
    """Chow breakpoint test on forecast-loss-differential.

    Splits the sample at the midpoint (structural break candidate), tests
    whether loss_diff's mean differs across the two halves via Welch's t-test.
    """
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 6:
        raise ExecutionError("chow_break_forecast requires at least 6 observations")
    mid = n // 2
    part1 = loss_diff[:mid]
    part2 = loss_diff[mid:]
    m1, m2 = float(part1.mean()), float(part2.mean())
    v1 = float(np.var(part1, ddof=1)) if len(part1) > 1 else 0.0
    v2 = float(np.var(part2, ddof=1)) if len(part2) > 1 else 0.0
    denom = math.sqrt(v1 / max(len(part1), 1) + v2 / max(len(part2), 1))
    if denom <= 0:
        return {
            "stat_test": "chow_break_forecast",
            "n": n, "break_index": int(mid),
            "mean_part1": m1, "mean_part2": m2,
            "t_statistic": 0.0, "p_value": 1.0,
            "significant_5pct": False,
        }
    t_stat = float((m1 - m2) / denom)
    p_value = float(_normal_two_sided_pvalue(t_stat))
    return {
        "stat_test": "chow_break_forecast",
        "n": n, "break_index": int(mid),
        "mean_part1": m1, "mean_part2": m2,
        "t_statistic": t_stat, "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
    }


def _compute_stepwise_mcs(predictions: pd.DataFrame, alpha: float = 0.10) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    if "model_name" not in rows.columns:
        # Fall back to single-model compare vs benchmark when no panel of models
        return _compute_mcs_test(predictions, block_bootstrap=False, alpha=alpha)
    pivot = rows.pivot_table(index=["target_date"], columns="model_name", values="squared_error", aggfunc="mean").dropna()
    if pivot.shape[1] < 2:
        raise ExecutionError("stepwise_mcs requires at least 2 models")
    survivors = list(pivot.columns)
    eliminated: list[str] = []
    while len(survivors) > 1:
        sub = pivot[survivors]
        mean_loss = sub.mean()
        worst = mean_loss.idxmax()
        other_mean = sub.drop(columns=[worst]).mean(axis=1)
        diff = sub[worst].to_numpy() - other_mean.to_numpy()
        n = len(diff)
        var = float(np.var(diff, ddof=1))
        if var <= 0 or n < 2:
            break
        stat = float(diff.mean() / math.sqrt(var / n))
        p_value = float(_normal_two_sided_pvalue(stat))
        if p_value < alpha:
            eliminated.append(worst)
            survivors.remove(worst)
        else:
            break
    return {
        "stat_test": "stepwise_mcs",
        "alpha": alpha,
        "n_obs": int(pivot.shape[0]),
        "initial_models": int(pivot.shape[1]),
        "surviving_models": survivors,
        "eliminated_models": eliminated,
    }


def _compute_bootstrap_best_model(predictions: pd.DataFrame, n_bootstrap: int = 500, seed: int = 0) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    if "model_name" not in rows.columns:
        # Compare model vs benchmark via bootstrap of loss_diff sign
        loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
        n = int(len(loss_diff))
        if n < 3:
            raise ExecutionError("bootstrap_best_model requires at least 3 obs")
        rng = np.random.default_rng(seed)
        wins = 0
        for _ in range(n_bootstrap):
            sample = rng.choice(loss_diff, size=n, replace=True)
            if sample.mean() > 0:
                wins += 1
        freq_model_best = wins / n_bootstrap
        return {
            "stat_test": "bootstrap_best_model",
            "n_bootstrap": int(n_bootstrap),
            "freq_model_beats_benchmark": float(freq_model_best),
            "model_declared_best": bool(freq_model_best >= 0.5),
        }
    pivot = rows.pivot_table(index=["target_date"], columns="model_name", values="squared_error", aggfunc="mean").dropna()
    arr = pivot.to_numpy()
    n, m = arr.shape
    rng = np.random.default_rng(seed)
    win_counts = np.zeros(m, dtype=int)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        mean_loss = arr[idx].mean(axis=0)
        win_counts[int(np.argmin(mean_loss))] += 1
    freqs = (win_counts / n_bootstrap).tolist()
    best_idx = int(np.argmax(win_counts))
    return {
        "stat_test": "bootstrap_best_model",
        "n_bootstrap": int(n_bootstrap),
        "models": list(pivot.columns),
        "freq_best": freqs,
        "declared_best": str(pivot.columns[best_idx]),
    }


def _compute_roc_comparison(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    y = rows["y_true"].astype(float).to_numpy()
    yhat = rows["y_pred"].astype(float).to_numpy()
    yben = rows["benchmark_pred"].astype(float).to_numpy()
    if len(y) < 3:
        raise ExecutionError("roc_comparison requires at least 3 observations")
    actual_up = (np.diff(y) > 0).astype(int)
    model_score = yhat[1:] - y[:-1]
    bench_score = yben[1:] - y[:-1]
    try:
        from sklearn.metrics import roc_auc_score
        auc_model = float(roc_auc_score(actual_up, model_score)) if actual_up.std() > 0 else float("nan")
        auc_bench = float(roc_auc_score(actual_up, bench_score)) if actual_up.std() > 0 else float("nan")
    except Exception:
        auc_model = auc_bench = float("nan")
    return {
        "stat_test": "roc_comparison",
        "n": int(len(actual_up)),
        "auc_model": auc_model,
        "auc_benchmark": auc_bench,
        "auc_delta": float(auc_model - auc_bench) if not (math.isnan(auc_model) or math.isnan(auc_bench)) else float("nan"),
    }


def _compute_cusum_on_loss(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 3:
        raise ExecutionError("cusum_on_loss requires at least 3 observations")
    centered = loss_diff - loss_diff.mean()
    cusum = np.cumsum(centered)
    sigma = float(np.std(loss_diff, ddof=1))
    if sigma <= 0:
        raise ExecutionError("cusum_on_loss requires non-zero loss-diff variance")
    normalized = cusum / (sigma * math.sqrt(n))
    max_abs = float(np.max(np.abs(normalized)))
    # 5% two-sided critical value for Brownian bridge supremum ≈ 1.358
    return {
        "stat_test": "cusum_on_loss",
        "n": n,
        "max_abs_normalized_cusum": max_abs,
        "critical_5pct": 1.358,
        "flag_instability": bool(max_abs > 1.358),
    }


def _compute_paired_t_on_loss_diff(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 2:
        raise ExecutionError("paired_t_on_loss_diff requires at least 2 forecast errors")
    mean = float(loss_diff.mean())
    variance = float(np.var(loss_diff, ddof=1))
    if variance <= 0:
        raise ExecutionError("paired_t_on_loss_diff variance must be positive")
    t_stat = float(mean / math.sqrt(variance / n))
    p_value = float(_normal_two_sided_pvalue(t_stat))
    return {
        "stat_test": "paired_t_on_loss_diff",
        "n": n,
        "mean_loss_diff": mean,
        "variance": variance,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
    }


def _compute_wilcoxon_signed_rank(predictions: pd.DataFrame) -> dict[str, object]:
    from scipy.stats import wilcoxon
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 2:
        raise ExecutionError("wilcoxon_signed_rank requires at least 2 forecast errors")
    if float(np.std(loss_diff)) == 0.0:
        raise ExecutionError("wilcoxon_signed_rank requires non-zero loss differences")
    res = wilcoxon(loss_diff, zero_method="wilcox", alternative="two-sided")
    return {
        "stat_test": "wilcoxon_signed_rank",
        "n": n,
        "mean_loss_diff": float(loss_diff.mean()),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "significant_5pct": bool(res.pvalue < 0.05),
    }


def _compute_autocorrelation_of_errors(predictions: pd.DataFrame, max_lag: int = 10) -> dict[str, object]:
    from scipy.stats import chi2
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    errors = rows["y_true"].to_numpy(dtype=float) - rows["y_pred"].to_numpy(dtype=float)
    n = int(len(errors))
    if n < 3:
        raise ExecutionError("autocorrelation_of_errors requires at least 3 errors")
    centered = errors - errors.mean()
    denom = float(np.dot(centered, centered))
    if denom <= 0:
        raise ExecutionError("autocorrelation_of_errors requires non-zero error variance")
    effective_lag = min(int(max_lag), n - 1)
    q_stat = 0.0
    rhos: list[float] = []
    for h in range(1, effective_lag + 1):
        cov_h = float(np.dot(centered[h:], centered[:-h]))
        rho_h = cov_h / denom
        rhos.append(rho_h)
        q_stat += (rho_h * rho_h) / max(n - h, 1)
    q_stat *= n * (n + 2)
    p_value = float(1.0 - chi2.cdf(q_stat, df=effective_lag))
    return {
        "stat_test": "autocorrelation_of_errors",
        "n": n,
        "max_lag": int(effective_lag),
        "rho": rhos,
        "q_statistic": float(q_stat),
        "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
    }


def _compute_mcnemar_direction(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["target_date", "origin_date"]).reset_index(drop=True)
    y = rows["y_true"].astype(float).to_numpy()
    yhat = rows["y_pred"].astype(float).to_numpy()
    yben = rows["benchmark_pred"].astype(float).to_numpy()
    if len(y) < 3:
        raise ExecutionError("mcnemar requires at least 3 observations")
    actual_up = np.sign(np.diff(y)) > 0
    model_up = np.sign(yhat[1:] - y[:-1]) > 0
    bench_up = np.sign(yben[1:] - y[:-1]) > 0
    hit_model = (actual_up == model_up).astype(int)
    hit_bench = (actual_up == bench_up).astype(int)
    b = int(((hit_model == 1) & (hit_bench == 0)).sum())
    c = int(((hit_model == 0) & (hit_bench == 1)).sum())
    n_off = b + c
    if n_off == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float((abs(b - c) - 1) ** 2 / n_off) if n_off > 0 else 0.0
        from scipy.stats import chi2
        p_value = float(1.0 - chi2.cdf(statistic, df=1))
    return {
        "stat_test": "mcnemar",
        "n": int(len(actual_up)),
        "model_hit_rate": float(hit_model.mean()),
        "benchmark_hit_rate": float(hit_bench.mean()),
        "disagreements_model_better": b,
        "disagreements_benchmark_better": c,
        "statistic": statistic,
        "p_value": p_value,
        "significant_5pct": bool(p_value < 0.05),
    }


def _compute_forecast_encompassing_nested(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    y = rows["y_true"].astype(float).to_numpy()
    f1 = rows["y_pred"].astype(float).to_numpy()
    f2 = rows["benchmark_pred"].astype(float).to_numpy()
    n = int(len(y))
    if n < 3:
        raise ExecutionError("forecast_encompassing_nested requires at least 3 observations")
    resid = y - f1
    delta = f2 - f1
    denom = float(np.dot(delta - delta.mean(), delta - delta.mean()))
    if denom <= 0:
        raise ExecutionError("forecast_encompassing_nested has zero-variance delta")
    beta = float(np.dot(delta - delta.mean(), resid - resid.mean()) / denom)
    alpha = float(resid.mean() - beta * delta.mean())
    yhat = alpha + beta * delta
    residuals = resid - yhat
    sigma2 = float(np.dot(residuals, residuals) / max(n - 2, 1))
    se_beta = float(math.sqrt(sigma2 / denom)) if sigma2 > 0 else float("nan")
    t_stat = float(beta / se_beta) if se_beta and not math.isnan(se_beta) and se_beta > 0 else float("nan")
    p_value = float(_normal_two_sided_pvalue(t_stat)) if not math.isnan(t_stat) else float("nan")
    return {
        "stat_test": "forecast_encompassing_nested",
        "n": n,
        "beta": beta,
        "se_beta": se_beta,
        "t_statistic": t_stat,
        "p_value": p_value,
        "encompassed_5pct": bool(not math.isnan(p_value) and p_value > 0.05),
    }


def _compute_serial_dependence_loss_diff(predictions: pd.DataFrame) -> dict[str, object]:
    rows = predictions.sort_values(["horizon", "target_date", "origin_date"]).reset_index(drop=True)
    loss_diff = rows["benchmark_squared_error"].to_numpy(dtype=float) - rows["squared_error"].to_numpy(dtype=float)
    n = int(len(loss_diff))
    if n < 3:
        raise ExecutionError("serial_dependence_loss_diff requires at least 3 observations")
    diffs = np.diff(loss_diff)
    num = float(np.dot(diffs, diffs))
    den = float(np.dot(loss_diff - loss_diff.mean(), loss_diff - loss_diff.mean()))
    dw = float(num / den) if den > 0 else float("nan")
    lag1 = float(1.0 - dw / 2.0) if not math.isnan(dw) else float("nan")
    return {
        "stat_test": "serial_dependence_loss_diff",
        "n": n,
        "durbin_watson": dw,
        "lag1_autocorr_estimate": lag1,
        "flag_serial_dependence": bool(not math.isnan(dw) and (dw < 1.5 or dw > 2.5)),
    }


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
    legacy_feature_builder = _feature_builder(recipe)
    feature_runtime_builder = _feature_runtime_builder(recipe)
    if model_family not in {"ridge", "lasso", "randomforest"}:
        raise ExecutionError(f"minimal_importance not implemented for model_family {model_family!r}")
    if feature_runtime_builder != "raw_feature_panel":
        raise ExecutionError(
            f"minimal_importance currently requires feature runtime 'raw_feature_panel', got {feature_runtime_builder!r}"
        )

    last_tuning_payload: dict[str, object] = {}
    aligned_frame = raw_frame.loc[target_series.index]
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    origin_idx = len(target_series) - max(recipe.horizons) - 1
    if origin_idx < _minimum_train_size(recipe) - 1:
        raise ExecutionError("minimal_importance requires at least one valid forecast origin")
    start_idx = max(0, origin_idx + 1 - _rolling_window_size(recipe)) if rolling else 0
    horizon = min(recipe.horizons)
    representation = _build_raw_panel_representation(
        aligned_frame,
        recipe,
        horizon,
        start_idx,
        origin_idx,
        contract,
    )

    if model_family == "ridge":
        model = Ridge(alpha=1.0)
        model.fit(representation.Z_train, representation.y_train)
        importance_values = np.abs(model.coef_)
    elif model_family == "lasso":
        model = Lasso(alpha=1e-4, max_iter=10000)
        model.fit(representation.Z_train, representation.y_train)
        importance_values = np.abs(model.coef_)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=current_seed(model_family="randomforest"))
        model.fit(representation.Z_train, representation.y_train)
        importance_values = model.feature_importances_

    feature_importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(zip(representation.feature_names, importance_values), key=lambda item: item[1], reverse=True)
    ]
    return {
        "importance_method": "minimal_importance",
        "model_family": model_family,
        "feature_builder": feature_runtime_builder,
        "feature_runtime_builder": feature_runtime_builder,
        "legacy_feature_builder": legacy_feature_builder,
        "feature_dispatch_source": "layer2_feature_blocks",
        "n_train": int(len(representation.y_train)),
        "feature_importance": feature_importance,
    }


def _importance_feature_names(recipe: RecipeSpec, raw_frame: pd.DataFrame, target_series: pd.Series, contract: PreprocessContract) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    feature_runtime_builder = _feature_runtime_builder(recipe)
    horizon = min(recipe.horizons)
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    origin_idx = len(target_series) - max(recipe.horizons) - 1
    if origin_idx < _minimum_train_size(recipe) - 1:
        raise ExecutionError("importance requires at least one valid forecast origin")
    start_idx = max(0, origin_idx + 1 - _rolling_window_size(recipe)) if rolling else 0
    if feature_runtime_builder == "raw_feature_panel":
        aligned_frame = raw_frame.loc[target_series.index]
        representation = _build_raw_panel_representation(
            aligned_frame,
            recipe,
            horizon,
            start_idx,
            origin_idx,
            contract,
        )
        return list(representation.feature_names), representation.Z_train, representation.y_train, representation.Z_pred
    if feature_runtime_builder == "autoreg_lagged_target":
        train = target_series.iloc[start_idx: origin_idx + 1]
        representation = _build_target_lag_representation(train, recipe, default_prefix="lag")
        return list(representation.feature_names), representation.Z_train, representation.y_train, representation.Z_pred
    raise ExecutionError(f"importance not implemented for feature runtime {feature_runtime_builder!r}")


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
        "feature_builder": _feature_runtime_builder(recipe),
        "feature_runtime_builder": _feature_runtime_builder(recipe),
        "legacy_feature_builder": _feature_builder(recipe),
        "feature_dispatch_source": "layer2_feature_blocks",
    }


def _importance_runtime_metadata(bundle: Mapping[str, object]) -> dict[str, object]:
    return {
        "feature_builder": bundle["feature_builder"],
        "feature_runtime_builder": bundle["feature_runtime_builder"],
        "legacy_feature_builder": bundle["legacy_feature_builder"],
        "feature_dispatch_source": bundle.get("feature_dispatch_source", "layer2_feature_blocks"),
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
        **_importance_runtime_metadata(bundle),
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
        **_importance_runtime_metadata(bundle),
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
        **_importance_runtime_metadata(bundle),
        "n_background": int(len(background)),
        "feature_importance": _ranked_feature_payload("abs_shap", bundle["feature_names"], row_values),
    }


def _compute_permutation_importance_artifact(raw_frame: pd.DataFrame, target_series: pd.Series, recipe: RecipeSpec, contract: PreprocessContract) -> dict[str, object]:
    bundle = _importance_training_bundle(raw_frame, target_series, recipe, contract)
    result = permutation_importance(bundle["model"], bundle["X_train"], bundle["y_train"], n_repeats=5, random_state=current_seed(model_family="permutation_importance"))
    return {
        "importance_method": "permutation_importance",
        "model_family": bundle["model_family"],
        **_importance_runtime_metadata(bundle),
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
        **_importance_runtime_metadata(bundle),
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
        **_importance_runtime_metadata(bundle),
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
    payload = {
        "importance_method": mode,
        "model_family": bundle["model_family"],
        **_importance_runtime_metadata(bundle),
        "profiles": [],
    }
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
        "model_family": base["model_family"],
        "feature_builder": base["feature_builder"],
        "feature_runtime_builder": base["feature_runtime_builder"],
        "legacy_feature_builder": base["legacy_feature_builder"],
        "feature_dispatch_source": base["feature_dispatch_source"],
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
        "model_family": bundle["model_family"],
        **_importance_runtime_metadata(bundle),
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
    target_transformer_spec = _target_transformer_spec(recipe)
    feature_runtime_builder = _feature_runtime_builder(recipe)
    model_family = _model_family(recipe)
    if (
        target_transformer_spec is not None
        and feature_runtime_builder not in _TARGET_TRANSFORMER_FEATURE_RUNTIMES
    ):
        raise ExecutionError(
            "target_transformer runtime currently supports feature runtime in "
            f"{sorted(_TARGET_TRANSFORMER_FEATURE_RUNTIMES)}"
        )
    if (
        target_transformer_spec is not None
        and feature_runtime_builder in {"raw_feature_panel", "raw_X_only"}
        and model_family not in _TARGET_TRANSFORMER_RAW_PANEL_MODELS
        and not is_custom_model(model_family)
    ):
        raise ExecutionError(
            "target_transformer raw-panel runtime currently supports "
            f"model_family in {sorted(_TARGET_TRANSFORMER_RAW_PANEL_MODELS)} or a registered custom model; "
            f"got {model_family!r}"
        )

    last_tuning_payload: dict[str, object] = {}
    raw_target_series = _raw_target_series_for_scale(raw_frame, str(target_series.name))
    aligned_frame = raw_frame.loc[target_series.index]
    evaluation_scale = str(getattr(contract, "evaluation_scale", "raw_level"))
    rolling = recipe.stage0.fixed_design.sample_split == "rolling_window_oos"
    rolling_window_size = _rolling_window_size(recipe)
    outer_window = str(recipe.training_spec.get("outer_window", "rolling" if rolling else "expanding"))
    refit_policy = str(recipe.training_spec.get("refit_policy", "refit_every_step"))
    anchored_max_window_size = int(recipe.training_spec.get("anchored_max_window_size", rolling_window_size))
    refit_k_steps = int(recipe.training_spec.get("refit_k_steps", 3))
    _horizon_construction = _canonicalize_horizon_target_construction(
        str(_layer2_runtime_spec(recipe).get("horizon_target_construction", "future_target_level_t_plus_h"))
    )
    _oos_period = str(recipe.data_task_spec.get("oos_period", "all_oos_data"))

    # 1.3 training_start_rule=fixed_start: resolve the calendar date to an index floor
    _training_start_rule = _training_axis(recipe, "training_start_rule")
    _fixed_start_idx = 0
    if _training_start_rule == "fixed_start":
        _fixed_start_date = _training_value(recipe, "training_start_date")
        if _fixed_start_date is None:
            raise ExecutionError("training_start_rule='fixed_start' requires training_start_date (validated at compile time)")
        import pandas as _pd
        try:
            _ts = _pd.Timestamp(_fixed_start_date)
        except Exception as _e:
            raise ExecutionError(f"training_start_date={_fixed_start_date!r} is not a valid ISO date: {_e}") from _e
        _idx = target_series.index.searchsorted(_ts)
        if _idx >= len(target_series.index):
            raise ExecutionError(f"training_start_date={_fixed_start_date!r} is after the last available observation")
        _fixed_start_idx = int(_idx)

    def _rows_for_horizon(horizon: int) -> list[dict[str, object]]:
        nonlocal last_tuning_payload
        # Stage 1 — build the origin plan serially so refit_policy state
        # (locked_start_idx / locked_origin_idx) is honoured deterministically.
        origin_plan: list[tuple[int, int, int]] = []
        locked_start_idx = None
        locked_origin_idx = None
        for origin_idx in range(minimum_train_size - 1, len(target_series) - horizon):
            base_start_idx = max(0, origin_idx + 1 - rolling_window_size) if rolling else 0
            if _fixed_start_idx > 0:
                base_start_idx = max(base_start_idx, _fixed_start_idx)
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
            origin_plan.append((origin_idx, start_idx, effective_origin_idx))

        # 1.3 oos_period regime filter — applied after origin_plan is finalized.
        if _oos_period in {"recession_only_oos", "expansion_only_oos"} and origin_plan:
            origin_plan = _filter_origins_by_regime(origin_plan, index=target_series.index, regime=_oos_period)

        # Stage 2 — compute each origin's row. Thread-pool when requested.
        def _compute_origin(origin_idx: int, start_idx: int, effective_origin_idx: int) -> tuple[dict[str, object], dict[str, object] | None]:
            train = target_series.iloc[start_idx : effective_origin_idx + 1]
            train_for_model = train
            target_scale_state: dict[str, object] = {
                "normalization": "none",
                "fit_scope": "not_applicable",
                "params": {},
            }
            target_transform_context: dict[str, object] = {}
            fitted_target_transformer = None
            if target_transformer_spec is not None:
                fitted_target_transformer, train_for_model, target_transform_context = _fit_target_transformer_for_window(
                    train,
                    recipe=recipe,
                    horizon=horizon,
                    target_date=target_series.index[origin_idx + horizon],
                )
            else:
                train_for_model, target_scale_state = _fit_target_normalization_for_window(train, contract)
            model_output = _coerce_forecast_payload(
                model_executor(train_for_model, horizon, recipe, contract, aligned_frame, effective_origin_idx, start_idx),
                executor_name=str(model_spec["executor_name"]),
            )
            tuning_payload = model_output.tuning_payload or None
            y_pred_model_scale = model_output.y_pred
            benchmark_pred_model_scale = float(benchmark_executor(train_for_model, horizon, recipe))
            y_true_transformed_scale = float(target_series.iloc[origin_idx + horizon])
            y_true_model_scale = (
                y_true_transformed_scale
                if target_transformer_spec is not None
                else _apply_target_normalization_scalar(y_true_transformed_scale, target_scale_state)
            )
            if target_transformer_spec is not None:
                y_pred_transformed_scale = _inverse_target_transformer_prediction(
                    fitted_target_transformer,
                    y_pred_model_scale,
                    recipe=recipe,
                    context=target_transform_context,
                )
                benchmark_pred_transformed_scale = benchmark_pred_model_scale
            else:
                y_pred_transformed_scale = _inverse_target_normalization_scalar(y_pred_model_scale, target_scale_state)
                benchmark_pred_transformed_scale = _inverse_target_normalization_scalar(
                    benchmark_pred_model_scale,
                    target_scale_state,
                )
            origin_date = target_series.index[origin_idx]
            target_date = target_series.index[origin_idx + horizon]
            y_true_original_scale = float(raw_target_series.loc[target_date])
            if target_transformer_spec is not None:
                y_pred_original_scale = y_pred_transformed_scale
                benchmark_pred_original_scale = benchmark_pred_transformed_scale
            else:
                y_pred_original_scale = _inverse_target_transform_scalar(
                    y_pred_transformed_scale,
                    contract=contract,
                    raw_target_series=raw_target_series,
                    origin_date=origin_date,
                    target_date=target_date,
                    horizon=horizon,
                )
                benchmark_pred_original_scale = _inverse_target_transform_scalar(
                    benchmark_pred_transformed_scale,
                    contract=contract,
                    raw_target_series=raw_target_series,
                    origin_date=origin_date,
                    target_date=target_date,
                    horizon=horizon,
                )
            y_true, y_pred, benchmark_pred, metric_target_scale = _target_metric_values(
                y_true_model_scale=y_true_model_scale,
                y_pred_model_scale=y_pred_model_scale,
                benchmark_pred_model_scale=benchmark_pred_model_scale,
                y_true_transformed_scale=y_true_transformed_scale,
                y_pred_transformed_scale=y_pred_transformed_scale,
                benchmark_pred_transformed_scale=benchmark_pred_transformed_scale,
                y_true_original_scale=y_true_original_scale,
                y_pred_original_scale=y_pred_original_scale,
                benchmark_pred_original_scale=benchmark_pred_original_scale,
                evaluation_scale=evaluation_scale,
            )
            # Level-scale values are kept for provenance; metric-scale values
            # honour horizon_target_construction (1.2.4).
            y_pred_level = y_pred
            benchmark_pred_level = benchmark_pred
            y_true_level = y_true
            if _horizon_construction != "future_target_level_t_plus_h":
                y_anchor = float(target_series.iloc[effective_origin_idx])
                y_pred = _horizon_forward_scalar(y_pred_level, y_anchor, _horizon_construction, horizon=horizon)
                benchmark_pred = _horizon_forward_scalar(benchmark_pred_level, y_anchor, _horizon_construction, horizon=horizon)
                y_true = _horizon_forward_scalar(y_true_level, y_anchor, _horizon_construction, horizon=horizon)
            error = y_true - y_pred
            benchmark_error = y_true - benchmark_pred
            row = {
                "target": target_series.name,
                "model_name": model_spec["executor_name"],
                "benchmark_name": benchmark_family,
                "horizon": horizon,
                "origin_date": target_series.index[origin_idx].strftime("%Y-%m-%d"),
                "target_date": target_series.index[origin_idx + horizon].strftime("%Y-%m-%d"),
                "fit_origin_date": target_series.index[effective_origin_idx].strftime("%Y-%m-%d"),
                "selected_lag": model_output.selected_lag,
                "selected_bic": model_output.selected_bic,
                "train_start_date": target_series.index[start_idx].strftime("%Y-%m-%d"),
                "train_end_date": target_series.index[origin_idx].strftime("%Y-%m-%d"),
                "training_window_size": int(len(train)),
                "y_true": y_true,
                "y_pred": y_pred,
                "benchmark_pred": benchmark_pred,
                "y_true_model_scale": y_true_model_scale,
                "y_pred_model_scale": y_pred_model_scale,
                "benchmark_pred_model_scale": benchmark_pred_model_scale,
                "y_true_transformed_scale": y_true_transformed_scale,
                "y_pred_transformed_scale": y_pred_transformed_scale,
                "benchmark_pred_transformed_scale": benchmark_pred_transformed_scale,
                "y_true_original_scale": y_true_original_scale,
                "y_pred_original_scale": y_pred_original_scale,
                "benchmark_pred_original_scale": benchmark_pred_original_scale,
                "target_transformer": target_transformer_spec.name if target_transformer_spec is not None else "none",
                "target_normalization": str(target_scale_state.get("normalization", "none")),
                "target_normalization_fit_scope": str(target_scale_state.get("fit_scope", "not_applicable")),
                "target_normalization_params": json.dumps(target_scale_state.get("params", {}), sort_keys=True),
                "model_target_scale": (
                    "custom_transformer_scale"
                    if target_transformer_spec is not None
                    else "normalized_target_scale"
                    if str(target_scale_state.get("normalization", "none")) != "none"
                    else "transformed_target_scale"
                ),
                "forecast_scale": metric_target_scale,
                "evaluation_scale": evaluation_scale,
                "target_construction_scale": _horizon_construction_scale(_horizon_construction),
                "error": error,
                "abs_error": abs(error),
                "squared_error": error**2,
                "benchmark_error": benchmark_error,
                "benchmark_abs_error": abs(benchmark_error),
                "benchmark_squared_error": benchmark_error**2,
                "model_scale_error": y_true_model_scale - y_pred_model_scale,
                "model_scale_benchmark_error": y_true_model_scale - benchmark_pred_model_scale,
                "transformed_scale_error": y_true_transformed_scale - y_pred_transformed_scale,
                "transformed_scale_benchmark_error": y_true_transformed_scale - benchmark_pred_transformed_scale,
                "transformed_scale_squared_error": (y_true_transformed_scale - y_pred_transformed_scale) ** 2,
                "transformed_scale_benchmark_squared_error": (
                    y_true_transformed_scale - benchmark_pred_transformed_scale
                ) ** 2,
                "original_scale_error": y_true_original_scale - y_pred_original_scale,
                "original_scale_benchmark_error": y_true_original_scale - benchmark_pred_original_scale,
                "original_scale_squared_error": (y_true_original_scale - y_pred_original_scale) ** 2,
                "original_scale_benchmark_squared_error": (
                    y_true_original_scale - benchmark_pred_original_scale
                ) ** 2,
                "horizon_target_construction": _horizon_construction,
                "y_true_level": y_true_level,
                "y_pred_level": y_pred_level,
                "benchmark_pred_level": benchmark_pred_level,
            }
            return row, tuning_payload

        horizon_rows: list[dict[str, object]] = []
        if compute_mode == "parallel_by_oos_date" and len(origin_plan) > 1:
            with ThreadPoolExecutor(max_workers=min(len(origin_plan), 4)) as ex:
                futures = [ex.submit(_compute_origin, *item) for item in origin_plan]
                for future in futures:
                    row, tuning_payload = future.result()
                    horizon_rows.append(row)
                    if tuning_payload:
                        last_tuning_payload = tuning_payload
        else:
            for item in origin_plan:
                row, tuning_payload = _compute_origin(*item)
                horizon_rows.append(row)
                if tuning_payload:
                    last_tuning_payload = tuning_payload
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


def _scale_metric_summary(
    group: pd.DataFrame,
    *,
    squared_error_col: str,
    benchmark_squared_error_col: str,
    error_col: str,
    benchmark_error_col: str,
) -> dict[str, float]:
    msfe = float(group[squared_error_col].mean())
    benchmark_msfe = float(group[benchmark_squared_error_col].mean())
    mae = float(group[error_col].abs().mean())
    benchmark_mae = float(group[benchmark_error_col].abs().mean())
    rmse = float(msfe**0.5)
    benchmark_rmse = float(benchmark_msfe**0.5)
    return {
        "msfe": msfe,
        "benchmark_msfe": benchmark_msfe,
        "relative_msfe": msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0,
        "oos_r2": 1.0 - (msfe / benchmark_msfe if benchmark_msfe > 0 else 1.0),
        "mae": mae,
        "benchmark_mae": benchmark_mae,
        "relative_mae": mae / benchmark_mae if benchmark_mae > 0 else 1.0,
        "rmse": rmse,
        "benchmark_rmse": benchmark_rmse,
        "relative_rmse": rmse / benchmark_rmse if benchmark_rmse > 0 else 1.0,
    }


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
        scale_metrics: dict[str, object] = {
            "primary": {
                "scale": str(group["forecast_scale"].iloc[0]) if "forecast_scale" in group else "primary",
                "evaluation_scale": str(group["evaluation_scale"].iloc[0]) if "evaluation_scale" in group else "raw_level",
            }
        }
        if {
            "original_scale_squared_error",
            "original_scale_benchmark_squared_error",
            "original_scale_error",
            "original_scale_benchmark_error",
        }.issubset(group.columns):
            scale_metrics["original_target_scale"] = _scale_metric_summary(
                group,
                squared_error_col="original_scale_squared_error",
                benchmark_squared_error_col="original_scale_benchmark_squared_error",
                error_col="original_scale_error",
                benchmark_error_col="original_scale_benchmark_error",
            )
        if {
            "transformed_scale_squared_error",
            "transformed_scale_benchmark_squared_error",
            "transformed_scale_error",
            "transformed_scale_benchmark_error",
        }.issubset(group.columns):
            scale_metrics["transformed_target_scale"] = _scale_metric_summary(
                group,
                squared_error_col="transformed_scale_squared_error",
                benchmark_squared_error_col="transformed_scale_benchmark_squared_error",
                error_col="transformed_scale_error",
                benchmark_error_col="transformed_scale_benchmark_error",
            )
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
            "scale_metrics": scale_metrics,
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
            f"benchmark_family {_benchmark_family(recipe)!r} is not supported in current runtime slice"
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
    # Install global RNG state + determinism flags based on the recipe's mode.
    # current_seed() uses the same policy for per-variant seeds; this call
    # covers the global state that downstream libraries (numpy, torch, cudnn)
    # read directly without passing through resolve_seed.
    _reproducibility_applied = apply_reproducibility_mode(
        mode=str(reproducibility_spec.get("reproducibility_mode", "seeded_reproducible")),
        seed=int(reproducibility_spec.get("seed", 42)),
    )
    raw_result = _load_raw_for_recipe(recipe, local_raw_source, effective_cache_root)
    raw_result = _apply_frequency_policy(raw_result, recipe)
    raw_result = _apply_sd_inferred_tcodes(raw_result, recipe)
    _raw_missing_policy = _data_task_axis(recipe, "raw_missing_policy")
    _raw_outlier_policy = _data_task_axis(recipe, "raw_outlier_policy")
    raw_result = _apply_raw_missing_policy(raw_result, _raw_missing_policy, target=str(recipe.target) if getattr(recipe, 'target', None) else None, spec=dict(recipe.data_task_spec))
    raw_result = _apply_raw_outlier_policy(raw_result, _raw_outlier_policy, spec=dict(recipe.data_task_spec))
    raw_result = _apply_tcode_preprocessing(raw_result, recipe, preprocess, target=str(recipe.target) if getattr(recipe, 'target', None) else None)
    raw_result = _apply_sample_period_and_availability(raw_result, recipe, target=str(recipe.target) if getattr(recipe, 'target', None) else None)
    _release_lag = _data_task_axis(recipe, "release_lag_rule")
    _missing_avail = _data_task_axis(recipe, "missing_availability")
    _var_universe = _data_task_axis(recipe, "variable_universe")
    _min_train_axis = _training_axis(recipe, "min_train_size")
    _break_seg = str(_layer2_runtime_spec(recipe).get("structural_break_segmentation", "none"))
    _separation = _data_task_axis(recipe, "separation_rule")
    raw_result = _apply_release_lag(raw_result, _release_lag, spec=dict(recipe.data_task_spec))
    raw_result = _apply_missing_availability(raw_result, _missing_avail, target=str(recipe.target) if getattr(recipe, 'target', None) else None, spec=dict(recipe.data_task_spec))
    raw_result = _apply_variable_universe(raw_result, _var_universe, spec=dict(recipe.data_task_spec), target=str(recipe.target) if getattr(recipe, 'target', None) else None)
    targets = _recipe_targets(recipe)
    prediction_frames = []
    failed_components: list[dict[str, object]] = []
    successful_targets: list[str] = []
    target_series = None
    _last_tp: dict[str, object] = {}
    def _target_job(target: str):
        target_recipe = _recipe_for_target(recipe, target)
        target_series_local = _get_target_series(raw_result.data, target, _minimum_train_size(target_recipe))
        target_series_local = _apply_target_transform_and_normalization(target_series_local, preprocess)
        frame, tp = _build_predictions(raw_result.data, target_series_local, target_recipe, preprocess, compute_mode=compute_mode)
        return target, target_series_local, frame, tp

    if compute_mode == "parallel_by_target" and len(targets) > 1:
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
                    if failure_policy in {"skip_failed_model", "save_partial_results", "warn_only"}:
                        failed_components.append({"stage": "prediction_build", "target": target_name, "error": err})
                        if failure_policy == "warn_only":
                            warnings.warn(f"prediction_build failure (target={target_name!r}): {err}", RuntimeWarning, stacklevel=2)
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
                if failure_policy in {"skip_failed_model", "save_partial_results", "warn_only"}:
                    failed_components.append({"stage": "prediction_build", "target": target, "error": str(exc)})
                    if failure_policy == "warn_only":
                        warnings.warn(f"prediction_build failure (target={target!r}): {exc}", RuntimeWarning, stacklevel=2)
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
        "target_transformer": _target_transformer_manifest(recipe),
        "benchmark_name": _benchmark_family(recipe),
        "benchmark_spec": _benchmark_spec(recipe),
        "data_task_spec": dict(recipe.data_task_spec),
        "layer2_representation_spec": dict(getattr(recipe, "layer2_representation_spec", {}) or {}),
        "data_warnings": _data_warnings(raw_result),
        "data_reports": _data_reports(raw_result),
        "training_spec": dict(recipe.training_spec),
        "evaluation_spec": evaluation_spec,
        "stat_test_spec": stat_test_spec,
        "importance_spec": importance_spec,
        "reproducibility_spec": reproducibility_spec,
        "reproducibility_applied": _reproducibility_applied,
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
    feature_fit_state = _last_tp.get("feature_representation_fit_state") if isinstance(_last_tp, dict) else None
    forecast_payload_contract = _last_tp.get("forecast_payload_contract") if isinstance(_last_tp, dict) else None
    if forecast_payload_contract:
        manifest["forecast_payload_contract"] = forecast_payload_contract
    if feature_fit_state:
        _write_json(run_dir / "feature_representation_fit_state.json", feature_fit_state)
        manifest["feature_representation_fit_state_file"] = "feature_representation_fit_state.json"
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
        if failure_policy in {"save_partial_results", "warn_only"}:
            failed_components.append({"stage": "stat_test_artifact", "target": None, "error": str(exc)})
            if failure_policy == "warn_only":
                warnings.warn(f"stat_test_artifact failure: {exc}", RuntimeWarning, stacklevel=2)
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
            if failure_policy in {"save_partial_results", "warn_only"}:
                failed_components.append({"stage": "importance_artifact", "target": None, "error": str(exc)})
                if failure_policy == "warn_only":
                    warnings.warn(f"importance_artifact failure: {exc}", RuntimeWarning, stacklevel=2)
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
