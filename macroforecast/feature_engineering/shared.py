from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd

from macroforecast.data import (
    DataBundle,
    DataSpec,
    as_panel,
    attach_metadata,
    validate_panel,
)
from macroforecast.preprocessing import PreprocessedData
from macroforecast.feature_engineering.types import FeatureInput, _InputBundle

TargetTransform = Literal[
    "level",
    "change",
    "growth",
    "log_growth",
    "average_change",
    "average_growth",
    "average_log_growth",
]
PathTransform = Literal["change", "growth", "log_growth"]
TargetMode = Literal["direct", "path"]
FitPolicy = Literal["expanding", "full_sample"]
_FEATURE_METADATA_COLUMNS = (
    "feature",
    "step",
    "block",
    "operation",
    "source",
    "parameter",
    "lag",
    "window",
    "component",
    "fit_policy",
    "inputs",
    "included",
)
_TARGET_METADATA_COLUMNS = (
    "target_column",
    "source",
    "horizon",
    "step",
    "mode",
    "transform",
    "operation",
    "formula",
    "aggregation",
    "used_for_horizons",
)

def _zscore_frame(
    frame: pd.DataFrame,
    *,
    fit_policy: str,
    min_train_size: int,
    ddof: int,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    if fit_policy == "full_sample":
        complete = frame.dropna()
        if len(complete) < min_train_size:
            return result
        center = complete.mean(axis=0)
        scale = complete.std(axis=0, ddof=ddof).replace(0.0, np.nan)
        return (frame - center) / scale
    for position, date in enumerate(frame.index):
        train = frame.iloc[: position + 1].dropna()
        current = frame.iloc[[position]]
        if len(train) < min_train_size or current.isna().any(axis=None):
            continue
        center = train.mean(axis=0)
        scale = train.std(axis=0, ddof=ddof).replace(0.0, np.nan)
        result.loc[date, :] = ((current - center) / scale).iloc[0]
    return result


def _scale_frame(
    frame: pd.DataFrame,
    *,
    method: str,
    fit_policy: str,
    min_train_size: int,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    if fit_policy == "full_sample":
        complete = frame.dropna()
        if len(complete) < min_train_size:
            return result
        center, scale = _scale_parameters(complete, method=method)
        return (frame - center) / scale
    for position, date in enumerate(frame.index):
        train = frame.iloc[: position + 1].dropna()
        current = frame.iloc[[position]]
        if len(train) < min_train_size or current.isna().any(axis=None):
            continue
        center, scale = _scale_parameters(train, method=method)
        result.loc[date, :] = ((current - center) / scale).iloc[0]
    return result


def _scale_parameters(frame: pd.DataFrame, *, method: str) -> tuple[pd.Series, pd.Series]:
    if method == "zscore":
        center = frame.mean(axis=0)
        scale = frame.std(axis=0, ddof=0)
    elif method == "minmax":
        center = frame.min(axis=0)
        scale = frame.max(axis=0) - frame.min(axis=0)
    elif method == "robust":
        center = frame.median(axis=0)
        scale = frame.quantile(0.75, axis=0) - frame.quantile(0.25, axis=0)
    else:
        raise ValueError(f"unknown scale method {method!r}")
    return center, scale.replace(0.0, np.nan)


def _pca_frame(
    frame: pd.DataFrame,
    *,
    n_components: int,
    fit_policy: str,
    min_train_size: int,
    scale: bool,
    prefix: str,
    random_state: int | None,
) -> pd.DataFrame:
    columns = [f"{prefix}{index}" for index in range(1, n_components + 1)]
    result = pd.DataFrame(index=frame.index, columns=columns, dtype=float)
    if fit_policy == "full_sample":
        complete = frame.dropna()
        if len(complete) < min_train_size:
            return result
        transformed = _fit_transform_pca(
            complete,
            frame,
            n_components=n_components,
            scale=scale,
            random_state=random_state,
        )
        result.loc[transformed.index, :] = transformed.to_numpy()
        return result

    for position, date in enumerate(frame.index):
        train = frame.iloc[: position + 1].dropna()
        current = frame.iloc[[position]]
        if len(train) < min_train_size or current.isna().any(axis=None):
            continue
        transformed = _fit_transform_pca(
            train,
            current,
            n_components=n_components,
            scale=scale,
            random_state=random_state,
        )
        result.loc[date, :] = transformed.iloc[0].to_numpy()
    return result


def _fit_transform_pca(
    train: pd.DataFrame,
    data: pd.DataFrame,
    *,
    n_components: int,
    scale: bool,
    random_state: int | None,
) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    train_values = train.astype(float)
    data_values = data.dropna().astype(float)
    if data_values.empty:
        return pd.DataFrame(index=data.index, columns=[f"pc{idx}" for idx in range(1, n_components + 1)], dtype=float)
    if scale:
        center, divisor = _scale_parameters(train_values, method="zscore")
        divisor = divisor.fillna(1.0)
        train_values = (train_values - center) / divisor
        data_values = (data_values - center) / divisor
    model = PCA(n_components=n_components, random_state=random_state)
    model.fit(train_values.dropna())
    transformed = pd.DataFrame(
        model.transform(data_values),
        index=data_values.index,
        columns=[f"pc{idx}" for idx in range(1, n_components + 1)],
    )
    return transformed


def _reject_extra_params(params: Mapping[str, Any], step_name: str) -> None:
    if params:
        raise ValueError(f"unknown parameters for feature step {step_name!r}: {sorted(params)}")


def _step(
    *,
    name: str,
    method: str,
    input: str,
    include: bool,
    **params: Any,
) -> dict[str, Any]:
    step = {
        "name": str(name),
        "method": str(method),
        "input": str(input),
        "include": bool(include),
    }
    step.update({key: value for key, value in params.items() if value is not None})
    return step


def _coerce_input(data: FeatureInput, *, metadata: Mapping[str, Any] | None = None) -> _InputBundle:
    if isinstance(data, PreprocessedData):
        base = _InputBundle(
            panel=data.panel,
            metadata=dict(data.metadata),
            target=data.target,
            targets=data.targets,
            horizons=data.horizons,
            predictors=data.predictors,
        )
    elif isinstance(data, DataSpec):
        base = _InputBundle(
            panel=data.panel,
            metadata=dict(data.metadata),
            target=data.target,
            targets=data.targets,
            horizons=data.horizons,
            predictors=data.predictors,
        )
    elif isinstance(data, DataBundle):
        base = _InputBundle(panel=data.panel, metadata=dict(data.metadata))
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], pd.DataFrame):
        panel = as_panel(data[0], metadata=data[1])
        base = _InputBundle(panel=panel, metadata=dict(data[1]))
    elif isinstance(data, pd.DataFrame):
        existing = dict(data.attrs.get("macroforecast_metadata", {}))
        base = _InputBundle(panel=as_panel(data, metadata=existing), metadata=existing)
    else:
        raise TypeError("expected PreprocessedData, DataSpec, DataBundle, (panel, metadata), or pandas DataFrame")
    if metadata is None:
        return base
    merged = dict(base.metadata)
    merged.update(dict(metadata))
    panel = base.panel.copy()
    panel.attrs["macroforecast_metadata"] = merged
    return replace(base, panel=panel, metadata=merged)


def _resolve_targets(
    panel: pd.DataFrame,
    *,
    base: _InputBundle,
    target: str | None,
    targets: Iterable[str] | None,
) -> tuple[str, ...]:
    if target is not None and targets is not None:
        raise ValueError("provide either target or targets, not both")
    if targets is not None:
        values = _normalize_string_iterable(targets, name="targets")
    elif target is not None:
        values = (str(target),)
    elif base.targets:
        values = tuple(str(value) for value in base.targets)
    elif base.target:
        values = (str(base.target),)
    else:
        raise ValueError("target is required; pass target=... or use mf.data.spec(..., target=...)")
    if not values:
        raise ValueError("targets must not be empty")
    missing = [value for value in values if value not in panel.columns]
    if missing:
        raise ValueError(f"target columns are not in the panel: {missing}")
    return values


def _resolve_horizons(
    *,
    base: _InputBundle,
    horizon: int | None,
    horizons: Iterable[int] | int | None,
) -> tuple[int, ...]:
    if horizon is not None and horizons is not None:
        raise ValueError("provide either horizon or horizons, not both")
    if horizons is not None:
        return _normalize_positive_ints(horizons, name="horizons")
    if horizon is not None:
        return _normalize_positive_ints((horizon,), name="horizon")
    if base.horizons:
        return _normalize_positive_ints(base.horizons, name="horizons")
    return (1,)


def _resolve_predictors(
    panel: pd.DataFrame,
    *,
    base: _InputBundle,
    predictors: Literal["all"] | Iterable[str] | None,
    targets: tuple[str, ...],
) -> tuple[str, ...]:
    if predictors is None:
        predictors = base.predictors
    if predictors == "all":
        values = tuple(str(column) for column in panel.columns if str(column) not in set(targets))
    else:
        values = _normalize_string_iterable(predictors, name="predictors")
    if not values:
        raise ValueError("predictors must not be empty")
    overlap = sorted(set(values).intersection(targets))
    if overlap:
        raise ValueError(f"predictors must not include target columns: {overlap}")
    missing = [value for value in values if value not in panel.columns]
    if missing:
        raise ValueError(f"predictor columns are not in the panel: {missing}")
    return values


def _resolve_columns(panel: pd.DataFrame, *, columns: Iterable[str] | None) -> tuple[str, ...]:
    if columns is None:
        return tuple(str(column) for column in panel.columns)
    values = _normalize_string_iterable(columns, name="columns")
    if not values:
        raise ValueError("columns must not be empty")
    missing = [value for value in values if value not in panel.columns]
    if missing:
        raise ValueError(f"columns are not in the panel: {missing}")
    return values


def _normalize_lags(values: Iterable[int] | int, *, allow_zero: bool) -> tuple[int, ...]:
    if isinstance(values, int):
        if values < 0:
            raise ValueError("lags must be non-negative")
        if values == 0 and allow_zero:
            normalized = (0,)
        else:
            normalized = tuple(range(1, values + 1))
    else:
        normalized = tuple(dict.fromkeys(int(value) for value in values))
    if not normalized:
        raise ValueError("lags must not be empty")
    minimum = 0 if allow_zero else 1
    invalid = [value for value in normalized if value < minimum]
    if invalid:
        raise ValueError(f"lags must be >= {minimum}; got {invalid}")
    return normalized


def _prepend_zero_lag(values: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(dict.fromkeys((0, *values)))


def _normalize_maf_lags(*, max_lag: int, lags: Iterable[int] | None) -> tuple[int, ...]:
    if lags is None:
        value = int(max_lag)
        if value < 0:
            raise ValueError("max_lag must be non-negative")
        return tuple(range(0, value + 1))
    return _normalize_lags(lags, allow_zero=True)


def _normalize_positive_ints(values: Iterable[int] | int, *, name: str) -> tuple[int, ...]:
    if isinstance(values, int):
        normalized = (int(values),)
    else:
        normalized = tuple(dict.fromkeys(int(value) for value in values))
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    invalid = [value for value in normalized if value <= 0]
    if invalid:
        raise ValueError(f"{name} must contain positive integers; got {invalid}")
    return normalized


def _power_of_two_windows(max_window: int) -> tuple[int, ...]:
    value = int(max_window)
    if value <= 0:
        raise ValueError("max_window must be positive")
    windows: list[int] = []
    current = 1
    while current <= value:
        windows.append(current)
        current *= 2
    return tuple(windows)


def _normalize_string_iterable(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be an iterable of strings, not a single string")
    return tuple(dict.fromkeys(str(value) for value in values))


def _normalize_column_groups(groups: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    if not isinstance(groups, Mapping):
        raise TypeError("groups must be a mapping from group name to column names")
    if not groups:
        raise ValueError("groups must not be empty")
    result: dict[str, tuple[str, ...]] = {}
    for raw_name, raw_columns in groups.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("group names must be non-empty")
        if name in result:
            raise ValueError(f"duplicate group name after normalization: {name!r}")
        columns = _normalize_string_iterable(raw_columns, name=f"groups[{name!r}]")
        if not columns:
            raise ValueError(f"group {name!r} must contain at least one column")
        result[name] = columns
    return result


def _resolve_group_components(
    n_components: int | Mapping[str, int],
    *,
    groups: Mapping[str, tuple[str, ...]],
) -> dict[str, int]:
    if isinstance(n_components, Mapping):
        values: dict[str, int] = {}
        for group_name in groups:
            if group_name not in n_components:
                raise ValueError(f"n_components is missing group {group_name!r}")
            value = int(n_components[group_name])
            if value <= 0:
                raise ValueError(f"n_components for group {group_name!r} must be positive")
            values[group_name] = value
        extra = sorted(set(map(str, n_components)).difference(groups))
        if extra:
            raise ValueError(f"n_components contains unknown groups: {extra}")
        return values
    value = int(n_components)
    if value <= 0:
        raise ValueError("n_components must be positive")
    return {group_name: value for group_name in groups}


def _target_column_name(name: str, *, horizon: int, transform: str) -> str:
    return f"{name}_{transform}_h{horizon}"


def _path_target_column_name(name: str, *, step: int, transform: str) -> str:
    return f"{name}_{transform}_step{step}"


def _normalize_target_transform(value: str) -> TargetTransform:
    aliases = {
        "level": "level",
        "future_level": "level",
        "change": "change",
        "diff": "change",
        "growth": "growth",
        "pct_change": "growth",
        "simple_growth": "growth",
        "log_growth": "log_growth",
        "log_change": "log_growth",
        "log_diff": "log_growth",
        "average_change": "average_change",
        "avg_change": "average_change",
        "mean_change": "average_change",
        "direct_average_change": "average_change",
        "average_growth": "average_growth",
        "avg_growth": "average_growth",
        "mean_growth": "average_growth",
        "direct_average_growth": "average_growth",
        "average_log_growth": "average_log_growth",
        "avg_log_growth": "average_log_growth",
        "mean_log_growth": "average_log_growth",
        "direct_average_log_growth": "average_log_growth",
    }
    if not isinstance(value, str):
        raise TypeError("target transform must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"target transform must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]  # type: ignore[return-value]


def _normalize_path_transform(value: str) -> PathTransform:
    aliases = {
        "change": "change",
        "diff": "change",
        "growth": "growth",
        "pct_change": "growth",
        "simple_growth": "growth",
        "log_growth": "log_growth",
        "log_change": "log_growth",
        "log_diff": "log_growth",
    }
    if not isinstance(value, str):
        raise TypeError("path transform must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"path transform must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]  # type: ignore[return-value]


def _one_period_future_transform(series: pd.Series, *, step: int, transform: PathTransform) -> pd.Series:
    current = series.shift(-(step - 1))
    future = series.shift(-step)
    if transform == "change":
        return future - current
    result = pd.Series(np.nan, index=series.index, dtype=float)
    if transform == "growth":
        valid = current.notna() & future.notna() & (current != 0)
        result.loc[valid] = future.loc[valid] / current.loc[valid] - 1.0
        return result
    if transform == "log_growth":
        valid = current.notna() & future.notna() & (current > 0) & (future > 0)
        result.loc[valid] = np.log(future.loc[valid]) - np.log(current.loc[valid])
        return result
    raise ValueError(f"unsupported path transform {transform!r}")


def _average_future_path(series: pd.Series, *, horizon: int, transform: PathTransform) -> pd.Series:
    components = [
        _one_period_future_transform(series, step=step, transform=transform)
        for step in range(1, int(horizon) + 1)
    ]
    return pd.concat(components, axis=1).mean(axis=1, skipna=False)


def _normalize_target_mode(value: str) -> TargetMode:
    aliases = {
        "direct": "direct",
        "direct_average": "direct",
        "single_target": "direct",
        "path": "path",
        "path_average": "path",
        "path_avg": "path",
    }
    if not isinstance(value, str):
        raise TypeError("target_mode must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"target_mode must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]  # type: ignore[return-value]


def _target_transform_to_path_transform(value: str) -> PathTransform:
    transform = _normalize_target_transform(value)
    if transform.startswith("average_"):
        return _normalize_path_transform(transform.removeprefix("average_"))
    if transform in {"change", "growth", "log_growth"}:
        return transform  # type: ignore[return-value]
    raise ValueError("target_mode='path' requires change, growth, log_growth, or an average_* transform")


def _maf_component_prefix(column: str, *, prefix: str) -> str:
    label = str(prefix).strip()
    return f"{column}_{label}" if label else f"{column}_maf"


def _group_component_prefix(group_name: str, *, prefix: str | None) -> str:
    if prefix is None or not str(prefix).strip():
        return str(group_name)
    return f"{prefix}_{group_name}"


def _normalize_feature_matrix_specification(specification: str | Iterable[str]) -> tuple[str, ...]:
    aliases = {
        "X": "X",
        "F": "F",
        "FACTOR": "F",
        "FACTORS": "F",
        "PC": "F",
        "PCS": "F",
        "MARX": "MARX",
        "MAF": "MAF",
        "H": "LEVEL",
        "LEVEL": "LEVEL",
        "LEVELS": "LEVEL",
    }
    if isinstance(specification, str):
        raw_values = [
            part.strip()
            for part in specification.replace("+", "-").replace("_", "-").split("-")
            if part.strip()
        ]
    else:
        raw_values = [str(value).strip() for value in specification]
    if not raw_values:
        raise ValueError("feature matrix specification must not be empty")

    blocks: list[str] = []
    for raw_value in raw_values:
        key = raw_value.upper()
        if key not in aliases:
            raise ValueError(f"unsupported feature matrix block {raw_value!r}; allowed blocks: {sorted(aliases)}")
        block = aliases[key]
        if block not in blocks:
            blocks.append(block)
    return tuple(blocks)


def _prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [f"{prefix}__{column}" for column in result.columns]
    return result


def _feature_matrix_records(
    frame: pd.DataFrame,
    *,
    block: str,
    source_columns: tuple[str, ...],
    fit_policy: str | None = None,
    scale_marx: bool | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column in frame.columns:
        feature = str(column)
        inner = feature.split("__", 1)[1] if "__" in feature else feature
        source = _source_for_feature(inner, source_columns)
        lag = _parse_lag(inner)
        window = _parse_window(inner)
        component = _parse_component(inner)
        operation = {
            "X": "lag",
            "F": "factor_lag",
            "MARX": "marx",
            "MAF": "maf",
            "LEVEL": "level_lag",
        }.get(block, f"feature_matrix.{block.lower()}")
        parameter_parts: list[str] = []
        if lag is not None:
            parameter_parts.append(f"lag={lag}")
        if window is not None:
            parameter_parts.append(f"window={window}")
        if component is not None:
            parameter_parts.append(f"component={component}")
        if scale_marx is not None:
            parameter_parts.append(f"scale_lags={bool(scale_marx)}")
        records.append(
            {
                "feature": feature,
                "block": block,
                "operation": operation,
                "source": source or block,
                "parameter": ";".join(parameter_parts) or None,
                "lag": lag,
                "window": window,
                "component": component,
                "fit_policy": fit_policy,
                "inputs": ",".join(source_columns),
                "included": True,
            }
        )
    return records


def _normalize_feature_method(value: str) -> str:
    aliases = {
        "lag": "lag",
        "lags": "lag",
        "rolling_mean": "rolling_mean",
        "ma_window": "rolling_mean",
        "moving_average": "rolling_mean",
        "moving_average_ladder": "moving_average_ladder",
        "ma_ladder": "moving_average_ladder",
        "marx_ladder": "moving_average_ladder",
        "scale": "scale",
        "standardize": "scale",
        "maf": "maf",
        "moving_average_factors": "maf",
        "group_pca": "group_pca",
        "grouped_pca": "group_pca",
        "pca": "pca",
        "principal_components": "pca",
        "time": "time",
        "time_features": "time",
    }
    if not value:
        raise ValueError("feature step requires method")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"feature method must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]


def _normalize_scale_method(value: str) -> str:
    aliases = {
        "zscore": "zscore",
        "standard": "zscore",
        "standardize": "zscore",
        "minmax": "minmax",
        "min_max": "minmax",
        "robust": "robust",
    }
    if not isinstance(value, str):
        raise TypeError("scale method must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"scale method must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]


def _normalize_fit_policy(value: str) -> FitPolicy:
    aliases = {
        "expanding": "expanding",
        "expanding_window": "expanding",
        "walk_forward": "expanding",
        "full_sample": "full_sample",
        "full_sample_once": "full_sample",
    }
    if not isinstance(value, str):
        raise TypeError("fit_policy must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"fit_policy must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]  # type: ignore[return-value]


def _normalize_min_train_size(value: int | None, *, minimum: int) -> int:
    if value is None:
        return max(int(minimum), 5)
    size = int(value)
    if size < minimum:
        raise ValueError(f"min_train_size must be >= {minimum}")
    return size


def _records_for_columns(
    frame: pd.DataFrame,
    *,
    operation: str,
    sources: tuple[str, ...],
    block: str | None = None,
    fit_policy: str | None = None,
    included: bool | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column in frame.columns:
        name = str(column)
        source = _source_for_feature(name, sources)
        lag = _parse_lag(name)
        window = _parse_window(name)
        component = _parse_component(name) if operation in {"pca", "group_pca", "maf", "factor_lag"} else None
        records.append(
            {
                "feature": name,
                "block": block,
                "operation": operation,
                "source": source,
                "parameter": _feature_parameter(name, operation=operation),
                "lag": lag,
                "window": window,
                "component": component,
                "fit_policy": fit_policy,
                "inputs": ",".join(sources),
                "included": included,
            }
        )
    return records


def _feature_parameter(name: str, *, operation: str) -> str | None:
    lag = _parse_lag(name)
    window = _parse_window(name)
    component = _parse_component(name)
    parts: list[str] = []
    if lag is not None:
        parts.append(f"lag={lag}")
    if window is not None:
        parts.append(f"window={window}")
    if component is not None and operation in {"pca", "group_pca", "maf", "factor_lag"}:
        parts.append(f"component={component}")
    if operation in {"scale"}:
        parts.append("method=scale")
    return ";".join(parts) or None


def _component_records(
    frame: pd.DataFrame,
    *,
    operation: str,
    source: str,
    inputs: tuple[str, ...],
    fit_policy: str | None,
    block: str | None = None,
    included: bool | None = True,
) -> list[dict[str, Any]]:
    return [
        {
            "feature": str(column),
            "block": block,
            "operation": operation,
            "source": source,
            "parameter": f"component={idx}",
            "component": idx,
            "fit_policy": fit_policy,
            "inputs": ",".join(inputs),
            "included": included,
        }
        for idx, column in enumerate(frame.columns, start=1)
    ]


def _target_record(
    *,
    target_column: str,
    source: str,
    horizon: int | None,
    step: int | None,
    mode: str,
    transform: str,
    operation: str,
    formula: str,
    aggregation: str | None,
    used_for_horizons: Iterable[int],
) -> dict[str, Any]:
    return {
        "target_column": str(target_column),
        "source": str(source),
        "horizon": int(horizon) if horizon is not None else None,
        "step": int(step) if step is not None else None,
        "mode": str(mode),
        "transform": str(transform),
        "operation": str(operation),
        "formula": str(formula),
        "aggregation": aggregation,
        "used_for_horizons": ",".join(str(int(value)) for value in used_for_horizons),
    }


def _target_metadata_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(list(records))
    for column in _TARGET_METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    extra_columns = [column for column in frame.columns if column not in _TARGET_METADATA_COLUMNS]
    return frame.loc[:, list(_TARGET_METADATA_COLUMNS) + extra_columns]


def _metadata_frame(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(list(records))
    for column in _FEATURE_METADATA_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    extra_columns = [column for column in frame.columns if column not in _FEATURE_METADATA_COLUMNS]
    return frame.loc[:, list(_FEATURE_METADATA_COLUMNS) + extra_columns]


def _source_for_feature(name: str, sources: tuple[str, ...]) -> str | None:
    for source in sorted(sources, key=len, reverse=True):
        if name == source or name.startswith(f"{source}_"):
            return source
    return None


def _parse_lag(name: str) -> int | None:
    marker = "_lag"
    if marker not in name:
        return None
    suffix = name.rsplit(marker, 1)[1]
    digits = ""
    for char in suffix:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


def _parse_window(name: str) -> int | None:
    for marker in ("_roll", "_ma"):
        if marker not in name:
            continue
        suffix = name.split(marker, 1)[1]
        digits = ""
        for char in suffix:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return int(digits)
    return None


def _parse_component(name: str) -> int | None:
    for marker in ("_maf", "pc", "F"):
        if marker == "F" and not name.startswith("F"):
            continue
        if marker != "F" and marker not in name:
            continue
        suffix = name.split(marker, 1)[1]
        digits = ""
        for char in suffix:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            return int(digits)
    trailing = ""
    for char in reversed(name):
        if char.isdigit():
            trailing = char + trailing
        else:
            break
    return int(trailing) if trailing else None


def _warn_if_no_preprocessing_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("preprocessing"):
        return
    warnings.warn(
        "feature engineering works best with PreprocessedData from mf.preprocessing.reprocess(). "
        "Proceeding with the supplied canonical panel.",
        UserWarning,
        stacklevel=3,
    )


def _warn_if_full_sample_fit(fit_policy: str, *, context: str, enabled: bool) -> None:
    if not enabled or _normalize_fit_policy(fit_policy) != "full_sample":
        return
    warnings.warn(
        f"{context} uses fit_policy='full_sample'. This fits transformation parameters on the "
        "whole supplied sample; use only for exploratory work or data that has already been "
        "split into a training-only panel.",
        UserWarning,
        stacklevel=3,
    )


def _target_formula(source: str, *, horizon: int, transform: str) -> str:
    if transform == "level":
        return f"{source}[t+{horizon}]"
    if transform == "change":
        return f"{source}[t+{horizon}] - {source}[t]"
    if transform == "growth":
        return f"{source}[t+{horizon}] / {source}[t] - 1"
    if transform == "log_growth":
        return f"log({source}[t+{horizon}]) - log({source}[t])"
    if transform.startswith("average_"):
        inner = transform.removeprefix("average_")
        return f"mean({_path_target_formula(source, step='s', transform=inner)} for s=1..{horizon})"
    raise ValueError(f"unsupported target transform {transform!r}")


def _path_target_formula(source: str, *, step: int | str, transform: str) -> str:
    previous = f"{source}[t+{step}-1]" if isinstance(step, str) else f"{source}[t+{step - 1}]"
    current = f"{source}[t+{step}]"
    if transform == "change":
        return f"{current} - {previous}"
    if transform == "growth":
        return f"{current} / {previous} - 1"
    if transform == "log_growth":
        return f"log({current}) - log({previous})"
    raise ValueError(f"unsupported path transform {transform!r}")
