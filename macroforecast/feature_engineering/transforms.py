from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata, validate_panel
from macroforecast.feature_engineering.types import FeatureInput
from macroforecast.feature_engineering.shared import (
    FitPolicy,
    _coerce_input,
    _component_records,
    _group_component_prefix,
    _maf_component_prefix,
    _metadata_frame,
    _normalize_column_groups,
    _normalize_fit_policy,
    _normalize_lags,
    _normalize_maf_lags,
    _normalize_min_train_size,
    _normalize_positive_ints,
    _normalize_scale_method,
    _pca_frame,
    _power_of_two_windows,
    _records_for_columns,
    _resolve_columns,
    _resolve_group_components,
    _scale_frame,
    _source_for_feature,
    _warn_if_full_sample_fit,
)

def lag(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (1,),
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create lagged predictor columns from a canonical panel.

    ``lags=3`` means lags ``1, 2, 3``. Pass an iterable such as ``(0, 1, 3)``
    when the current value and specific lag lengths should be included.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    lag_values = _normalize_lags(lags, allow_zero=True)

    pieces: list[pd.Series] = []
    for column in selected:
        for lag in lag_values:
            feature = panel[column].shift(lag).rename(f"{column}_lag{lag}")
            feature.attrs = {}
            pieces.append(feature)
    result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_lag",
        {"columns": list(selected), "lags": list(lag_values), "drop_missing": bool(drop_missing)},
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _records_for_columns(result, operation="lag", sources=selected, included=True)
    )
    return result


def rolling_mean(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    windows: Iterable[int] | int = (3,),
    min_periods: int | None = None,
    shift: int = 0,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create rolling-mean feature columns from a canonical panel."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    window_values = _normalize_positive_ints(windows, name="windows")
    if min_periods is not None and int(min_periods) <= 0:
        raise ValueError("min_periods must be positive")
    if int(shift) < 0:
        raise ValueError("shift must be non-negative")
    shift_value = int(shift)

    pieces: list[pd.Series] = []
    for column in selected:
        base_series = panel[column].shift(shift_value) if shift_value else panel[column]
        for window in window_values:
            required = min_periods if min_periods is not None else window
            feature = base_series.rolling(window=window, min_periods=required).mean()
            suffix = f"{column}_roll{window}_mean"
            if shift_value:
                suffix = f"{suffix}_lag{shift_value}"
            feature = feature.rename(suffix)
            feature.attrs = {}
            pieces.append(feature)
    result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_rolling_mean",
        {
            "columns": list(selected),
            "windows": list(window_values),
            "min_periods": min_periods,
            "shift": shift_value,
            "drop_missing": bool(drop_missing),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _records_for_columns(result, operation="rolling_mean", sources=selected, included=True)
    )
    return result


def moving_average_ladder(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    windows: Iterable[int] | None = None,
    max_window: int = 12,
    min_periods: int | None = None,
    shift: int = 0,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create a multi-scale trailing moving-average feature block.

    This function is the moving-average ladder used by MARX-style macro-ML
    feature pipelines. In this package, the paper notation ``marx_features(P)``
    is expressed as ``moving_average_ladder(..., windows=range(1, P + 1),
    shift=1)``: increasing-order moving averages of lagged ``X``. With the
    default ``max_window=12`` it builds windows ``1, 2, 4, 8``. It does not run
    PCA. Moving-average PCA is a separate composition: first call
    ``moving_average_ladder(...)`` to create the stacked moving-average block,
    then apply a fit-aware PCA/factor step.

    The default ladder uses powers of two because those windows give a compact
    short/medium/long persistence basis without manually choosing every lag.
    Pass ``windows=...`` for an exact window set such as ``(1, 2, 4, 8, 12)``.
    """

    if windows is None:
        window_values = _power_of_two_windows(max_window)
    else:
        window_values = _normalize_positive_ints(windows, name="windows")
    result = rolling_mean(
        data,
        metadata=metadata,
        columns=columns,
        windows=window_values,
        min_periods=min_periods,
        shift=shift,
        drop_missing=drop_missing,
    )
    result = result.rename(
        columns={
            str(column): str(column).replace("_roll", "_ma").replace("_mean", "")
            for column in result.columns
        }
    )
    base = _coerce_input(data, metadata=metadata)
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_moving_average_ladder",
        {
            "columns": list(_resolve_columns(base.panel, columns=columns)),
            "windows": list(window_values),
            "max_window": int(max_window),
            "min_periods": min_periods,
            "shift": int(shift),
            "drop_missing": bool(drop_missing),
            "note": (
                "Moving-average ladder only. PCA/factor extraction is a separate "
                "fit-aware step to avoid full-sample leakage."
            ),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _records_for_columns(result, operation="moving_average_ladder", sources=tuple(_resolve_columns(base.panel, columns=columns)), included=True)
    )
    return result


def scale_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    method: str = "zscore",
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Scale features with either expanding or explicit full-sample fitting.

    ``fit_policy='expanding'`` estimates scaling parameters using observations
    available through each row's date. ``fit_policy='full_sample'`` is useful
    for exploratory transforms but should not be used before a forecasting
    split unless the caller deliberately accepts full-sample leakage.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    source = panel.loc[:, selected]
    method_value = _normalize_scale_method(method)
    fit_value = _normalize_fit_policy(fit_policy)
    _warn_if_full_sample_fit(fit_value, context="scale_features()", enabled=warn_full_sample)
    min_size = _normalize_min_train_size(min_train_size, minimum=2)
    result = _scale_frame(source, method=method_value, fit_policy=fit_value, min_train_size=min_size)
    result = result.add_suffix(f"_{method_value}")
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_scale",
        {
            "columns": list(selected),
            "method": method_value,
            "fit_policy": fit_value,
            "min_train_size": min_size,
            "drop_missing": bool(drop_missing),
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "scale",
            "source": _source_for_feature(str(feature).removesuffix(f"_{method_value}"), selected),
            "parameter": f"method={method_value}",
            "fit_policy": fit_value,
            "inputs": ",".join(selected),
            "included": True,
        }
        for feature in result.columns
    )
    return result


def pca_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 1,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str = "pc",
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create principal-component features with a declared fit policy.

    PCA is a fitted transformation, so the default is expanding-window fitting.
    Use ``fit_policy='full_sample'`` only for exploratory analysis or after an
    external split has already made the input sample training-only.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    source = panel.loc[:, selected]
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    if n_value > len(selected):
        raise ValueError("n_components must be <= the number of selected columns")
    fit_value = _normalize_fit_policy(fit_policy)
    _warn_if_full_sample_fit(fit_value, context="pca_features()", enabled=warn_full_sample)
    min_size = _normalize_min_train_size(min_train_size, minimum=n_value + 1)
    result = _pca_frame(
        source,
        n_components=n_value,
        fit_policy=fit_value,
        min_train_size=min_size,
        scale=bool(scale),
        prefix=str(prefix),
        random_state=random_state,
    )
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_pca",
        {
            "columns": list(selected),
            "n_components": n_value,
            "fit_policy": fit_value,
            "min_train_size": min_size,
            "scale": bool(scale),
            "prefix": str(prefix),
            "drop_missing": bool(drop_missing),
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="pca",
            source=",".join(selected),
            inputs=selected,
            fit_policy=fit_value,
        )
    )
    return result


def group_pca(
    data: FeatureInput,
    *,
    groups: Mapping[str, Iterable[str]],
    metadata: Mapping[str, Any] | None = None,
    n_components: int | Mapping[str, int] = 1,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = True,
    prefix: str | None = None,
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create PCA factors separately within named column groups.

    This is useful when factors should be extracted from economically
    meaningful blocks rather than from the whole predictor panel at once.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    group_map = _normalize_column_groups(groups)
    component_counts = _resolve_group_components(n_components, groups=group_map)
    fit_value = _normalize_fit_policy(fit_policy)
    _warn_if_full_sample_fit(fit_value, context="group_pca()", enabled=warn_full_sample)

    pieces: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    group_records: list[dict[str, Any]] = []
    for group_name, group_columns in group_map.items():
        selected = _resolve_columns(panel, columns=group_columns)
        n_value = component_counts[group_name]
        if n_value > len(selected):
            raise ValueError(
                f"n_components for group {group_name!r} must be <= the number of group columns"
            )
        min_size = _normalize_min_train_size(min_train_size, minimum=n_value + 1)
        source = panel.loc[:, selected]
        component_prefix = _group_component_prefix(group_name, prefix=prefix)
        transformed = _pca_frame(
            source,
            n_components=n_value,
            fit_policy=fit_value,
            min_train_size=min_size,
            scale=bool(scale),
            prefix=component_prefix,
            random_state=random_state,
        )
        pieces.append(transformed)
        records.extend(
            {
                "feature": str(feature),
                "block": None,
                "operation": "group_pca",
                "source": str(group_name),
                "parameter": f"columns={list(selected)};component={idx}",
                "component": idx,
                "fit_policy": fit_value,
                "inputs": ",".join(selected),
                "included": True,
            }
            for idx, feature in enumerate(transformed.columns, start=1)
        )
        group_records.append(
            {
                "group": group_name,
                "columns": list(selected),
                "n_components": n_value,
                "output_columns": [str(column) for column in transformed.columns],
            }
        )

    result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)
    if result.columns.has_duplicates:
        duplicate_columns = result.columns[result.columns.duplicated()].unique()
        raise ValueError(f"group PCA produced duplicate columns: {list(map(str, duplicate_columns))}")
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    metadata_payload = attach_metadata(
        base.metadata,
        "feature_engineering_group_pca",
        {
            "groups": group_records,
            "fit_policy": fit_value,
            "min_train_size": min_train_size,
            "scale": bool(scale),
            "prefix": prefix,
            "drop_missing": bool(drop_missing),
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_metadata"] = metadata_payload
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(records)
    return result


def maf_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    max_lag: int = 12,
    lags: Iterable[int] | None = None,
    n_components: int = 2,
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    scale: bool = False,
    prefix: str = "maf",
    drop_missing: bool = False,
    random_state: int | None = None,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create Moving Average Factors from variable-specific lag panels.

    For each selected variable ``x_k``, this builds
    ``[x_k, L x_k, ..., L^P x_k]`` and extracts PCA components from that
    variable-specific lag panel. This is the MAF construction from
    macro-ML forecasting papers; it is not global PCA over all variables.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    lag_values = _normalize_maf_lags(max_lag=max_lag, lags=lags)
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    if n_value > len(lag_values):
        raise ValueError("n_components must be <= the number of MAF lag columns")
    fit_value = _normalize_fit_policy(fit_policy)
    _warn_if_full_sample_fit(fit_value, context="maf_features()", enabled=warn_full_sample)
    min_size = _normalize_min_train_size(min_train_size, minimum=n_value + 1)

    pieces: list[pd.DataFrame] = []
    feature_records: list[dict[str, Any]] = []
    for column in selected:
        lag_block = pd.DataFrame(
            {
                f"{column}_lag{lag}": panel[column].shift(lag)
                for lag in lag_values
            },
            index=panel.index,
        )
        component_prefix = _maf_component_prefix(column, prefix=prefix)
        transformed = _pca_frame(
            lag_block,
            n_components=n_value,
            fit_policy=fit_value,
            min_train_size=min_size,
            scale=bool(scale),
            prefix=component_prefix,
            random_state=random_state,
        )
        pieces.append(transformed)
        feature_records.extend(
            {
                "feature": str(feature),
                "operation": "maf",
                "source": str(column),
                "parameter": f"lags={list(lag_values)};component={idx}",
                "component": idx,
                "fit_policy": fit_value,
                "inputs": ",".join(f"{column}_lag{lag}" for lag in lag_values),
                "included": True,
            }
            for idx, feature in enumerate(transformed.columns, start=1)
        )

    result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    metadata_payload = attach_metadata(
        base.metadata,
        "feature_engineering_maf",
        {
            "columns": list(selected),
            "lags": list(lag_values),
            "max_lag": int(max_lag),
            "n_components": n_value,
            "fit_policy": fit_value,
            "min_train_size": min_size,
            "scale": bool(scale),
            "prefix": str(prefix),
            "drop_missing": bool(drop_missing),
            "warn_full_sample": bool(warn_full_sample),
            "note": (
                "MAF extracts PCA components separately within each variable's "
                "own lag panel; it is not global PCA over all variables."
            ),
        },
    )
    result.attrs["macroforecast_metadata"] = metadata_payload
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(feature_records)
    return result


def time_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    trend: bool = True,
    month: bool = False,
    quarter: bool = False,
    year: bool = False,
) -> pd.DataFrame:
    """Create deterministic date-index features."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    result = pd.DataFrame(index=panel.index)
    if trend:
        result["trend"] = np.arange(1, len(panel) + 1, dtype=float)
    if month:
        for value in range(1, 13):
            result[f"month_{value:02d}"] = (panel.index.month == value).astype(float)
    if quarter:
        for value in range(1, 5):
            result[f"quarter_{value}"] = (panel.index.quarter == value).astype(float)
    if year:
        result["year"] = panel.index.year.astype(float)
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_time",
        {"trend": bool(trend), "month": bool(month), "quarter": bool(quarter), "year": bool(year)},
    )
    records: list[dict[str, Any]] = []
    for column in result.columns:
        feature = str(column)
        if feature == "trend":
            parameter = "trend=True"
        elif feature.startswith("month_"):
            parameter = f"month={feature.removeprefix('month_')}"
        elif feature.startswith("quarter_"):
            parameter = f"quarter={feature.removeprefix('quarter_')}"
        elif feature == "year":
            parameter = "year=True"
        else:
            parameter = None
        records.append(
            {
                "feature": feature,
                "operation": "time",
                "source": "date",
                "parameter": parameter,
                "inputs": "date",
                "included": True,
            }
        )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(records)
    return result
