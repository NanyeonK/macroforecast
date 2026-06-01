from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from itertools import combinations, combinations_with_replacement
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.kernel_approximation import Nystroem
from sklearn.random_projection import GaussianRandomProjection
from statsmodels.tsa.filters.hp_filter import hpfilter

from macroforecast.data import attach_metadata, validate_panel
from macroforecast.feature_engineering.types import FeatureInput
from macroforecast.feature_engineering.shared import (
    FitPolicy,
    _coerce_input,
    _component_records,
    _apply_sparse_pca_chen_rohe,
    _apply_varimax_rotation,
    _effective_pls_components,
    _fit_sparse_factor_var1,
    _fit_sparse_pca_chen_rohe,
    _fit_varimax_rotation,
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


def mixed_frequency_lags(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    target: str | None = None,
    anchor_dates: Iterable[Any] | None = None,
    columns: Iterable[str] | None = None,
    lags: Iterable[int] | int = (0, 1, 2),
    frequency_by_column: Mapping[str, str] | None = None,
    target_frequency: str | None = None,
    anchor_position: str = "date",
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Build exact-date lag blocks for mixed-frequency regressions.

    Lags are measured in each source column's native frequency. For example,
    monthly predictors with ``lags=(0, 1, 2)`` produce the current, previous,
    and two-month-lag values at each target anchor date. Quarterly predictors
    with the same lags use quarter steps.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    target_name = str(target) if target is not None else base.target
    if target_name is not None and target_name not in panel.columns:
        raise ValueError(f"target {target_name!r} is not in the panel")
    selected = _resolve_mixed_lag_columns(panel, columns=columns, target=target_name)
    lag_values = _normalize_lags(lags, allow_zero=True)
    frequency_map = _mixed_frequency_map(
        panel,
        metadata=base.metadata,
        frequency_by_column=frequency_by_column,
    )
    anchor_index = _resolve_anchor_dates(
        panel,
        target=target_name,
        anchor_dates=anchor_dates,
        target_frequency=target_frequency,
        frequency_map=frequency_map,
        anchor_position=anchor_position,
    )

    result = pd.DataFrame(index=anchor_index)
    records: list[dict[str, Any]] = []
    n_rows_before_drop = int(len(result))
    for column in selected:
        native_frequency = frequency_map.get(column, "unknown")
        source = _source_series_by_period(
            panel[column],
            frequency=native_frequency,
        )
        for lag_value in lag_values:
            lookup_dates = _lagged_lookup_dates(anchor_index, lag=lag_value, frequency=native_frequency)
            values = source.reindex(lookup_dates).to_numpy(dtype=float)
            feature_name = f"{column}_lag{lag_value}"
            result[feature_name] = values
            records.append(
                {
                    "feature": feature_name,
                    "operation": "mixed_frequency_lag",
                    "source": column,
                    "parameter": f"frequency={native_frequency};lag={lag_value}",
                    "lag": int(lag_value),
                    "inputs": ",".join(selected),
                    "included": True,
                    "source_frequency": native_frequency,
                    "anchor_position": str(anchor_position),
                    "lookup_start": lookup_dates[0].strftime("%Y-%m-%d") if len(lookup_dates) else None,
                    "lookup_end": lookup_dates[-1].strftime("%Y-%m-%d") if len(lookup_dates) else None,
                    "lookup_calendar": "source_period_start",
                }
            )
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_mixed_frequency_lags",
        {
            "target": target_name,
            "anchor_dates": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in anchor_index],
            "columns": list(selected),
            "lags": list(lag_values),
            "target_frequency": target_frequency,
            "anchor_position": str(anchor_position),
            "frequency_by_column": {column: frequency_map.get(column, "unknown") for column in selected},
            "lookup_calendar": "source_period_start",
            "n_rows_before_drop": n_rows_before_drop,
            "n_rows_after_drop": int(len(result)),
            "drop_missing": bool(drop_missing),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(records)
    return result


def transform_features(
    data: FeatureInput,
    *,
    transform: str,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    periods: int = 1,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Apply a simple column-wise transformation as feature engineering.

    These helpers are separate from preprocessing t-codes. Use them when the
    model feature set needs extra ML transforms after the canonical panel has
    already been cleaned.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    period_value = int(periods)
    if period_value <= 0:
        raise ValueError("periods must be positive")
    transform_value = _normalize_simple_transform(transform)
    result = pd.DataFrame(index=panel.index)
    for column in selected:
        series = panel[column]
        if transform_value == "log":
            output = _positive_log(series)
        elif transform_value == "diff":
            output = series.diff(period_value)
        elif transform_value == "log_diff":
            output = _positive_log(series).diff(period_value)
        elif transform_value == "pct_change":
            output = series.pct_change(period_value)
        elif transform_value == "cumsum":
            output = series.cumsum()
        else:
            raise ValueError(f"unsupported transform {transform!r}")
        result[f"{column}_{transform_value}"] = output
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        f"feature_engineering_{transform_value}",
        {
            "columns": list(selected),
            "transform": transform_value,
            "periods": period_value,
            "drop_missing": bool(drop_missing),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": transform_value,
            "source": _source_for_feature(str(feature).removesuffix(f"_{transform_value}"), selected),
            "parameter": f"periods={period_value}",
            "inputs": ",".join(selected),
            "included": True,
        }
        for feature in result.columns
    )
    return result


def log_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create log features; non-positive values become missing."""

    return transform_features(
        data,
        transform="log",
        metadata=metadata,
        columns=columns,
        drop_missing=drop_missing,
    )


def diff_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    periods: int = 1,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create difference features."""

    return transform_features(
        data,
        transform="diff",
        metadata=metadata,
        columns=columns,
        periods=periods,
        drop_missing=drop_missing,
    )


def log_diff_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    periods: int = 1,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create log-difference features; non-positive values become missing."""

    return transform_features(
        data,
        transform="log_diff",
        metadata=metadata,
        columns=columns,
        periods=periods,
        drop_missing=drop_missing,
    )


def pct_change_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    periods: int = 1,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create simple-growth features."""

    return transform_features(
        data,
        transform="pct_change",
        metadata=metadata,
        columns=columns,
        periods=periods,
        drop_missing=drop_missing,
    )


def cumsum_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create cumulative-sum features."""

    return transform_features(
        data,
        transform="cumsum",
        metadata=metadata,
        columns=columns,
        drop_missing=drop_missing,
    )


def seasonal_lag(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    season_length: int = 12,
    lags: Iterable[int] | int = (1,),
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create seasonal lag features such as 12-month or 4-quarter lags."""

    season_length_value = int(season_length)
    if season_length_value <= 0:
        raise ValueError("season_length must be positive")
    seasonal_lag_values = _normalize_lags(lags, allow_zero=False)
    lag_values = tuple(season_length_value * value for value in seasonal_lag_values)
    result = lag(
        data,
        metadata=metadata,
        columns=columns,
        lags=lag_values,
        drop_missing=drop_missing,
    )
    result = result.rename(columns={column: column.replace("_lag", "_seasonlag") for column in result.columns})
    base = _coerce_input(data, metadata=metadata)
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_seasonal_lag",
        {
            "columns": list(_resolve_columns(base.panel, columns=columns)),
            "season_length": season_length_value,
            "seasonal_lags": list(seasonal_lag_values),
            "actual_lags": list(lag_values),
            "drop_missing": bool(drop_missing),
        },
    )
    selected = _resolve_columns(base.panel, columns=columns)
    records: list[dict[str, Any]] = []
    for column in selected:
        for seasonal_lag_value, actual_lag in zip(seasonal_lag_values, lag_values, strict=True):
            records.append(
                {
                    "feature": f"{column}_seasonlag{actual_lag}",
                    "operation": "seasonal_lag",
                    "source": column,
                    "parameter": (
                        f"season_length={season_length_value};"
                        f"seasonal_lag={seasonal_lag_value};actual_lag={actual_lag}"
                    ),
                    "lag": actual_lag,
                    "inputs": ",".join(selected),
                    "included": True,
                }
            )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(records)
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


def sparse_pca_chen_rohe_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 4,
    zeta: float = 0.0,
    max_iter: int = 200,
    var_innovations: bool = False,
    prefix: str | None = None,
    min_train_size: int | None = None,
    drop_missing: bool = False,
    random_state: int | None = 0,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create Chen-Rohe sparse component analysis factors.

    The loading matrix is learned from complete input rows and then applied to
    the same panel, so this direct callable is a full-sample transform. Use
    ``sparse_pca_chen_rohe_step()`` inside ``feature_spec()`` when the runner
    must fit loadings only on each training window.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    zeta_value = float(zeta)
    if zeta_value < 0:
        raise ValueError("zeta must be non-negative")
    iter_value = int(max_iter)
    if iter_value <= 0:
        raise ValueError("max_iter must be positive")
    minimum = 3 if var_innovations else 1
    min_size = _normalize_min_train_size(min_train_size, minimum=minimum)
    _warn_if_full_sample_fit(
        "full_sample",
        context="sparse_pca_chen_rohe_features()",
        enabled=warn_full_sample,
    )

    source = panel.loc[:, selected]
    complete = source.dropna().astype(float)
    resolved_components = max(1, min(n_value, len(complete), len(selected))) if len(complete) else n_value
    output_prefix = str(prefix) if prefix is not None else ("scaf" if var_innovations else "sca")
    if len(complete) < min_size:
        result = pd.DataFrame(
            index=panel.index,
            columns=[f"{output_prefix}{index}" for index in range(1, resolved_components + 1)],
            dtype=float,
        )
        result.index.name = "date"
        zeta_resolved = zeta_value if zeta_value > 0 else float(resolved_components)
        n_iter = 0
        objective = np.nan
        n_fit_rows = int(len(complete))
    else:
        center, theta, zeta_resolved, n_iter, objective = _fit_sparse_pca_chen_rohe(
            complete,
            n_components=n_value,
            zeta=zeta_value,
            max_iter=iter_value,
            random_state=random_state,
        )
        train_scores = (complete - center).to_numpy(dtype=float) @ theta
        var_coef = _fit_sparse_factor_var1(train_scores) if var_innovations else None
        if var_innovations and var_coef is None:
            raise ValueError("var_innovations requires at least three complete rows")
        result = _apply_sparse_pca_chen_rohe(
            source,
            columns=selected,
            center=center,
            theta=theta,
            prefix=output_prefix,
            var_coef=var_coef,
        )
        resolved_components = int(theta.shape[1])
        n_fit_rows = int(len(complete))
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_sparse_pca_chen_rohe",
        {
            "columns": list(selected),
            "n_components": n_value,
            "resolved_n_components": int(resolved_components),
            "zeta": zeta_value,
            "zeta_resolved": float(zeta_resolved),
            "max_iter": iter_value,
            "n_iter": int(n_iter),
            "objective": None if np.isnan(objective) else float(objective),
            "var_innovations": bool(var_innovations),
            "prefix": output_prefix,
            "min_train_size": min_size,
            "n_fit_rows": n_fit_rows,
            "drop_missing": bool(drop_missing),
            "random_state": random_state,
            "fit_policy": "full_input_complete_rows",
            "warn_full_sample": bool(warn_full_sample),
            "note": (
                "Chen-Rohe sparse component analysis with an L1 loading budget; "
                "this is not sklearn SparsePCA."
            ),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="sparse_pca_chen_rohe",
            source=",".join(selected),
            inputs=selected,
            fit_policy="full_input_complete_rows",
        )
    )
    return result


def varimax_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    max_iter: int = 50,
    tol: float = 1e-7,
    prefix: str = "varimax",
    min_train_size: int | None = None,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Rotate factor-score columns with an orthogonal varimax rotation.

    This direct callable fits the rotation on all complete rows. Use
    ``varimax_step()`` inside ``feature_spec()`` when the rotation must be fit
    only on a forecasting window's feature-fit panel.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    iter_value = int(max_iter)
    if iter_value <= 0:
        raise ValueError("max_iter must be positive")
    tol_value = float(tol)
    if tol_value < 0:
        raise ValueError("tol must be non-negative")
    min_size = _normalize_min_train_size(min_train_size, minimum=1)
    _warn_if_full_sample_fit(
        "full_sample",
        context="varimax_features()",
        enabled=warn_full_sample,
    )
    source = panel.loc[:, selected]
    complete = source.dropna().astype(float)
    if len(complete) < min_size:
        result = pd.DataFrame(
            index=panel.index,
            columns=[f"{prefix}{index}" for index in range(1, len(selected) + 1)],
            dtype=float,
        )
        result.index.name = "date"
        n_iter = 0
        n_fit_rows = int(len(complete))
    else:
        rotation, n_iter = _fit_varimax_rotation(complete, max_iter=iter_value, tol=tol_value)
        result = _apply_varimax_rotation(
            source,
            columns=selected,
            rotation=rotation,
            prefix=str(prefix),
        )
        n_fit_rows = int(len(complete))
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_varimax",
        {
            "columns": list(selected),
            "max_iter": iter_value,
            "tol": tol_value,
            "n_iter": int(n_iter),
            "prefix": str(prefix),
            "min_train_size": min_size,
            "n_fit_rows": n_fit_rows,
            "drop_missing": bool(drop_missing),
            "fit_policy": "full_input_complete_rows",
            "warn_full_sample": bool(warn_full_sample),
            "note": "Orthogonal varimax rotation for already-created factor-score columns.",
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="varimax",
            source=",".join(selected),
            inputs=selected,
            fit_policy="full_input_complete_rows",
        )
    )
    return result


def _univariate_slope(x: pd.Series, y: pd.Series) -> float:
    common = x.dropna().index.intersection(y.dropna().index)
    if len(common) < 2:
        return 0.0
    x_values = x.loc[common].astype(float).to_numpy()
    y_values = y.loc[common].astype(float).to_numpy()
    x_centered = x_values - float(np.mean(x_values))
    y_centered = y_values - float(np.mean(y_values))
    denominator = float(np.dot(x_centered, x_centered))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def sliced_inverse_regression_features(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 3,
    n_slices: int = 10,
    scaling_policy: str = "scaled_pca",
    prefix: str = "sir",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create Sliced Inverse Regression factors from a target signal.

    SIR is target-aware, so this direct callable fits on all target-aligned rows.
    For runner-safe use, call ``sliced_inverse_regression_step()`` inside
    ``feature_spec()`` so the directions are fitted only on each fit window.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    if isinstance(target, str):
        if target not in panel.columns:
            raise ValueError(f"target {target!r} is not in the panel")
        target_name = target
        target_series = panel[target_name]
    elif isinstance(target, pd.Series):
        target_name = target.name if target.name is not None else "target"
        target_series = target.astype(float)
    elif target is None and base.target is not None:
        if base.target not in panel.columns:
            raise ValueError(f"target {base.target!r} is not in the panel")
        target_name = base.target
        target_series = panel[target_name]
    else:
        raise ValueError("sliced_inverse_regression_features() requires target or input target metadata")

    if columns is None:
        selected = tuple(str(column) for column in panel.columns if str(column) != str(target_name))
        if not selected:
            raise ValueError("sliced_inverse_regression_features() requires at least one predictor column")
    else:
        selected = _resolve_columns(panel, columns=columns)
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    slice_value = int(n_slices)
    if slice_value < 2:
        raise ValueError("n_slices must be at least 2")
    scaling_value = str(scaling_policy)
    if scaling_value not in {"scaled_pca", "marginal_R2", "none"}:
        raise ValueError("scaling_policy must be 'scaled_pca', 'marginal_R2', or 'none'")
    _warn_if_full_sample_fit(
        "full_sample",
        context="sliced_inverse_regression_features()",
        enabled=warn_full_sample,
    )

    source = panel.loc[:, selected].astype(float)
    common_index = source.index.intersection(target_series.dropna().index)
    train_x = source.loc[common_index].dropna(axis=0, how="any")
    train_y = pd.Series(target_series, index=target_series.index).reindex(train_x.index).dropna()
    train_x = train_x.loc[train_y.index]
    if len(train_x) < 2:
        raise ValueError("sliced_inverse_regression_features() requires at least two target-aligned complete rows")
    n_effective = min(n_value, train_x.shape[1])

    center = train_x.mean(axis=0)
    divisor = train_x.std(axis=0, ddof=1).replace(0.0, np.nan).fillna(1.0)
    x_scaled = (train_x - center) / divisor
    beta = None
    if scaling_value in {"scaled_pca", "marginal_R2"}:
        beta = np.array([_univariate_slope(x_scaled[column], train_y) for column in x_scaled.columns], dtype=float)
        if scaling_value == "marginal_R2":
            beta = np.sign(beta) * np.abs(beta)
        x_scaled = x_scaled * beta

    order = np.argsort(train_y.to_numpy(dtype=float))
    z_sorted = x_scaled.to_numpy(dtype=float)[order]
    n_total = z_sorted.shape[0]
    n_slices_resolved = min(slice_value, n_total)
    slice_size = max(1, n_total // n_slices_resolved)
    slice_means: list[np.ndarray] = []
    slice_weights: list[float] = []
    for slice_index in range(n_slices_resolved):
        start = slice_index * slice_size
        end = (slice_index + 1) * slice_size if slice_index < n_slices_resolved - 1 else n_total
        values = z_sorted[start:end]
        if values.size == 0:
            slice_means.append(np.zeros(z_sorted.shape[1]))
            slice_weights.append(0.0)
        else:
            slice_means.append(values.mean(axis=0))
            slice_weights.append(values.shape[0] / max(n_total, 1))
    mean_matrix = np.vstack(slice_means)
    weights = np.asarray(slice_weights, dtype=float)
    between_slice_cov = (mean_matrix * weights[:, None]).T @ mean_matrix
    try:
        values, vectors = np.linalg.eigh(between_slice_cov)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate numerical fallback
        values = np.zeros(between_slice_cov.shape[0])
        vectors = np.eye(between_slice_cov.shape[0])
    selected_order = np.argsort(-np.abs(values))[:n_effective]
    directions = vectors[:, selected_order]
    for component in range(directions.shape[1]):
        max_index = int(np.argmax(np.abs(directions[:, component])))
        if directions[max_index, component] < 0:
            directions[:, component] = -directions[:, component]

    x_full = ((source - center) / divisor).fillna(0.0)
    if beta is not None:
        x_full = x_full * beta
    scores = x_full.to_numpy(dtype=float) @ directions
    if scores.shape[1] < n_value:
        scores = np.hstack([scores, np.zeros((scores.shape[0], n_value - scores.shape[1]))])
    result = pd.DataFrame(
        scores,
        index=panel.index,
        columns=[f"{prefix}{index}" for index in range(1, n_value + 1)],
    )
    if drop_missing:
        result = result.loc[source.dropna().index]
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_sliced_inverse_regression",
        {
            "target": str(target_name),
            "columns": list(selected),
            "n_components": n_value,
            "resolved_n_components": int(n_effective),
            "n_slices": slice_value,
            "resolved_n_slices": int(n_slices_resolved),
            "scaling_policy": scaling_value,
            "prefix": str(prefix),
            "n_fit_rows": int(len(train_x)),
            "drop_missing": bool(drop_missing),
            "missing_policy": "standardized_mean_fill_for_projection",
            "fit_policy": "full_input_target_aligned_rows",
            "warn_full_sample": bool(warn_full_sample),
            "note": (
                "Target-aware SIR direct callable. For runner-safe use, call "
                "sliced_inverse_regression_step() inside feature_spec()."
            ),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="sliced_inverse_regression",
            source=",".join(selected),
            inputs=selected,
            fit_policy="full_input_target_aligned_rows",
        )
    )
    return result


def partial_least_squares_features(
    data: FeatureInput,
    target: str | pd.Series | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 3,
    prefix: str = "pls",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create target-aware PLS latent-component scores."""

    from sklearn.cross_decomposition import PLSRegression

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    target_name, target_series = _resolve_target_argument(base, panel, target, context="partial_least_squares_features()")
    selected = (
        tuple(str(column) for column in panel.columns if str(column) != str(target_name))
        if columns is None
        else _resolve_columns(panel, columns=columns)
    )
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    source = panel.loc[:, selected].astype(float)
    joined = pd.concat([source, target_series.rename("__target__")], axis=1).dropna()
    if len(joined) < 2:
        raise ValueError("partial_least_squares_features() requires at least two target-aligned complete rows")
    resolved = _effective_pls_components(joined.loc[:, selected], n_value)
    _warn_if_full_sample_fit(
        "full_sample",
        context="partial_least_squares_features()",
        enabled=warn_full_sample,
    )
    result = pd.DataFrame(index=panel.index, dtype=float)
    if resolved > 0:
        model = PLSRegression(n_components=resolved)
        scores = model.fit_transform(
            joined.loc[:, selected].to_numpy(dtype=float),
            joined["__target__"].to_numpy(dtype=float).reshape(-1, 1),
        )[0]
        result = pd.DataFrame(
            scores,
            index=joined.index,
            columns=[f"{prefix}{idx}" for idx in range(1, resolved + 1)],
        ).reindex(panel.index)
    if resolved < n_value:
        for idx in range(resolved + 1, n_value + 1):
            result[f"{prefix}{idx}"] = np.nan
    result = result[[f"{prefix}{idx}" for idx in range(1, n_value + 1)]]
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_partial_least_squares",
        {
            "target": str(target_name),
            "columns": list(selected),
            "n_components": n_value,
            "resolved_n_components": int(resolved),
            "prefix": str(prefix),
            "n_fit_rows": int(len(joined)),
            "drop_missing": bool(drop_missing),
            "fit_policy": "full_input_target_aligned_rows",
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="partial_least_squares",
            source=",".join(selected),
            inputs=selected,
            fit_policy="full_input_target_aligned_rows",
        )
    )
    return result


def dfm_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_factors: int = 3,
    prefix: str = "dfm",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create static dynamic-factor approximation features by standardized PCA."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    n_value = int(n_factors)
    if n_value <= 0:
        raise ValueError("n_factors must be positive")
    _warn_if_full_sample_fit("full_sample", context="dfm_features()", enabled=warn_full_sample)
    result = _pca_frame(
        panel.loc[:, selected],
        n_components=min(n_value, len(selected)),
        fit_policy="full_sample",
        min_train_size=min(n_value + 1, len(panel)),
        scale=True,
        prefix=str(prefix),
        random_state=None,
    )
    if result.shape[1] < n_value:
        for idx in range(result.shape[1] + 1, n_value + 1):
            result[f"{prefix}{idx}"] = np.nan
    result = result[[f"{prefix}{idx}" for idx in range(1, n_value + 1)]]
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_dfm",
        {
            "columns": list(selected),
            "n_factors": n_value,
            "prefix": str(prefix),
            "drop_missing": bool(drop_missing),
            "fit_policy": "full_input_complete_rows",
            "warn_full_sample": bool(warn_full_sample),
            "note": "Static DFM approximation: standardized PCA factor scores.",
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="dfm",
            source=",".join(selected),
            inputs=selected,
            fit_policy="full_input_complete_rows",
        )
    )
    return result


def custom_features(
    data: FeatureInput,
    func: Callable[..., Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    name: str | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Apply a user supplied feature-engineering callable to a panel."""

    if not callable(func):
        raise TypeError("custom feature func must be callable")
    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    source = panel.loc[:, selected].copy()
    output = func(source, metadata=dict(base.metadata), **params)
    result = _coerce_custom_feature_output(output, index=source.index)
    validate_panel(result)
    step_name = name or _callable_name(func)
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_custom",
        {
            "name": str(step_name),
            "callable": _callable_name(func),
            "columns": list(selected),
            "params": _json_ready(params),
            "output_columns": [str(column) for column in result.columns],
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _records_for_columns(
            result,
            operation="custom",
            sources=selected,
            included=True,
        )
    )
    return result


def asymmetric_trim_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    prefix: str = "rank_",
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Sort each row across columns into rank-space features."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    if len(selected) < 2:
        raise ValueError("asymmetric_trim_features() requires at least two columns")
    values = np.sort(panel.loc[:, selected].to_numpy(dtype=float), axis=1)
    result = pd.DataFrame(
        values,
        index=panel.index,
        columns=[f"{prefix}{idx}" for idx in range(1, len(selected) + 1)],
    )
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_asymmetric_trim",
        {"columns": list(selected), "prefix": str(prefix), "drop_missing": bool(drop_missing)},
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="asymmetric_trim",
            source=",".join(selected),
            inputs=selected,
            fit_policy="rowwise",
        )
    )
    return result


def wavelet_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_levels: int = 3,
    wavelet: str = "db4",
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create rolling multi-resolution approximation/detail features."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    levels = int(n_levels)
    if levels <= 0:
        raise ValueError("n_levels must be positive")
    pieces: list[pd.Series] = []
    for column in selected:
        series = panel[column].astype(float)
        for level in range(1, levels + 1):
            window = 2**level
            approx = series.rolling(window=window, min_periods=1).mean().rename(f"{column}_wA{level}")
            detail = (series - approx).rename(f"{column}_wD{level}")
            pieces.extend([approx, detail])
    result = pd.concat(pieces, axis=1) if pieces else pd.DataFrame(index=panel.index)
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_wavelet",
        {
            "columns": list(selected),
            "n_levels": levels,
            "wavelet": str(wavelet),
            "drop_missing": bool(drop_missing),
            "note": "Causal rolling multi-resolution approximation; wavelet name recorded for compatibility.",
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="wavelet",
            source=",".join(selected),
            inputs=selected,
            fit_policy="causal_rolling",
        )
    )
    return result


def adaptive_ma_rf_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_estimators: int = 100,
    min_samples_leaf: int = 40,
    sided: str = "two",
    random_state: int | None = 0,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create adaptive moving-average smoothers using random forests over time."""

    from sklearn.ensemble import RandomForestRegressor

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    n_tree = int(n_estimators)
    min_leaf = int(min_samples_leaf)
    if n_tree <= 0:
        raise ValueError("n_estimators must be positive")
    if min_leaf <= 0:
        raise ValueError("min_samples_leaf must be positive")
    sided_value = str(sided).lower()
    if sided_value not in {"one", "two"}:
        raise ValueError("sided must be 'one' or 'two'")
    if sided_value == "two":
        _warn_if_full_sample_fit("full_sample", context="adaptive_ma_rf_features()", enabled=warn_full_sample)
    time_values: np.ndarray = np.arange(len(panel), dtype=float).reshape(-1, 1)
    result = pd.DataFrame(index=panel.index)
    for offset, column in enumerate(selected):
        series = panel[column].astype(float)
        if sided_value == "two":
            mask = series.notna().to_numpy()
            fitted: np.ndarray = np.full(len(series), np.nan, dtype=float)
            if int(mask.sum()) >= 2:
                model = RandomForestRegressor(
                    n_estimators=n_tree,
                    min_samples_leaf=min(min_leaf, int(mask.sum())),
                    random_state=None if random_state is None else int(random_state) + offset,
                )
                model.fit(time_values[mask], series.to_numpy(dtype=float)[mask])
                fitted = model.predict(time_values)
            result[f"{column}_albama"] = fitted
        else:
            result[f"{column}_albama"] = _one_sided_adaptive_ma_rf(
                time_values,
                series,
                n_estimators=n_tree,
                min_samples_leaf=min_leaf,
                random_state=None if random_state is None else int(random_state) + offset,
            )
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_adaptive_ma_rf",
        {
            "columns": list(selected),
            "n_estimators": n_tree,
            "min_samples_leaf": min_leaf,
            "sided": sided_value,
            "random_state": random_state,
            "drop_missing": bool(drop_missing),
            "fit_policy": "full_sample" if sided_value == "two" else "one_sided_expanding",
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _records_for_columns(result, operation="adaptive_ma_rf", sources=selected, included=True)
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


def fourier_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    period: int = 12,
    order: int = 2,
    prefix: str = "fourier",
) -> pd.DataFrame:
    """Create deterministic Fourier seasonal terms."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    period_value = int(period)
    order_value = int(order)
    if period_value <= 1:
        raise ValueError("period must be greater than 1")
    if order_value <= 0:
        raise ValueError("order must be positive")
    if order_value > period_value // 2:
        raise ValueError("order must be <= period // 2")
    t: np.ndarray = np.arange(len(panel), dtype=float)
    result = pd.DataFrame(index=panel.index)
    for k in range(1, order_value + 1):
        result[f"{prefix}_sin{k}_p{period_value}"] = np.sin(2.0 * np.pi * k * t / period_value)
        result[f"{prefix}_cos{k}_p{period_value}"] = np.cos(2.0 * np.pi * k * t / period_value)
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_fourier",
        {"period": period_value, "order": order_value, "prefix": str(prefix)},
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "fourier",
            "source": "date",
            "parameter": f"period={period_value};order={order_value}",
            "inputs": "date",
            "included": True,
        }
        for feature in result.columns
    )
    return result


def season_dummy(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    frequency: str = "auto",
    drop_first: bool = False,
) -> pd.DataFrame:
    """Create month or quarter seasonal dummies from the date index."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    freq = str(frequency).lower()
    if freq == "auto":
        inferred = pd.infer_freq(panel.index) or ""
        freq = "quarter" if inferred.startswith("Q") or inferred.startswith("QE") else "month"
    if freq not in {"month", "quarter"}:
        raise ValueError("frequency must be 'auto', 'month', or 'quarter'")
    result = pd.DataFrame(index=panel.index)
    if freq == "month":
        values = range(1, 13)
        labels = panel.index.month
        prefix = "month"
    else:
        values = range(1, 5)
        labels = panel.index.quarter
        prefix = "quarter"
    for value in values:
        if drop_first and value == 1:
            continue
        result[f"{prefix}_{value:02d}" if freq == "month" else f"{prefix}_{value}"] = (
            labels == value
        ).astype(float)
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_season_dummy",
        {"frequency": freq, "drop_first": bool(drop_first)},
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "season_dummy",
            "source": "date",
            "parameter": f"frequency={freq};drop_first={bool(drop_first)}",
            "inputs": "date",
            "included": True,
        }
        for feature in result.columns
    )
    return result


def polynomial_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create polynomial expansion features with readable column names."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    degree_value = int(degree)
    if degree_value < 1:
        raise ValueError("degree must be at least 1")
    result = pd.DataFrame(index=panel.index)
    if include_bias:
        result["bias"] = 1.0
    for deg in range(1, degree_value + 1):
        iterator = combinations(selected, deg) if interaction_only else combinations_with_replacement(selected, deg)
        for terms in iterator:
            name = _term_name(terms, power_separator="^")
            values = pd.Series(1.0, index=panel.index)
            for term in terms:
                values = values * panel[term]
            result[f"poly_{name}"] = values
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_polynomial",
        {
            "columns": list(selected),
            "degree": degree_value,
            "include_bias": bool(include_bias),
            "interaction_only": bool(interaction_only),
            "drop_missing": bool(drop_missing),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "polynomial",
            "source": ",".join(selected),
            "parameter": f"degree={degree_value}",
            "inputs": ",".join(selected),
            "included": True,
        }
        for feature in result.columns
    )
    return result


def interaction_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    order: int = 2,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Create pure interaction terms without powers."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    order_value = int(order)
    if order_value < 2:
        raise ValueError("order must be at least 2 for interaction features")
    if order_value > len(selected):
        raise ValueError("order must be <= the number of selected columns")
    result = pd.DataFrame(index=panel.index)
    for terms in combinations(selected, order_value):
        values = pd.Series(1.0, index=panel.index)
        for term in terms:
            values = values * panel[term]
        result[f"interaction_{'__'.join(terms)}"] = values
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_interaction",
        {
            "columns": list(selected),
            "order": order_value,
            "drop_missing": bool(drop_missing),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "interaction",
            "source": ",".join(str(feature).removeprefix("interaction_").split("__")),
            "parameter": f"order={order_value}",
            "inputs": ",".join(selected),
            "included": True,
        }
        for feature in result.columns
    )
    return result


def hp_filter_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    lamb: float = 129600.0,
    component: str = "cycle",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create Hodrick-Prescott filter cycle or trend features.

    HP filtering is two-sided on the supplied sample. It is direct-only and
    warns by default because using it before a forecasting split can leak future
    information into trend/cycle features.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    component_value = str(component).lower()
    if component_value not in {"cycle", "trend", "both"}:
        raise ValueError("component must be 'cycle', 'trend', or 'both'")
    _warn_if_full_sample_fit(
        "full_sample",
        context="hp_filter_features()",
        enabled=warn_full_sample,
    )
    result = pd.DataFrame(index=panel.index)
    for column in selected:
        series = panel[column].astype(float)
        if series.isna().any():
            series = series.interpolate(limit_direction="both")
        cycle, trend = hpfilter(series, lamb=float(lamb))
        if component_value in {"cycle", "both"}:
            result[f"{column}_hp_cycle"] = cycle
        if component_value in {"trend", "both"}:
            result[f"{column}_hp_trend"] = trend
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_hp_filter",
        {
            "columns": list(selected),
            "lambda": float(lamb),
            "component": component_value,
            "drop_missing": bool(drop_missing),
            "missing_policy": "linear_interpolate_before_filter",
            "fit_policy": "full_input_two_sided",
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "hp_filter",
            "source": _source_for_feature(str(feature).split("_hp_")[0], selected),
            "parameter": f"lambda={float(lamb)};component={component_value}",
            "fit_policy": "full_input_two_sided",
            "inputs": ",".join(selected),
            "included": True,
        }
        for feature in result.columns
    )
    return result


def hamilton_filter_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    h: int = 8,
    p: int = 4,
    component: str = "cycle",
    fit_policy: FitPolicy = "expanding",
    min_train_size: int | None = None,
    missing: str = "drop",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create Hamilton-filter cycle or trend features.

    The Hamilton filter regresses ``y[t+h]`` on a constant and
    ``y[t], y[t-1], ..., y[t-p+1]``. The trend is the fitted value and the
    cycle is the residual, both labeled at ``t+h``. Defaults ``h=8, p=4``
    match the common quarterly specification. For monthly data, callers often
    use ``h=24, p=12``.

    ``fit_policy='expanding'`` estimates each row with only earlier completed
    target rows, which avoids full-sample leakage when the output is used as a
    forecasting feature. ``fit_policy='full_sample'`` reproduces the usual
    in-sample filter style and emits a warning by default.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    h_value = int(h)
    p_value = int(p)
    if h_value <= 0:
        raise ValueError("h must be positive")
    if p_value <= 0:
        raise ValueError("p must be positive")
    component_value = str(component).lower()
    if component_value not in {"cycle", "trend", "both"}:
        raise ValueError("component must be 'cycle', 'trend', or 'both'")
    fit_value = _normalize_fit_policy(fit_policy)
    _warn_if_full_sample_fit(
        fit_value,
        context="hamilton_filter_features()",
        enabled=warn_full_sample,
    )
    min_size = _normalize_min_train_size(min_train_size, minimum=p_value + 1)
    missing_value = str(missing).lower()
    if missing_value not in {"drop", "interpolate"}:
        raise ValueError("missing must be 'drop' or 'interpolate'")

    result = pd.DataFrame(index=panel.index)
    for column in selected:
        series = panel[column].astype(float)
        if missing_value == "interpolate" and series.isna().any():
            series = series.interpolate(limit_direction="both")
        cycle, trend = _hamilton_filter_series(
            series,
            h=h_value,
            p=p_value,
            fit_policy=fit_value,
            min_train_size=min_size,
        )
        if component_value in {"cycle", "both"}:
            result[f"{column}_hamilton_cycle"] = cycle
        if component_value in {"trend", "both"}:
            result[f"{column}_hamilton_trend"] = trend
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_hamilton_filter",
        {
            "columns": list(selected),
            "h": h_value,
            "p": p_value,
            "component": component_value,
            "fit_policy": fit_value,
            "min_train_size": min_size,
            "missing": missing_value,
            "drop_missing": bool(drop_missing),
            "warn_full_sample": bool(warn_full_sample),
            "label_alignment": "components are labeled at t+h",
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "hamilton_filter",
            "source": _source_for_feature(str(feature).split("_hamilton_")[0], selected),
            "parameter": f"h={h_value};p={p_value};component={component_value}",
            "fit_policy": fit_value,
            "inputs": ",".join(selected),
            "included": True,
        }
        for feature in result.columns
    )
    return result


def savitzky_golay_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    window_length: int = 5,
    polyorder: int = 2,
    derivative: int = 0,
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Smooth columns with a centered Savitzky-Golay filter.

    The scipy filter uses a centered local window by default. It is direct-only
    and warns by default because centered smoothing can leak future values
    relative to a forecasting origin.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    window = int(window_length)
    order = int(polyorder)
    deriv = int(derivative)
    if window <= 0 or window % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    if order >= window:
        raise ValueError("polyorder must be smaller than window_length")
    _warn_if_full_sample_fit(
        "full_sample",
        context="savitzky_golay_features()",
        enabled=warn_full_sample,
    )
    result = pd.DataFrame(index=panel.index)
    for column in selected:
        series = panel[column].astype(float)
        if series.isna().any():
            series = series.interpolate(limit_direction="both")
        values = savgol_filter(series.to_numpy(dtype=float), window_length=window, polyorder=order, deriv=deriv)
        result[f"{column}_savgol"] = values
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_savitzky_golay",
        {
            "columns": list(selected),
            "window_length": window,
            "polyorder": order,
            "derivative": deriv,
            "drop_missing": bool(drop_missing),
            "missing_policy": "linear_interpolate_before_filter",
            "fit_policy": "full_input_centered_window",
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        {
            "feature": str(feature),
            "operation": "savitzky_golay",
            "source": str(feature).removesuffix("_savgol"),
            "parameter": f"window_length={window};polyorder={order};derivative={deriv}",
            "fit_policy": "full_input_centered_window",
            "inputs": ",".join(selected),
            "included": True,
        }
        for feature in result.columns
    )
    return result


def _resolve_target_argument(
    base: Any,
    panel: pd.DataFrame,
    target: str | pd.Series | None,
    *,
    context: str,
) -> tuple[str, pd.Series]:
    if isinstance(target, str):
        if target not in panel.columns:
            raise ValueError(f"target {target!r} is not in the panel")
        return target, panel[target].astype(float)
    if isinstance(target, pd.Series):
        name = str(target.name) if target.name is not None else "target"
        return name, target.astype(float)
    if base.target is not None:
        if base.target not in panel.columns:
            raise ValueError(f"target {base.target!r} is not in the panel")
        return str(base.target), panel[str(base.target)].astype(float)
    raise ValueError(f"{context} requires target or input target metadata")


def _resolve_feature_keep_count(n_features: int | float, *, n_columns: int) -> int:
    if n_columns <= 0:
        raise ValueError("at least one column is required")
    if isinstance(n_features, float) and 0.0 < n_features <= 1.0:
        return max(1, min(n_columns, int(np.ceil(n_features * n_columns))))
    count = int(n_features)
    if count <= 0:
        raise ValueError("n_features must be a positive count or a fraction in (0, 1]")
    return max(1, min(n_columns, count))


def _one_sided_adaptive_ma_rf(
    time_values: np.ndarray,
    series: pd.Series,
    *,
    n_estimators: int,
    min_samples_leaf: int,
    random_state: int | None,
) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    out: np.ndarray = np.full(len(series), np.nan, dtype=float)
    values = series.to_numpy(dtype=float)
    observed = np.isfinite(values)
    row_numbers = np.arange(len(series))
    for row in range(len(series)):
        mask = observed & (row_numbers <= row)
        n_fit = int(mask.sum())
        if n_fit < 2:
            out[row] = values[row] if np.isfinite(values[row]) else np.nan
            continue
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min(min_samples_leaf, n_fit),
            random_state=None if random_state is None else int(random_state) + row,
        )
        model.fit(time_values[mask], values[mask])
        out[row] = float(model.predict(time_values[row : row + 1])[0])
    return out


def random_projection_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 2,
    random_state: int | None = None,
    prefix: str = "rp",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create Gaussian random-projection features."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    source = panel.loc[:, selected].dropna()
    if source.empty:
        raise ValueError("random projection requires at least one complete row")
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    _warn_if_full_sample_fit(
        "full_sample",
        context="random_projection_features()",
        enabled=warn_full_sample,
    )
    transformer = GaussianRandomProjection(n_components=n_value, random_state=random_state)
    transformed = transformer.fit_transform(source.to_numpy(dtype=float))
    result = pd.DataFrame(
        transformed,
        index=source.index,
        columns=[f"{prefix}{idx}" for idx in range(1, n_value + 1)],
    ).reindex(panel.index)
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_random_projection",
        {
            "columns": list(selected),
            "n_components": n_value,
            "random_state": random_state,
            "drop_missing": bool(drop_missing),
            "fit_policy": "full_input_complete_rows",
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="random_projection",
            source=",".join(selected),
            inputs=selected,
            fit_policy="full_input_complete_rows",
        )
    )
    return result


def nystroem_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    n_components: int = 10,
    kernel: str = "rbf",
    gamma: float | None = None,
    random_state: int | None = None,
    prefix: str = "nys",
    drop_missing: bool = False,
    warn_full_sample: bool = True,
) -> pd.DataFrame:
    """Create Nystroem kernel-approximation features."""

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    source = panel.loc[:, selected].dropna()
    if source.empty:
        raise ValueError("Nystroem features require at least one complete row")
    n_value = int(n_components)
    if n_value <= 0:
        raise ValueError("n_components must be positive")
    _warn_if_full_sample_fit(
        "full_sample",
        context="nystroem_features()",
        enabled=warn_full_sample,
    )
    transformer = Nystroem(
        kernel=str(kernel),
        gamma=gamma,
        n_components=min(n_value, len(source)),
        random_state=random_state,
    )
    transformed = transformer.fit_transform(source.to_numpy(dtype=float))
    result = pd.DataFrame(
        transformed,
        index=source.index,
        columns=[f"{prefix}{idx}" for idx in range(1, transformed.shape[1] + 1)],
    ).reindex(panel.index)
    if drop_missing:
        result = result.dropna()
    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "feature_engineering_nystroem",
        {
            "columns": list(selected),
            "n_components": n_value,
            "kernel": str(kernel),
            "gamma": gamma,
            "random_state": random_state,
            "drop_missing": bool(drop_missing),
            "fit_policy": "full_input_complete_rows",
            "warn_full_sample": bool(warn_full_sample),
        },
    )
    result.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _component_records(
            result,
            operation="nystroem",
            source=",".join(selected),
            inputs=selected,
            fit_policy="full_input_complete_rows",
        )
    )
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


def _resolve_mixed_lag_columns(
    panel: pd.DataFrame,
    *,
    columns: Iterable[str] | None,
    target: str | None,
) -> tuple[str, ...]:
    if columns is None:
        return tuple(str(column) for column in panel.columns if str(column) != target)
    return _resolve_columns(panel, columns=columns)


def _mixed_frequency_map(
    panel: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
    frequency_by_column: Mapping[str, str] | None,
) -> dict[str, str]:
    raw_map = frequency_by_column
    if raw_map is None:
        raw_map = metadata.get("native_frequency_by_column", {})
    result: dict[str, str] = {}
    if isinstance(raw_map, Mapping):
        unknown = sorted(set(str(column) for column in raw_map).difference(str(column) for column in panel.columns))
        if unknown:
            raise ValueError(f"frequency_by_column includes unknown columns: {unknown}")
        result.update({str(column): _normalize_frequency_label(value) for column, value in raw_map.items()})
    for column in panel.columns:
        name = str(column)
        result.setdefault(name, _infer_frequency(panel[column]))
    return result


def _resolve_anchor_dates(
    panel: pd.DataFrame,
    *,
    target: str | None,
    anchor_dates: Iterable[Any] | None,
    target_frequency: str | None,
    frequency_map: Mapping[str, str],
    anchor_position: str,
) -> pd.DatetimeIndex:
    if anchor_dates is not None:
        dates = pd.DatetimeIndex(pd.to_datetime(list(anchor_dates))).sort_values()
    elif target is not None:
        dates = pd.DatetimeIndex(panel[target].dropna().index)
    else:
        dates = pd.DatetimeIndex(panel.index)
    if dates.empty:
        raise ValueError("anchor_dates are empty")
    freq = _normalize_frequency_label(target_frequency) if target_frequency is not None else None
    if freq is None and target is not None:
        freq = frequency_map.get(target, _infer_frequency(panel[target]))
    positioned = _anchor_dates_to_position(dates, frequency=freq or "unknown", anchor_position=anchor_position)
    positioned.name = "date"
    return positioned


def _anchor_dates_to_position(
    dates: pd.DatetimeIndex,
    *,
    frequency: str,
    anchor_position: str,
) -> pd.DatetimeIndex:
    key = str(anchor_position).lower()
    if key in {"date", "as_is", "asis"}:
        return pd.DatetimeIndex(dates.to_period("M").to_timestamp())
    if key not in {"period_start", "start", "period_end", "end"}:
        raise ValueError("anchor_position must be one of ['date', 'period_start', 'period_end']")
    how = "start" if key in {"period_start", "start"} else "end"
    if frequency == "quarterly":
        return pd.DatetimeIndex(dates.to_period("Q").asfreq("M", how=how).to_timestamp())
    if frequency == "annual":
        return pd.DatetimeIndex(dates.to_period("Y").asfreq("M", how=how).to_timestamp())
    if frequency in {"monthly", "unknown", "irregular", "weekly"}:
        return pd.DatetimeIndex(dates.to_period("M").to_timestamp())
    raise ValueError(f"cannot position anchors for frequency {frequency!r}")


def _lagged_lookup_dates(anchor_index: pd.DatetimeIndex, *, lag: int, frequency: str) -> pd.DatetimeIndex:
    freq = _normalize_frequency_label(frequency)
    if freq == "monthly":
        dates = anchor_index - pd.DateOffset(months=int(lag))
    elif freq == "quarterly":
        dates = anchor_index - pd.DateOffset(months=3 * int(lag))
    elif freq == "annual":
        dates = anchor_index - pd.DateOffset(years=int(lag))
    elif freq == "weekly":
        dates = anchor_index - pd.DateOffset(weeks=int(lag))
    elif freq in {"unknown", "irregular"}:
        dates = anchor_index - pd.DateOffset(months=int(lag))
    else:  # pragma: no cover - guarded by _normalize_frequency_label
        raise ValueError(f"unsupported frequency {frequency!r}")
    return _dates_to_source_period(dates, frequency=freq)


def _dates_to_source_period(dates: pd.DatetimeIndex, *, frequency: str) -> pd.DatetimeIndex:
    if frequency == "quarterly":
        return pd.DatetimeIndex(dates.to_period("Q").to_timestamp())
    if frequency == "annual":
        return pd.DatetimeIndex(dates.to_period("Y").to_timestamp())
    if frequency == "weekly":
        return pd.DatetimeIndex(dates.to_period("W").start_time.normalize())
    return pd.DatetimeIndex(dates.to_period("M").to_timestamp())


def _source_series_by_period(series: pd.Series, *, frequency: str) -> pd.Series:
    """Index a source series by the same period-start dates used for lookup.

    Data loaders and user panels may carry monthly/quarterly observations at
    period start or period end. MIDAS lag lookup should depend on the native
    period, not the cosmetic timestamp convention, so duplicate periods keep
    the last non-missing observation.
    """

    observed = series.dropna().astype(float)
    if observed.empty:
        return observed
    period_index = _dates_to_source_period(
        pd.DatetimeIndex(observed.index),
        frequency=_normalize_frequency_label(frequency),
    )
    normalized = pd.Series(observed.to_numpy(dtype=float), index=period_index, name=series.name)
    normalized = normalized.groupby(level=0).last().sort_index()
    normalized.index.name = "date"
    return normalized


def _hamilton_filter_series(
    series: pd.Series,
    *,
    h: int,
    p: int,
    fit_policy: str,
    min_train_size: int,
) -> tuple[pd.Series, pd.Series]:
    values = pd.Series(series, index=series.index, dtype=float)
    cycle = pd.Series(np.nan, index=values.index, name="cycle", dtype=float)
    trend = pd.Series(np.nan, index=values.index, name="trend", dtype=float)
    if len(values) <= h + p:
        return cycle, trend

    rows: list[tuple[int, np.ndarray, float]] = []
    arr = values.to_numpy(dtype=float)
    for anchor_pos in range(p - 1, len(arr) - h):
        target_pos = anchor_pos + h
        regressors = np.array([arr[anchor_pos - lag] for lag in range(p)], dtype=float)
        target = float(arr[target_pos])
        if not np.isfinite(target) or not np.isfinite(regressors).all():
            continue
        rows.append((target_pos, np.r_[1.0, regressors], target))
    if len(rows) < min_train_size:
        return cycle, trend

    target_positions = np.array([row[0] for row in rows], dtype=int)
    x_matrix = np.vstack([row[1] for row in rows]).astype(float)
    y_vector = np.array([row[2] for row in rows], dtype=float)

    if fit_policy == "full_sample":
        fitted = _ols_predict(x_matrix, y_vector, x_matrix)
        trend.iloc[target_positions] = fitted
        cycle.iloc[target_positions] = y_vector - fitted
        return cycle, trend

    for row_idx, target_pos in enumerate(target_positions):
        train_mask = target_positions < target_pos
        if int(train_mask.sum()) < min_train_size:
            continue
        fitted_value = _ols_predict(
            x_matrix[train_mask],
            y_vector[train_mask],
            x_matrix[row_idx : row_idx + 1],
        )[0]
        trend.iloc[target_pos] = fitted_value
        cycle.iloc[target_pos] = y_vector[row_idx] - fitted_value
    return cycle, trend


def _ols_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_pred: np.ndarray,
) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    return np.asarray(x_pred @ beta, dtype=float).reshape(-1)


def _normalize_frequency_label(value: Any) -> str:
    key = str(value).strip().lower()
    aliases = {
        "m": "monthly",
        "month": "monthly",
        "monthly": "monthly",
        "state_monthly": "monthly",
        "q": "quarterly",
        "quarter": "quarterly",
        "quarterly": "quarterly",
        "w": "weekly",
        "week": "weekly",
        "weekly": "weekly",
        "a": "annual",
        "annual": "annual",
        "yearly": "annual",
        "irregular": "irregular",
        "unknown": "unknown",
    }
    if key not in aliases:
        allowed = sorted(set(aliases.values()))
        raise ValueError(f"frequency must be one of {allowed}; got {value!r}")
    return aliases[key]


def _infer_frequency(series: pd.Series) -> str:
    observed = series.dropna()
    if observed.shape[0] < 2:
        return "unknown"
    periods = pd.DatetimeIndex(observed.index).to_period("M")
    deltas = [
        int(right.ordinal - left.ordinal)
        for left, right in zip(periods[:-1], periods[1:], strict=False)
        if right.ordinal > left.ordinal
    ]
    if not deltas:
        return "unknown"
    most_common = int(pd.Series(deltas).mode().iloc[0])
    if most_common == 1:
        return "monthly"
    if most_common == 3:
        return "quarterly"
    if most_common == 12:
        return "annual"
    return "irregular"


def _coerce_custom_feature_output(output: Any, *, index: pd.Index) -> pd.DataFrame:
    if isinstance(output, pd.Series):
        frame = output.to_frame()
    elif isinstance(output, pd.DataFrame):
        frame = output.copy()
    else:
        values = np.asarray(output)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2:
            raise TypeError("custom feature output must be a Series, DataFrame, or 1D/2D array-like")
        frame = pd.DataFrame(
            values,
            columns=[f"custom_{idx}" for idx in range(1, values.shape[1] + 1)],
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        if len(frame.index) == len(index):
            frame.index = index
        else:
            raise ValueError("custom feature output must keep the input DatetimeIndex or have matching length")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "date"
    frame.columns = [str(column) for column in frame.columns]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_index()


def _callable_name(func: Callable[..., Any]) -> str:
    module = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", getattr(func, "__name__", repr(func)))
    return f"{module}.{qualname}" if module else str(qualname)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return _callable_name(value)
    return value


def _normalize_simple_transform(transform: str) -> str:
    key = str(transform).lower()
    aliases = {
        "log_change": "log_diff",
        "logdiff": "log_diff",
        "growth": "pct_change",
        "simple_growth": "pct_change",
        "cum_sum": "cumsum",
    }
    key = aliases.get(key, key)
    allowed = {"log", "diff", "log_diff", "pct_change", "cumsum"}
    if key not in allowed:
        raise ValueError(f"transform must be one of {sorted(allowed)}")
    return key


def _positive_log(series: pd.Series) -> pd.Series:
    values = series.astype(float).where(series.astype(float) > 0.0)
    return np.log(values)


def _term_name(terms: tuple[str, ...], *, power_separator: str) -> str:
    counts = {term: terms.count(term) for term in dict.fromkeys(terms)}
    pieces = [
        term if power == 1 else f"{term}{power_separator}{power}"
        for term, power in counts.items()
    ]
    return "_x_".join(pieces)
