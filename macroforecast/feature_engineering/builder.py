from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

import pandas as pd

from macroforecast.data import DataBundle, attach_metadata, panel_info, validate_panel
from macroforecast.feature_engineering.types import FeatureInput, FeatureSet
from macroforecast.feature_engineering.compose import compose_features
from macroforecast.feature_engineering.matrix import feature_matrix
from macroforecast.feature_engineering.shared import (
    FitPolicy,
    TargetMode,
    TargetTransform,
    _coerce_input,
    _metadata_frame,
    _normalize_feature_matrix_specification,
    _normalize_fit_policy,
    _normalize_lags,
    _normalize_positive_ints,
    _normalize_target_mode,
    _normalize_target_transform,
    _records_for_columns,
    _resolve_horizons,
    _resolve_predictors,
    _resolve_targets,
    _target_metadata_frame,
    _target_transform_to_path_transform,
    _warn_if_no_preprocessing_metadata,
)
from macroforecast.feature_engineering.targets import direct_target, path_targets
from macroforecast.feature_engineering.transforms import lag, rolling_mean, time_features

def build_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    predictors: Literal["all"] | Iterable[str] | None = None,
    lags: Iterable[int] | int = (0, 1),
    rolling_windows: Iterable[int] | int | None = None,
    rolling_min_periods: int | None = None,
    add_time: bool = False,
    time_trend: bool = True,
    time_month: bool = False,
    time_quarter: bool = False,
    time_year: bool = False,
    feature_steps: Iterable[Mapping[str, Any]] | None = None,
    feature_specification: str | Iterable[str] | None = None,
    include_original: bool = False,
    level_data: FeatureInput | None = None,
    max_lag: int = 12,
    n_factors: int = 8,
    n_maf_components: int = 2,
    feature_fit_policy: FitPolicy = "expanding",
    feature_min_train_size: int | None = None,
    feature_warn_full_sample: bool = True,
    include_current_factor: bool = True,
    scale_factors: bool = True,
    scale_marx: bool = False,
    scale_maf: bool = False,
    target_transform: TargetTransform = "level",
    target_mode: TargetMode = "direct",
    drop_missing: bool = True,
) -> FeatureSet:
    """Build an aligned predictor matrix and target matrix."""

    base = _coerce_input(data, metadata=metadata)
    _warn_if_no_preprocessing_metadata(base.metadata)
    panel = base.panel
    validate_panel(panel)
    target_values = _resolve_targets(panel, base=base, target=target, targets=targets)
    horizon_values = _resolve_horizons(base=base, horizon=horizon, horizons=horizons)
    predictor_values = _resolve_predictors(
        panel,
        base=base,
        predictors=predictors,
        targets=target_values,
    )

    rolling_values: tuple[int, ...] = ()
    lag_values: tuple[int, ...] = ()
    step_list: list[dict[str, Any]] | None = None
    if feature_steps is not None and feature_specification is not None:
        raise ValueError("provide either feature_steps or feature_specification, not both")
    if feature_specification is not None:
        lag_values = _normalize_lags(lags, allow_zero=True)
        source = DataBundle(panel.loc[:, predictor_values], base.metadata)
        X = feature_matrix(
            source,
            specification=feature_specification,
            columns=predictor_values,
            level_data=level_data,
            lags=lag_values,
            max_lag=max_lag,
            n_factors=n_factors,
            n_maf_components=n_maf_components,
            fit_policy=feature_fit_policy,
            min_train_size=feature_min_train_size,
            include_current_factor=include_current_factor,
            scale_factors=scale_factors,
            scale_marx=scale_marx,
            scale_maf=scale_maf,
            drop_missing=False,
            warn_full_sample=feature_warn_full_sample,
        )
        feature_metadata = X.attrs.get("macroforecast_feature_metadata", pd.DataFrame())
    elif feature_steps is not None:
        step_list = [dict(step) for step in feature_steps]
        source = DataBundle(panel.loc[:, predictor_values], base.metadata)
        X = compose_features(
            source,
            step_list,
            include_original=include_original,
            drop_missing=False,
        )
        feature_metadata = X.attrs.get("macroforecast_feature_metadata", pd.DataFrame())
    else:
        feature_parts: list[pd.DataFrame] = []
        feature_records: list[dict[str, Any]] = []

        lag_values = _normalize_lags(lags, allow_zero=True)
        lagged = lag(panel, metadata=base.metadata, columns=predictor_values, lags=lag_values)
        feature_parts.append(lagged)
        feature_records.extend(_records_for_columns(lagged, operation="lag", sources=predictor_values, included=True))

        if rolling_windows is not None:
            rolling_values = _normalize_positive_ints(rolling_windows, name="rolling_windows")
            rolled = rolling_mean(
                panel,
                metadata=base.metadata,
                columns=predictor_values,
                windows=rolling_values,
                min_periods=rolling_min_periods,
            )
            feature_parts.append(rolled)
            feature_records.extend(
                _records_for_columns(rolled, operation="rolling_mean", sources=predictor_values, included=True)
            )

        if add_time:
            timed = time_features(
                panel,
                metadata=base.metadata,
                trend=time_trend,
                month=time_month,
                quarter=time_quarter,
                year=time_year,
            )
            feature_parts.append(timed)
            feature_records.extend(_records_for_columns(timed, operation="time", sources=(), included=True))

        X = pd.concat(feature_parts, axis=1) if feature_parts else pd.DataFrame(index=panel.index)
        feature_metadata = _metadata_frame(feature_records)
    if X.empty:
        raise ValueError("feature choices produced an empty predictor matrix")
    target_mode_value = _normalize_target_mode(target_mode)
    if target_mode_value == "path":
        path_transform = _target_transform_to_path_transform(target_transform)
        y = path_targets(
            panel,
            metadata=base.metadata,
            targets=target_values,
            horizons=horizon_values,
            transform=path_transform,
        )
    else:
        y = direct_target(
            panel,
            metadata=base.metadata,
            targets=target_values,
            horizons=horizon_values,
            transform=target_transform,
        )
    target_metadata = y.attrs.get("macroforecast_target_metadata", _target_metadata_frame([]))
    path_columns_by_horizon = (
        y.attrs.get("macroforecast_metadata", {}).get("path_target", {}).get("columns_by_horizon")
        if target_mode_value == "path"
        else None
    )

    if drop_missing:
        aligned = pd.concat([X, y], axis=1).dropna()
        X = aligned.loc[:, X.columns]
        y = aligned.loc[:, y.columns]
    if X.empty or y.empty:
        raise ValueError("feature construction leaves an empty aligned sample")

    if feature_metadata.empty:
        feature_metadata = _metadata_frame([])
    stage = {
        "input_panel": panel_info(DataBundle(panel, base.metadata)),
        "predictors": list(predictor_values),
        "targets": list(target_values),
        "horizons": list(horizon_values),
        "target_transform": _normalize_target_transform(target_transform),
        "target_mode": target_mode_value,
        "path_target_columns_by_horizon": path_columns_by_horizon,
        "lags": list(lag_values),
        "rolling_windows": list(rolling_values),
        "rolling_min_periods": rolling_min_periods,
        "feature_specification": (
            "-".join(_normalize_feature_matrix_specification(feature_specification))
            if feature_specification is not None
            else None
        ),
        "feature_matrix": {
            "level_data": level_data is not None,
            "max_lag": int(max_lag),
            "n_factors": int(n_factors),
            "n_maf_components": int(n_maf_components),
            "fit_policy": _normalize_fit_policy(feature_fit_policy),
            "min_train_size": feature_min_train_size,
            "warn_full_sample": bool(feature_warn_full_sample),
            "include_current_factor": bool(include_current_factor),
            "scale_factors": bool(scale_factors),
            "scale_marx": bool(scale_marx),
            "scale_maf": bool(scale_maf),
        }
        if feature_specification is not None
        else None,
        "feature_steps": step_list,
        "include_original": bool(include_original),
        "time": {
            "enabled": bool(add_time),
            "trend": bool(time_trend),
            "month": bool(time_month),
            "quarter": bool(time_quarter),
            "year": bool(time_year),
        },
        "drop_missing": bool(drop_missing),
        "output": {
            "n_observations": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_targets": int(y.shape[1]),
            "start": X.index[0].strftime("%Y-%m-%d") if len(X) else None,
            "end": X.index[-1].strftime("%Y-%m-%d") if len(X) else None,
        },
    }
    updated_metadata = attach_metadata(base.metadata, "feature_engineering", stage)
    X = X.copy()
    y = y.copy()
    feature_metadata = feature_metadata.copy()
    target_metadata = target_metadata.copy()
    X.attrs["macroforecast_metadata"] = updated_metadata
    y.attrs["macroforecast_metadata"] = updated_metadata
    y.attrs["macroforecast_target_metadata"] = target_metadata
    feature_metadata.attrs["macroforecast_metadata"] = updated_metadata
    target_metadata.attrs["macroforecast_metadata"] = updated_metadata
    return FeatureSet(
        X=X,
        y=y,
        metadata=updated_metadata,
        feature_metadata=feature_metadata,
        target_metadata=target_metadata,
        target=target_values[0] if len(target_values) == 1 else None,
        targets=target_values,
        horizons=horizon_values,
        predictors=predictor_values,
    )
