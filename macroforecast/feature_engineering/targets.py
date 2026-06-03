from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata, validate_panel
from macroforecast.feature_engineering.types import FeatureInput
from macroforecast.feature_engineering.shared import (
    PathTransform,
    TargetTransform,
    _average_future_path,
    _coerce_input,
    _normalize_path_transform,
    _normalize_target_transform,
    _one_period_future_transform,
    _path_target_column_name,
    _path_target_formula,
    _resolve_horizons,
    _resolve_targets,
    _target_column_name,
    _target_formula,
    _target_metadata_frame,
    _target_record,
)

def direct_target(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    transform: TargetTransform = "level",
) -> pd.DataFrame:
    """Construct direct-forecast target columns from a canonical panel.

    For date ``t`` and horizon ``h``, ``transform="level"`` or
    ``transform="value"`` returns
    ``target[t + h]`` aligned on row ``t``. Other transforms compare
    ``target[t + h]`` with ``target[t]``. ``average_*`` transforms build direct
    average targets from steps ``t + 1`` through ``t + h``; use
    ``average_value`` when the input series is already a one-period transformed
    forecasting object such as monthly growth or monthly difference.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    target_values = _resolve_targets(panel, base=base, target=target, targets=targets)
    horizon_values = _resolve_horizons(base=base, horizon=horizon, horizons=horizons)
    transform_method = _normalize_target_transform(transform)

    result = pd.DataFrame(index=panel.index)
    target_records: list[dict[str, Any]] = []
    for name in target_values:
        series = panel[name]
        for h in horizon_values:
            future = series.shift(-h)
            column = _target_column_name(name, horizon=h, transform=transform_method)
            if transform_method in {"level", "value"}:
                result[column] = future
            elif transform_method == "change":
                result[column] = future - series
            elif transform_method == "growth":
                valid = series != 0
                result[column] = np.where(valid, future / series - 1.0, np.nan)
            elif transform_method == "log_growth":
                valid = (future > 0) & (series > 0)
                result[column] = np.where(valid, np.log(future) - np.log(series), np.nan)
            elif transform_method.startswith("average_"):
                path_transform = _normalize_path_transform(transform_method.removeprefix("average_"))
                result[column] = _average_future_path(series, horizon=h, transform=path_transform)
            target_records.append(
                _target_record(
                    target_column=column,
                    source=name,
                    horizon=h,
                    step=None,
                    mode="direct",
                    transform=transform_method,
                    operation=(
                        "direct_average_target"
                        if transform_method.startswith("average_")
                        else "direct_target"
                    ),
                    formula=_target_formula(name, horizon=h, transform=transform_method),
                    aggregation=(
                        f"mean_step_{transform_method.removeprefix('average_')}"
                        if transform_method.startswith("average_")
                        else None
                    ),
                    used_for_horizons=(h,),
                )
            )

    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "target",
        {
            "targets": list(target_values),
            "horizons": list(horizon_values),
            "transform": transform_method,
            "columns": [str(column) for column in result.columns],
        },
    )
    target_metadata = _target_metadata_frame(target_records)
    target_metadata.attrs["macroforecast_metadata"] = result.attrs["macroforecast_metadata"]
    result.attrs["macroforecast_target_metadata"] = target_metadata
    return result

def average_target(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    transform: PathTransform = "change",
) -> pd.DataFrame:
    """Construct direct average targets over each forecast horizon.

    For horizon ``h``, this returns the average of one-period target
    transformations over steps ``1, ..., h``. It is the target used when one
    model is fit directly to an average growth or average difference object.
    """

    path_transform = _normalize_path_transform(transform)
    return direct_target(
        data,
        metadata=metadata,
        target=target,
        targets=targets,
        horizon=horizon,
        horizons=horizons,
        transform=f"average_{path_transform}",  # type: ignore[arg-type]
    )


def path_targets(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    transform: PathTransform = "change",
) -> pd.DataFrame:
    """Construct one-period target columns for path-average forecasting.

    A path-average workflow fits and forecasts one model per future step; a
    later evaluation stage can average the step forecasts. This function
    creates the ``step1`` through ``stepH`` target columns required by the
    model stage. Use ``transform="value"`` when the input series is already a
    one-period transformed forecasting object.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    target_values = _resolve_targets(panel, base=base, target=target, targets=targets)
    horizon_values = _resolve_horizons(base=base, horizon=horizon, horizons=horizons)
    transform_method = _normalize_path_transform(transform)
    max_horizon = max(horizon_values)

    result = pd.DataFrame(index=panel.index)
    target_records: list[dict[str, Any]] = []
    columns_by_horizon: dict[str, dict[str, list[str]]] = {}
    for name in target_values:
        series = panel[name]
        step_columns: list[str] = []
        for step in range(1, max_horizon + 1):
            column = _path_target_column_name(name, step=step, transform=transform_method)
            result[column] = _one_period_future_transform(series, step=step, transform=transform_method)
            step_columns.append(column)
            target_records.append(
                _target_record(
                    target_column=column,
                    source=name,
                    horizon=None,
                    step=step,
                    mode="path",
                    transform=transform_method,
                    operation="path_step_target",
                    formula=_path_target_formula(name, step=step, transform=transform_method),
                    aggregation="average_step_forecasts_in_evaluation",
                    used_for_horizons=tuple(h for h in horizon_values if h >= step),
                )
            )
        for horizon_value in horizon_values:
            horizon_key = f"h{horizon_value}"
            columns_by_horizon.setdefault(horizon_key, {})[name] = step_columns[:horizon_value]

    result.index.name = "date"
    result.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "path_target",
        {
            "targets": list(target_values),
            "horizons": list(horizon_values),
            "transform": transform_method,
            "max_horizon": int(max_horizon),
            "columns": [str(column) for column in result.columns],
            "columns_by_horizon": columns_by_horizon,
            "note": (
                "Path-average target columns are step-level outcomes. Later "
                "model code should fit and forecast each step separately; "
                "evaluation code can average step forecasts for each requested "
                "horizon."
            ),
        },
    )
    target_metadata = _target_metadata_frame(target_records)
    target_metadata.attrs["macroforecast_metadata"] = result.attrs["macroforecast_metadata"]
    result.attrs["macroforecast_target_metadata"] = target_metadata
    return result
