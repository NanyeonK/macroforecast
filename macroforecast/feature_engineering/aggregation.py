from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata, validate_panel
from macroforecast.feature_engineering.targets import average_target
from macroforecast.feature_engineering.transforms import asymmetric_trim_features
from macroforecast.feature_engineering.types import FeatureInput
from macroforecast.feature_engineering.shared import (
    PathTransform,
    _coerce_input,
    _metadata_frame,
    _records_for_columns,
    _resolve_columns,
)

MovingAverageMethod = Literal["compound_percent", "compound_decimal", "mean"]

ALBACORE_SOURCE = (
    "Goulet Coulombe, Klieber, Barrette, and Goebel, "
    "Maximally Forward-Looking Core Inflation; R package assemblage."
)


def forward_average_target(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizon: int | None = None,
    horizons: Iterable[int] | int | None = None,
    transform: PathTransform = "change",
) -> pd.DataFrame:
    """Construct the forward average target used by assemblage-style models.

    This is a named, reusable wrapper around :func:`average_target`. Its source
    cue is Albacore/assemblage, where the target is future average aggregate
    inflation. The function itself is generic: it can build a future average
    target for any aggregate macro series.
    """

    result = average_target(
        data,
        metadata=metadata,
        target=target,
        targets=targets,
        horizon=horizon,
        horizons=horizons,
        transform=transform,
    )
    result.attrs["macroforecast_metadata"] = attach_metadata(
        result.attrs.get("macroforecast_metadata", {}),
        "forward_average_target",
        {
            "source_method": "assemblage_forward_target",
            "source_reference": ALBACORE_SOURCE,
            "note": (
                "Generic helper derived from the Albacore target convention: "
                "predict the average future aggregate path over horizon h."
            ),
        },
    )
    target_metadata = result.attrs.get("macroforecast_target_metadata")
    if isinstance(target_metadata, pd.DataFrame) and not target_metadata.empty:
        target_metadata = target_metadata.copy()
        target_metadata["source_method"] = "assemblage_forward_target"
        target_metadata["source_reference"] = ALBACORE_SOURCE
        result.attrs["macroforecast_target_metadata"] = target_metadata
    return result


def rank_space_features(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    prefix: str = "rank_",
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Sort each row into rank-space features for supervised aggregation.

    The primitive is generic order-statistic feature construction. It is the
    reusable form of ``x.transformation`` in the R ``assemblage`` package used
    for Albacoreranks.
    """

    result = asymmetric_trim_features(
        data,
        metadata=metadata,
        columns=columns,
        prefix=prefix,
        drop_missing=drop_missing,
    )
    result.attrs["macroforecast_metadata"] = attach_metadata(
        result.attrs.get("macroforecast_metadata", {}),
        "rank_space_features",
        {
            "source_method": "assemblage_x_transformation_rank_space",
            "source_reference": ALBACORE_SOURCE,
            "note": (
                "Rows are sorted from low to high. Downstream weights are "
                "rank weights, not original component weights."
            ),
        },
    )
    feature_metadata = result.attrs.get("macroforecast_feature_metadata")
    if isinstance(feature_metadata, pd.DataFrame) and not feature_metadata.empty:
        feature_metadata = feature_metadata.copy()
        feature_metadata["source_method"] = "assemblage_x_transformation"
        feature_metadata["source_reference"] = ALBACORE_SOURCE
        result.attrs["macroforecast_feature_metadata"] = feature_metadata
    return result


def moving_average_changes(
    data: FeatureInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    window: int = 3,
    method: MovingAverageMethod = "compound_percent",
    suffix: str | None = None,
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Convert one-period component changes to a trailing moving-average unit.

    ``method="compound_percent"`` follows the R ``assemblage``
    ``x.transformation`` convention for month-over-month percentage changes:
    ``prod(1 + x / 100) - 1``, returned in percent units. Other methods are
    provided so the same helper can be reused outside inflation applications.
    """

    if int(window) < 1:
        raise ValueError("window must be at least 1")
    method_value = _normalize_moving_average_method(method)
    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    source = panel.loc[:, selected].astype(float)
    if int(window) == 1:
        out = source.copy()
    elif method_value == "compound_percent":
        out = (1.0 + source / 100.0).rolling(int(window), min_periods=int(window)).apply(
            np.prod,
            raw=True,
        )
        out = (out - 1.0) * 100.0
    elif method_value == "compound_decimal":
        out = (1.0 + source).rolling(int(window), min_periods=int(window)).apply(
            np.prod,
            raw=True,
        )
        out = out - 1.0
    else:
        out = source.rolling(int(window), min_periods=int(window)).mean()
    column_suffix = suffix if suffix is not None else f"_ma{int(window)}"
    out = out.rename(columns={column: f"{column}{column_suffix}" for column in selected})
    if drop_missing:
        out = out.dropna()
    out.index.name = "date"
    out.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "moving_average_changes",
        {
            "columns": list(selected),
            "window": int(window),
            "method": method_value,
            "source_method": "assemblage_x_transformation_moving_average",
            "source_reference": ALBACORE_SOURCE,
        },
    )
    out.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _records_for_columns(
            out,
            operation="moving_average_changes",
            sources=tuple(selected),
            included=True,
        )
    )
    return out


def align_reference_weights(
    weights: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float],
    columns: Iterable[str],
    *,
    normalize: bool = True,
    fill_value: float = 0.0,
    name: str = "reference_weight",
) -> pd.Series:
    """Align component reference weights to a model column order.

    This is the generic version of ``weight.transformation`` in the R
    ``assemblage`` package. It accepts official basket weights for Albacore,
    but can also align sector, state, or survey-item reference weights.
    """

    column_values = tuple(str(column) for column in columns)
    if not column_values:
        raise ValueError("columns must not be empty")
    if isinstance(weights, pd.DataFrame):
        if weights.empty:
            values = pd.Series(fill_value, index=column_values, dtype=float)
        else:
            values = weights.astype(float).mean(axis=0).reindex(column_values)
    elif isinstance(weights, pd.Series):
        values = weights.astype(float).reindex(column_values)
    elif isinstance(weights, Mapping):
        values = pd.Series({str(key): float(value) for key, value in weights.items()})
        values = values.reindex(column_values)
    else:
        arr = np.asarray(tuple(weights), dtype=float).reshape(-1)
        if len(arr) != len(column_values):
            raise ValueError("weights sequence must have one value per column")
        values = pd.Series(arr, index=column_values, dtype=float)
    values = values.fillna(float(fill_value)).astype(float)
    if normalize:
        total = float(values.sum())
        if not np.isfinite(total) or abs(total) <= 1e-12:
            raise ValueError("reference weights cannot be normalized because they sum to zero")
        values = values / total
    values.name = name
    values.attrs["macroforecast_metadata"] = {
        "operation": "align_reference_weights",
        "normalize": bool(normalize),
        "fill_value": float(fill_value),
        "source_method": "assemblage_weight_transformation",
        "source_reference": ALBACORE_SOURCE,
    }
    return values


def weighted_aggregate(
    data: FeatureInput,
    weights: Mapping[str, float] | pd.Series | pd.DataFrame | Sequence[float],
    *,
    metadata: Mapping[str, Any] | None = None,
    columns: Iterable[str] | None = None,
    normalize: bool = True,
    name: str = "weighted_aggregate",
    drop_missing: bool = False,
) -> pd.DataFrame:
    """Apply aligned component weights to a panel and return one aggregate.

    Albacore uses this object as a learned core-inflation index. This callable
    keeps the operation generic for any component-to-aggregate macro panel.
    """

    base = _coerce_input(data, metadata=metadata)
    panel = base.panel
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    aligned = align_reference_weights(weights, selected, normalize=normalize)
    values = panel.loc[:, selected].astype(float)
    out = pd.DataFrame(
        {name: values.to_numpy(dtype=float) @ aligned.to_numpy(dtype=float)},
        index=panel.index,
    )
    if drop_missing:
        out = out.dropna()
    out.index.name = "date"
    out.attrs["macroforecast_metadata"] = attach_metadata(
        base.metadata,
        "weighted_aggregate",
        {
            "columns": list(selected),
            "weights": {str(key): float(value) for key, value in aligned.items()},
            "normalize": bool(normalize),
            "source_method": "assemblage_weighted_core_measure",
            "source_reference": ALBACORE_SOURCE,
        },
    )
    out.attrs["macroforecast_feature_metadata"] = _metadata_frame(
        _records_for_columns(
            out,
            operation="weighted_aggregate",
            sources=tuple(selected),
            included=True,
        )
    )
    return out


def _normalize_moving_average_method(method: str) -> MovingAverageMethod:
    key = str(method).lower().replace("-", "_")
    aliases = {
        "compound_percent": "compound_percent",
        "percent": "compound_percent",
        "pct": "compound_percent",
        "assemblage": "compound_percent",
        "compound_decimal": "compound_decimal",
        "decimal": "compound_decimal",
        "mean": "mean",
        "average": "mean",
    }
    if key not in aliases:
        raise ValueError(f"method must be one of {sorted(aliases)}; got {method!r}")
    return aliases[key]  # type: ignore[return-value]


__all__ = [
    "ALBACORE_SOURCE",
    "align_reference_weights",
    "forward_average_target",
    "moving_average_changes",
    "rank_space_features",
    "weighted_aggregate",
]
