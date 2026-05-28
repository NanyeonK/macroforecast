from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date as date_type
from calendar import monthrange
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypeAlias

import pandas as pd

from .types import RawLoadResult


PredictorSelection = Literal["all"] | tuple[str, ...]


@dataclass(frozen=True)
class DataBundle:
    """Canonical data payload: a pandas panel plus explicit metadata."""

    panel: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.panel
        yield self.metadata

    def attach(self, stage: str, values: Mapping[str, Any]) -> DataBundle:
        return replace(self, metadata=attach_metadata(self.metadata, stage, values))


@dataclass(frozen=True)
class DataSpec:
    """Panel plus target, horizon, sample, and predictor choices for a run."""

    panel: pd.DataFrame
    metadata: dict[str, Any]
    target: str | None
    targets: tuple[str, ...]
    horizons: tuple[int, ...]
    start: str | None = None
    end: str | None = None
    predictors: PredictorSelection = "all"

    def __iter__(self):
        yield self.panel
        yield self.metadata

    def attach(self, stage: str, values: Mapping[str, Any]) -> DataSpec:
        return replace(self, metadata=attach_metadata(self.metadata, stage, values))


PanelInput: TypeAlias = DataBundle | DataSpec | tuple[pd.DataFrame, Mapping[str, Any]] | pd.DataFrame


def as_panel(
    frame: pd.DataFrame,
    *,
    date: str | None = None,
    columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return ``frame`` as macroforecast's canonical date-indexed panel."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    panel = frame.copy()

    if date is not None:
        if date not in panel.columns:
            raise ValueError(f"date column {date!r} is not in the DataFrame")
        panel[date] = pd.to_datetime(panel[date], errors="coerce")
        panel = panel[panel[date].notna()].set_index(date)
    elif not isinstance(panel.index, pd.DatetimeIndex):
        if not len(panel.columns):
            raise ValueError("panel must have a DatetimeIndex or a date column")
        first_column = panel.columns[0]
        panel[first_column] = pd.to_datetime(panel[first_column], errors="coerce")
        panel = panel[panel[first_column].notna()].set_index(first_column)

    if columns is not None:
        selected = [str(column) for column in columns]
        missing = [column for column in selected if column not in panel.columns]
        if missing:
            raise ValueError(f"columns are not in the panel: {missing}")
        panel = panel[selected]

    if rename:
        panel = panel.rename(columns=dict(rename))

    panel.index = pd.DatetimeIndex(panel.index)
    panel.index.name = "date"
    panel = panel.sort_index()
    if panel.index.has_duplicates:
        duplicated = panel.index[panel.index.duplicated()].unique()
        sample = ", ".join(ts.strftime("%Y-%m-%d") for ts in duplicated[:3])
        raise ValueError(f"panel has duplicate dates: {sample}")

    for column in panel.columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    validate_panel(panel)

    attrs = dict(getattr(frame, "attrs", {}) or {})
    attrs.update(getattr(panel, "attrs", {}) or {})
    if metadata is not None:
        attrs["macroforecast_metadata"] = dict(metadata)
    panel.attrs.update(attrs)
    return panel


def validate_panel(panel: pd.DataFrame) -> None:
    """Validate macroforecast's canonical panel contract."""

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("panel index must be a pandas DatetimeIndex")
    if panel.index.name != "date":
        raise ValueError("panel index name must be 'date'")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted in ascending date order")
    if panel.index.has_duplicates:
        raise ValueError("panel index must not contain duplicate dates")
    non_numeric = [
        str(column)
        for column, dtype in panel.dtypes.items()
        if not pd.api.types.is_numeric_dtype(dtype)
    ]
    if non_numeric:
        raise TypeError(f"panel columns must be numeric: {non_numeric}")


def panel_info(panel: PanelInput) -> dict[str, Any]:
    """Return a compact diagnostic summary for a canonical panel."""

    bundle = _coerce_bundle(panel)
    frame = bundle.panel
    validate_panel(frame)
    return {
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "start": frame.index[0].strftime("%Y-%m-%d") if len(frame) else None,
        "end": frame.index[-1].strftime("%Y-%m-%d") if len(frame) else None,
        "columns": [str(column) for column in frame.columns],
        "missing_values": int(frame.isna().sum().sum()),
        "frequency": pd.infer_freq(frame.index) if len(frame.index) >= 3 else None,
    }


def metadata(obj: PanelInput | RawLoadResult) -> dict[str, Any]:
    """Return metadata from a bundle, spec, frame, tuple, or raw load result."""

    if isinstance(obj, RawLoadResult):
        return metadata_from_result(obj)
    return dict(_coerce_bundle(obj).metadata)


def spec(
    data: PanelInput,
    *,
    metadata: Mapping[str, Any] | None = None,
    target: str | None = None,
    targets: Iterable[str] | None = None,
    horizons: Iterable[int] | int | None = None,
    start: str | None = None,
    end: str | None = None,
    predictors: Literal["all"] | Iterable[str] = "all",
) -> DataSpec:
    """Build a run-level data specification from a canonical panel."""

    bundle = _coerce_bundle(data, metadata=metadata)
    panel = bundle.panel
    validate_panel(panel)
    target_value, target_values = _normalize_targets(target=target, targets=targets)
    horizon_values = _normalize_horizons(horizons, metadata=bundle.metadata)
    start_iso = _normalize_date(start, end_of_period=False)
    end_iso = _normalize_date(end, end_of_period=True)
    if start_iso is not None:
        panel = panel.loc[panel.index >= pd.Timestamp(start_iso)]
    if end_iso is not None:
        panel = panel.loc[panel.index <= pd.Timestamp(end_iso)]
    if start_iso is not None and end_iso is not None and pd.Timestamp(end_iso) < pd.Timestamp(start_iso):
        raise ValueError("end must be greater than or equal to start")

    predictor_values: PredictorSelection
    if predictors == "all":
        predictor_values = "all"
    else:
        predictor_values = tuple(str(value) for value in predictors)
        if not predictor_values:
            raise ValueError("predictors must not be empty")

    required_columns = set(target_values)
    if predictor_values != "all":
        required_columns.update(predictor_values)
    missing = [column for column in sorted(required_columns) if column not in panel.columns]
    if missing:
        raise ValueError(f"requested columns are not in the panel: {missing}")

    if predictor_values != "all":
        ordered = list(dict.fromkeys([*predictor_values, *target_values]))
        panel = panel[ordered]
    if panel.empty:
        raise ValueError("sample window leaves an empty panel")

    spec_metadata = attach_metadata(
        bundle.metadata,
        "data_spec",
        {
            "target": target_value,
            "targets": list(target_values),
            "horizons": list(horizon_values),
            "start": start_iso,
            "end": end_iso,
            "predictors": predictor_values if predictor_values == "all" else list(predictor_values),
            "panel": panel_info(DataBundle(panel, bundle.metadata)),
        },
    )
    panel = panel.copy()
    panel.attrs["macroforecast_metadata"] = spec_metadata
    return DataSpec(
        panel=panel,
        metadata=spec_metadata,
        target=target_value,
        targets=target_values,
        horizons=horizon_values,
        start=start_iso,
        end=end_iso,
        predictors=predictor_values,
    )


def bundle_from_result(result: RawLoadResult) -> DataBundle:
    """Convert a raw loader envelope into the public bundle contract."""

    meta = metadata_from_result(result)
    existing = dict(result.data.attrs.get("macroforecast_metadata", {}) or {})
    if existing:
        meta = {**existing, **meta}
    frame = as_panel(result.data, metadata=meta)
    if result.transform_codes:
        frame.attrs["macroforecast_transform_codes"] = dict(result.transform_codes)
    frame.attrs["macroforecast_metadata"] = meta
    return DataBundle(panel=frame, metadata=meta)


def metadata_from_result(result: RawLoadResult) -> dict[str, Any]:
    return {
        "dataset": result.dataset_metadata.dataset,
        "source_family": result.dataset_metadata.source_family,
        "frequency": result.dataset_metadata.frequency,
        "version_mode": result.dataset_metadata.version_mode,
        "vintage": result.dataset_metadata.vintage,
        "data_through": result.dataset_metadata.data_through,
        "support_tier": result.dataset_metadata.support_tier,
        "parse_notes": result.dataset_metadata.parse_notes,
        "artifact": {
            "source_url": result.artifact.source_url,
            "local_path": result.artifact.local_path,
            "file_format": result.artifact.file_format,
            "downloaded_at": result.artifact.downloaded_at,
            "file_sha256": result.artifact.file_sha256,
            "file_size_bytes": result.artifact.file_size_bytes,
            "cache_hit": result.artifact.cache_hit,
            "manifest_version": result.artifact.manifest_version,
        },
        "transform_codes": dict(result.transform_codes),
    }


def attach_metadata(metadata: Mapping[str, Any], stage: str, values: Mapping[str, Any]) -> dict[str, Any]:
    if not stage:
        raise ValueError("metadata stage must be non-empty")
    updated = dict(metadata)
    updated[stage] = dict(values)
    return updated


def _coerce_bundle(data: PanelInput, *, metadata: Mapping[str, Any] | None = None) -> DataBundle:
    if isinstance(data, DataSpec):
        base = DataBundle(data.panel, data.metadata)
    elif isinstance(data, DataBundle):
        base = data
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], pd.DataFrame):
        base = DataBundle(as_panel(data[0], metadata=data[1]), dict(data[1]))
    elif isinstance(data, pd.DataFrame):
        existing = dict(data.attrs.get("macroforecast_metadata", {}))
        base = DataBundle(as_panel(data, metadata=existing), existing)
    else:
        raise TypeError("expected DataBundle, DataSpec, (panel, metadata), or pandas DataFrame")
    if metadata is not None:
        merged = dict(base.metadata)
        merged.update(dict(metadata))
        panel = base.panel.copy()
        panel.attrs["macroforecast_metadata"] = merged
        return DataBundle(panel=panel, metadata=merged)
    return base


def _normalize_targets(*, target: str | None, targets: Iterable[str] | None) -> tuple[str | None, tuple[str, ...]]:
    if target is not None and targets is not None:
        raise ValueError("provide either target or targets, not both")
    if targets is None:
        if not isinstance(target, str) or not target:
            raise ValueError("target is required")
        return target, (target,)
    values = tuple(str(value) for value in targets)
    if not values:
        raise ValueError("targets must not be empty")
    return None, values


def _normalize_horizons(values: Iterable[int] | int | None, *, metadata: Mapping[str, Any]) -> tuple[int, ...]:
    if values is None:
        frequency = metadata.get("frequency")
        if frequency == "monthly":
            return (1, 3, 6, 12)
        if frequency == "quarterly":
            return (1, 2, 4, 8)
        return (1,)
    if isinstance(values, int):
        horizons = (values,)
    else:
        horizons = tuple(int(value) for value in values)
    if not horizons:
        raise ValueError("horizons must not be empty")
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive integers")
    return horizons


def _normalize_date(value: str | None, *, end_of_period: bool) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("date values must be strings")
    try:
        return date_type.fromisoformat(value).isoformat()
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}-\d{2}", value):
        year, month = int(value[:4]), int(value[5:7])
        if not 1 <= month <= 12:
            raise ValueError(f"invalid date value: {value!r}")
        day = monthrange(year, month)[1] if end_of_period else 1
        return f"{year:04d}-{month:02d}-{day:02d}"
    if re.fullmatch(r"\d{4}", value):
        return f"{value}-12-31" if end_of_period else f"{value}-01-01"
    raise ValueError(f"invalid date value: {value!r}")
