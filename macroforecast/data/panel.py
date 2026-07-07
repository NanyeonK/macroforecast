from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import date as date_type
from calendar import monthrange
import re
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

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
    strict: bool = True,
) -> pd.DataFrame:
    """Return ``frame`` as macroforecast's canonical date-indexed panel.

    ``strict=True`` is intentional. A forecasting panel should not silently
    lose rows because date parsing failed, nor should string cells such as
    ``"missing"`` become ``NaN`` without the caller noticing. Official FRED
    files use real missing-value markers that are already parsed upstream; this
    guard is mainly for custom CSV/Parquet inputs and ad hoc DataFrames.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    panel = frame.copy()
    input_rows = int(panel.shape[0])
    input_columns = [str(column) for column in panel.columns]
    date_source: str | None = None
    invalid_date_rows = 0

    if date is not None:
        if date not in panel.columns:
            raise ValueError(f"date column {date!r} is not in the DataFrame")
        date_source = str(date)
        panel[date] = pd.to_datetime(panel[date], errors="coerce")
        invalid_date_rows = int(panel[date].isna().sum())
        if invalid_date_rows and strict:
            raise ValueError(
                f"date column {date!r} has {invalid_date_rows} invalid or missing date values"
            )
        panel = panel[panel[date].notna()].set_index(date)
    elif not isinstance(panel.index, pd.DatetimeIndex):
        if not len(panel.columns):
            raise ValueError("panel must have a DatetimeIndex or a date column")
        first_column = panel.columns[0]
        date_source = str(first_column)
        panel[first_column] = pd.to_datetime(panel[first_column], errors="coerce")
        invalid_date_rows = int(panel[first_column].isna().sum())
        if invalid_date_rows and strict:
            raise ValueError(
                f"date column {str(first_column)!r} has {invalid_date_rows} invalid or missing date values"
            )
        panel = panel[panel[first_column].notna()].set_index(first_column)
    else:
        date_source = "index"

    if columns is not None:
        selected = [str(column) for column in columns]
        missing = [column for column in selected if column not in panel.columns]
        if missing:
            raise ValueError(f"columns are not in the panel: {missing}")
        panel = panel[selected]

    if rename:
        panel = panel.rename(columns=dict(rename))

    panel.index = pd.DatetimeIndex(panel.index)
    invalid_index_rows = int(panel.index.isna().sum())
    if invalid_index_rows and strict:
        raise ValueError(f"panel index has {invalid_index_rows} invalid or missing date values")
    panel = panel[panel.index.notna()]
    panel.index.name = "date"
    panel = panel.sort_index()
    if panel.index.has_duplicates:
        duplicated = panel.index[panel.index.duplicated()].unique()
        sample = ", ".join(ts.strftime("%Y-%m-%d") for ts in duplicated[:3])
        raise ValueError(f"panel has duplicate dates: {sample}{_long_format_hint(panel)}")

    coercion_report = _numeric_coercion_report(panel)
    for column in panel.columns:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    if coercion_report["coerced_cells"] and strict:
        examples = coercion_report["examples"]
        raise ValueError(
            "non-numeric panel values would be coerced to NaN; "
            f"coerced_cells={coercion_report['coerced_cells']}, examples={examples}"
        )
    inf_report = _infinite_value_report(panel)
    if inf_report["inf_cells"]:
        raise ValueError(
            "panel contains infinite values; replace them with finite values or missing values "
            f"before loading. inf_cells={inf_report['inf_cells']}, examples={inf_report['examples']}"
        )
    validate_panel(panel)

    attrs = dict(getattr(frame, "attrs", {}) or {})
    attrs.update(getattr(panel, "attrs", {}) or {})
    panel_report = {
        "contract": "macroforecast_panel_v1",
        "strict": bool(strict),
        "input_rows": input_rows,
        "output_rows": int(panel.shape[0]),
        "input_columns": input_columns,
        "output_columns": [str(column) for column in panel.columns],
        "date_source": date_source,
        "invalid_date_rows_dropped": int(invalid_date_rows + invalid_index_rows),
        "numeric_coercion": coercion_report,
    }
    attrs["macroforecast_panel_report"] = panel_report
    if metadata is not None:
        # Keep panel-normalization metadata beside the data-source metadata.
        # Later stages read this to tell whether a custom load required any
        # lossy normalization. The source metadata itself remains unchanged
        # unless the bundle constructor explicitly adopts this attrs payload.
        attrs["macroforecast_metadata"] = attach_metadata(metadata, "panel", panel_report)
    panel.attrs.update(attrs)
    return panel


def validate_panel(panel: pd.DataFrame) -> None:
    """Validate macroforecast's canonical panel contract."""

    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError(f"panel must not be empty; got shape {panel.shape}")
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
    inf_report = _infinite_value_report(panel)
    if inf_report["inf_cells"]:
        raise ValueError(
            "panel must contain only finite numeric values or NaN; "
            f"inf_cells={inf_report['inf_cells']}, examples={inf_report['examples']}"
        )


def panel_info(panel: PanelInput) -> dict[str, Any]:
    """Return a compact diagnostic summary for a canonical panel."""

    bundle = _coerce_bundle(panel)
    frame = bundle.panel
    validate_panel(frame)
    index_frequency = pd.infer_freq(frame.index) if len(frame.index) >= 3 else None
    metadata_frequency = bundle.metadata.get("frequency")
    info = {
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "start": frame.index[0].strftime("%Y-%m-%d") if len(frame) else None,
        "end": frame.index[-1].strftime("%Y-%m-%d") if len(frame) else None,
        "columns": [str(column) for column in frame.columns],
        "missing_values": int(frame.isna().sum().sum()),
        "frequency": metadata_frequency or index_frequency,
        "index_frequency": index_frequency,
    }
    if metadata_frequency is not None:
        info["metadata_frequency"] = metadata_frequency
    native_frequency_counts = bundle.metadata.get("native_frequency_counts")
    if native_frequency_counts is not None:
        info["native_frequency_counts"] = dict(native_frequency_counts)
    output_frequency_counts = bundle.metadata.get("output_frequency_counts")
    if output_frequency_counts is not None:
        info["output_frequency_counts"] = dict(output_frequency_counts)
    return info


def metadata(obj: PanelInput) -> dict[str, Any]:
    """Return metadata from a bundle, spec, tuple, or DataFrame."""

    return dict(_coerce_bundle(obj).metadata)


def custom_dataset(
    frame: pd.DataFrame,
    *,
    date: str | None = None,
    columns: Iterable[str] | None = None,
    rename: Mapping[str, str] | None = None,
    dataset: str = "custom",
    source_family: str = "custom",
    frequency: str = "unknown",
    frequency_by_column: Mapping[str, str] | None = None,
    transform_codes: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> DataBundle:
    """Build a canonical custom ``DataBundle`` from an in-memory DataFrame."""

    base_metadata = dict(metadata or {})
    base_metadata.update(
        {
            "dataset": str(dataset),
            "source_family": str(source_family),
            "frequency": _normalize_frequency_label(frequency),
        }
    )
    panel = as_panel(
        frame,
        date=date,
        columns=columns,
        rename=rename,
        metadata=base_metadata,
        strict=strict,
    )
    updated = dict(panel.attrs.get("macroforecast_metadata", base_metadata))
    if transform_codes is not None:
        code_map = {str(column): int(code) for column, code in transform_codes.items()}
        missing = sorted(set(code_map) - {str(column) for column in panel.columns})
        if missing:
            raise ValueError(f"transform code keys are not in the panel: {missing}")
        updated["transform_codes"] = dict(sorted(code_map.items()))
    bundle = DataBundle(panel=panel, metadata=updated)
    if frequency_by_column is not None:
        bundle = set_frequencies(
            bundle,
            frequency_by_column,
            default_frequency=frequency,
            frequency=frequency,
        )
        updated = dict(bundle.metadata)
    updated = attach_metadata(
        updated,
        "custom_dataset",
        {
            "dataset": str(dataset),
            "source_family": str(source_family),
            "frequency": updated.get("frequency"),
            "columns": [str(column) for column in panel.columns],
            "strict": bool(strict),
        },
    )
    panel = bundle.panel.copy()
    panel.attrs["macroforecast_metadata"] = updated
    return DataBundle(panel=panel, metadata=updated)


def set_frequencies(
    data: PanelInput,
    frequency_by_column: Mapping[str, str],
    *,
    default_frequency: str | None = None,
    output_frequency_by_column: Mapping[str, str] | None = None,
    frequency: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DataBundle:
    """Attach a column-level frequency contract to a panel or bundle."""

    bundle = _coerce_bundle(data, metadata=metadata)
    panel = bundle.panel.copy()
    validate_panel(panel)
    native = _normalize_frequency_map(
        frequency_by_column,
        columns=panel.columns,
        default_frequency=default_frequency,
        label="frequency_by_column",
    )
    output = (
        dict(native)
        if output_frequency_by_column is None
        else _normalize_frequency_map(
            output_frequency_by_column,
            columns=panel.columns,
            default_frequency=default_frequency,
            label="output_frequency_by_column",
        )
    )
    frequency_label = _resolve_frequency_label(native, frequency=frequency)
    updated = dict(bundle.metadata)
    updated.update(
        {
            "frequency": frequency_label,
            "native_frequency_by_column": dict(sorted(native.items())),
            "native_frequency_counts": dict(sorted(Counter(native.values()).items())),
            "output_frequency_by_column": dict(sorted(output.items())),
            "output_frequency_counts": dict(sorted(Counter(output.values()).items())),
        }
    )
    panel.attrs["macroforecast_metadata"] = updated
    return DataBundle(panel=panel, metadata=updated)


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
        # ``predictors='all'`` means all non-target columns. Recording the
        # expanded tuple prevents a later model stage from accidentally using
        # the target column as its own predictor when it consumes metadata.
        predictor_values = tuple(str(column) for column in panel.columns if str(column) not in set(target_values))
    else:
        predictor_values = tuple(dict.fromkeys(str(value) for value in predictors))
        overlap = sorted(set(predictor_values).intersection(target_values))
        if overlap:
            raise ValueError(f"predictors must not include target columns: {overlap}")

    required_columns = set(target_values)
    if predictor_values != "all":
        required_columns.update(predictor_values)
    missing = [column for column in sorted(required_columns) if column not in panel.columns]
    if missing:
        raise ValueError(f"requested columns are not in the panel: {missing}")

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
            "predictors": list(predictor_values),
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


def _numeric_coercion_report(panel: pd.DataFrame) -> dict[str, Any]:
    coerced_cells = 0
    examples: list[dict[str, Any]] = []
    for column in panel.columns:
        before = panel[column]
        after = pd.to_numeric(before, errors="coerce")
        mask = before.notna() & after.isna()
        count = int(mask.sum())
        coerced_cells += count
        if count and len(examples) < 5:
            for index, value in before[mask].head(5 - len(examples)).items():
                examples.append(
                    {
                        "date": pd.Timestamp(index).strftime("%Y-%m-%d"),
                        "column": str(column),
                        "value": str(value),
                    }
                )
    return {"coerced_cells": int(coerced_cells), "examples": examples}


def _long_format_hint(panel: pd.DataFrame) -> str:
    if panel.empty:
        return ""
    max_unique = max(2, min(20, int(len(panel) * 0.5)))
    for column in panel.columns:
        series = panel[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        unique_values = int(series.nunique(dropna=True))
        if 1 < unique_values <= max_unique:
            return (
                "; data appears to be in long format; pivot to wide "
                "(one column per series) - e.g. "
                "df.pivot(index=..., columns=..., values=...)"
            )
    return ""


def _infinite_value_report(panel: pd.DataFrame) -> dict[str, Any]:
    numeric = panel.select_dtypes("number")
    if numeric.empty:
        return {"inf_cells": 0, "examples": []}
    values = numeric.to_numpy(dtype=float, copy=False)
    mask = np.isinf(values)
    examples: list[dict[str, Any]] = []
    if mask.any():
        row_positions, column_positions = np.where(mask)
        for row_pos, column_pos in zip(row_positions[:5], column_positions[:5], strict=False):
            examples.append(
                {
                    "date": pd.Timestamp(numeric.index[int(row_pos)]).strftime("%Y-%m-%d"),
                    "column": str(numeric.columns[int(column_pos)]),
                    "value": float(values[int(row_pos), int(column_pos)]),
                }
            )
    return {"inf_cells": int(mask.sum()), "examples": examples}


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


def _normalize_frequency_map(
    frequency_by_column: Mapping[str, str],
    *,
    columns: pd.Index,
    default_frequency: str | None,
    label: str,
) -> dict[str, str]:
    column_names = [str(column) for column in columns]
    provided = {str(column): _normalize_frequency_label(value) for column, value in frequency_by_column.items()}
    unknown = sorted(set(provided).difference(column_names))
    if unknown:
        raise ValueError(f"{label} includes unknown columns: {unknown}")
    if default_frequency is not None:
        default = _normalize_frequency_label(default_frequency)
        for column in column_names:
            provided.setdefault(column, default)
    missing = [column for column in column_names if column not in provided]
    if missing:
        raise ValueError(f"{label} must include every panel column or default_frequency; missing={missing}")
    return {column: provided[column] for column in column_names}


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


def _resolve_frequency_label(native: Mapping[str, str], *, frequency: str | None) -> str:
    if frequency is not None:
        if str(frequency).strip().lower() == "mixed":
            return "mixed"
        explicit = _normalize_frequency_label(frequency)
        if explicit == "unknown":
            unique = set(native.values())
            return unique.pop() if len(unique) == 1 else "mixed"
        return explicit
    unique = set(native.values())
    return unique.pop() if len(unique) == 1 else "mixed"


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
        horizons: tuple[int, ...] = (values,)
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
