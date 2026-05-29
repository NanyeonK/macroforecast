from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata


DistributionMetric = Literal[
    "mean_change",
    "sd_change",
    "sd_ratio",
    "skew_change",
    "kurtosis_change",
    "ks_statistic",
]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]

DEFAULT_DISTRIBUTION_METRICS: tuple[DistributionMetric, ...] = (
    "mean_change",
    "sd_change",
    "sd_ratio",
    "skew_change",
    "kurtosis_change",
    "ks_statistic",
)


@dataclass(frozen=True)
class DataAnalysisReport:
    """Container returned by :func:`analyze_data`."""

    comparison: dict[str, Any]
    missing_shift: pd.DataFrame
    distribution_shift: pd.DataFrame
    correlation_shift: pd.DataFrame | None = None
    cleaning_effect_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "comparison": self.comparison,
            "missing_shift": self.missing_shift.to_dict(orient="index"),
            "distribution_shift": self.distribution_shift.to_dict(orient="index"),
            "cleaning_effect_summary": self.cleaning_effect_summary,
            "metadata": dict(self.metadata),
        }
        if self.correlation_shift is not None:
            out["correlation_shift"] = self.correlation_shift.to_dict()
        return out


def compare_panels(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Compare raw and cleaned panels at panel, column, index, and cell level."""

    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    common_columns = list(raw.columns.intersection(clean.columns))
    common_index = raw.index.intersection(clean.index)
    changed_cells = _count_changed_cells(
        raw.loc[common_index, common_columns],
        clean.loc[common_index, common_columns],
        tolerance=tolerance,
    )
    return {
        "raw_shape": tuple(raw.shape),
        "clean_shape": tuple(clean.shape),
        "raw_index_type": type(raw.index).__name__,
        "clean_index_type": type(clean.index).__name__,
        "raw_start": _index_value(raw.index.min()) if len(raw.index) else None,
        "raw_end": _index_value(raw.index.max()) if len(raw.index) else None,
        "clean_start": _index_value(clean.index.min()) if len(clean.index) else None,
        "clean_end": _index_value(clean.index.max()) if len(clean.index) else None,
        "raw_missing_total": int(raw.isna().sum().sum()),
        "clean_missing_total": int(clean.isna().sum().sum()),
        "common_columns": common_columns,
        "raw_only_columns": list(raw.columns.difference(clean.columns)),
        "clean_only_columns": list(clean.columns.difference(raw.columns)),
        "common_index_count": int(len(common_index)),
        "raw_only_index_count": int(len(raw.index.difference(clean.index))),
        "clean_only_index_count": int(len(clean.index.difference(raw.index))),
        "changed_cell_count": int(changed_cells),
    }


def missing_shift(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missing-count and missing-rate changes."""

    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    columns = list(raw.columns.union(clean.columns))
    rows: list[dict[str, Any]] = []
    for column in columns:
        raw_series = raw[column] if column in raw else pd.Series(dtype="float64")
        clean_series = clean[column] if column in clean else pd.Series(dtype="float64")
        raw_missing = int(raw_series.isna().sum())
        clean_missing = int(clean_series.isna().sum())
        raw_n = int(len(raw_series))
        clean_n = int(len(clean_series))
        rows.append(
            {
                "column": column,
                "raw_n": raw_n,
                "clean_n": clean_n,
                "raw_missing": raw_missing,
                "clean_missing": clean_missing,
                "delta_missing": clean_missing - raw_missing,
                "raw_missing_rate": _safe_ratio(raw_missing, raw_n),
                "clean_missing_rate": _safe_ratio(clean_missing, clean_n),
                "delta_missing_rate": _safe_ratio(clean_missing, clean_n)
                - _safe_ratio(raw_missing, raw_n),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def distribution_shift(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    metrics: Sequence[DistributionMetric] | None = None,
) -> pd.DataFrame:
    """Return per-series distribution changes from raw to cleaned data."""

    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    selected = tuple(metrics or DEFAULT_DISTRIBUTION_METRICS)
    unknown = sorted(set(selected) - set(DEFAULT_DISTRIBUTION_METRICS))
    if unknown:
        raise ValueError(f"unknown distribution metric(s): {unknown}")
    common = [
        column
        for column in raw.select_dtypes("number").columns
        if column in clean.select_dtypes("number").columns
    ]
    rows: list[dict[str, Any]] = []
    for column in common:
        raw_series = raw[column].dropna()
        clean_series = clean[column].dropna()
        raw_sd = raw_series.std()
        clean_sd = clean_series.std()
        row: dict[str, Any] = {
            "column": column,
            "raw_n": int(raw_series.shape[0]),
            "clean_n": int(clean_series.shape[0]),
        }
        if "mean_change" in selected:
            row["mean_change"] = _float_or_none(clean_series.mean() - raw_series.mean())
        if "sd_change" in selected:
            row["sd_change"] = _float_or_none(clean_sd - raw_sd)
        if "sd_ratio" in selected:
            row["sd_ratio"] = _float_or_none(clean_sd / raw_sd) if raw_sd else None
        if "skew_change" in selected:
            row["skew_change"] = _float_or_none(clean_series.skew() - raw_series.skew())
        if "kurtosis_change" in selected:
            row["kurtosis_change"] = _float_or_none(
                clean_series.kurtosis() - raw_series.kurtosis()
            )
        if "ks_statistic" in selected:
            row["ks_statistic"] = _ks_statistic(raw_series, clean_series)
        rows.append(row)
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def correlation_shift(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    method: CorrelationMethod = "pearson",
    fill_value: float | None = None,
) -> pd.DataFrame:
    """Return cleaned-minus-raw correlation matrix for common numeric columns."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    common = [
        column
        for column in raw.select_dtypes("number").columns
        if column in clean.select_dtypes("number").columns
    ]
    if len(common) < 2:
        return pd.DataFrame(index=common, columns=common, dtype="float64")
    delta = clean[common].corr(method=method) - raw[common].corr(method=method)
    return delta.fillna(fill_value) if fill_value is not None else delta


def cleaning_effect_summary(
    *,
    cleaning_metadata: Mapping[str, Any] | None = None,
    cleaning_log: Mapping[str, Any] | None = None,
    transform_map_applied: Mapping[str, int] | None = None,
    n_imputed_cells: int | None = None,
    n_outliers_flagged: int | None = None,
    n_truncated_obs: int | None = None,
    column_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize preprocessing metadata into a compact data analysis summary."""

    metadata = dict(cleaning_metadata or {})
    return {
        "n_imputed_cells": int(_coalesce(n_imputed_cells, metadata.get("n_imputed_cells"), 0)),
        "n_outliers_flagged": int(
            _coalesce(n_outliers_flagged, metadata.get("n_outliers_flagged"), 0)
        ),
        "n_truncated_obs": int(_coalesce(n_truncated_obs, metadata.get("n_truncated_obs"), 0)),
        "transform_map_applied": dict(
            _coalesce(transform_map_applied, metadata.get("transform_map_applied"), {})
        ),
        "cleaning_log": dict(_coalesce(cleaning_log, metadata.get("cleaning_log"), {})),
        "column_metadata": dict(
            _coalesce(column_metadata, metadata.get("column_metadata"), {})
        ),
    }


def analyze_data(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    distribution_metrics: Sequence[DistributionMetric] | None = None,
    include_correlation: bool = False,
    correlation_method: CorrelationMethod = "pearson",
    cleaning_metadata: Mapping[str, Any] | None = None,
    cleaning_log: Mapping[str, Any] | None = None,
    transform_map_applied: Mapping[str, int] | None = None,
    n_imputed_cells: int | None = None,
    n_outliers_flagged: int | None = None,
    n_truncated_obs: int | None = None,
    column_metadata: Mapping[str, Any] | None = None,
    tolerance: float = 0.0,
) -> DataAnalysisReport:
    """Run the standard data analysis suite on raw and cleaned panels."""

    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    selected_metrics = tuple(distribution_metrics or DEFAULT_DISTRIBUTION_METRICS)
    comparison = compare_panels(raw, clean, tolerance=tolerance)
    missing = missing_shift(raw, clean)
    distribution = distribution_shift(raw, clean, metrics=selected_metrics)
    correlation = (
        correlation_shift(raw, clean, method=correlation_method)
        if include_correlation
        else None
    )
    effects = cleaning_effect_summary(
        cleaning_metadata=cleaning_metadata,
        cleaning_log=cleaning_log,
        transform_map_applied=transform_map_applied,
        n_imputed_cells=n_imputed_cells,
        n_outliers_flagged=n_outliers_flagged,
        n_truncated_obs=n_truncated_obs,
        column_metadata=column_metadata,
    )
    metadata = _data_analysis_metadata(
        raw=raw,
        clean=clean,
        comparison=comparison,
        effects=effects,
        distribution_metrics=selected_metrics,
        include_correlation=include_correlation,
        correlation_method=correlation_method,
        tolerance=tolerance,
    )
    return DataAnalysisReport(
        comparison=comparison,
        missing_shift=missing,
        distribution_shift=distribution,
        correlation_shift=correlation,
        cleaning_effect_summary=effects,
        metadata=metadata,
    )


def _validate_panel(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.columns.has_duplicates:
        raise ValueError(f"{name} must not have duplicate column names")
    return frame


def _data_analysis_metadata(
    *,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    comparison: Mapping[str, Any],
    effects: Mapping[str, Any],
    distribution_metrics: Sequence[DistributionMetric],
    include_correlation: bool,
    correlation_method: CorrelationMethod,
    tolerance: float,
) -> dict[str, Any]:
    raw_metadata = _frame_metadata(raw)
    clean_metadata = _frame_metadata(clean)
    base_metadata = dict(raw_metadata)
    base_metadata.update(clean_metadata)
    return attach_metadata(
        base_metadata,
        "data_analysis",
        {
            "before": _panel_snapshot(raw),
            "after": _panel_snapshot(clean),
            "common": {
                "n_columns": int(len(comparison["common_columns"])),
                "n_rows": int(comparison["common_index_count"]),
                "changed_cells": int(comparison["changed_cell_count"]),
            },
            "options": {
                "distribution_metrics": list(distribution_metrics),
                "include_correlation": bool(include_correlation),
                "correlation_method": correlation_method if include_correlation else None,
                "tolerance": float(tolerance),
            },
            "effects": _effect_snapshot(effects),
            "metadata_keys": {
                "before": sorted(str(key) for key in raw_metadata),
                "after": sorted(str(key) for key in clean_metadata),
            },
        },
    )


def _frame_metadata(frame: pd.DataFrame) -> dict[str, Any]:
    return dict(getattr(frame, "attrs", {}).get("macroforecast_metadata", {}) or {})


def _panel_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "start": _index_value(frame.index.min()) if len(frame.index) else None,
        "end": _index_value(frame.index.max()) if len(frame.index) else None,
        "missing_values": int(frame.isna().sum().sum()),
    }


def _effect_snapshot(effects: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_imputed_cells": int(effects.get("n_imputed_cells", 0)),
        "n_outliers_flagged": int(effects.get("n_outliers_flagged", 0)),
        "n_truncated_obs": int(effects.get("n_truncated_obs", 0)),
        "n_transform_codes": int(len(effects.get("transform_map_applied", {}) or {})),
        "n_column_metadata": int(len(effects.get("column_metadata", {}) or {})),
        "has_cleaning_log": bool(effects.get("cleaning_log")),
    }


def _count_changed_cells(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    tolerance: float,
) -> int:
    if raw.empty or clean.empty:
        return 0
    equal = raw.eq(clean) | (raw.isna() & clean.isna())
    numeric_columns = [
        column
        for column in raw.select_dtypes("number").columns
        if column in clean.select_dtypes("number").columns
    ]
    if numeric_columns and tolerance > 0:
        close = (raw[numeric_columns] - clean[numeric_columns]).abs() <= tolerance
        equal.loc[:, numeric_columns] = equal[numeric_columns] | close
    return int((~equal).sum().sum())


def _ks_statistic(raw: pd.Series, clean: pd.Series) -> float | None:
    x = np.asarray(raw.dropna(), dtype=float)
    y = np.asarray(clean.dropna(), dtype=float)
    if x.size == 0 or y.size == 0:
        return None
    values = np.sort(np.unique(np.concatenate([x, y])))
    x_cdf = np.searchsorted(np.sort(x), values, side="right") / x.size
    y_cdf = np.searchsorted(np.sort(y), values, side="right") / y.size
    return float(np.max(np.abs(x_cdf - y_cdf)))


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _index_value(value: Any) -> Any:
    return value.isoformat() if hasattr(value, "isoformat") else value


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
