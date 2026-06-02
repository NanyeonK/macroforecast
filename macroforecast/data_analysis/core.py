from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata, validate_panel


DistributionMetric = Literal[
    "mean_change",
    "sd_change",
    "sd_ratio",
    "skew_change",
    "kurtosis_change",
    "ks_statistic",
]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
AnalysisSample = Literal["common_index", "full"]

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
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    common_columns = list(raw.columns.intersection(clean.columns))
    common_index = raw.index.intersection(clean.index)
    changed_cells = changed_cell_count(raw, clean, tolerance=tolerance)
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


def panel_snapshots(raw: pd.DataFrame, clean: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return compact before/after panel snapshots."""

    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    return {"before": _panel_snapshot(raw), "after": _panel_snapshot(clean)}


def changed_cells(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    """Return a boolean mask of changed common-index/common-column cells."""

    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    common_columns = list(raw.columns.intersection(clean.columns))
    common_index = raw.index.intersection(clean.index)
    return _changed_cell_mask(
        raw.loc[common_index, common_columns],
        clean.loc[common_index, common_columns],
        tolerance=tolerance,
    )


def changed_cell_count(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    tolerance: float = 0.0,
) -> int:
    """Return the number of changed common-index/common-column cells."""

    mask = changed_cells(raw, clean, tolerance=tolerance)
    return int(mask.sum().sum()) if not mask.empty else 0


def changed_cell_summary(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    """Return changed-cell count and rate for the common sample."""

    mask = changed_cells(raw, clean, tolerance=tolerance)
    n_cells = int(mask.size)
    n_changed = int(mask.sum().sum()) if n_cells else 0
    return {
        "common_rows": int(mask.shape[0]),
        "common_columns": int(mask.shape[1]),
        "common_cells": n_cells,
        "changed_cells": n_changed,
        "changed_cell_rate": float(n_changed / n_cells) if n_cells else 0.0,
        "tolerance": float(tolerance),
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
        if column in raw and column in clean:
            column_status = "common"
        elif column in raw:
            column_status = "raw_only"
        else:
            column_status = "clean_only"
        raw_missing = int(raw_series.isna().sum())
        clean_missing = int(clean_series.isna().sum())
        raw_n = int(len(raw_series))
        clean_n = int(len(clean_series))
        rows.append(
            {
                "column": column,
                "column_status": column_status,
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
    sample: AnalysisSample = "common_index",
) -> pd.DataFrame:
    """Return per-series distribution changes from raw to cleaned data."""

    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    _validate_sample(sample)
    selected = tuple(metrics or DEFAULT_DISTRIBUTION_METRICS)
    unknown = sorted(set(selected) - set(DEFAULT_DISTRIBUTION_METRICS))
    if unknown:
        raise ValueError(f"unknown distribution metric(s): {unknown}")
    common = [
        column
        for column in raw.select_dtypes("number").columns
        if column in clean.select_dtypes("number").columns
    ]
    raw_aligned, clean_aligned, index_n = _analysis_frames(raw, clean, common, sample=sample)
    rows: list[dict[str, Any]] = []
    for column in common:
        raw_series = raw_aligned[column].dropna()
        clean_series = clean_aligned[column].dropna()
        raw_sd = raw_series.std()
        clean_sd = clean_series.std()
        row: dict[str, Any] = {
            "column": column,
            "sample": sample,
            "sample_n": index_n,
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
    sample: AnalysisSample = "common_index",
) -> pd.DataFrame:
    """Return cleaned-minus-raw correlation matrix for common numeric columns."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    raw = _validate_panel(raw, "raw")
    clean = _validate_panel(clean, "clean")
    _validate_sample(sample)
    common = [
        column
        for column in raw.select_dtypes("number").columns
        if column in clean.select_dtypes("number").columns
    ]
    if len(common) < 2:
        return pd.DataFrame(index=common, columns=common, dtype="float64")
    raw_aligned, clean_aligned, _index_n = _analysis_frames(raw, clean, common, sample=sample)
    delta = clean_aligned[common].corr(method=method) - raw_aligned[common].corr(method=method)
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
    transform_map = _coalesce(
        transform_map_applied,
        metadata.get("transform_map_applied"),
        _transform_map_from_steps(metadata),
        {},
    )
    log = _coalesce(cleaning_log, metadata.get("cleaning_log"), _cleaning_log_from_metadata(metadata), {})
    columns = _coalesce(column_metadata, metadata.get("column_metadata"), metadata.get("transform_state"), {})
    return {
        "n_imputed_cells": int(
            _coalesce(n_imputed_cells, metadata.get("n_imputed_cells"), _imputed_cells_from_steps(metadata), 0)
        ),
        "n_outliers_flagged": int(
            _coalesce(n_outliers_flagged, metadata.get("n_outliers_flagged"), _outliers_from_steps(metadata), 0)
        ),
        "n_truncated_obs": int(
            _coalesce(n_truncated_obs, metadata.get("n_truncated_obs"), _truncated_obs_from_steps(metadata), 0)
        ),
        "transform_map_applied": dict(transform_map),
        "cleaning_log": dict(log),
        "column_metadata": dict(columns),
    }


def analyze_data(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    distribution_metrics: Sequence[DistributionMetric] | None = None,
    include_correlation: bool = False,
    correlation_method: CorrelationMethod = "pearson",
    sample: AnalysisSample = "common_index",
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
    distribution = distribution_shift(raw, clean, metrics=selected_metrics, sample=sample)
    correlation = (
        correlation_shift(raw, clean, method=correlation_method, sample=sample)
        if include_correlation
        else None
    )
    auto_cleaning_metadata = cleaning_metadata
    if auto_cleaning_metadata is None:
        auto_cleaning_metadata = _preprocessing_metadata(clean)
    effects = cleaning_effect_summary(
        cleaning_metadata=auto_cleaning_metadata,
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
        sample=sample,
        tolerance=tolerance,
    )
    _attach_metadata(missing, metadata)
    _attach_metadata(distribution, metadata)
    _attach_metadata(correlation, metadata)
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
    if frame.index.has_duplicates:
        raise ValueError(f"{name} must not have duplicate index values")
    try:
        validate_panel(frame)
    except Exception as exc:
        raise ValueError(f"{name} must satisfy the macroforecast canonical panel contract") from exc
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
    sample: AnalysisSample,
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
            "analysis_type": "raw_vs_processed",
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
                "sample": sample,
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


def _preprocessing_metadata(frame: pd.DataFrame) -> Mapping[str, Any] | None:
    metadata = _frame_metadata(frame)
    stage = metadata.get("preprocessing")
    return stage if isinstance(stage, Mapping) else None


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


def _attach_metadata(frame: pd.DataFrame | None, metadata: Mapping[str, Any]) -> None:
    if frame is not None:
        frame.attrs["macroforecast_metadata"] = dict(metadata)


def _changed_cell_mask(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    *,
    tolerance: float,
) -> pd.DataFrame:
    if raw.empty or clean.empty:
        return pd.DataFrame(False, index=raw.index, columns=raw.columns)
    equal = raw.eq(clean) | (raw.isna() & clean.isna())
    numeric_columns = [
        column
        for column in raw.select_dtypes("number").columns
        if column in clean.select_dtypes("number").columns
    ]
    if numeric_columns and tolerance > 0:
        close = (raw[numeric_columns] - clean[numeric_columns]).abs() <= tolerance
        equal.loc[:, numeric_columns] = equal[numeric_columns] | close
    return ~equal


def _analysis_frames(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    columns: Sequence[Any],
    *,
    sample: AnalysisSample,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if sample == "common_index":
        index = raw.index.intersection(clean.index)
        if len(index) == 0:
            raise ValueError("sample='common_index' requires at least one common date")
        return raw.loc[index, list(columns)], clean.loc[index, list(columns)], int(len(index))
    if sample == "full":
        return raw.loc[:, list(columns)], clean.loc[:, list(columns)], int(max(len(raw.index), len(clean.index)))
    raise ValueError("sample must be one of 'common_index' or 'full'")


def _validate_sample(sample: AnalysisSample) -> None:
    if sample not in {"common_index", "full"}:
        raise ValueError("sample must be one of 'common_index' or 'full'")


def _steps(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_steps = metadata.get("steps", ())
    if not isinstance(raw_steps, (list, tuple)):
        return []
    return [step for step in raw_steps if isinstance(step, Mapping)]


def _transform_map_from_steps(metadata: Mapping[str, Any]) -> dict[str, int]:
    for step in _steps(metadata):
        if step.get("step") != "transform":
            continue
        applied = step.get("applied", {})
        if isinstance(applied, Mapping):
            return {str(key): int(value) for key, value in applied.items()}
    return {}


def _imputed_cells_from_steps(metadata: Mapping[str, Any]) -> int:
    return int(sum(_safe_int(step.get("missing_filled")) for step in _steps(metadata) if step.get("step") == "impute"))


def _outliers_from_steps(metadata: Mapping[str, Any]) -> int:
    return int(sum(_safe_int(step.get("missing_added")) for step in _steps(metadata) if step.get("step") == "outliers"))


def _truncated_obs_from_steps(metadata: Mapping[str, Any]) -> int:
    total = 0
    for step in _steps(metadata):
        if step.get("step") not in {"frame", "tcode_lag"}:
            continue
        if "rows_removed" in step:
            total += _safe_int(step.get("rows_removed"))
            continue
        input_shape = step.get("input_shape")
        output_shape = step.get("output_shape")
        if isinstance(input_shape, (list, tuple)) and isinstance(output_shape, (list, tuple)):
            if input_shape and output_shape:
                total += max(_safe_int(input_shape[0]) - _safe_int(output_shape[0]), 0)
    return int(total)


def _cleaning_log_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    steps = _steps(metadata)
    if not steps:
        return {}
    return {"steps": [dict(step) for step in steps]}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
