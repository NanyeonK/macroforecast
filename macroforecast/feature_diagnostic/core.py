from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import DataBundle, DataSpec, attach_metadata, validate_panel
from macroforecast.feature_engineering.types import FeatureSet


CorrelationMethod = Literal["pearson", "spearman", "kendall"]

FACTOR_OPERATIONS: tuple[str, ...] = (
    "pca",
    "group_pca",
    "maf",
    "factor",
    "factor_lag",
    "scaled_pca",
    "supervised_pca",
    "supervised_scaled_pca",
)
LAG_OPERATIONS: tuple[str, ...] = (
    "lag",
    "mixed_frequency_lag",
    "seasonal_lag",
    "factor_lag",
    "rolling_mean",
    "moving_average",
    "marx",
)


@dataclass(frozen=True)
class FeatureDiagnosticReport:
    """Container returned by :func:`diagnose_features`."""

    overview: dict[str, Any]
    correlation: pd.DataFrame | None = None
    factors: pd.DataFrame | None = None
    lags: pd.DataFrame | None = None
    marx: pd.DataFrame | None = None
    selection_stability: pd.DataFrame | None = None
    stage_comparison: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "overview": self.overview,
            "metadata": dict(self.metadata),
        }
        if self.correlation is not None:
            out["correlation"] = self.correlation.to_dict(orient="records")
        if self.factors is not None:
            out["factors"] = self.factors.to_dict(orient="records")
        if self.lags is not None:
            out["lags"] = self.lags.to_dict(orient="records")
        if self.marx is not None:
            out["marx"] = self.marx.to_dict(orient="records")
        if self.selection_stability is not None:
            out["selection_stability"] = self.selection_stability.to_dict(orient="index")
        if self.stage_comparison is not None:
            out["stage_comparison"] = self.stage_comparison.to_dict(orient="index")
        return out


def feature_overview(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    high_missing_threshold: float = 0.5,
) -> dict[str, Any]:
    """Return shape, missingness, variance, and metadata coverage for features."""

    _validate_probability(high_missing_threshold, "high_missing_threshold")
    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X
    metadata = _coerce_feature_metadata(bundle.feature_metadata)
    missing_rate = X.isna().mean(axis=0)
    variance = X.var(axis=0, skipna=True)
    operation_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    included_count: int | None = None
    excluded_count: int | None = None
    if not metadata.empty:
        if "operation" in metadata:
            operation_counts = _value_counts(metadata["operation"])
        if "source" in metadata:
            source_counts = _value_counts(metadata["source"])
        if "included" in metadata:
            included = metadata["included"].map(_metadata_bool)
            included_count = int(included.sum())
            excluded_count = int((~included).sum())
    return {
        "n_observations": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "start": X.index[0].strftime("%Y-%m-%d") if len(X.index) else None,
        "end": X.index[-1].strftime("%Y-%m-%d") if len(X.index) else None,
        "missing_total": int(X.isna().sum().sum()),
        "missing_rate": _safe_ratio(int(X.isna().sum().sum()), int(X.size)),
        "high_missing_threshold": float(high_missing_threshold),
        "high_missing_features": missing_rate[missing_rate > float(high_missing_threshold)].index.astype(str).tolist(),
        "zero_variance_features": variance[(variance.fillna(0.0) == 0.0)].index.astype(str).tolist(),
        "feature_metadata_available": bool(not metadata.empty),
        "feature_metadata_rows": int(metadata.shape[0]),
        "operation_counts": operation_counts,
        "source_counts": source_counts,
        "included_count": included_count,
        "excluded_count": excluded_count,
    }


def compare_feature_stages(
    stages: Mapping[str, Any] | None = None,
    **named_stages: Any,
) -> pd.DataFrame:
    """Compare feature-like panels across named construction stages."""

    stage_map: dict[str, Any] = {}
    if stages is not None:
        stage_map.update(dict(stages))
    stage_map.update(named_stages)
    if not stage_map:
        raise ValueError("at least one feature stage is required")

    rows: list[dict[str, Any]] = []
    previous_columns: set[str] | None = None
    for name, value in stage_map.items():
        bundle = _coerce_feature_input(value)
        X = bundle.X
        columns = {str(column) for column in X.columns}
        numeric = X.select_dtypes("number")
        row: dict[str, Any] = {
            "stage": str(name),
            "n_observations": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "start": X.index[0].strftime("%Y-%m-%d") if len(X.index) else None,
            "end": X.index[-1].strftime("%Y-%m-%d") if len(X.index) else None,
            "missing_total": int(X.isna().sum().sum()),
            "missing_rate": _safe_ratio(int(X.isna().sum().sum()), int(X.size)),
            "zero_variance_count": int((numeric.var(axis=0, skipna=True).fillna(0.0) == 0.0).sum()),
        }
        if previous_columns is None:
            row["common_with_previous"] = None
            row["added_from_previous"] = None
            row["removed_from_previous"] = None
        else:
            row["common_with_previous"] = int(len(columns & previous_columns))
            row["added_from_previous"] = int(len(columns - previous_columns))
            row["removed_from_previous"] = int(len(previous_columns - columns))
        rows.append(row)
        previous_columns = columns

    out = pd.DataFrame(rows).set_index("stage")
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "feature_stage_comparison",
        "version": 1,
    }
    return out


def feature_correlation(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    method: CorrelationMethod = "pearson",
    min_periods: int = 3,
    threshold: float | None = 0.9,
    absolute: bool = True,
    max_pairs: int | None = None,
) -> pd.DataFrame:
    """Return long-form high-correlation feature pairs."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    if int(min_periods) < 1:
        raise ValueError("min_periods must be a positive integer")
    if threshold is not None:
        _validate_probability(threshold, "threshold")
    if max_pairs is not None and int(max_pairs) < 1:
        raise ValueError("max_pairs must be a positive integer or None")

    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X.select_dtypes("number")
    corr = X.corr(method=method, min_periods=int(min_periods))
    metadata = _metadata_lookup(bundle.feature_metadata)
    rows: list[dict[str, Any]] = []
    columns = list(corr.columns)
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            score = abs(float(value)) if absolute else float(value)
            if threshold is not None and score < float(threshold):
                continue
            row = {
                "feature_a": str(left),
                "feature_b": str(right),
                "correlation": float(value),
                "abs_correlation": abs(float(value)),
            }
            row.update(_pair_metadata(metadata, str(left), str(right)))
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "correlation",
                "abs_correlation",
                "operation_a",
                "operation_b",
                "source_a",
                "source_b",
            ]
        )
    else:
        sort_column = "abs_correlation" if absolute else "correlation"
        out = out.sort_values(sort_column, ascending=False).reset_index(drop=True)
        if max_pairs is not None:
            out = out.head(int(max_pairs)).reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "feature_correlation_pairs",
        "version": 1,
        "method": method,
        "threshold": threshold,
    }
    return out


def factor_diagnostics(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    operations: Sequence[str] = FACTOR_OPERATIONS,
    prefixes: Sequence[str] = ("pc", "factor", "maf"),
) -> pd.DataFrame:
    """Summarize factor/component feature columns."""

    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X
    metadata = _coerce_feature_metadata(bundle.feature_metadata)
    candidates = _factor_candidates(X, metadata, operations=operations, prefixes=prefixes)
    rows: list[dict[str, Any]] = []
    for feature, meta in candidates.items():
        if feature not in X:
            continue
        series = X[feature]
        operation = _str_or_none(meta.get("operation")) or _inferred_factor_operation(feature, prefixes)
        block = _str_or_none(meta.get("block"))
        source = _str_or_none(meta.get("source"))
        group = block or source or operation or "factor"
        variance = _float_or_none(series.var(skipna=True))
        rows.append(
            {
                "feature": feature,
                "group": group,
                "operation": operation,
                "block": block,
                "source": source,
                "component": _int_or_none(meta.get("component")) or _parse_component(feature),
                "n_obs": int(series.notna().sum()),
                "missing_rate": float(series.isna().mean()),
                "mean": _float_or_none(series.mean(skipna=True)),
                "sd": _float_or_none(series.std(skipna=True)),
                "variance": variance,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "feature",
                "group",
                "operation",
                "block",
                "source",
                "component",
                "n_obs",
                "missing_rate",
                "mean",
                "sd",
                "variance",
                "variance_share",
            ]
        )
    else:
        out["variance_share"] = _variance_share(out)
        out = out.sort_values(["group", "component", "feature"], na_position="last").reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "factor_diagnostics", "version": 1}
    return out


def lag_diagnostics(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    operations: Sequence[str] = LAG_OPERATIONS,
) -> pd.DataFrame:
    """Summarize feature columns that carry lag or window information."""

    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X
    metadata = _coerce_feature_metadata(bundle.feature_metadata)
    candidates = _lag_candidates(X, metadata, operations=operations)
    rows = [_lag_row(X, feature, meta) for feature, meta in candidates.items() if feature in X]
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "feature",
                "operation",
                "source",
                "lag",
                "window",
                "n_obs",
                "missing_rate",
                "first_valid",
                "last_valid",
            ]
        )
    else:
        out = out.sort_values(["source", "operation", "lag", "window", "feature"], na_position="last").reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "lag_diagnostics", "version": 1}
    return out


def marx_diagnostics(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize MARX-style moving-average lag features."""

    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X
    metadata = _coerce_feature_metadata(bundle.feature_metadata)
    rows: list[dict[str, Any]] = []
    for feature, meta in _marx_candidates(X, metadata).items():
        if feature not in X:
            continue
        parsed = _parse_marx_name(feature)
        source = _str_or_none(meta.get("source")) or (parsed[0] if parsed else None)
        window = _int_or_none(meta.get("window")) or (parsed[1] if parsed else _parse_window(feature))
        lag_value = _int_or_none(meta.get("lag")) or (parsed[2] if parsed else _parse_lag(feature))
        row = _lag_row(X, feature, meta)
        row.update(
            {
                "source": source,
                "window": window,
                "lag": lag_value,
                "marx_formula": f"mean({source}[t-1]...{source}[t-{window}])" if source and window else None,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "feature",
                "operation",
                "source",
                "lag",
                "window",
                "n_obs",
                "missing_rate",
                "first_valid",
                "last_valid",
                "marx_formula",
            ]
        )
    else:
        out = out.sort_values(["source", "window", "lag", "feature"], na_position="last").reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "marx_diagnostics", "version": 1}
    return out


def selection_stability(
    selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame,
    *,
    all_features: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return selection frequency by feature across folds, windows, or origins."""

    explicit_features = set(str(name) for name in all_features) if all_features is not None else set()
    if isinstance(selections, pd.DataFrame) and not {"feature", "selected"}.issubset(selections.columns):
        explicit_features.update(str(column) for column in selections.columns)
    origin_map = _coerce_selections(selections)
    if not origin_map:
        raise ValueError("selections must contain at least one origin")
    all_names = set(explicit_features)
    for selected in origin_map.values():
        all_names.update(selected)
    rows: list[dict[str, Any]] = []
    n_origins = len(origin_map)
    for feature in sorted(all_names):
        selected_origins = [origin for origin, values in origin_map.items() if feature in values]
        rows.append(
            {
                "feature": feature,
                "selected_count": int(len(selected_origins)),
                "selection_rate": _safe_ratio(len(selected_origins), n_origins),
                "n_origins": int(n_origins),
                "first_selected_origin": selected_origins[0] if selected_origins else None,
                "last_selected_origin": selected_origins[-1] if selected_origins else None,
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    out = out.sort_values(["selected_count", "selection_rate"], ascending=False)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "feature_selection_stability", "version": 1}
    return out


def diagnose_features(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    stages: Mapping[str, Any] | None = None,
    include_correlation: bool = False,
    correlation_method: CorrelationMethod = "pearson",
    correlation_threshold: float | None = 0.9,
    correlation_min_periods: int = 3,
    high_missing_threshold: float = 0.5,
    include_factors: bool = True,
    include_lags: bool = True,
    include_marx: bool = True,
    selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame | None = None,
) -> FeatureDiagnosticReport:
    """Run the standard feature-diagnostic suite on a feature matrix."""

    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    overview = feature_overview(
        bundle.X,
        feature_metadata=bundle.feature_metadata,
        high_missing_threshold=high_missing_threshold,
    )
    correlation = (
        feature_correlation(
            bundle.X,
            feature_metadata=bundle.feature_metadata,
            method=correlation_method,
            min_periods=correlation_min_periods,
            threshold=correlation_threshold,
        )
        if include_correlation
        else None
    )
    factors = factor_diagnostics(bundle.X, feature_metadata=bundle.feature_metadata) if include_factors else None
    lags = lag_diagnostics(bundle.X, feature_metadata=bundle.feature_metadata) if include_lags else None
    marx = marx_diagnostics(bundle.X, feature_metadata=bundle.feature_metadata) if include_marx else None
    stability = selection_stability(selections, all_features=bundle.X.columns) if selections is not None else None
    stage_comparison = compare_feature_stages(stages) if stages is not None else None
    stage = {
        "overview": {
            "n_observations": overview["n_observations"],
            "n_features": overview["n_features"],
            "missing_total": overview["missing_total"],
            "high_missing_feature_count": len(overview["high_missing_features"]),
            "zero_variance_feature_count": len(overview["zero_variance_features"]),
        },
        "options": {
            "include_correlation": bool(include_correlation),
            "correlation_method": correlation_method,
            "correlation_threshold": correlation_threshold,
            "correlation_min_periods": int(correlation_min_periods),
            "high_missing_threshold": float(high_missing_threshold),
            "include_factors": bool(include_factors),
            "include_lags": bool(include_lags),
            "include_marx": bool(include_marx),
            "include_selection_stability": selections is not None,
            "include_stage_comparison": stages is not None,
        },
        "tables": {
            "correlation_pairs": None if correlation is None else int(correlation.shape[0]),
            "factor_rows": None if factors is None else int(factors.shape[0]),
            "lag_rows": None if lags is None else int(lags.shape[0]),
            "marx_rows": None if marx is None else int(marx.shape[0]),
            "selection_rows": None if stability is None else int(stability.shape[0]),
            "stage_rows": None if stage_comparison is None else int(stage_comparison.shape[0]),
        },
    }
    updated_metadata = attach_metadata(bundle.metadata, "feature_diagnostic", stage)
    _attach_metadata(correlation, updated_metadata)
    _attach_metadata(factors, updated_metadata)
    _attach_metadata(lags, updated_metadata)
    _attach_metadata(marx, updated_metadata)
    _attach_metadata(stability, updated_metadata)
    _attach_metadata(stage_comparison, updated_metadata)
    return FeatureDiagnosticReport(
        overview=overview,
        correlation=correlation,
        factors=factors,
        lags=lags,
        marx=marx,
        selection_stability=stability,
        stage_comparison=stage_comparison,
        metadata=updated_metadata,
    )


def custom_feature_diagnostic(
    data: Any,
    func: Callable[..., Any],
    *,
    name: str | None = None,
    feature_metadata: pd.DataFrame | None = None,
    metadata: Mapping[str, Any] | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Run a user-supplied feature diagnostic and attach macroforecast metadata."""

    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    resolved_name = str(name or _callable_name(func) or "custom_feature_diagnostic")
    call_metadata = dict(bundle.metadata)
    call_metadata.update(dict(metadata or {}))
    result = func(
        bundle.X.copy(),
        feature_metadata=bundle.feature_metadata,
        metadata=call_metadata,
        **params,
    )
    table = _coerce_custom_table(result)
    stage = {
        "name": resolved_name,
        "callable": _callable_name(func),
        "params": dict(params),
        "input": {
            "n_observations": int(bundle.X.shape[0]),
            "n_features": int(bundle.X.shape[1]),
        },
        "output": {
            "rows": int(table.shape[0]),
            "columns": [str(column) for column in table.columns],
        },
        "user_metadata": dict(metadata or {}),
    }
    updated_metadata = attach_metadata(bundle.metadata, "custom_feature_diagnostic", stage)
    table.attrs["macroforecast_metadata_schema"] = {
        "kind": "custom_feature_diagnostic",
        "version": 1,
        "method": resolved_name,
        "columns": [str(column) for column in table.columns],
        "metadata": stage,
    }
    _attach_metadata(table, updated_metadata)
    return table


@dataclass(frozen=True)
class _FeatureBundle:
    X: pd.DataFrame
    metadata: dict[str, Any]
    feature_metadata: pd.DataFrame | None = None


def _coerce_feature_input(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
) -> _FeatureBundle:
    if isinstance(data, FeatureSet):
        X = data.X.copy()
        metadata = dict(data.metadata)
        meta = data.feature_metadata if feature_metadata is None else feature_metadata
    elif isinstance(data, DataBundle):
        X = data.panel.copy()
        metadata = dict(data.metadata)
        meta = feature_metadata
    elif isinstance(data, DataSpec):
        X = data.panel.copy()
        metadata = dict(data.metadata)
        meta = feature_metadata
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], pd.DataFrame):
        X = data[0].copy()
        metadata = dict(data[1])
        meta = feature_metadata
    elif isinstance(data, pd.DataFrame):
        X = data.copy()
        metadata = dict(data.attrs.get("macroforecast_metadata", {}) or {})
        meta = feature_metadata if feature_metadata is not None else data.attrs.get("macroforecast_feature_metadata")
    else:
        raise TypeError("data must be a FeatureSet, DataFrame, DataBundle, DataSpec, or (DataFrame, metadata) tuple")
    validate_panel(X)
    return _FeatureBundle(X=X, metadata=metadata, feature_metadata=_coerce_feature_metadata(meta))


def _coerce_feature_metadata(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if not isinstance(value, pd.DataFrame):
        raise TypeError("feature_metadata must be a pandas DataFrame")
    return value.copy()


def _metadata_lookup(metadata: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    meta = _coerce_feature_metadata(metadata)
    if meta.empty or "feature" not in meta:
        return {}
    return {
        str(row["feature"]): {str(key): value for key, value in row.items()}
        for row in meta.to_dict(orient="records")
    }


def _pair_metadata(metadata: Mapping[str, Mapping[str, Any]], left: str, right: str) -> dict[str, Any]:
    left_meta = metadata.get(left, {})
    right_meta = metadata.get(right, {})
    return {
        "operation_a": _str_or_none(left_meta.get("operation")),
        "operation_b": _str_or_none(right_meta.get("operation")),
        "source_a": _str_or_none(left_meta.get("source")),
        "source_b": _str_or_none(right_meta.get("source")),
    }


def _factor_candidates(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    operations: Sequence[str],
    prefixes: Sequence[str],
) -> dict[str, dict[str, Any]]:
    lookup = _metadata_lookup(metadata)
    selected: dict[str, dict[str, Any]] = {}
    operation_set = {str(value) for value in operations}
    for feature, meta in lookup.items():
        operation = _str_or_none(meta.get("operation"))
        component = _int_or_none(meta.get("component"))
        if operation in operation_set or component is not None:
            selected[feature] = dict(meta)
    for feature in X.columns:
        name = str(feature)
        if name not in selected and _matches_prefix_component(name, prefixes):
            selected[name] = {"feature": name}
    return selected


def _lag_candidates(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    operations: Sequence[str],
) -> dict[str, dict[str, Any]]:
    lookup = _metadata_lookup(metadata)
    selected: dict[str, dict[str, Any]] = {}
    operation_set = {str(value) for value in operations}
    for feature, meta in lookup.items():
        operation = _str_or_none(meta.get("operation"))
        lag_value = _int_or_none(meta.get("lag"))
        window_value = _int_or_none(meta.get("window"))
        if operation in operation_set or lag_value is not None or window_value is not None:
            selected[feature] = dict(meta)
    for feature in X.columns:
        name = str(feature)
        if name not in selected and (_parse_lag(name) is not None or _parse_window(name) is not None):
            selected[name] = {"feature": name}
    return selected


def _marx_candidates(X: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup = _metadata_lookup(metadata)
    selected: dict[str, dict[str, Any]] = {}
    for feature, meta in lookup.items():
        operation = _str_or_none(meta.get("operation"))
        if operation == "marx" or _parse_marx_name(feature) is not None:
            selected[feature] = dict(meta)
    for feature in X.columns:
        name = str(feature)
        if name not in selected and _parse_marx_name(name) is not None:
            selected[name] = {"feature": name, "operation": "marx"}
    return selected


def _lag_row(X: pd.DataFrame, feature: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    series = X[feature]
    observed = series.dropna()
    return {
        "feature": feature,
        "operation": _str_or_none(meta.get("operation")) or _inferred_lag_operation(feature),
        "source": _str_or_none(meta.get("source")) or _parse_source(feature),
        "lag": _int_or_none(meta.get("lag")) or _parse_lag(feature),
        "window": _int_or_none(meta.get("window")) or _parse_window(feature),
        "n_obs": int(series.notna().sum()),
        "missing_rate": float(series.isna().mean()),
        "first_valid": _index_value(observed.index.min()) if len(observed) else None,
        "last_valid": _index_value(observed.index.max()) if len(observed) else None,
    }


def _coerce_selections(
    selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame,
) -> dict[str, set[str]]:
    if isinstance(selections, pd.DataFrame):
        return _coerce_selection_frame(selections)
    if isinstance(selections, Mapping):
        return {
            str(origin): {str(feature) for feature in values}
            for origin, values in selections.items()
        }
    if isinstance(selections, Sequence) and not isinstance(selections, (str, bytes)):
        return {
            str(position): {str(feature) for feature in values}
            for position, values in enumerate(selections)
        }
    raise TypeError("selections must be a mapping, sequence of feature iterables, or DataFrame")


def _coerce_selection_frame(frame: pd.DataFrame) -> dict[str, set[str]]:
    if {"feature", "selected"}.issubset(frame.columns):
        origin_column = _first_existing(frame, ("origin", "window", "fold", "split"))
        if origin_column is None:
            selected = frame.loc[frame["selected"].map(bool), "feature"].astype(str)
            return {"0": set(selected)}
        out: dict[str, set[str]] = {}
        for origin, group in frame.groupby(origin_column, sort=False):
            out[str(origin)] = set(group.loc[group["selected"].map(bool), "feature"].astype(str))
        return out
    out = {}
    for idx, row in frame.iterrows():
        out[str(idx)] = {str(column) for column, value in row.items() if _truthy_selection(value)}
    return out


def _attach_metadata(frame: pd.DataFrame | None, metadata: Mapping[str, Any]) -> None:
    if frame is not None:
        frame.attrs["macroforecast_metadata"] = dict(metadata)


def _coerce_custom_table(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        name = "value" if value.name is None else str(value.name)
        return value.rename(name).to_frame()
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, (list, tuple)):
        return pd.DataFrame(value)
    raise TypeError("custom feature diagnostic must return a DataFrame, Series, mapping, or sequence")


def _callable_name(func: Any) -> str:
    return str(getattr(func, "__name__", func.__class__.__name__))


def _value_counts(series: pd.Series) -> dict[str, int]:
    values = series.dropna().map(str)
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _variance_share(frame: pd.DataFrame) -> pd.Series:
    shares = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _group, group in frame.groupby("group", dropna=False):
        total = group["variance"].fillna(0.0).sum()
        if total > 0:
            shares.loc[group.index] = group["variance"].fillna(0.0) / total
    return shares


def _validate_probability(value: float, name: str) -> None:
    if not 0 <= float(value) <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if text else None


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "n"}
    return bool(value)


def _truthy_selection(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, np.number)):
        return float(value) != 0.0
    return bool(value)


def _first_existing(frame: pd.DataFrame, columns: Sequence[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _matches_prefix_component(name: str, prefixes: Sequence[str]) -> bool:
    return any(re.match(rf"^{re.escape(prefix)}[_-]?\d+$", name) for prefix in prefixes)


def _parse_component(name: str) -> int | None:
    match = re.search(r"(?:component|pc|factor|maf)[_-]?(\d+)$", name)
    return int(match.group(1)) if match else None


def _parse_lag(name: str) -> int | None:
    match = re.search(r"(?:^|_)lag(\d+)(?:$|_)", name)
    return int(match.group(1)) if match else None


def _parse_window(name: str) -> int | None:
    match = re.search(r"(?:^|_)(?:ma|window|roll|rolling)(\d+)(?:$|_)", name)
    return int(match.group(1)) if match else None


def _parse_marx_name(name: str) -> tuple[str, int, int] | None:
    match = re.match(r"^(?P<source>.+)_ma(?P<window>\d+)_lag(?P<lag>\d+)$", name)
    if match is None:
        return None
    return match.group("source"), int(match.group("window")), int(match.group("lag"))


def _parse_source(name: str) -> str | None:
    for token in ("_lag", "_ma", "_rolling", "_roll", "_window"):
        if token in name:
            return name.split(token, 1)[0]
    return None


def _inferred_factor_operation(name: str, prefixes: Sequence[str]) -> str | None:
    for prefix in prefixes:
        if re.match(rf"^{re.escape(prefix)}[_-]?\d+$", name):
            return "factor" if prefix == "factor" else prefix
    return None


def _inferred_lag_operation(name: str) -> str | None:
    if _parse_marx_name(name) is not None:
        return "marx"
    if "_lag" in name:
        return "lag"
    if _parse_window(name) is not None:
        return "moving_average"
    return None


def _index_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


__all__ = [
    "FeatureDiagnosticReport",
    "compare_feature_stages",
    "custom_feature_diagnostic",
    "diagnose_features",
    "factor_diagnostics",
    "feature_correlation",
    "feature_overview",
    "lag_diagnostics",
    "marx_diagnostics",
    "selection_stability",
]
