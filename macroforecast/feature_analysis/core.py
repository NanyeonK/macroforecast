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
CorrelationOrder = Literal["original", "clustered"]
CorrelationScope = Literal["all", "within_block", "cross_block"]
AutocorrelationKind = Literal["acf", "pacf"]
SelectionSimilarityMetric = Literal["jaccard", "kuncheva"]

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
    correlation_matrix: pd.DataFrame | None = None
    target_correlation: pd.DataFrame | None = None
    factors: pd.DataFrame | None = None
    factor_variance: pd.DataFrame | None = None
    factor_loadings: pd.DataFrame | None = None
    factor_timeseries: pd.DataFrame | None = None
    lags: pd.DataFrame | None = None
    lag_autocorrelation: pd.DataFrame | None = None
    lag_correlation_decay: pd.DataFrame | None = None
    marx: pd.DataFrame | None = None
    marx_weight_decay: pd.DataFrame | None = None
    selection_stability: pd.DataFrame | None = None
    selection_similarity: pd.DataFrame | None = None
    stage_comparison: pd.DataFrame | None = None
    stage_distribution_shift: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "overview": self.overview,
            "metadata": dict(self.metadata),
        }
        if self.correlation is not None:
            out["correlation"] = self.correlation.to_dict(orient="records")
        if self.correlation_matrix is not None:
            out["correlation_matrix"] = self.correlation_matrix.to_dict(orient="index")
        if self.target_correlation is not None:
            out["target_correlation"] = self.target_correlation.to_dict(orient="records")
        if self.factors is not None:
            out["factors"] = self.factors.to_dict(orient="records")
        if self.factor_variance is not None:
            out["factor_variance"] = self.factor_variance.to_dict(orient="records")
        if self.factor_loadings is not None:
            out["factor_loadings"] = self.factor_loadings.to_dict(orient="records")
        if self.factor_timeseries is not None:
            out["factor_timeseries"] = self.factor_timeseries.to_dict(orient="records")
        if self.lags is not None:
            out["lags"] = self.lags.to_dict(orient="records")
        if self.lag_autocorrelation is not None:
            out["lag_autocorrelation"] = self.lag_autocorrelation.to_dict(orient="records")
        if self.lag_correlation_decay is not None:
            out["lag_correlation_decay"] = self.lag_correlation_decay.to_dict(orient="records")
        if self.marx is not None:
            out["marx"] = self.marx.to_dict(orient="records")
        if self.marx_weight_decay is not None:
            out["marx_weight_decay"] = self.marx_weight_decay.to_dict(orient="records")
        if self.selection_stability is not None:
            out["selection_stability"] = self.selection_stability.to_dict(orient="index")
        if self.selection_similarity is not None:
            out["selection_similarity"] = self.selection_similarity.to_dict(orient="records")
        if self.stage_comparison is not None:
            out["stage_comparison"] = self.stage_comparison.to_dict(orient="index")
        if self.stage_distribution_shift is not None:
            out["stage_distribution_shift"] = self.stage_distribution_shift.to_dict(orient="records")
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


def stage_distribution_shift(
    stages: Mapping[str, Any] | None = None,
    *,
    columns: Iterable[str] | None = None,
    min_obs: int = 3,
    **named_stages: Any,
) -> pd.DataFrame:
    """Return distribution-shift diagnostics between adjacent feature stages."""

    min_obs_value = int(min_obs)
    if min_obs_value < 1:
        raise ValueError("min_obs must be a positive integer")
    stage_map: dict[str, Any] = {}
    if stages is not None:
        stage_map.update(dict(stages))
    stage_map.update(named_stages)
    if len(stage_map) < 2:
        raise ValueError("at least two feature stages are required")
    selected_columns = None if columns is None else {str(column) for column in columns}
    bundles = [(str(name), _coerce_feature_input(value).X.select_dtypes("number")) for name, value in stage_map.items()]
    rows: list[dict[str, Any]] = []
    for (left_name, left), (right_name, right) in zip(bundles[:-1], bundles[1:], strict=False):
        common = [str(column) for column in left.columns if str(column) in set(map(str, right.columns))]
        if selected_columns is not None:
            common = [column for column in common if column in selected_columns]
        for column in common:
            left_values = pd.to_numeric(left[column], errors="coerce")
            right_values = pd.to_numeric(right[column], errors="coerce")
            left_obs = left_values.dropna().astype(float)
            right_obs = right_values.dropna().astype(float)
            enough = len(left_obs) >= min_obs_value and len(right_obs) >= min_obs_value
            rows.append(
                {
                    "stage_a": left_name,
                    "stage_b": right_name,
                    "feature": column,
                    "n_a": int(len(left_obs)),
                    "n_b": int(len(right_obs)),
                    "mean_a": _float_or_none(left_obs.mean()) if len(left_obs) else None,
                    "mean_b": _float_or_none(right_obs.mean()) if len(right_obs) else None,
                    "mean_shift": _float_or_none(right_obs.mean() - left_obs.mean()) if enough else None,
                    "sd_a": _float_or_none(left_obs.std(ddof=1)) if len(left_obs) > 1 else None,
                    "sd_b": _float_or_none(right_obs.std(ddof=1)) if len(right_obs) > 1 else None,
                    "sd_ratio": _safe_divide(right_obs.std(ddof=1), left_obs.std(ddof=1)) if enough else None,
                    "median_a": _float_or_none(left_obs.median()) if len(left_obs) else None,
                    "median_b": _float_or_none(right_obs.median()) if len(right_obs) else None,
                    "missing_rate_a": float(left_values.isna().mean()),
                    "missing_rate_b": float(right_values.isna().mean()),
                    "missing_rate_shift": float(right_values.isna().mean() - left_values.isna().mean()),
                    "ks_statistic": _ks_statistic(left_obs, right_obs) if enough else None,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "stage_a",
                "stage_b",
                "feature",
                "n_a",
                "n_b",
                "mean_a",
                "mean_b",
                "mean_shift",
                "sd_a",
                "sd_b",
                "sd_ratio",
                "median_a",
                "median_b",
                "missing_rate_a",
                "missing_rate_b",
                "missing_rate_shift",
                "ks_statistic",
            ]
        )
    else:
        out = out.sort_values(["stage_a", "stage_b", "ks_statistic", "feature"], ascending=[True, True, False, True]).reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "stage_distribution_shift", "version": 1}
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
    scope: CorrelationScope = "all",
    block_column: str = "block",
) -> pd.DataFrame:
    """Return long-form high-correlation feature pairs."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    if scope not in {"all", "within_block", "cross_block"}:
        raise ValueError("scope must be one of 'all', 'within_block', or 'cross_block'")
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
            block_left = _feature_block(metadata.get(str(left), {}), block_column=block_column)
            block_right = _feature_block(metadata.get(str(right), {}), block_column=block_column)
            if scope == "within_block" and (block_left is None or block_left != block_right):
                continue
            if scope == "cross_block" and (block_left is None or block_right is None or block_left == block_right):
                continue
            score = abs(float(value)) if absolute else float(value)
            if threshold is not None and score < float(threshold):
                continue
            row = {
                "feature_a": str(left),
                "feature_b": str(right),
                "correlation": float(value),
                "abs_correlation": abs(float(value)),
                "block_a": block_left,
                "block_b": block_right,
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
                "block_a",
                "block_b",
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
        "scope": scope,
    }
    return out


def feature_target_correlation(
    data: Any,
    target: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    method: CorrelationMethod = "pearson",
    min_periods: int = 3,
    absolute: bool = True,
    max_features: int | None = None,
) -> pd.DataFrame:
    """Return feature-to-target correlations."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    if int(min_periods) < 1:
        raise ValueError("min_periods must be a positive integer")
    if max_features is not None and int(max_features) < 1:
        raise ValueError("max_features must be a positive integer or None")
    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X.select_dtypes("number")
    y = _coerce_target_series(target, X)
    metadata = _metadata_lookup(bundle.feature_metadata)
    rows: list[dict[str, Any]] = []
    for feature in X.columns:
        if str(feature) == str(y.name):
            continue
        aligned = pd.concat([X[feature], y], axis=1, join="inner").dropna()
        value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method) if len(aligned) >= int(min_periods) else np.nan
        meta = metadata.get(str(feature), {})
        rows.append(
            {
                "feature": str(feature),
                "target": str(y.name or "target"),
                "correlation": _float_or_none(value),
                "abs_correlation": None if pd.isna(value) else abs(float(value)),
                "operation": _str_or_none(meta.get("operation")),
                "source": _str_or_none(meta.get("source")),
                "block": _feature_block(meta, block_column="block"),
                "n_obs": int(len(aligned)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["feature", "target", "correlation", "abs_correlation", "operation", "source", "block", "n_obs"])
    else:
        sort_column = "abs_correlation" if absolute else "correlation"
        out = out.sort_values(sort_column, ascending=False, na_position="last").reset_index(drop=True)
        if max_features is not None:
            out = out.head(int(max_features)).reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "feature_target_correlation",
        "version": 1,
        "method": method,
    }
    return out


def feature_correlation_matrix(
    data: Any,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int = 3,
    order: CorrelationOrder = "original",
    absolute_distance: bool = True,
) -> pd.DataFrame:
    """Return a full feature-correlation matrix, optionally cluster-ordered."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    if int(min_periods) < 1:
        raise ValueError("min_periods must be a positive integer")
    if order not in {"original", "clustered"}:
        raise ValueError("order must be 'original' or 'clustered'")

    bundle = _coerce_feature_input(data)
    X = bundle.X.select_dtypes("number")
    corr = X.corr(method=method, min_periods=int(min_periods))
    if order == "clustered" and corr.shape[0] > 2:
        ordered = _cluster_correlation_order(corr, absolute_distance=absolute_distance)
        corr = corr.loc[ordered, ordered]
    corr.attrs["macroforecast_metadata_schema"] = {
        "kind": "feature_correlation_matrix",
        "version": 1,
        "method": method,
        "min_periods": int(min_periods),
        "order": order,
        "absolute_distance": bool(absolute_distance),
    }
    return corr


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


def factor_variance(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    operations: Sequence[str] = FACTOR_OPERATIONS,
    prefixes: Sequence[str] = ("pc", "factor", "maf"),
) -> pd.DataFrame:
    """Return scree-style variance and cumulative variance share by factor group."""

    diagnostics = factor_diagnostics(
        data,
        feature_metadata=feature_metadata,
        operations=operations,
        prefixes=prefixes,
    )
    if diagnostics.empty:
        out = pd.DataFrame(
            columns=[
                "group",
                "feature",
                "component",
                "variance",
                "variance_share",
                "cumulative_variance_share",
            ]
        )
    else:
        work = diagnostics.copy()
        work = work.sort_values(["group", "component", "feature"], na_position="last")
        work["cumulative_variance_share"] = (
            work.groupby("group", dropna=False)["variance_share"].cumsum()
        )
        out = work.loc[
            :,
            [
                "group",
                "feature",
                "component",
                "variance",
                "variance_share",
                "cumulative_variance_share",
            ],
        ].reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "factor_variance", "version": 1}
    return out


def factor_loadings(
    data: Any,
    *,
    source_data: Any | None = None,
    feature_metadata: pd.DataFrame | None = None,
    operations: Sequence[str] = FACTOR_OPERATIONS,
    prefixes: Sequence[str] = ("pc", "factor", "maf"),
    method: CorrelationMethod = "pearson",
    max_sources: int | None = None,
) -> pd.DataFrame:
    """Approximate factor loadings as source-column correlations with factors.

    If `source_data` is supplied, its numeric columns are treated as original
    source variables. Otherwise, non-factor columns in `data` are used. This
    keeps the callable pandas-native without requiring a fitted PCA object.
    """

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    if max_sources is not None and int(max_sources) < 1:
        raise ValueError("max_sources must be a positive integer or None")

    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X.select_dtypes("number")
    factor_rows = factor_diagnostics(
        bundle.X,
        feature_metadata=bundle.feature_metadata,
        operations=operations,
        prefixes=prefixes,
    )
    factor_names = [name for name in factor_rows.get("feature", pd.Series(dtype=object)).astype(str) if name in X]
    if source_data is None:
        sources = X.loc[:, [column for column in X.columns if str(column) not in set(factor_names)]]
    else:
        sources = _coerce_feature_input(source_data).X.select_dtypes("number")
    rows: list[dict[str, Any]] = []
    for factor in factor_names:
        factor_series = X[factor]
        loadings: list[dict[str, Any]] = []
        for source in sources.columns:
            aligned = pd.concat([sources[source], factor_series], axis=1, join="inner").dropna()
            value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method) if len(aligned) >= 3 else np.nan
            loadings.append(
                {
                    "factor": str(factor),
                    "source": str(source),
                    "loading": _float_or_none(value),
                    "abs_loading": None if pd.isna(value) else abs(float(value)),
                }
            )
        loadings = sorted(
            loadings,
            key=lambda item: float("inf")
            if item["abs_loading"] is None
            else -float(item["abs_loading"]),
        )
        if max_sources is not None:
            loadings = loadings[: int(max_sources)]
        rows.extend(loadings)
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["factor", "source", "loading", "abs_loading"])
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "factor_loadings",
        "version": 1,
        "method": method,
        "max_sources": max_sources,
    }
    return out


def factor_timeseries(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    operations: Sequence[str] = FACTOR_OPERATIONS,
    prefixes: Sequence[str] = ("pc", "factor", "maf"),
    max_factors: int | None = None,
) -> pd.DataFrame:
    """Return factor/component columns in long time-series form."""

    if max_factors is not None and int(max_factors) < 1:
        raise ValueError("max_factors must be a positive integer or None")
    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X.select_dtypes("number")
    factor_rows = factor_diagnostics(
        bundle.X,
        feature_metadata=bundle.feature_metadata,
        operations=operations,
        prefixes=prefixes,
    )
    factor_names = [str(name) for name in factor_rows.get("feature", pd.Series(dtype=object)) if str(name) in X]
    if max_factors is not None:
        factor_names = factor_names[: int(max_factors)]
    lookup = {
        str(row["feature"]): row
        for row in factor_rows.to_dict(orient="records")
        if row.get("feature") is not None
    }
    rows: list[dict[str, Any]] = []
    for factor in factor_names:
        meta = lookup.get(factor, {})
        for date_value, value in X[factor].items():
            rows.append(
                {
                    "date": _index_value(date_value),
                    "factor": factor,
                    "value": _float_or_none(value),
                    "group": meta.get("group"),
                    "operation": meta.get("operation"),
                    "component": meta.get("component"),
                    "source": meta.get("source"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["date", "factor", "value", "group", "operation", "component", "source"])
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "factor_timeseries",
        "version": 1,
        "max_factors": max_factors,
    }
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


def lag_autocorrelation(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    columns: Iterable[str] | None = None,
    max_lag: int = 12,
    kind: AutocorrelationKind = "acf",
    operations: Sequence[str] = LAG_OPERATIONS,
) -> pd.DataFrame:
    """Return ACF or PACF values for lag/window feature columns."""

    max_lag_value = int(max_lag)
    if max_lag_value < 0:
        raise ValueError("max_lag must be non-negative")
    if kind not in {"acf", "pacf"}:
        raise ValueError("kind must be 'acf' or 'pacf'")
    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X.select_dtypes("number")
    if columns is None:
        lag_rows = lag_diagnostics(X, feature_metadata=bundle.feature_metadata, operations=operations)
        selected = [str(feature) for feature in lag_rows.get("feature", pd.Series(dtype=object)) if str(feature) in X]
    else:
        selected = [str(column) for column in columns]
    missing = [column for column in selected if column not in X]
    if missing:
        raise ValueError(f"columns are not in the feature matrix: {missing}")
    rows: list[dict[str, Any]] = []
    for feature in selected:
        series = X[feature].dropna().astype(float)
        for lag_value in range(max_lag_value + 1):
            value = _acf_value(series, lag_value) if kind == "acf" else _pacf_value(series, lag_value)
            rows.append(
                {
                    "feature": feature,
                    "lag": int(lag_value),
                    kind: _float_or_none(value),
                    "n_obs": int(series.shape[0]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["feature", "lag", kind, "n_obs"])
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "lag_autocorrelation",
        "version": 1,
        "autocorrelation_kind": kind,
        "max_lag": max_lag_value,
    }
    return out


def lag_correlation_decay(
    data: Any,
    *,
    target: str | pd.Series | None = None,
    feature_metadata: pd.DataFrame | None = None,
    operations: Sequence[str] = LAG_OPERATIONS,
    method: CorrelationMethod = "pearson",
) -> pd.DataFrame:
    """Return correlation decay across lag/window features.

    If `target` is supplied, lag features are correlated with that target. If
    not, each lag feature is correlated with the same source's lag-0/current
    column when one is available.
    """

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    bundle = _coerce_feature_input(data, feature_metadata=feature_metadata)
    X = bundle.X.select_dtypes("number")
    lags = lag_diagnostics(X, feature_metadata=bundle.feature_metadata, operations=operations)
    if target is None:
        target_map = _source_target_map(X, lags)
    elif isinstance(target, pd.Series):
        target_map = {str(source): target for source in lags["source"].dropna().astype(str).unique()}
    else:
        target_name = str(target)
        if target_name not in X:
            raise ValueError(f"target column {target_name!r} is not in the feature matrix")
        target_map = {str(source): X[target_name] for source in lags["source"].dropna().astype(str).unique()}
    rows: list[dict[str, Any]] = []
    for _, row in lags.iterrows():
        feature = str(row["feature"])
        source = _str_or_none(row.get("source"))
        comparator = target_map.get(str(source)) if source is not None else None
        if feature not in X or comparator is None:
            continue
        aligned = pd.concat([X[feature], comparator], axis=1, join="inner").dropna()
        value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1], method=method) if len(aligned) >= 3 else np.nan
        rows.append(
            {
                "feature": feature,
                "source": source,
                "lag": row.get("lag"),
                "window": row.get("window"),
                "correlation": _float_or_none(value),
                "abs_correlation": None if pd.isna(value) else abs(float(value)),
                "method": method,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["feature", "source", "lag", "window", "correlation", "abs_correlation", "method"])
    else:
        out = out.sort_values(["source", "lag", "window", "feature"], na_position="last").reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "lag_correlation_decay", "version": 1}
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


def marx_weight_decay(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return implied equal lag weights for MARX moving-average features."""

    marx = marx_diagnostics(data, feature_metadata=feature_metadata)
    rows: list[dict[str, Any]] = []
    if not marx.empty:
        for item in marx.to_dict(orient="records"):
            window = _int_or_none(item.get("window"))
            if window is None or window < 1:
                continue
            for lag_position in range(1, window + 1):
                rows.append(
                    {
                        "feature": item.get("feature"),
                        "source": item.get("source"),
                        "window": int(window),
                        "lag": int(lag_position),
                        "weight": 1.0 / float(window),
                        "cumulative_weight": float(lag_position) / float(window),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["feature", "source", "window", "lag", "weight", "cumulative_weight"])
    out.attrs["macroforecast_metadata_schema"] = {"kind": "marx_weight_decay", "version": 1}
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


def selection_similarity(
    selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame,
    *,
    metric: SelectionSimilarityMetric = "jaccard",
    all_features: Iterable[str] | None = None,
    n_features: int | None = None,
) -> pd.DataFrame:
    """Return pairwise feature-selection stability across origins/folds/windows."""

    if metric not in {"jaccard", "kuncheva"}:
        raise ValueError("metric must be 'jaccard' or 'kuncheva'")
    origin_map = _coerce_selections(selections)
    if not origin_map:
        raise ValueError("selections must contain at least one origin")
    universe = {str(name) for values in origin_map.values() for name in values}
    if all_features is not None:
        universe.update(str(name) for name in all_features)
    if n_features is not None:
        n_total = int(n_features)
        if n_total < len(universe):
            raise ValueError("n_features must be at least the number of observed unique features")
    else:
        n_total = len(universe)
    origins = list(origin_map)
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(origins):
        left_set = origin_map[left]
        for right in origins[i + 1 :]:
            right_set = origin_map[right]
            overlap = len(left_set & right_set)
            union = len(left_set | right_set)
            score = (
                _safe_ratio(overlap, union)
                if metric == "jaccard"
                else _kuncheva_similarity(overlap, len(left_set), len(right_set), n_total)
            )
            rows.append(
                {
                    "origin_a": left,
                    "origin_b": right,
                    "metric": metric,
                    "score": _float_or_none(score),
                    "overlap": int(overlap),
                    "selected_a": int(len(left_set)),
                    "selected_b": int(len(right_set)),
                    "union": int(union),
                    "n_features": int(n_total),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["origin_a", "origin_b", "metric", "score", "overlap", "selected_a", "selected_b", "union", "n_features"])
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "feature_selection_similarity",
        "version": 1,
        "metric": metric,
    }
    return out


def diagnose_features(
    data: Any,
    *,
    feature_metadata: pd.DataFrame | None = None,
    stages: Mapping[str, Any] | None = None,
    include_correlation: bool = False,
    include_correlation_matrix: bool = False,
    correlation_method: CorrelationMethod = "pearson",
    correlation_threshold: float | None = 0.9,
    correlation_min_periods: int = 3,
    correlation_order: CorrelationOrder = "original",
    correlation_scope: CorrelationScope = "all",
    target: Any | None = None,
    include_target_correlation: bool = False,
    high_missing_threshold: float = 0.5,
    include_factors: bool = True,
    include_factor_variance: bool = True,
    include_factor_loadings: bool = False,
    include_factor_timeseries: bool = False,
    factor_source_data: Any | None = None,
    include_lags: bool = True,
    include_lag_autocorrelation: bool = False,
    include_lag_correlation_decay: bool = False,
    include_marx: bool = True,
    include_marx_weight_decay: bool = True,
    include_stage_distribution_shift: bool = True,
    selections: Mapping[Any, Iterable[str]] | Sequence[Iterable[str]] | pd.DataFrame | None = None,
    selection_similarity_metric: SelectionSimilarityMetric | None = None,
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
            scope=correlation_scope,
        )
        if include_correlation
        else None
    )
    corr_matrix = (
        feature_correlation_matrix(
            bundle.X,
            method=correlation_method,
            min_periods=correlation_min_periods,
            order=correlation_order,
        )
        if include_correlation_matrix
        else None
    )
    target_corr = (
        feature_target_correlation(
            bundle.X,
            target,
            feature_metadata=bundle.feature_metadata,
            method=correlation_method,
            min_periods=correlation_min_periods,
        )
        if include_target_correlation and target is not None
        else None
    )
    factors = factor_diagnostics(bundle.X, feature_metadata=bundle.feature_metadata) if include_factors else None
    factor_var = factor_variance(bundle.X, feature_metadata=bundle.feature_metadata) if include_factor_variance else None
    factor_load = (
        factor_loadings(
            bundle.X,
            source_data=factor_source_data,
            feature_metadata=bundle.feature_metadata,
            method=correlation_method,
        )
        if include_factor_loadings
        else None
    )
    factor_ts = (
        factor_timeseries(bundle.X, feature_metadata=bundle.feature_metadata)
        if include_factor_timeseries
        else None
    )
    lags = lag_diagnostics(bundle.X, feature_metadata=bundle.feature_metadata) if include_lags else None
    lag_acf = (
        lag_autocorrelation(bundle.X, feature_metadata=bundle.feature_metadata)
        if include_lag_autocorrelation
        else None
    )
    lag_decay = (
        lag_correlation_decay(bundle.X, feature_metadata=bundle.feature_metadata, method=correlation_method)
        if include_lag_correlation_decay
        else None
    )
    marx = marx_diagnostics(bundle.X, feature_metadata=bundle.feature_metadata) if include_marx else None
    marx_decay = (
        marx_weight_decay(bundle.X, feature_metadata=bundle.feature_metadata)
        if include_marx and include_marx_weight_decay
        else None
    )
    stability = selection_stability(selections, all_features=bundle.X.columns) if selections is not None else None
    similarity = (
        selection_similarity(
            selections,
            metric=selection_similarity_metric,
            all_features=bundle.X.columns,
        )
        if selections is not None and selection_similarity_metric is not None
        else None
    )
    stage_comparison = compare_feature_stages(stages) if stages is not None else None
    stage_shift = (
        stage_distribution_shift(stages)
        if stages is not None and include_stage_distribution_shift
        else None
    )
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
            "include_correlation_matrix": bool(include_correlation_matrix),
            "correlation_method": correlation_method,
            "correlation_threshold": correlation_threshold,
            "correlation_min_periods": int(correlation_min_periods),
            "correlation_order": correlation_order,
            "correlation_scope": correlation_scope,
            "include_target_correlation": bool(include_target_correlation),
            "high_missing_threshold": float(high_missing_threshold),
            "include_factors": bool(include_factors),
            "include_factor_variance": bool(include_factor_variance),
            "include_factor_loadings": bool(include_factor_loadings),
            "include_factor_timeseries": bool(include_factor_timeseries),
            "include_lags": bool(include_lags),
            "include_lag_autocorrelation": bool(include_lag_autocorrelation),
            "include_lag_correlation_decay": bool(include_lag_correlation_decay),
            "include_marx": bool(include_marx),
            "include_marx_weight_decay": bool(include_marx_weight_decay),
            "include_selection_stability": selections is not None,
            "selection_similarity_metric": selection_similarity_metric,
            "include_stage_comparison": stages is not None,
            "include_stage_distribution_shift": bool(include_stage_distribution_shift),
        },
        "tables": {
            "correlation_pairs": None if correlation is None else int(correlation.shape[0]),
            "correlation_matrix_shape": None if corr_matrix is None else list(corr_matrix.shape),
            "target_correlation_rows": None if target_corr is None else int(target_corr.shape[0]),
            "factor_rows": None if factors is None else int(factors.shape[0]),
            "factor_variance_rows": None if factor_var is None else int(factor_var.shape[0]),
            "factor_loading_rows": None if factor_load is None else int(factor_load.shape[0]),
            "factor_timeseries_rows": None if factor_ts is None else int(factor_ts.shape[0]),
            "lag_rows": None if lags is None else int(lags.shape[0]),
            "lag_autocorrelation_rows": None if lag_acf is None else int(lag_acf.shape[0]),
            "lag_correlation_decay_rows": None if lag_decay is None else int(lag_decay.shape[0]),
            "marx_rows": None if marx is None else int(marx.shape[0]),
            "marx_weight_decay_rows": None if marx_decay is None else int(marx_decay.shape[0]),
            "selection_rows": None if stability is None else int(stability.shape[0]),
            "selection_similarity_rows": None if similarity is None else int(similarity.shape[0]),
            "stage_rows": None if stage_comparison is None else int(stage_comparison.shape[0]),
            "stage_distribution_shift_rows": None if stage_shift is None else int(stage_shift.shape[0]),
        },
    }
    updated_metadata = attach_metadata(bundle.metadata, "feature_analysis", stage)
    _attach_metadata(correlation, updated_metadata)
    _attach_metadata(corr_matrix, updated_metadata)
    _attach_metadata(target_corr, updated_metadata)
    _attach_metadata(factors, updated_metadata)
    _attach_metadata(factor_var, updated_metadata)
    _attach_metadata(factor_load, updated_metadata)
    _attach_metadata(factor_ts, updated_metadata)
    _attach_metadata(lags, updated_metadata)
    _attach_metadata(lag_acf, updated_metadata)
    _attach_metadata(lag_decay, updated_metadata)
    _attach_metadata(marx, updated_metadata)
    _attach_metadata(marx_decay, updated_metadata)
    _attach_metadata(stability, updated_metadata)
    _attach_metadata(similarity, updated_metadata)
    _attach_metadata(stage_comparison, updated_metadata)
    _attach_metadata(stage_shift, updated_metadata)
    return FeatureDiagnosticReport(
        overview=overview,
        correlation=correlation,
        correlation_matrix=corr_matrix,
        target_correlation=target_corr,
        factors=factors,
        factor_variance=factor_var,
        factor_loadings=factor_load,
        factor_timeseries=factor_ts,
        lags=lags,
        lag_autocorrelation=lag_acf,
        lag_correlation_decay=lag_decay,
        marx=marx,
        marx_weight_decay=marx_decay,
        selection_stability=stability,
        selection_similarity=similarity,
        stage_comparison=stage_comparison,
        stage_distribution_shift=stage_shift,
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


def _feature_block(meta: Mapping[str, Any], *, block_column: str) -> str | None:
    block = _str_or_none(meta.get(block_column))
    if block is not None:
        return block
    operation = _str_or_none(meta.get("operation"))
    if operation is not None:
        return operation
    return _str_or_none(meta.get("source"))


def _coerce_target_series(target: Any, X: pd.DataFrame) -> pd.Series:
    if isinstance(target, str):
        if target not in X:
            raise ValueError(f"target column {target!r} is not in the feature matrix")
        return X[target].rename(target)
    if isinstance(target, pd.Series):
        return pd.to_numeric(target.copy(), errors="coerce").rename(target.name or "target")
    if isinstance(target, pd.DataFrame):
        numeric = target.select_dtypes("number")
        if numeric.shape[1] != 1:
            raise ValueError("target DataFrame must have exactly one numeric column")
        column = numeric.columns[0]
        return numeric[column].rename(str(column))
    values = np.asarray(target, dtype=float).reshape(-1)
    if len(values) != len(X.index):
        raise ValueError("array-like target must have the same length as the feature matrix")
    return pd.Series(values, index=X.index, name="target")


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


def _cluster_correlation_order(corr: pd.DataFrame, *, absolute_distance: bool) -> list[Any]:
    remaining = list(corr.columns)
    if not remaining:
        return remaining
    filled = corr.fillna(0.0)
    score = filled.abs().sum(axis=1) if absolute_distance else filled.sum(axis=1)
    current = score.sort_values(ascending=False).index[0]
    ordered = [current]
    remaining.remove(current)
    while remaining:
        row = filled.loc[current, remaining]
        if absolute_distance:
            row = row.abs()
        current = row.sort_values(ascending=False).index[0]
        ordered.append(current)
        remaining.remove(current)
    return ordered


def _source_target_map(X: pd.DataFrame, lag_rows: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if lag_rows.empty:
        return out
    for source in lag_rows["source"].dropna().astype(str).unique():
        candidates = [
            source,
            f"{source}_lag0",
            f"{source}_ma1_lag0",
            f"{source}_rolling1",
        ]
        for candidate in candidates:
            if candidate in X:
                out[source] = X[candidate]
                break
    return out


def _acf_value(series: pd.Series, lag: int) -> float | None:
    values = series.dropna().astype(float)
    if lag == 0:
        return 1.0 if len(values) else None
    if len(values) <= lag:
        return None
    left = values.iloc[lag:].to_numpy(dtype=float)
    right = values.iloc[:-lag].to_numpy(dtype=float)
    if np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _pacf_value(series: pd.Series, lag: int) -> float | None:
    values = series.dropna().astype(float).to_numpy(dtype=float)
    if lag == 0:
        return 1.0 if values.size else None
    if values.size <= lag + 1:
        return None
    y = values[lag:]
    X = np.column_stack([values[lag - k : values.size - k] for k in range(1, lag + 1)])
    X = np.column_stack([np.ones(X.shape[0]), X])
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return float(beta[-1])


def _kuncheva_similarity(overlap: int, left_size: int, right_size: int, n_features: int) -> float | None:
    if n_features <= 0:
        return None
    expected = left_size * right_size / n_features
    max_overlap = min(left_size, right_size)
    denominator = max_overlap - expected
    if denominator == 0:
        return 1.0 if overlap == max_overlap else None
    return float((overlap - expected) / denominator)


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


def _safe_divide(numerator: Any, denominator: Any) -> float | None:
    top = _float_or_none(numerator)
    bottom = _float_or_none(denominator)
    if top is None or bottom is None or bottom == 0.0:
        return None
    return float(top / bottom)


def _ks_statistic(left: pd.Series, right: pd.Series) -> float | None:
    left_values = np.sort(left.dropna().to_numpy(dtype=float))
    right_values = np.sort(right.dropna().to_numpy(dtype=float))
    if left_values.size == 0 or right_values.size == 0:
        return None
    grid = np.sort(np.unique(np.concatenate([left_values, right_values])))
    left_cdf = np.searchsorted(left_values, grid, side="right") / left_values.size
    right_cdf = np.searchsorted(right_values, grid, side="right") / right_values.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


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
    "factor_loadings",
    "factor_timeseries",
    "factor_variance",
    "feature_correlation",
    "feature_correlation_matrix",
    "feature_overview",
    "feature_target_correlation",
    "lag_autocorrelation",
    "lag_correlation_decay",
    "lag_diagnostics",
    "marx_diagnostics",
    "marx_weight_decay",
    "selection_similarity",
    "selection_stability",
    "stage_distribution_shift",
]
