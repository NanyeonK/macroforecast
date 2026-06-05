from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import (
    DataBundle,
    DataSpec,
    as_panel,
    attach_metadata,
    panel_info,
    validate_panel,
)


SummaryMetric = Literal["mean", "sd", "min", "max", "skew", "kurtosis", "n_obs", "n_missing"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
OutlierMethod = Literal["iqr", "zscore", "multi", "both"]
StationarityTest = Literal["adf", "pp", "kpss", "multi", "none"]
StationarityScope = Literal["all", "target_and_predictors", "target_only", "predictors_only"]

DEFAULT_SUMMARY_METRICS: tuple[SummaryMetric, ...] = (
    "mean",
    "sd",
    "min",
    "max",
    "n_obs",
    "n_missing",
)


@dataclass(frozen=True)
class DataSummaryReport:
    """Container returned by :func:`summarize_data`."""

    overview: dict[str, Any]
    coverage: pd.DataFrame
    univariate: pd.DataFrame
    missing: pd.DataFrame
    correlation: pd.DataFrame | None = None
    outliers: pd.DataFrame | None = None
    stationarity: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "overview": self.overview,
            "coverage": self.coverage.to_dict(orient="index"),
            "univariate": self.univariate.to_dict(orient="index"),
            "missing": self.missing.to_dict(orient="index"),
            "metadata": dict(self.metadata),
        }
        if self.correlation is not None:
            out["correlation"] = self.correlation.to_dict()
        if self.outliers is not None:
            out["outliers"] = self.outliers.to_dict(orient="index")
        if self.stationarity is not None:
            out["stationarity"] = self.stationarity
        return out


def panel_overview(data: Any) -> dict[str, Any]:
    """Return panel-level shape, date range, frequency, and missingness."""

    panel, metadata = _coerce_panel(data)
    info = panel_info(DataBundle(panel, metadata))
    info["metadata_keys"] = sorted(str(key) for key in metadata)
    return info


def panel_snapshot(data: Any) -> dict[str, Any]:
    """Return a compact single-panel snapshot for reports and provenance."""

    panel, metadata = _coerce_panel(data)
    return _compact_panel_info(DataBundle(panel, metadata))


def sample_coverage(data: Any) -> pd.DataFrame:
    """Return per-series sample start, end, observation count, and missingness."""

    panel, _metadata = _coerce_panel(data)
    rows: list[dict[str, Any]] = []
    n_rows = int(panel.shape[0])
    for column in panel.columns:
        series = panel[column]
        observed = series.dropna()
        n_obs = int(observed.shape[0])
        n_missing = int(series.isna().sum())
        rows.append(
            {
                "column": str(column),
                "first_valid": _date_string(observed.index.min()) if n_obs else None,
                "last_valid": _date_string(observed.index.max()) if n_obs else None,
                "n_obs": n_obs,
                "n_missing": n_missing,
                "missing_rate": _safe_ratio(n_missing, n_rows),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def observation_counts(data: Any) -> pd.Series:
    """Return per-series non-missing observation counts."""

    coverage = sample_coverage(data)
    if coverage.empty:
        return pd.Series(dtype="int64", name="n_obs")
    return coverage["n_obs"].rename("n_obs")


def missing_rates(data: Any) -> pd.Series:
    """Return per-series missing rates."""

    coverage = sample_coverage(data)
    if coverage.empty:
        return pd.Series(dtype="float64", name="missing_rate")
    return coverage["missing_rate"].rename("missing_rate")


def univariate_summary(
    data: Any,
    *,
    metrics: Sequence[SummaryMetric] | None = None,
) -> pd.DataFrame:
    """Return per-series descriptive statistics for numeric panel columns."""

    panel, _metadata = _coerce_panel(data)
    selected = tuple(metrics or DEFAULT_SUMMARY_METRICS)
    _validate_summary_metrics(selected)
    rows: list[dict[str, Any]] = []
    for column in panel.select_dtypes("number").columns:
        series = panel[column].dropna()
        row: dict[str, Any] = {"column": str(column)}
        if "mean" in selected:
            row["mean"] = _float_or_none(series.mean())
        if "sd" in selected:
            row["sd"] = _float_or_none(series.std())
        if "min" in selected:
            row["min"] = _float_or_none(series.min())
        if "max" in selected:
            row["max"] = _float_or_none(series.max())
        if "skew" in selected:
            row["skew"] = _float_or_none(series.skew())
        if "kurtosis" in selected:
            row["kurtosis"] = _float_or_none(series.kurtosis())
        if "n_obs" in selected:
            row["n_obs"] = int(series.shape[0])
        if "n_missing" in selected:
            row["n_missing"] = int(panel[column].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def missing_summary(data: Any) -> pd.DataFrame:
    """Return per-series missing-count, missing-rate, and longest-gap summary."""

    panel, _metadata = _coerce_panel(data)
    rows: list[dict[str, Any]] = []
    n_rows = int(panel.shape[0])
    for column in panel.columns:
        missing = panel[column].isna()
        n_missing = int(missing.sum())
        rows.append(
            {
                "column": str(column),
                "n_missing": n_missing,
                "missing_rate": _safe_ratio(n_missing, n_rows),
                "longest_missing_run": _longest_true_run(missing),
            }
        )
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def correlation_matrix(
    data: Any,
    *,
    method: CorrelationMethod = "pearson",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Return a numeric correlation matrix for one panel."""

    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be one of 'pearson', 'spearman', or 'kendall'")
    if int(min_periods) < 1:
        raise ValueError("min_periods must be a positive integer")
    panel, _metadata = _coerce_panel(data)
    return panel.select_dtypes("number").corr(method=method, min_periods=int(min_periods))


def outlier_summary(
    data: Any,
    *,
    method: OutlierMethod = "iqr",
    iqr_threshold: float = 10.0,
    zscore_threshold: float = 3.0,
) -> pd.DataFrame:
    """Return per-series outlier counts and rates for one panel."""

    if method not in {"iqr", "zscore", "multi", "both"}:
        raise ValueError("method must be one of 'iqr', 'zscore', 'multi', or 'both'")
    _validate_positive(iqr_threshold, "iqr_threshold")
    _validate_positive(zscore_threshold, "zscore_threshold")
    panel, _metadata = _coerce_panel(data)
    numeric = panel.select_dtypes("number")
    rows: list[dict[str, Any]] = []
    include_iqr = method in {"iqr", "multi", "both"}
    include_zscore = method in {"zscore", "multi", "both"}
    if include_iqr:
        median = numeric.median()
        iqr = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0, pd.NA)
        iqr_mask = (numeric - median).abs() > float(iqr_threshold) * iqr
    else:
        iqr_mask = pd.DataFrame(False, index=numeric.index, columns=numeric.columns)
    if include_zscore:
        # Match preprocessing.zscore_outlier_clean: population standard
        # deviation with ddof=0. data_analysis should describe the same outlier
        # rule that preprocessing would apply, not a sample-std variant.
        sd = numeric.std(ddof=0).replace(0, pd.NA)
        zscore_mask = ((numeric - numeric.mean()).abs() / sd) > float(zscore_threshold)
    else:
        zscore_mask = pd.DataFrame(False, index=numeric.index, columns=numeric.columns)
    for column in numeric.columns:
        observed_n = int(numeric[column].notna().sum())
        row: dict[str, Any] = {"column": str(column), "n_obs": observed_n}
        if include_iqr:
            count = int(iqr_mask[column].fillna(False).sum())
            row["iqr_outlier_count"] = count
            row["iqr_outlier_rate"] = _safe_ratio(count, observed_n)
        if include_zscore:
            count = int(zscore_mask[column].fillna(False).sum())
            row["zscore_outlier_count"] = count
            row["zscore_outlier_rate"] = _safe_ratio(count, observed_n)
        rows.append(row)
    return pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def stationarity_tests(
    data: Any,
    *,
    test: StationarityTest = "multi",
    scope: StationarityScope = "all",
    target: str | None = None,
    targets: Sequence[str] | None = None,
    alpha: float = 0.05,
    adf_regression: str = "c",
) -> dict[str, Any]:
    """Run ADF, Phillips-Perron, KPSS, or all three on one panel.

    ``adf_regression`` is the ADF deterministic specification passed to
    statsmodels ``adfuller`` ('n', 'c', 'ct', 'ctt'); the default 'c' (constant
    only) follows statsmodels. Note ``tseries::adf.test`` instead defaults to 'ct'
    (constant + linear trend) with a fixed lag, so pass ``adf_regression='ct'`` to
    reproduce that reference.
    """

    if test not in {"adf", "pp", "kpss", "multi", "none"}:
        raise ValueError("test must be one of 'adf', 'pp', 'kpss', 'multi', or 'none'")
    if adf_regression not in {"n", "c", "ct", "ctt"}:
        raise ValueError("adf_regression must be one of 'n', 'c', 'ct', 'ctt'")
    if scope not in {"all", "target_and_predictors", "target_only", "predictors_only"}:
        raise ValueError("scope must be one of 'all', 'target_and_predictors', 'target_only', or 'predictors_only'")
    _validate_alpha(alpha)
    panel, _metadata = _coerce_panel(data)
    columns = _scope_columns(panel, data, scope=scope, target=target, targets=targets)
    if test == "none":
        return {
            "test": "none",
            "scope": scope,
            "alpha": float(alpha),
            "n_series": 0,
            "by_series": {},
        }
    selected_tests = ("adf", "pp", "kpss") if test == "multi" else (test,)
    results: dict[str, dict[str, Any]] = {}
    for column in columns:
        series = pd.to_numeric(panel[column], errors="coerce").dropna()
        name = str(column)
        if series.size < 12 or series.std(ddof=0) == 0:
            results[name] = {"status": "insufficient_data", "n_obs": int(series.size)}
            continue
        column_result: dict[str, Any] = {"n_obs": int(series.size)}
        for selected in selected_tests:
            try:
                column_result[selected] = _run_stationarity(
                    selected, series, float(alpha), adf_regression=adf_regression
                )
            except Exception as exc:  # pragma: no cover - defensive
                column_result[selected] = {"status": "error", "error": str(exc)}
        results[name] = column_result
    return {
        "test": test,
        "scope": scope,
        "alpha": float(alpha),
        "n_series": len(columns),
        "by_series": results,
    }


def phillips_perron_test(values: Sequence[float] | np.ndarray, *, alpha: float = 0.05) -> dict[str, Any]:
    """Run the native Phillips-Perron Z_tau unit-root test."""

    _validate_alpha(alpha)
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 8:
        return {"status": "insufficient_data", "n_obs": int(n)}
    y_t = y[1:]
    y_lag = y[:-1]
    x = np.column_stack([np.ones(n - 1), y_lag])
    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return {"status": "singular_design", "n_obs": int(n)}
    coef = xtx_inv @ x.T @ y_t
    rho = float(coef[1])
    resid = y_t - x @ coef
    sigma2 = float((resid @ resid) / max(n - 2 - 1, 1))
    bandwidth = max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))
    gamma0 = float(np.dot(resid, resid) / n)
    lr_var = gamma0
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1)
        cov = float(np.dot(resid[lag:], resid[:-lag]) / n)
        lr_var += 2.0 * weight * cov
    se_rho = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    t_rho = (rho - 1.0) / se_rho
    z_tau = float(
        np.sqrt(gamma0 / max(lr_var, 1e-12)) * t_rho
        - 0.5
        * (lr_var - gamma0)
        * np.sqrt(n)
        * np.sqrt(xtx_inv[1, 1])
        / np.sqrt(max(lr_var, 1e-12))
    )
    p_value = mackinnon_pp_pvalue(z_tau, n=n, regression="c")
    return {
        "statistic": z_tau,
        "p_value": p_value,
        "reject_unit_root": bool(p_value < alpha),
        "n_obs": int(n),
        "bandwidth_lags": bandwidth,
    }


_MACKINNON_PP_C_TABLE = {
    0.01: {25: -3.75, 50: -3.58, 100: -3.51, 250: -3.46, 500: -3.44, 1000: -3.43},
    0.05: {25: -3.00, 50: -2.93, 100: -2.89, 250: -2.88, 500: -2.87, 1000: -2.86},
    0.10: {25: -2.63, 50: -2.60, 100: -2.58, 250: -2.57, 500: -2.57, 1000: -2.57},
}


def mackinnon_pp_pvalue(z_tau: float, *, n: int, regression: str = "c") -> float:
    """Approximate MacKinnon p-value for the Phillips-Perron Z_tau statistic."""

    if not np.isfinite(float(z_tau)):
        raise ValueError("z_tau must be finite")
    if int(n) < 1:
        raise ValueError("n must be a positive integer")
    if regression != "c":
        from scipy import stats as _stats

        return float(_stats.norm.cdf(z_tau))
    table = _MACKINNON_PP_C_TABLE
    sizes = sorted(next(iter(table.values())).keys())
    n_clamped = max(sizes[0], min(sizes[-1], int(n)))
    lower_n = max((size for size in sizes if size <= n_clamped), default=sizes[0])
    upper_n = min((size for size in sizes if size >= n_clamped), default=sizes[-1])
    weight = 0.0 if lower_n == upper_n else (n_clamped - lower_n) / (upper_n - lower_n)
    critical_values: dict[float, float] = {}
    for level, by_n in table.items():
        critical_values[level] = by_n[lower_n] * (1 - weight) + by_n[upper_n] * weight
    levels = sorted(critical_values)
    values = [critical_values[level] for level in levels]
    if z_tau <= values[0]:
        return 0.005
    if z_tau >= values[-1]:
        return 0.50
    for idx in range(len(levels) - 1):
        if values[idx] <= z_tau <= values[idx + 1]:
            span = values[idx + 1] - values[idx]
            if span <= 0:
                return float(levels[idx])
            point = (z_tau - values[idx]) / span
            return float(levels[idx] * (1 - point) + levels[idx + 1] * point)
    return 0.50


def summarize_data(
    data: Any,
    *,
    metrics: Sequence[SummaryMetric] | None = None,
    include_correlation: bool = False,
    correlation_method: CorrelationMethod = "pearson",
    include_outliers: bool = False,
    outlier_method: OutlierMethod = "iqr",
    include_stationarity: bool = False,
    stationarity_test: StationarityTest = "multi",
    stationarity_scope: StationarityScope = "all",
) -> DataSummaryReport:
    """Run the standard single-panel summary suite."""

    panel, metadata = _coerce_panel(data)
    bundle = DataBundle(panel, metadata)
    selected_metrics = tuple(metrics or DEFAULT_SUMMARY_METRICS)
    _validate_summary_metrics(selected_metrics)
    report_metadata = attach_metadata(
        metadata,
        "data_analysis",
        {
            "analysis_type": "single_panel",
            "metrics": list(selected_metrics),
            "include_correlation": bool(include_correlation),
            "correlation_method": correlation_method if include_correlation else None,
            "include_outliers": bool(include_outliers),
            "outlier_method": outlier_method if include_outliers else None,
            "include_stationarity": bool(include_stationarity),
            "stationarity_test": stationarity_test if include_stationarity else None,
            "stationarity_scope": stationarity_scope if include_stationarity else None,
            "panel": _compact_panel_info(bundle),
            "input": _metadata_input_summary(metadata),
            "outputs": {
                "coverage": True,
                "univariate": True,
                "missing": True,
                "correlation": bool(include_correlation),
                "outliers": bool(include_outliers),
                "stationarity": bool(include_stationarity),
            },
        },
    )
    coverage = sample_coverage(bundle)
    univariate = univariate_summary(bundle, metrics=selected_metrics)
    missing = missing_summary(bundle)
    correlation = (
        correlation_matrix(bundle, method=correlation_method)
        if include_correlation
        else None
    )
    outliers = (
        outlier_summary(bundle, method=outlier_method)
        if include_outliers
        else None
    )
    stationarity = (
        stationarity_tests(
            data,
            test=stationarity_test,
            scope=stationarity_scope,
        )
        if include_stationarity
        else None
    )
    _attach_metadata(coverage, report_metadata)
    _attach_metadata(univariate, report_metadata)
    _attach_metadata(missing, report_metadata)
    _attach_metadata(correlation, report_metadata)
    _attach_metadata(outliers, report_metadata)
    return DataSummaryReport(
        overview=panel_overview(bundle),
        coverage=coverage,
        univariate=univariate,
        missing=missing,
        correlation=correlation,
        outliers=outliers,
        stationarity=stationarity,
        metadata=report_metadata,
    )


def _coerce_panel(data: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(data, DataSpec):
        panel = data.panel
        metadata = dict(data.metadata)
    elif isinstance(data, DataBundle):
        panel = data.panel
        metadata = dict(data.metadata)
    elif hasattr(data, "panel") and isinstance(getattr(data, "panel"), pd.DataFrame):
        panel = getattr(data, "panel")
        metadata = dict(getattr(data, "metadata", {}) or {})
    elif isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], pd.DataFrame):
        metadata = dict(data[1])
        panel = as_panel(data[0], metadata=metadata)
    elif isinstance(data, pd.DataFrame):
        metadata = dict(data.attrs.get("macroforecast_metadata", {}))
        panel = as_panel(data, metadata=metadata)
    else:
        raise TypeError("expected DataBundle, DataSpec, PreprocessedData, (panel, metadata), or pandas DataFrame")
    validate_panel(panel)
    return panel, metadata


def _validate_summary_metrics(metrics: Sequence[str]) -> None:
    allowed = set(SummaryMetric.__args__)  # type: ignore[attr-defined]
    unknown = sorted(set(metrics) - allowed)
    if unknown:
        raise ValueError(f"unknown summary metric(s): {unknown}")


def _compact_panel_info(bundle: DataBundle) -> dict[str, Any]:
    info = panel_info(bundle)
    return {
        "n_rows": info["n_rows"],
        "n_columns": info["n_columns"],
        "start": info["start"],
        "end": info["end"],
        "missing_values": info["missing_values"],
        "frequency": info["frequency"],
    }


def _target_names(data: Any, *, target: str | None, targets: Sequence[str] | None) -> set[str]:
    names = set(str(value) for value in (targets or ()))
    if target is not None:
        names.add(str(target))
    if not names:
        data_target = getattr(data, "target", None)
        data_targets = getattr(data, "targets", ())
        if data_target is not None:
            names.add(str(data_target))
        names.update(str(value) for value in data_targets)
    return names


def _scope_columns(
    panel: pd.DataFrame,
    data: Any,
    *,
    scope: StationarityScope,
    target: str | None,
    targets: Sequence[str] | None,
) -> list[str]:
    if scope == "all":
        return list(panel.columns)
    target_names = _target_names(data, target=target, targets=targets)
    if scope in {"target_only", "predictors_only"} and not target_names:
        raise ValueError(f"scope={scope!r} requires target or targets")
    panel_names = {str(column) for column in panel.columns}
    missing_targets = sorted(target_names - panel_names)
    if missing_targets:
        raise ValueError(f"target columns are not in the panel: {missing_targets}")
    if scope == "target_only":
        return [column for column in panel.columns if str(column) in target_names]
    if scope == "predictors_only":
        return [column for column in panel.columns if str(column) not in target_names]
    return list(panel.columns)


def _metadata_input_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": metadata.get("dataset"),
        "source_family": metadata.get("source_family"),
        "frequency": metadata.get("frequency"),
        "has_panel_report": "panel" in metadata,
        "has_preprocessing": "preprocessing" in metadata,
        "metadata_keys": sorted(str(key) for key in metadata),
    }


def _attach_metadata(frame: pd.DataFrame | None, metadata: Mapping[str, Any]) -> None:
    if frame is not None:
        frame.attrs["macroforecast_metadata"] = dict(metadata)


def _run_stationarity(
    name: str, series: pd.Series, alpha: float, *, adf_regression: str = "c"
) -> dict[str, Any]:
    if name == "adf":
        from statsmodels.tsa.stattools import adfuller

        stat, pvalue, *_ = adfuller(
            series.values, autolag="AIC", regression=adf_regression
        )
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "reject_unit_root": bool(pvalue < alpha),
            "regression": adf_regression,
        }
    if name == "kpss":
        from statsmodels.tsa.stattools import kpss as _kpss
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, pvalue, *_ = _kpss(series.values, regression="c", nlags="auto")
        return {
            "statistic": float(stat),
            "p_value": float(pvalue),
            "reject_stationarity": bool(pvalue < alpha),
        }
    if name == "pp":
        try:
            from arch.unitroot import PhillipsPerron  # type: ignore

            pp = PhillipsPerron(series.values, trend="c")
            return {
                "statistic": float(pp.stat),
                "p_value": float(pp.pvalue),
                "reject_unit_root": bool(pp.pvalue < alpha),
                "implementation": "arch",
            }
        except ImportError:
            pass
        result = phillips_perron_test(series.values, alpha=alpha)
        result["implementation"] = "native"
        return result
    return {"status": "unknown_test", "test": name}


def _date_string(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_positive(value: float, name: str) -> None:
    if float(value) <= 0:
        raise ValueError(f"{name} must be > 0")


def _validate_alpha(value: float) -> None:
    alpha = float(value)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _longest_true_run(values: pd.Series) -> int:
    longest = 0
    current = 0
    for value in values.astype(bool).tolist():
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


__all__ = [
    "DEFAULT_SUMMARY_METRICS",
    "DataSummaryReport",
    "correlation_matrix",
    "mackinnon_pp_pvalue",
    "missing_summary",
    "missing_rates",
    "observation_counts",
    "outlier_summary",
    "panel_overview",
    "panel_snapshot",
    "phillips_perron_test",
    "sample_coverage",
    "stationarity_tests",
    "summarize_data",
    "univariate_summary",
]
