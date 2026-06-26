from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd

from macroforecast.data.panel import (
    DataBundle,
    DataSpec,
    PanelInput,
    attach_metadata,
    validate_panel,
)

SamePeriodPolicy = Literal["allow", "lag", "drop", "forbid"]
RegimeDirection = Literal["above", "below", "equal", "not_equal"]


def align_frequency(
    data: PanelInput,
    *,
    method: str = "keep",
    quarterly_to_monthly: str = "step_forward",
    weekly_to_monthly: str = "mean",
    monthly_to_quarterly: str = "quarterly_average",
    weekly_to_quarterly: str = "mean",
    chow_lin_indicator: str | Mapping[str, str] | None = None,
    chow_lin_aggregation: str = "mean",
    chow_lin_rho: float | None = None,
    chow_lin_rho_method: str = "fixed",
) -> DataBundle:
    """Keep, filter, or align a panel to a common data frequency.

    This is a data-level callable because it changes the panel's calendar and
    column frequency contract. Statistical cleaning such as t-code transforms,
    outlier handling, imputation, and standardization stays in
    ``macroforecast.preprocessing``.
    """

    bundle = _coerce_bundle(data)
    panel = bundle.panel.copy()
    validate_panel(panel)
    method_value = _normalize_frequency_alignment(method)
    if method_value == "keep":
        result = panel.copy()
        meta = attach_metadata(
            bundle.metadata,
            "data_frequency_alignment",
            {
                "method": "keep",
                "input_frequency_source": infer_frequencies(panel)[1],
                "input_frequencies": infer_frequencies(panel)[0],
                "output_frequency": bundle.metadata.get("frequency"),
                "output_frequency_by_column": bundle.metadata.get("output_frequency_by_column"),
            },
        )
        result.attrs.update(dict(getattr(panel, "attrs", {}) or {}))
        result.attrs["macroforecast_metadata"] = meta
        return DataBundle(result, meta)

    frequencies, frequency_source = infer_frequencies(panel)
    _warn_frequency_hardening_issues(frequencies)
    if method_value in {"drop_non_monthly", "drop_non_quarterly"}:
        target_frequency = "monthly" if method_value == "drop_non_monthly" else "quarterly"
        columns = [
            column
            for column, frequency in frequencies.items()
            if frequency == target_frequency
        ]
        if not columns:
            raise ValueError(f"frequency alignment method {method!r} leaves no columns")
        result = panel[columns].copy()
        output_frequency = target_frequency
    elif method_value == "monthly":
        result = _align_to_monthly(
            panel,
            frequencies=frequencies,
            quarterly_to_monthly=quarterly_to_monthly,
            weekly_to_monthly=weekly_to_monthly,
            chow_lin_indicator=chow_lin_indicator,
            chow_lin_aggregation=chow_lin_aggregation,
            chow_lin_rho=chow_lin_rho,
            chow_lin_rho_method=chow_lin_rho_method,
        )
        output_frequency = "monthly"
    elif method_value == "quarterly":
        result = _align_to_quarterly(
            panel,
            frequencies=frequencies,
            monthly_to_quarterly=monthly_to_quarterly,
            weekly_to_quarterly=weekly_to_quarterly,
        )
        output_frequency = "quarterly"
    else:  # pragma: no cover - guarded by _normalize_frequency_alignment
        raise ValueError(f"unknown frequency alignment method {method!r}")

    result.attrs.update(dict(getattr(panel, "attrs", {}) or {}))
    output_frequency_by_column = {str(column): output_frequency for column in result.columns}
    meta = dict(bundle.metadata)
    native = {
        str(column): frequencies[str(column)]
        for column in result.columns
        if str(column) in frequencies
    }
    meta.update(
        {
            "frequency": output_frequency,
            "native_frequency_by_column": native,
            "native_frequency_counts": _frequency_counts(native),
            "output_frequency_by_column": output_frequency_by_column,
            "output_frequency_counts": _frequency_counts(output_frequency_by_column),
        }
    )
    meta = attach_metadata(
        meta,
        "data_frequency_alignment",
        {
            "method": method_value,
            "input_frequency_source": frequency_source,
            "input_frequencies": frequencies,
            "output_frequency": output_frequency,
            "output_frequency_by_column": output_frequency_by_column,
            "quarterly_to_monthly": quarterly_to_monthly if method_value == "monthly" else None,
            "weekly_to_monthly": weekly_to_monthly if method_value == "monthly" else None,
            "chow_lin_indicator": chow_lin_indicator if method_value == "monthly" else None,
            "chow_lin_aggregation": chow_lin_aggregation if method_value == "monthly" else None,
            "chow_lin_rho": chow_lin_rho if method_value == "monthly" else None,
            "chow_lin_rho_method": chow_lin_rho_method if method_value == "monthly" else None,
            "monthly_to_quarterly": monthly_to_quarterly if method_value == "quarterly" else None,
            "weekly_to_quarterly": weekly_to_quarterly if method_value == "quarterly" else None,
            "input_shape": tuple(int(value) for value in panel.shape),
            "output_shape": tuple(int(value) for value in result.shape),
        },
    )
    result.attrs["macroforecast_metadata"] = meta
    return DataBundle(result, meta)


def chow_lin_disaggregate(
    low_frequency: pd.Series,
    indicator: pd.Series | pd.DataFrame,
    *,
    aggregation: str = "mean",
    rho: float | None = None,
    rho_method: str = "fixed",
) -> pd.Series:
    """Disaggregate a low-frequency series with a high-frequency indicator.

    This implements the standard Chow-Lin regression-distribution identity with
    an AR(1) high-frequency residual covariance. The returned high-frequency
    series conserves the supplied low-frequency observations under
    ``aggregation='mean'`` or ``aggregation='sum'``.
    """

    y_low = pd.Series(low_frequency).dropna().astype(float)
    if isinstance(indicator, pd.DataFrame):
        if indicator.empty:
            raise ValueError("indicator DataFrame must contain at least one column")
        x_high = indicator.iloc[:, 0]
    else:
        x_high = pd.Series(indicator)
    x_high = x_high.dropna().astype(float)
    aggregation_value = str(aggregation).lower()
    if aggregation_value not in {"mean", "sum"}:
        raise ValueError("aggregation must be 'sum' or 'mean'")
    rho_method_value = str(rho_method).lower()
    if rho_method_value not in {"fixed", "min_chi_squared", "max_likelihood"}:
        raise ValueError("rho_method must be 'fixed', 'min_chi_squared', or 'max_likelihood'")
    if rho is not None and not (-1.0 < float(rho) < 1.0):
        raise ValueError("rho must be in the open interval (-1, 1)")
    if not isinstance(y_low.index, pd.DatetimeIndex) or not isinstance(x_high.index, pd.DatetimeIndex):
        return y_low.reindex(x_high.index).bfill().ffill().rename(x_high.name)
    if len(y_low) < 2 or len(x_high) < 2:
        return y_low.reindex(x_high.index).bfill().ffill().rename(x_high.name)

    design = _chow_lin_design(y_low, x_high, aggregation=aggregation_value)
    if design is None:
        return y_low.reindex(x_high.index).bfill().ffill().rename(x_high.name)
    y_vector, x_matrix, conversion = design
    rho_value = (
        float(rho)
        if rho is not None
        else _estimate_chow_lin_rho(
            y_vector,
            x_matrix,
            conversion,
            method=rho_method_value,
        )
    )
    high_cov = _ar1_covariance(rho_value, x_matrix.shape[0])
    low_cov = conversion @ high_cov @ conversion.T
    low_cov_inv = np.linalg.pinv(low_cov)
    beta = np.linalg.pinv(x_matrix.T @ conversion.T @ low_cov_inv @ conversion @ x_matrix) @ (
        x_matrix.T @ conversion.T @ low_cov_inv @ y_vector
    )
    predicted_high = x_matrix @ beta
    low_residual = y_vector - conversion @ predicted_high
    adjusted_high = predicted_high + high_cov @ conversion.T @ low_cov_inv @ low_residual
    return pd.Series(adjusted_high, index=x_high.index[: len(adjusted_high)], name=y_low.name or x_high.name)


def infer_frequencies(data: PanelInput | pd.DataFrame) -> tuple[dict[str, str], str]:
    """Infer or read native frequency by panel column.

    Metadata from ``set_frequencies`` / ``combine(..., frequency="native")`` is
    preferred, then FRED-SD series reports, then observed-date spacing.
    """

    panel: pd.DataFrame
    metadata: Mapping[str, Any]
    if isinstance(data, pd.DataFrame):
        panel = data
        metadata = dict(data.attrs.get("macroforecast_metadata", {}))
    else:
        bundle = _coerce_bundle(data)
        panel = bundle.panel
        metadata = bundle.metadata
    validate_panel(panel)
    metadata_map = metadata.get("native_frequency_by_column")
    if isinstance(metadata_map, Mapping) and metadata_map:
        normalized = {
            str(column): _normalize_frequency_label(value)
            for column, value in metadata_map.items()
            if str(column) in panel.columns
        }
        if normalized:
            return (
                {
                    str(column): normalized.get(str(column), _infer_column_frequency(panel[column]))
                    for column in panel.columns
                },
                "native_frequency_by_column",
            )
    report_map = _frequency_map_from_reports(panel)
    if report_map:
        return (
            {
                str(column): report_map.get(str(column), _infer_column_frequency(panel[column]))
                for column in panel.columns
            },
            "fred_sd_series_metadata",
        )
    return ({str(column): _infer_column_frequency(panel[column]) for column in panel.columns}, "observed_dates")


def frequency_hardening_issues(
    frequencies: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Return frequency-classification issues that should be surfaced."""

    issues: list[dict[str, Any]] = []
    for frequency in ("unknown", "irregular", "annual"):
        columns = sorted(column for column, value in frequencies.items() if value == frequency)
        if columns:
            issues.append({"frequency": frequency, "columns": columns, "n_columns": len(columns)})
    return issues


def availability_lag(
    data: PanelInput,
    *,
    lags: int | Mapping[str, int] = 1,
    columns: Iterable[str] | None = None,
    drop_missing: bool = False,
) -> DataBundle:
    """Delay selected columns to match an information-availability policy.

    A positive lag means the value dated ``t`` is treated as usable only from
    later forecast origins. For example, ``lags=1`` shifts ``x[t-1]`` onto row
    ``t``. This is the direct callable replacement for the old release-lag
    data policy module; release calendars can be expressed by passing a per-column
    lag mapping.
    """

    bundle = _coerce_bundle(data)
    panel = bundle.panel.copy()
    validate_panel(panel)
    selected = _resolve_columns(panel, columns=columns)
    lag_map = _resolve_lag_map(selected, lags)
    for column, lag in lag_map.items():
        panel[column] = panel[column].shift(lag)
    if drop_missing:
        panel = panel.dropna(subset=list(selected))
    panel.attrs["macroforecast_metadata"] = attach_metadata(
        bundle.metadata,
        "data_availability_lag",
        {
            "columns": list(selected),
            "lags": dict(lag_map),
            "drop_missing": bool(drop_missing),
            "meaning": "positive lags delay dated observations before forecasting use",
        },
    )
    return DataBundle(panel=panel, metadata=panel.attrs["macroforecast_metadata"])


def same_period_predictors(
    data: DataSpec,
    *,
    policy: SamePeriodPolicy = "allow",
    lag: int = 1,
    columns: Iterable[str] | None = None,
    drop_missing: bool = False,
) -> DataSpec:
    """Apply a same-period predictor policy to a run-level data spec.

    ``allow`` records that same-period predictors are intentionally allowed.
    ``lag`` shifts selected predictors by ``lag`` periods. ``drop`` removes
    selected predictors from the spec. ``forbid`` raises when selected
    same-period predictors are present. Targets are never shifted by this
    helper.
    """

    if not isinstance(data, DataSpec):
        raise TypeError("same_period_predictors() requires a DataSpec from mf.data.spec()")
    if int(lag) < 0:
        raise ValueError("lag must be non-negative")
    if policy not in {"allow", "lag", "drop", "forbid"}:
        raise ValueError("policy must be one of 'allow', 'lag', 'drop', or 'forbid'")
    panel = data.panel.copy()
    validate_panel(panel)
    predictors = tuple(data.predictors) if data.predictors != "all" else tuple(
        column for column in panel.columns if column not in set(data.targets)
    )
    selected = tuple(columns) if columns is not None else predictors
    selected = tuple(str(column) for column in selected)
    unknown = sorted(set(selected).difference(predictors))
    if unknown:
        raise ValueError(f"same-period policy columns are not active predictors: {unknown}")

    final_predictors = predictors
    action: dict[str, Any] = {"policy": policy, "columns": list(selected), "lag": int(lag)}
    if policy == "forbid" and selected:
        raise ValueError(
            "same-period predictors are present but policy='forbid': "
            f"{list(selected)}"
        )
    if policy == "lag":
        for column in selected:
            panel[column] = panel[column].shift(int(lag))
        if drop_missing:
            panel = panel.dropna(subset=list(selected))
        action["drop_missing"] = bool(drop_missing)
    elif policy == "drop":
        panel = panel.drop(columns=list(selected))
        final_predictors = tuple(column for column in predictors if column not in set(selected))

    meta = attach_metadata(data.metadata, "same_period_predictors", action)
    panel.attrs["macroforecast_metadata"] = meta
    return DataSpec(
        panel=panel,
        metadata=meta,
        target=data.target,
        targets=data.targets,
        horizons=data.horizons,
        start=data.start,
        end=data.end,
        predictors=final_predictors,
    )


def define_regime(
    data: PanelInput,
    *,
    name: str = "regime",
    column: str | None = None,
    threshold: float | None = None,
    direction: RegimeDirection = "above",
    dates: Iterable[str | pd.Timestamp] | None = None,
    values: Sequence[bool | int | float] | pd.Series | None = None,
    append: bool = False,
    output_column: str | None = None,
) -> DataBundle:
    """Attach a binary regime series to panel metadata.

    Regimes can be built from a threshold rule, explicit regime dates, or an
    aligned vector/Series of values. The panel is unchanged unless
    ``append=True``.
    """

    if not name:
        raise ValueError("name must be non-empty")
    bundle = _coerce_bundle(data)
    panel = bundle.panel.copy()
    validate_panel(panel)
    regime = _build_regime_series(
        panel,
        column=column,
        threshold=threshold,
        direction=direction,
        dates=dates,
        values=values,
    )
    out_col = output_column or f"{name}_regime"
    if append:
        panel[out_col] = regime.astype(float)
    regimes = dict(bundle.metadata.get("regimes", {}))
    regimes[name] = {
        "name": name,
        "column": column,
        "threshold": threshold,
        "direction": direction,
        "source": _regime_source(column=column, dates=dates, values=values),
        "output_column": out_col if append else None,
        "n_regime": int(regime.sum()),
        "n_observations": int(regime.notna().sum()),
        "series": {idx.strftime("%Y-%m-%d"): bool(value) for idx, value in regime.items() if pd.notna(value)},
    }
    meta = dict(bundle.metadata)
    meta["regimes"] = regimes
    meta = attach_metadata(
        meta,
        "data_regime",
        {
            "last_defined": name,
            "available_regimes": sorted(regimes),
            "appended": bool(append),
        },
    )
    panel.attrs["macroforecast_metadata"] = meta
    return DataBundle(panel=panel, metadata=meta)


def _coerce_bundle(data: PanelInput) -> DataBundle:
    if isinstance(data, DataSpec):
        return DataBundle(data.panel.copy(), dict(data.metadata))
    if isinstance(data, DataBundle):
        return DataBundle(data.panel.copy(), dict(data.metadata))
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], pd.DataFrame):
        return DataBundle(data[0].copy(), dict(data[1]))
    if isinstance(data, pd.DataFrame):
        return DataBundle(data.copy(), dict(data.attrs.get("macroforecast_metadata", {})))
    raise TypeError("expected DataBundle, DataSpec, (panel, metadata), or pandas DataFrame")


def _normalize_frequency_alignment(value: str) -> str:
    aliases = {
        "keep": "keep",
        "native": "keep",
        "mixed": "keep",
        "monthly": "monthly",
        "align_monthly": "monthly",
        "to_monthly": "monthly",
        "quarterly": "quarterly",
        "align_quarterly": "quarterly",
        "to_quarterly": "quarterly",
        "drop_non_monthly": "drop_non_monthly",
        "monthly_only": "drop_non_monthly",
        "drop_non_quarterly": "drop_non_quarterly",
        "quarterly_only": "drop_non_quarterly",
    }
    if not isinstance(value, str):
        raise TypeError("frequency alignment method must be a string")
    key = value.lower()
    if key not in aliases:
        raise ValueError(f"frequency alignment method must be one of {sorted(aliases)}; got {value!r}")
    return aliases[key]


def _warn_frequency_hardening_issues(frequencies: Mapping[str, str]) -> None:
    issues = frequency_hardening_issues(frequencies)
    for issue in issues:
        if issue["frequency"] == "irregular":
            continue
        sample = ", ".join(issue["columns"][:5])
        if issue["n_columns"] > 5:
            sample = f"{sample}, ..."
        warnings.warn(
            "frequency inference found "
            f"{issue['frequency']} columns before alignment: {sample}. "
            "Use data metadata when the source frequency is known.",
            UserWarning,
            stacklevel=3,
        )


def _frequency_map_from_reports(panel: pd.DataFrame) -> dict[str, str]:
    reports = getattr(panel, "attrs", {}).get("macrocast_reports", {})
    if not isinstance(reports, Mapping):
        return {}
    report = reports.get("fred_sd_series_metadata", {})
    if not isinstance(report, Mapping):
        return {}
    rows = report.get("series", ())
    if not isinstance(rows, (list, tuple)):
        return {}
    frequencies: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        column = row.get("column")
        frequency = row.get("native_frequency")
        if column is not None and frequency:
            frequencies[str(column)] = _normalize_frequency_label(frequency)
    return frequencies


def _infer_column_frequency(series: pd.Series) -> str:
    observed = series.dropna()
    if observed.shape[0] < 2:
        return "unknown"
    index = pd.DatetimeIndex(observed.index).sort_values()
    day_deltas = [
        (right - left).days
        for left, right in zip(index[:-1], index[1:], strict=False)
        if right > left
    ]
    if not day_deltas:
        return "unknown"
    median_days = float(pd.Series(day_deltas).median())
    if 5 <= median_days <= 10:
        return "weekly"
    if 25 <= median_days <= 35:
        return "monthly"
    if 80 <= median_days <= 100:
        return "quarterly"
    if 350 <= median_days <= 380:
        return "annual"
    return "irregular"


def _monthly_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    start = pd.DatetimeIndex(panel.index).min().to_period("M").to_timestamp()
    end = pd.DatetimeIndex(panel.index).max().to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS", name="date")


def _quarterly_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    start = pd.DatetimeIndex(panel.index).min().to_period("Q").start_time
    end = pd.DatetimeIndex(panel.index).max().to_period("Q").start_time
    return pd.date_range(start, end, freq="QS", name="date")


def _align_to_monthly(
    panel: pd.DataFrame,
    *,
    frequencies: Mapping[str, str],
    quarterly_to_monthly: str,
    weekly_to_monthly: str,
    chow_lin_indicator: str | Mapping[str, str] | None,
    chow_lin_aggregation: str,
    chow_lin_rho: float | None,
    chow_lin_rho_method: str,
) -> pd.DataFrame:
    index = _monthly_index(panel)
    columns: dict[str, pd.Series] = {}
    monthly_columns = [str(column) for column, frequency in frequencies.items() if frequency == "monthly"]
    for column in panel.columns:
        name = str(column)
        series = panel[column].dropna()
        frequency = frequencies.get(name, "unknown")
        if frequency == "weekly":
            aligned = _aggregate_to_monthly(series, weekly_to_monthly)
        elif frequency == "quarterly":
            indicator = _resolve_chow_lin_indicator(
                panel,
                target_column=name,
                monthly_columns=monthly_columns,
                indicator=chow_lin_indicator,
            )
            aligned = _quarterly_to_monthly(
                series,
                quarterly_to_monthly,
                index=index,
                indicator=indicator,
                chow_lin_aggregation=chow_lin_aggregation,
                chow_lin_rho=chow_lin_rho,
                chow_lin_rho_method=chow_lin_rho_method,
            )
        else:
            aligned = series.resample("MS").last()
        columns[name] = aligned.reindex(index)
    return pd.DataFrame(columns, index=index)


def _align_to_quarterly(
    panel: pd.DataFrame,
    *,
    frequencies: Mapping[str, str],
    monthly_to_quarterly: str,
    weekly_to_quarterly: str,
) -> pd.DataFrame:
    index = _quarterly_index(panel)
    columns: dict[str, pd.Series] = {}
    for column in panel.columns:
        name = str(column)
        series = panel[column].dropna()
        frequency = frequencies.get(name, "unknown")
        if frequency == "weekly":
            aligned = _aggregate_to_quarterly(series, weekly_to_quarterly)
        elif frequency == "monthly":
            aligned = _aggregate_to_quarterly(series, monthly_to_quarterly)
        else:
            aligned = series.resample("QS").last()
        columns[name] = aligned.reindex(index)
    return pd.DataFrame(columns, index=index)


def _quarterly_to_monthly(
    series: pd.Series,
    rule: str,
    *,
    index: pd.DatetimeIndex,
    indicator: pd.Series | None = None,
    chow_lin_aggregation: str = "mean",
    chow_lin_rho: float | None = None,
    chow_lin_rho_method: str = "fixed",
) -> pd.Series:
    key = rule.lower()
    quarterly = series.copy()
    quarterly.index = pd.DatetimeIndex(quarterly.index).to_period("Q").to_timestamp()
    quarterly = quarterly.groupby(level=0).last().sort_index()
    if key == "chow_lin":
        if indicator is None:
            raise ValueError("quarterly_to_monthly='chow_lin' requires a monthly indicator column")
        monthly = indicator.reindex(index).astype(float)
        return chow_lin_disaggregate(
            quarterly,
            monthly,
            aggregation=chow_lin_aggregation,
            rho=chow_lin_rho,
            rho_method=chow_lin_rho_method,
        ).reindex(index)
    if key in {"step_backward", "repeat_within_quarter", "repeat", "spread"}:
        by_quarter = dict(zip(pd.DatetimeIndex(quarterly.index).to_period("Q"), quarterly.to_numpy(), strict=False))
        values = [by_quarter.get(period) for period in pd.DatetimeIndex(index).to_period("Q")]
        return pd.Series(values, index=index, dtype="float64")
    if key in {"step_forward", "quarter_end_ffill", "ffill_from_quarter_end"}:
        observed = quarterly.copy()
        observed.index = pd.DatetimeIndex(observed.index).to_period("Q").asfreq("M", how="end").to_timestamp()
        return observed.reindex(index).ffill()
    if key in {"linear_interpolation", "linear"}:
        observed = quarterly.copy()
        observed.index = pd.DatetimeIndex(observed.index).to_period("Q").asfreq("M", how="end").to_timestamp()
        return observed.reindex(index.union(observed.index)).sort_index().interpolate(method="time").reindex(index)
    raise ValueError(
        "quarterly_to_monthly must be one of "
        "['step_backward', 'repeat_within_quarter', 'step_forward', 'quarter_end_ffill', 'linear_interpolation', 'chow_lin']"
    )


def _aggregate_to_monthly(series: pd.Series, rule: str) -> pd.Series:
    return _aggregate_resample(
        series,
        rule,
        frequency="MS",
        aliases={
            "last": "last",
            "endpoint": "last",
            "mean": "mean",
            "average": "mean",
            "sum": "sum",
        },
    )


def _aggregate_to_quarterly(series: pd.Series, rule: str) -> pd.Series:
    aliases = {
        "quarterly_average": "mean",
        "average": "mean",
        "mean": "mean",
        "quarterly_endpoint": "last",
        "endpoint": "last",
        "last": "last",
        "quarterly_sum": "sum",
        "sum": "sum",
    }
    return _aggregate_resample(series, rule, frequency="QS", aliases=aliases)


def _aggregate_resample(
    series: pd.Series,
    rule: str,
    *,
    frequency: str,
    aliases: Mapping[str, str],
) -> pd.Series:
    key = rule.lower()
    if key not in aliases:
        raise ValueError(f"aggregation rule must be one of {sorted(aliases)}; got {rule!r}")
    method = aliases[key]
    if method == "mean":
        return series.resample(frequency).mean()
    if method == "last":
        return series.resample(frequency).last()
    if method == "sum":
        return series.resample(frequency).sum(min_count=1)
    raise ValueError(f"unknown aggregation method {method!r}")


def _resolve_chow_lin_indicator(
    panel: pd.DataFrame,
    *,
    target_column: str,
    monthly_columns: Sequence[str],
    indicator: str | Mapping[str, str] | None,
) -> pd.Series | None:
    if isinstance(indicator, Mapping):
        indicator_column = indicator.get(target_column)
    elif indicator is None:
        indicator_column = next((column for column in monthly_columns if column != target_column), None)
    else:
        indicator_column = str(indicator)
    if indicator_column is None:
        return None
    if indicator_column not in panel.columns:
        raise ValueError(f"chow_lin_indicator column {indicator_column!r} is not in the panel")
    return panel[indicator_column]


def _chow_lin_design(
    y_low: pd.Series,
    x_high: pd.Series,
    *,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    y_periods = pd.DatetimeIndex(y_low.index).to_period("Q")
    x_periods = pd.DatetimeIndex(x_high.index).to_period("Q")
    rows: list[np.ndarray] = []
    y_values: list[float] = []
    for period, value in zip(y_periods, y_low.to_numpy(dtype=float), strict=False):
        positions = np.flatnonzero(x_periods == period)
        if len(positions) == 0 or not np.isfinite(value):
            continue
        row: np.ndarray = np.zeros(len(x_high), dtype=float)
        row[positions] = 1.0 if aggregation == "sum" else 1.0 / len(positions)
        rows.append(row)
        y_values.append(float(value))
    if len(rows) < 2:
        return None
    conversion = np.vstack(rows)
    x_matrix = np.column_stack([np.ones(len(x_high), dtype=float), x_high.to_numpy(dtype=float)])
    return np.asarray(y_values, dtype=float), x_matrix, conversion


def _estimate_chow_lin_rho(
    y_low: np.ndarray,
    x_high: np.ndarray,
    conversion: np.ndarray,
    *,
    method: str,
) -> float:
    if method == "fixed":
        return 0.0
    grid = np.linspace(-0.95, 0.95, 77)
    objective_values = [
        _chow_lin_objective(y_low, x_high, conversion, rho=float(candidate), method=method)
        for candidate in grid
    ]
    return float(grid[int(np.nanargmin(objective_values))])


def _chow_lin_objective(
    y_low: np.ndarray,
    x_high: np.ndarray,
    conversion: np.ndarray,
    *,
    rho: float,
    method: str,
) -> float:
    high_cov = _ar1_covariance(rho, x_high.shape[0])
    low_cov = conversion @ high_cov @ conversion.T
    low_cov_inv = np.linalg.pinv(low_cov)
    beta = np.linalg.pinv(x_high.T @ conversion.T @ low_cov_inv @ conversion @ x_high) @ (
        x_high.T @ conversion.T @ low_cov_inv @ y_low
    )
    residual = y_low - conversion @ x_high @ beta
    chi_squared = float(residual.T @ low_cov_inv @ residual)
    if method == "min_chi_squared":
        return chi_squared
    sign, logdet = np.linalg.slogdet(low_cov)
    if sign <= 0:
        return np.inf
    return float(logdet + chi_squared)


def _ar1_covariance(rho: float, size: int) -> np.ndarray:
    if not (-1.0 < float(rho) < 1.0):
        raise ValueError("rho must be in the open interval (-1, 1)")
    index = np.arange(int(size))
    return np.power(float(rho), np.abs(np.subtract.outer(index, index)))


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
        return key
    return aliases[key]


def _frequency_counts(frequencies: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in frequencies.values():
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _resolve_columns(panel: pd.DataFrame, *, columns: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(str(column) for column in (panel.columns if columns is None else columns))
    missing = [column for column in selected if column not in panel.columns]
    if missing:
        raise ValueError(f"columns are not in the panel: {missing}")
    return selected


def _resolve_lag_map(columns: tuple[str, ...], lags: int | Mapping[str, int]) -> dict[str, int]:
    if isinstance(lags, Mapping):
        lag_map: dict[str, int] = {}
        missing = [column for column in columns if column not in lags]
        if missing:
            raise ValueError(f"lags mapping is missing selected columns: {missing}")
        for column in columns:
            lag = int(lags[column])
            if lag < 0:
                raise ValueError("availability lags must be non-negative")
            lag_map[column] = lag
        return lag_map
    lag = int(lags)
    if lag < 0:
        raise ValueError("availability lags must be non-negative")
    return {column: lag for column in columns}


def _build_regime_series(
    panel: pd.DataFrame,
    *,
    column: str | None,
    threshold: float | None,
    direction: RegimeDirection,
    dates: Iterable[str | pd.Timestamp] | None,
    values: Sequence[bool | int | float] | pd.Series | None,
) -> pd.Series:
    n_sources = sum(source is not None for source in (column, dates, values))
    if n_sources != 1:
        raise ValueError("define exactly one regime source: column, dates, or values")
    if direction not in {"above", "below", "equal", "not_equal"}:
        raise ValueError("direction must be one of 'above', 'below', 'equal', or 'not_equal'")
    if column is not None:
        if threshold is None:
            raise ValueError("threshold is required when column is provided")
        if column not in panel.columns:
            raise ValueError(f"regime column {column!r} is not in the panel")
        series = panel[column]
        if direction == "above":
            return (series > float(threshold)).rename("regime")
        if direction == "below":
            return (series < float(threshold)).rename("regime")
        if direction == "equal":
            return (series == float(threshold)).rename("regime")
        return (series != float(threshold)).rename("regime")
    if dates is not None:
        regime_dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
        return pd.Series(panel.index.isin(regime_dates), index=panel.index, name="regime")
    if isinstance(values, pd.Series):
        aligned = values.reindex(panel.index)
        return aligned.astype(bool).rename("regime")
    if values is None:
        raise ValueError("values source is missing")
    if len(values) != len(panel):
        raise ValueError("values length must match the panel length")
    return pd.Series([bool(value) for value in values], index=panel.index, name="regime")


def _regime_source(
    *,
    column: str | None,
    dates: Iterable[str | pd.Timestamp] | None,
    values: Sequence[bool | int | float] | pd.Series | None,
) -> str:
    if column is not None:
        return "threshold"
    if dates is not None:
        return "dates"
    if values is not None:
        return "values"
    return "unknown"


__all__ = [
    "SamePeriodPolicy",
    "RegimeDirection",
    "align_frequency",
    "availability_lag",
    "chow_lin_disaggregate",
    "define_regime",
    "frequency_hardening_issues",
    "infer_frequencies",
    "same_period_predictors",
]
