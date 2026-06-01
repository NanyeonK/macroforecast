from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata
from macroforecast.forecasting import ForecastResult
from macroforecast.metrics import MetricLike, evaluate_forecasts, rank_forecasts


DEFAULT_METRICS: tuple[str, ...] = ("mse", "rmse", "mae")
DEFAULT_SCORE_BY: tuple[str, ...] = ("model", "horizon")
BENCHMARK_METRICS: tuple[str, ...] = (
    "mse",
    "mae",
    "relative_mse",
    "relative_mae",
    "mse_reduction",
    "r2_oos",
)


@dataclass(frozen=True)
class EvaluationReport:
    """Container returned by :func:`evaluate_report`."""

    scores: pd.DataFrame
    ranking: pd.DataFrame
    aggregations: dict[str, pd.DataFrame] = field(default_factory=dict)
    benchmark: pd.DataFrame | None = None
    regime: pd.DataFrame | None = None
    decomposition: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scores": self.scores.to_dict(orient="records"),
            "ranking": self.ranking.to_dict(orient="records"),
            "aggregations": {
                name: table.to_dict(orient="records")
                for name, table in self.aggregations.items()
            },
            "metadata": dict(self.metadata),
        }
        if self.benchmark is not None:
            out["benchmark"] = self.benchmark.to_dict(orient="records")
        if self.regime is not None:
            out["regime"] = self.regime.to_dict(orient="records")
        if self.decomposition is not None:
            out["decomposition"] = self.decomposition.to_dict(orient="records")
        return out


def evaluate_report(
    forecasts: Any,
    *,
    metrics: Sequence[str | MetricLike] = DEFAULT_METRICS,
    score_by: Sequence[str] = DEFAULT_SCORE_BY,
    aggregations: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | None = None,
    rank_metric: str | None = None,
    rank_by: Sequence[str] | None = None,
    benchmark_model: str | None = None,
    benchmark_metrics: Sequence[str | MetricLike] = BENCHMARK_METRICS,
    oos_start: Any | None = None,
    oos_end: Any | None = None,
    regimes: Mapping[Any, Any] | pd.Series | str | None = None,
    regime_column: str = "regime",
    target_column: str = "target",
    state_column: str = "state",
    time_frequency: str | None = None,
    time_column: str = "date",
    time_bucket_column: str = "time_bucket",
    include_decomposition: bool = False,
    decomposition_by: Sequence[str] | None = None,
    include_combined: bool = True,
) -> EvaluationReport:
    """Build a multi-slice forecast evaluation report."""

    table, base_metadata = _coerce_forecast_input(forecasts)
    table = filter_oos_period(
        table,
        start=oos_start,
        end=oos_end,
        date_column=time_column,
    )
    table = _prepare_table(
        table,
        regimes=regimes,
        regime_column=regime_column,
        time_frequency=time_frequency,
        time_column=time_column,
        time_bucket_column=time_bucket_column,
        include_combined=include_combined,
    )
    score_groups = _validate_grouping(score_by, table, label="score_by")
    score_table = evaluate_forecasts(
        table,
        by=score_groups,
        metrics=metrics,
        benchmark_model=benchmark_model,
    )
    resolved_rank_metric = _resolve_rank_metric(rank_metric, score_table, metrics)
    resolved_rank_by = (
        tuple(rank_by)
        if rank_by is not None
        else tuple(column for column in score_groups if column != "model")
    )
    ranking = (
        rank_forecasts(score_table, metric=resolved_rank_metric, by=resolved_rank_by)
        if resolved_rank_metric is not None and not score_table.empty
        else pd.DataFrame()
    )
    aggregation_groups = _resolve_aggregations(
        table,
        aggregations=aggregations,
        score_by=score_groups,
        target_column=target_column,
        state_column=state_column,
        regime_column=regime_column,
        time_bucket_column=time_bucket_column,
    )
    aggregation_tables = aggregate_scores(
        table,
        groupings=aggregation_groups,
        metrics=metrics,
        benchmark_model=benchmark_model,
    )
    benchmark = (
        benchmark_comparison(
            table,
            benchmark_model=benchmark_model,
            by=_benchmark_grouping(score_groups),
            metrics=benchmark_metrics,
        )
        if benchmark_model is not None
        else None
    )
    regime = (
        regime_scores(
            table,
            regimes=None,
            regime_column=regime_column,
            by=_regime_grouping(score_groups, regime_column),
            metrics=metrics,
            benchmark_model=benchmark_model,
        )
        if regime_column in table.columns
        else None
    )
    decomposition = (
        error_decomposition(
            table,
            by=decomposition_by or score_groups,
        )
        if include_decomposition
        else None
    )

    stage = {
        "options": {
            "metrics": [_metric_label(metric) for metric in metrics],
            "score_by": list(score_groups),
            "rank_metric": resolved_rank_metric,
            "rank_by": list(resolved_rank_by),
            "benchmark_model": benchmark_model,
            "benchmark_metrics": [_metric_label(metric) for metric in benchmark_metrics],
            "oos_start": None if oos_start is None else str(oos_start),
            "oos_end": None if oos_end is None else str(oos_end),
            "regime_column": regime_column,
            "target_column": target_column,
            "state_column": state_column,
            "time_frequency": time_frequency,
            "time_column": time_column,
            "time_bucket_column": time_bucket_column,
            "include_decomposition": bool(include_decomposition),
            "decomposition_by": list(decomposition_by or score_groups),
            "include_combined": bool(include_combined),
        },
        "tables": {
            "scores": int(score_table.shape[0]),
            "ranking": int(ranking.shape[0]),
            "aggregations": {name: int(table.shape[0]) for name, table in aggregation_tables.items()},
            "benchmark": None if benchmark is None else int(benchmark.shape[0]),
            "regime": None if regime is None else int(regime.shape[0]),
            "decomposition": None if decomposition is None else int(decomposition.shape[0]),
        },
        "input": _forecast_overview(table),
    }
    metadata = attach_metadata(base_metadata, "evaluation_report", stage)
    _attach_metadata(score_table, metadata)
    _attach_metadata(ranking, metadata)
    for table_value in aggregation_tables.values():
        _attach_metadata(table_value, metadata)
    _attach_metadata(benchmark, metadata)
    _attach_metadata(regime, metadata)
    _attach_metadata(decomposition, metadata)
    return EvaluationReport(
        scores=score_table,
        ranking=ranking,
        aggregations=aggregation_tables,
        benchmark=benchmark,
        regime=regime,
        decomposition=decomposition,
        metadata=metadata,
    )


def filter_oos_period(
    forecasts: Any,
    *,
    start: Any | None = None,
    end: Any | None = None,
    date_column: str = "date",
) -> pd.DataFrame:
    """Return forecast rows restricted to an out-of-sample date interval."""

    table, _metadata = _coerce_forecast_input(forecasts)
    if start is None and end is None:
        return table
    if date_column not in table.columns:
        raise ValueError(f"date_column {date_column!r} is not present")
    dates = pd.to_datetime(table[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"date_column {date_column!r} contains invalid dates")
    mask = pd.Series(True, index=table.index)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    out = table.loc[mask].reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "forecast_oos_period",
        "version": 1,
        "start": None if start is None else str(start),
        "end": None if end is None else str(end),
        "date_column": str(date_column),
    }
    return out


def error_decomposition(
    forecasts: Any,
    *,
    by: Sequence[str] = DEFAULT_SCORE_BY,
    actual: str = "actual",
    prediction: str = "prediction",
) -> pd.DataFrame:
    """Decompose forecast MSE into squared bias and residual variance."""

    table, _metadata = _coerce_forecast_input(forecasts)
    missing = [column for column in (actual, prediction) if column not in table.columns]
    if missing:
        raise ValueError(f"forecast table missing columns: {missing}")
    groups = _validate_grouping(by, table, label="by")
    rows: list[dict[str, Any]] = []
    iterator = table.groupby(list(groups), dropna=False, sort=True) if groups else [((), table)]
    for key, group in iterator:
        valid = group[[actual, prediction]].dropna()
        residual = valid[actual].astype(float) - valid[prediction].astype(float)
        row = _group_row(key, groups)
        n_obs = int(residual.shape[0])
        if n_obs == 0:
            row.update(
                {
                    "n": 0,
                    "mse": None,
                    "bias": None,
                    "bias_squared": None,
                    "residual_variance": None,
                    "bias_share": None,
                    "variance_share": None,
                }
            )
        else:
            mse_value = float(np.mean(residual.to_numpy(dtype=float) ** 2))
            bias_value = float(residual.mean())
            bias_squared = float(bias_value**2)
            variance = float(np.mean((residual - bias_value).to_numpy(dtype=float) ** 2))
            row.update(
                {
                    "n": n_obs,
                    "mse": mse_value,
                    "bias": bias_value,
                    "bias_squared": bias_squared,
                    "residual_variance": variance,
                    "bias_share": _safe_ratio(bias_squared, mse_value),
                    "variance_share": _safe_ratio(variance, mse_value),
                }
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "forecast_error_decomposition",
        "version": 1,
        "by": list(groups),
        "identity": "mse = bias_squared + residual_variance",
    }
    return out


def aggregate_scores(
    forecasts: Any,
    *,
    groupings: Mapping[str, Sequence[str]] | Sequence[Sequence[str]],
    metrics: Sequence[str | MetricLike] = DEFAULT_METRICS,
    benchmark_model: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Evaluate the same forecasts over multiple explicit groupings."""

    table, _metadata = _coerce_forecast_input(forecasts)
    grouping_map = _normalize_groupings(groupings)
    out: dict[str, pd.DataFrame] = {}
    for name, grouping in grouping_map.items():
        groups = _validate_grouping(grouping, table, label=f"grouping {name!r}")
        evaluated = evaluate_forecasts(
            table,
            by=groups,
            metrics=metrics,
            benchmark_model=benchmark_model,
        )
        evaluated.attrs["macroforecast_metadata_schema"] = {
            "kind": "forecast_evaluation_aggregation",
            "version": 1,
            "name": name,
            "by": list(groups),
        }
        out[name] = evaluated
    return out


def benchmark_comparison(
    forecasts: Any,
    *,
    benchmark_model: str,
    by: Sequence[str] = DEFAULT_SCORE_BY,
    metrics: Sequence[str | MetricLike] = BENCHMARK_METRICS,
) -> pd.DataFrame:
    """Evaluate candidate models relative to one benchmark model."""

    table, _metadata = _coerce_forecast_input(forecasts)
    if "model" not in table.columns:
        raise ValueError("forecast table must contain a 'model' column for benchmark comparison")
    if benchmark_model not in set(table["model"].dropna().astype(str)):
        raise ValueError(f"benchmark_model {benchmark_model!r} is not present in forecast rows")
    groups = _validate_grouping(by, table, label="by")
    if "model" not in groups:
        groups = ("model", *groups)
    evaluated = evaluate_forecasts(
        table,
        by=groups,
        metrics=metrics,
        benchmark_model=benchmark_model,
    )
    out = evaluated.loc[evaluated["model"].astype(str) != benchmark_model].reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "forecast_benchmark_comparison",
        "version": 1,
        "benchmark_model": str(benchmark_model),
        "by": list(groups),
    }
    return out


def regime_scores(
    forecasts: Any,
    *,
    regimes: Mapping[Any, Any] | pd.Series | str | None = None,
    regime_column: str = "regime",
    by: Sequence[str] = ("model", "horizon", "regime"),
    metrics: Sequence[str | MetricLike] = DEFAULT_METRICS,
    benchmark_model: str | None = None,
) -> pd.DataFrame:
    """Evaluate forecasts by regime labels."""

    table, _metadata = _coerce_forecast_input(forecasts)
    table = _attach_regimes(table, regimes=regimes, regime_column=regime_column)
    if regime_column not in table.columns:
        raise ValueError(
            f"regime column {regime_column!r} is not present; pass regimes=... or an existing column name"
        )
    groups = _validate_grouping(by, table, label="by")
    out = evaluate_forecasts(
        table,
        by=groups,
        metrics=metrics,
        benchmark_model=benchmark_model,
    )
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "forecast_regime_scores",
        "version": 1,
        "regime_column": regime_column,
        "by": list(groups),
    }
    return out


def _coerce_forecast_input(value: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(value, ForecastResult):
        return value.to_frame(), dict(value.metadata)
    if isinstance(value, pd.DataFrame):
        return value.copy(), dict(value.attrs.get("macroforecast_metadata", {}) or {})
    raise TypeError("forecasts must be a ForecastResult or pandas DataFrame")


def _prepare_table(
    table: pd.DataFrame,
    *,
    regimes: Mapping[Any, Any] | pd.Series | str | None,
    regime_column: str,
    time_frequency: str | None,
    time_column: str,
    time_bucket_column: str,
    include_combined: bool,
) -> pd.DataFrame:
    out = table.copy()
    if not include_combined and "combined" in out.columns:
        out = out.loc[~out["combined"].fillna(False).map(bool)].copy()
    out = _attach_regimes(out, regimes=regimes, regime_column=regime_column)
    if time_frequency is not None:
        if time_column not in out.columns:
            raise ValueError(f"time_column {time_column!r} is not present")
        dates = pd.to_datetime(out[time_column], errors="coerce")
        if dates.isna().any():
            raise ValueError(f"time_column {time_column!r} contains invalid dates")
        out[time_bucket_column] = dates.dt.to_period(str(time_frequency)).astype(str)
    return out


def _attach_regimes(
    table: pd.DataFrame,
    *,
    regimes: Mapping[Any, Any] | pd.Series | str | None,
    regime_column: str,
) -> pd.DataFrame:
    out = table.copy()
    if regimes is None:
        return out
    if isinstance(regimes, str):
        if regimes not in out.columns:
            raise ValueError(f"regime source column {regimes!r} is not present")
        if regimes != regime_column:
            out[regime_column] = out[regimes]
        return out
    if "date" not in out.columns:
        raise ValueError("date column is required when regimes is a mapping or Series")
    dates = pd.to_datetime(out["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("date column contains invalid dates")
    if isinstance(regimes, pd.Series):
        mapping = regimes.copy()
    else:
        mapping = pd.Series(dict(regimes))
    if not isinstance(mapping.index, pd.DatetimeIndex):
        mapping.index = pd.to_datetime(mapping.index, errors="coerce")
    out[regime_column] = dates.map(mapping)
    return out


def _resolve_aggregations(
    table: pd.DataFrame,
    *,
    aggregations: Mapping[str, Sequence[str]] | Sequence[Sequence[str]] | None,
    score_by: Sequence[str],
    target_column: str,
    state_column: str,
    regime_column: str,
    time_bucket_column: str,
) -> dict[str, tuple[str, ...]]:
    if aggregations is not None:
        return _normalize_groupings(aggregations)
    candidates: dict[str, tuple[str, ...]] = {
        "model": ("model",),
        "horizon": ("horizon",),
        "model_horizon": tuple(score_by),
    }
    optional = {
        "model_horizon_target": ("model", "horizon", target_column),
        "model_horizon_state": ("model", "horizon", state_column),
        "model_horizon_regime": ("model", "horizon", regime_column),
        "model_horizon_time": ("model", "horizon", time_bucket_column),
    }
    for name, grouping in optional.items():
        if all(column in table.columns for column in grouping):
            candidates[name] = grouping
    return candidates


def _normalize_groupings(
    groupings: Mapping[str, Sequence[str]] | Sequence[Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    if isinstance(groupings, Mapping):
        return {str(name): tuple(str(column) for column in grouping) for name, grouping in groupings.items()}
    out: dict[str, tuple[str, ...]] = {}
    for grouping in groupings:
        values = tuple(str(column) for column in grouping)
        out["_".join(values) or "overall"] = values
    return out


def _validate_grouping(grouping: Sequence[str], table: pd.DataFrame, *, label: str) -> tuple[str, ...]:
    values = tuple(str(column) for column in grouping)
    missing = [column for column in values if column not in table.columns]
    if missing:
        raise ValueError(f"{label} column(s) are not in the forecast table: {missing}")
    return values


def _group_row(key: Any, groups: Sequence[str]) -> dict[str, Any]:
    if not groups:
        return {}
    values = (key,) if len(groups) == 1 and not isinstance(key, tuple) else tuple(key)
    return {group: value for group, value in zip(groups, values, strict=False)}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    return float(numerator / denominator)


def _resolve_rank_metric(
    rank_metric: str | None,
    scores: pd.DataFrame,
    metrics: Sequence[str | MetricLike],
) -> str | None:
    if rank_metric is not None:
        if rank_metric not in scores.columns:
            raise ValueError(f"rank_metric {rank_metric!r} is not present in scores")
        return rank_metric
    preferred = ["rmse", "mse", "mae", "r2_oos", "relative_mse"]
    for metric in preferred:
        if metric in scores.columns:
            return metric
    for metric in metrics:
        label = _metric_label(metric)
        if label in scores.columns:
            return label
    return None


def _benchmark_grouping(score_by: Sequence[str]) -> tuple[str, ...]:
    groups = tuple(column for column in score_by if column != "model")
    return ("model", *groups) if groups else ("model",)


def _regime_grouping(score_by: Sequence[str], regime_column: str) -> tuple[str, ...]:
    groups = tuple(score_by)
    return groups if regime_column in groups else (*groups, regime_column)


def _forecast_overview(table: pd.DataFrame) -> dict[str, Any]:
    models = sorted(str(value) for value in table.get("model", pd.Series(dtype=object)).dropna().unique())
    horizons = sorted(_json_ready(value) for value in table.get("horizon", pd.Series(dtype=object)).dropna().unique())
    return {
        "n_rows": int(table.shape[0]),
        "n_models": int(len(models)),
        "models": models,
        "horizons": horizons,
        "columns": [str(column) for column in table.columns],
    }


def _metric_label(metric: str | MetricLike) -> str:
    if callable(metric):
        return getattr(metric, "__name__", "callable_metric")
    return str(metric)


def _attach_metadata(frame: pd.DataFrame | None, metadata: Mapping[str, Any]) -> None:
    if frame is not None:
        frame.attrs["macroforecast_metadata"] = dict(metadata)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


__all__ = [
    "BENCHMARK_METRICS",
    "DEFAULT_METRICS",
    "DEFAULT_SCORE_BY",
    "EvaluationReport",
    "aggregate_scores",
    "benchmark_comparison",
    "evaluate_report",
    "regime_scores",
]
