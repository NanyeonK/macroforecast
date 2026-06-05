from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata
from macroforecast.forecasting import ForecastResult


LossMetric = Literal["mse", "rmse", "mae", "bias"]
UnsupportedWeights = Literal["skip", "nan", "raise"]
ResidualACFKind = Literal["acf"]
OriginView = Literal["all_origins", "last_origin_only", "every_n_origins"]
ScaleView = Literal["transformed_only", "back_transformed_only", "both_overlay"]

FORECAST_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "origin",
    "origin_pos",
    "horizon",
    "model",
    "prediction",
    "actual",
)
DEFAULT_GROUPS: tuple[str, ...] = ("model", "horizon")


@dataclass(frozen=True)
class ForecastDiagnosticReport:
    """Container returned by :func:`diagnose_forecasts`."""

    overview: dict[str, Any]
    fitted: pd.DataFrame | None = None
    residuals: pd.DataFrame | None = None
    residual_acf: pd.DataFrame | None = None
    residual_qq: pd.DataFrame | None = None
    rolling_loss: pd.DataFrame | None = None
    forecast_scale: pd.DataFrame | None = None
    coefficients: pd.DataFrame | None = None
    parameter_stability: pd.DataFrame | None = None
    training_loss: pd.DataFrame | None = None
    rolling_training_loss: pd.DataFrame | None = None
    first_vs_last: pd.DataFrame | None = None
    tuning: pd.DataFrame | None = None
    tuning_objective: pd.DataFrame | None = None
    hyperparameters: pd.DataFrame | None = None
    tuning_scores: pd.DataFrame | None = None
    ensemble_weights: pd.DataFrame | None = None
    ensemble_concentration: pd.DataFrame | None = None
    member_contribution: pd.DataFrame | None = None
    dfm_idiosyncratic_acf: pd.DataFrame | None = None
    dfm_factor_stability: pd.DataFrame | None = None
    stage_updates: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "overview": self.overview,
            "metadata": dict(self.metadata),
        }
        if self.fitted is not None:
            out["fitted"] = self.fitted.to_dict(orient="records")
        if self.residuals is not None:
            out["residuals"] = self.residuals.to_dict(orient="records")
        if self.residual_acf is not None:
            out["residual_acf"] = self.residual_acf.to_dict(orient="records")
        if self.residual_qq is not None:
            out["residual_qq"] = self.residual_qq.to_dict(orient="records")
        if self.rolling_loss is not None:
            out["rolling_loss"] = self.rolling_loss.to_dict(orient="records")
        if self.forecast_scale is not None:
            out["forecast_scale"] = self.forecast_scale.to_dict(orient="records")
        if self.coefficients is not None:
            out["coefficients"] = self.coefficients.to_dict(orient="records")
        if self.parameter_stability is not None:
            out["parameter_stability"] = self.parameter_stability.to_dict(orient="records")
        if self.training_loss is not None:
            out["training_loss"] = self.training_loss.to_dict(orient="records")
        if self.rolling_training_loss is not None:
            out["rolling_training_loss"] = self.rolling_training_loss.to_dict(orient="records")
        if self.first_vs_last is not None:
            out["first_vs_last"] = self.first_vs_last.to_dict(orient="records")
        if self.tuning is not None:
            out["tuning"] = self.tuning.to_dict(orient="records")
        if self.tuning_objective is not None:
            out["tuning_objective"] = self.tuning_objective.to_dict(orient="records")
        if self.hyperparameters is not None:
            out["hyperparameters"] = self.hyperparameters.to_dict(orient="records")
        if self.tuning_scores is not None:
            out["tuning_scores"] = self.tuning_scores.to_dict(orient="records")
        if self.ensemble_weights is not None:
            out["ensemble_weights"] = self.ensemble_weights.to_dict(orient="records")
        if self.ensemble_concentration is not None:
            out["ensemble_concentration"] = self.ensemble_concentration.to_dict(orient="records")
        if self.member_contribution is not None:
            out["member_contribution"] = self.member_contribution.to_dict(orient="records")
        if self.dfm_idiosyncratic_acf is not None:
            out["dfm_idiosyncratic_acf"] = self.dfm_idiosyncratic_acf.to_dict(orient="records")
        if self.dfm_factor_stability is not None:
            out["dfm_factor_stability"] = self.dfm_factor_stability.to_dict(orient="records")
        if self.stage_updates is not None:
            out["stage_updates"] = self.stage_updates.to_dict(orient="records")
        return out


def forecast_overview(forecasts: Any) -> dict[str, Any]:
    """Return compact shape, model, horizon, and metadata coverage counts."""

    table, metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    if table.empty:
        return {
            "n_forecasts": 0,
            "n_models": 0,
            "models": [],
            "horizons": [],
            "start": None,
            "end": None,
            "missing_prediction_count": 0,
            "missing_actual_count": 0,
            "combined_count": 0,
            "stored_model_count": 0,
            "selection_count": 0,
            "retuned_count": 0,
            "variance_prediction_count": 0,
            "quantile_prediction_count": 0,
            "metadata_keys": sorted(str(key) for key in metadata),
        }

    dates = pd.to_datetime(table["date"], errors="coerce")
    combined = _bool_series(table.get("combined"), len(table))
    stored_model = table.get("stored_model", pd.Series([None] * len(table)))
    selection = table.get("model_selection", pd.Series([None] * len(table)))
    return {
        "n_forecasts": int(len(table)),
        "n_models": int(table["model"].nunique(dropna=True)),
        "models": sorted(str(value) for value in table["model"].dropna().unique()),
        "horizons": sorted(_json_scalar(value) for value in table["horizon"].dropna().unique()),
        "start": _date_string(dates.min()),
        "end": _date_string(dates.max()),
        "missing_prediction_count": int(pd.to_numeric(table["prediction"], errors="coerce").isna().sum()),
        "missing_actual_count": int(pd.to_numeric(table["actual"], errors="coerce").isna().sum()),
        "combined_count": int(combined.sum()),
        "base_model_count": int((~combined).sum()),
        "stored_model_count": int(stored_model.map(_is_mapping).sum()),
        "selection_count": int(selection.map(_is_mapping).sum()),
        "retuned_count": int(selection.map(_selection_retuned).sum()),
        "variance_prediction_count": int(table.get("variance_prediction", pd.Series(dtype=object)).notna().sum()),
        "quantile_prediction_count": int(table.get("quantile_predictions", pd.Series(dtype=object)).map(_is_mapping).sum()),
        "metadata_keys": sorted(str(key) for key in metadata),
    }


def fitted_vs_actual(
    forecasts: Any,
    *,
    include_combined: bool = True,
    drop_missing_actual: bool = True,
) -> pd.DataFrame:
    """Return forecast rows with residual and error columns."""

    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    if table.empty:
        return _empty_frame(
            [
                "date",
                "origin",
                "origin_pos",
                "horizon",
                "model",
                "model_spec",
                "combined",
                "prediction",
                "actual",
                "residual",
                "abs_error",
                "squared_error",
                "percent_error",
                "variance_prediction",
                "quantile_predictions",
            ],
            kind="fitted_vs_actual",
        )
    work = table.copy()
    if not include_combined and "combined" in work:
        work = work.loc[~_bool_series(work["combined"], len(work))].copy()
    work["prediction"] = pd.to_numeric(work["prediction"], errors="coerce")
    work["actual"] = pd.to_numeric(work["actual"], errors="coerce")
    if drop_missing_actual:
        work = work.loc[work["prediction"].notna() & work["actual"].notna()].copy()
    work["residual"] = work["actual"] - work["prediction"]
    work["abs_error"] = work["residual"].abs()
    work["squared_error"] = work["residual"] ** 2
    denominator = work["actual"].replace(0.0, np.nan).abs()
    work["percent_error"] = work["residual"] / denominator
    columns = [
        "date",
        "origin",
        "origin_pos",
        "horizon",
        "model",
        "model_spec",
        "combined",
        "prediction",
        "actual",
        "residual",
        "abs_error",
        "squared_error",
        "percent_error",
        "variance_prediction",
        "quantile_predictions",
    ]
    out = work.loc[:, [column for column in columns if column in work]].reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "fitted_vs_actual", "version": 1}
    return out


def residual_report(
    forecasts: Any,
    *,
    group_by: Sequence[str] = DEFAULT_GROUPS,
    include_combined: bool = True,
) -> pd.DataFrame:
    """Return grouped out-of-sample residual diagnostics."""

    fitted = fitted_vs_actual(
        forecasts,
        include_combined=include_combined,
        drop_missing_actual=True,
    )
    groups = _normalize_group_by(group_by, fitted)
    if fitted.empty:
        return _empty_frame(
            list(groups)
            + [
                "n",
                "bias",
                "mae",
                "mse",
                "rmse",
                "residual_sd",
                "residual_autocorr1",
                "mean_actual",
                "mean_prediction",
                "first_date",
                "last_date",
            ],
            kind="residual_report",
        )
    rows: list[dict[str, Any]] = []
    grouped = fitted.groupby(list(groups), dropna=False, sort=True) if groups else [((), fitted)]
    for key, group in grouped:
        # Residual diagnostics are time-series diagnostics. Sort by the same
        # origin/date order used by residual_autocorrelation so the lag-1
        # summary is invariant to input row order.
        group = group.sort_values(["origin_pos", "date"])
        values = group["residual"].dropna().astype(float)
        row = _group_row(key, groups)
        row.update(
            {
                "n": int(values.shape[0]),
                "bias": _float_or_none(values.mean()),
                "mae": _float_or_none(group["abs_error"].mean()),
                "mse": _float_or_none(group["squared_error"].mean()),
                "rmse": _float_or_none(np.sqrt(group["squared_error"].mean())),
                "residual_sd": _float_or_none(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "residual_autocorr1": _acf_value(values, 1),
                "mean_actual": _float_or_none(group["actual"].mean()),
                "mean_prediction": _float_or_none(group["prediction"].mean()),
                "first_date": _date_string(pd.to_datetime(group["date"], errors="coerce").min()),
                "last_date": _date_string(pd.to_datetime(group["date"], errors="coerce").max()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["macroforecast_metadata_schema"] = {"kind": "residual_report", "version": 1}
    return out


def residual_autocorrelation(
    forecasts: Any,
    *,
    max_lag: int = 12,
    group_by: Sequence[str] = DEFAULT_GROUPS,
    include_combined: bool = True,
) -> pd.DataFrame:
    """Return residual autocorrelation by model/horizon group."""

    max_lag_value = int(max_lag)
    if max_lag_value < 0:
        raise ValueError("max_lag must be non-negative")
    fitted = fitted_vs_actual(
        forecasts,
        include_combined=include_combined,
        drop_missing_actual=True,
    )
    groups = _normalize_group_by(group_by, fitted)
    columns = list(groups) + ["lag", "acf", "n"]
    if fitted.empty:
        return _empty_frame(columns, kind="residual_autocorrelation")
    rows: list[dict[str, Any]] = []
    grouped = fitted.groupby(list(groups), dropna=False, sort=True) if groups else [((), fitted)]
    for key, group in grouped:
        values = group.sort_values(["origin_pos", "date"])["residual"].dropna().astype(float)
        for lag in range(max_lag_value + 1):
            row = _group_row(key, groups)
            row.update({"lag": int(lag), "acf": _acf_value(values, lag), "n": int(values.shape[0])})
            rows.append(row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "residual_autocorrelation",
        "version": 1,
        "max_lag": max_lag_value,
    }
    return out


def residual_qq(
    forecasts: Any,
    *,
    n_quantiles: int = 21,
    group_by: Sequence[str] = DEFAULT_GROUPS,
    include_combined: bool = True,
) -> pd.DataFrame:
    """Return residual QQ table against a fitted normal reference."""

    n_value = int(n_quantiles)
    if n_value < 3:
        raise ValueError("n_quantiles must be at least 3")
    fitted = fitted_vs_actual(
        forecasts,
        include_combined=include_combined,
        drop_missing_actual=True,
    )
    groups = _normalize_group_by(group_by, fitted)
    columns = list(groups) + ["probability", "sample_quantile", "normal_quantile", "n"]
    if fitted.empty:
        return _empty_frame(columns, kind="residual_qq")
    probs = np.linspace(1.0 / (n_value + 1), n_value / (n_value + 1), n_value)
    rows: list[dict[str, Any]] = []
    normal = NormalDist()
    grouped = fitted.groupby(list(groups), dropna=False, sort=True) if groups else [((), fitted)]
    for key, group in grouped:
        values = group["residual"].dropna().astype(float)
        mean = float(values.mean()) if len(values) else 0.0
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for prob in probs:
            row = _group_row(key, groups)
            row.update(
                {
                    "probability": float(prob),
                    "sample_quantile": _float_or_none(values.quantile(float(prob))) if len(values) else None,
                    "normal_quantile": mean + sd * normal.inv_cdf(float(prob)) if sd > 0 else mean,
                    "n": int(values.shape[0]),
                }
            )
            rows.append(row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "residual_qq",
        "version": 1,
        "n_quantiles": n_value,
    }
    return out


def rolling_loss(
    forecasts: Any,
    *,
    metric: LossMetric = "rmse",
    window: int = 12,
    min_periods: int | None = None,
    group_by: Sequence[str] = DEFAULT_GROUPS,
    include_combined: bool = True,
) -> pd.DataFrame:
    """Return rolling out-of-sample loss by model and horizon."""

    metric_value = _normalize_loss_metric(metric)
    window_value = int(window)
    if window_value < 1:
        raise ValueError("window must be a positive integer")
    min_value = window_value if min_periods is None else int(min_periods)
    if min_value < 1:
        raise ValueError("min_periods must be a positive integer")
    fitted = fitted_vs_actual(
        forecasts,
        include_combined=include_combined,
        drop_missing_actual=True,
    )
    groups = _normalize_group_by(group_by, fitted)
    columns = list(groups) + ["date", "origin", "origin_pos", "n_window", f"rolling_{metric_value}"]
    if fitted.empty:
        return _empty_frame(columns, kind="rolling_loss")
    loss_column = _loss_values(fitted, metric_value)
    fitted = fitted.assign(_loss=loss_column)
    fitted = fitted.sort_values([*groups, "date", "origin_pos"] if groups else ["date", "origin_pos"])
    rows: list[dict[str, Any]] = []
    grouped = fitted.groupby(list(groups), dropna=False, sort=True) if groups else [((), fitted)]
    for key, group in grouped:
        values = group["_loss"].astype(float)
        if metric_value == "rmse":
            rolled = values.rolling(window_value, min_periods=min_value).mean().pow(0.5)
        else:
            rolled = values.rolling(window_value, min_periods=min_value).mean()
        counts = values.rolling(window_value, min_periods=min_value).count()
        for (_, row), score, n_window in zip(group.iterrows(), rolled, counts, strict=True):
            out_row = _group_row(key, groups)
            out_row.update(
                {
                    "date": row["date"],
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "n_window": None if pd.isna(n_window) else int(n_window),
                    f"rolling_{metric_value}": _float_or_none(score),
                }
            )
            rows.append(out_row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "rolling_loss",
        "version": 1,
        "metric": metric_value,
        "window": window_value,
        "min_periods": min_value,
    }
    return out


def forecast_scale_view(
    forecasts: Any,
    *,
    levels: Any | None = None,
    target: str | None = None,
    transform: str | None = None,
    view: ScaleView = "both_overlay",
    back_transform: Callable[..., Any] | None = None,
    include_combined: bool = True,
) -> pd.DataFrame:
    """Return transformed and back-transformed forecast rows when possible."""

    if view not in {"transformed_only", "back_transformed_only", "both_overlay"}:
        raise ValueError("view must be 'transformed_only', 'back_transformed_only', or 'both_overlay'")
    table, metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    if not include_combined and "combined" in table:
        table = table.loc[~_bool_series(table["combined"], len(table))].copy()
    level_series = _coerce_level_series(levels, target=target) if levels is not None else None
    default_transform = transform or _metadata_target_transform(metadata) or "level"
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        row_transform = str(row.get("target_transform") or default_transform)
        base = _forecast_row_base(row)
        if view in {"transformed_only", "both_overlay"}:
            rows.append(
                {
                    **base,
                    "scale": "transformed",
                    "target_transform": row_transform,
                    "prediction": _float_or_none(row.get("prediction")),
                    "actual": _float_or_none(row.get("actual")),
                    "residual": _residual_value(row.get("prediction"), row.get("actual")),
                    "back_transform_available": row_transform == "level",
                }
            )
        if view in {"back_transformed_only", "both_overlay"}:
            pred_level, actual_level, available = _back_transformed_values(
                row,
                row_transform,
                levels=level_series,
                back_transform=back_transform,
            )
            rows.append(
                {
                    **base,
                    "scale": "back_transformed",
                    "target_transform": row_transform,
                    "prediction": pred_level,
                    "actual": actual_level,
                    "residual": _residual_value(pred_level, actual_level),
                    "back_transform_available": bool(available),
                }
            )
    columns = [
        "date",
        "origin",
        "origin_pos",
        "horizon",
        "forecast_policy",
        "target",
        "model",
        "model_spec",
        "combined",
        "scale",
        "target_transform",
        "prediction",
        "actual",
        "residual",
        "back_transform_available",
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=columns)
    else:
        out = out.loc[:, [column for column in columns if column in out]]
    out.attrs["macroforecast_metadata_schema"] = {"kind": "forecast_scale_view", "version": 1, "view": view}
    return out


def select_forecast_origins(
    forecasts: Any,
    *,
    view: OriginView = "all_origins",
    every_n: int = 12,
    include_last: bool = True,
    include_combined: bool = True,
) -> pd.DataFrame:
    """Return a forecast table filtered to a requested origin view."""

    if view not in {"all_origins", "last_origin_only", "every_n_origins"}:
        raise ValueError("view must be 'all_origins', 'last_origin_only', or 'every_n_origins'")
    step = int(every_n)
    if step < 1:
        raise ValueError("every_n must be a positive integer")
    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    if not include_combined and "combined" in table:
        table = table.loc[~_bool_series(table["combined"], len(table))].copy()
    if table.empty or view == "all_origins":
        out = table.reset_index(drop=True)
    else:
        origin_pos = pd.to_numeric(table["origin_pos"], errors="coerce")
        valid_positions = origin_pos.dropna().astype(int)
        if valid_positions.empty:
            out = table.iloc[0:0].copy()
        elif view == "last_origin_only":
            out = table.loc[origin_pos == valid_positions.max()].copy()
        else:
            first = int(valid_positions.min())
            keep = ((origin_pos - first) % step == 0).fillna(False)
            if include_last:
                keep = keep | (origin_pos == valid_positions.max())
            out = table.loc[keep].copy()
    out = out.reset_index(drop=True)
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "select_forecast_origins",
        "version": 1,
        "view": view,
        "every_n": step,
        "include_last": bool(include_last),
    }
    return out


def first_vs_last_forecast(
    forecasts: Any,
    *,
    group_by: Sequence[str] = DEFAULT_GROUPS,
    include_combined: bool = True,
) -> pd.DataFrame:
    """Compare the first and last forecast row inside each group."""

    fitted = fitted_vs_actual(
        forecasts,
        include_combined=include_combined,
        drop_missing_actual=False,
    )
    groups = _normalize_group_by(group_by, fitted) if not fitted.empty else tuple(str(column) for column in group_by)
    columns = list(groups) + [
        "first_date",
        "first_origin",
        "first_origin_pos",
        "first_prediction",
        "first_actual",
        "first_residual",
        "last_date",
        "last_origin",
        "last_origin_pos",
        "last_prediction",
        "last_actual",
        "last_residual",
        "prediction_change",
        "actual_change",
        "residual_change",
    ]
    if fitted.empty:
        return _empty_frame(columns, kind="first_vs_last_forecast")
    rows: list[dict[str, Any]] = []
    sorted_frame = fitted.sort_values([*groups, "origin_pos", "date"] if groups else ["origin_pos", "date"])
    grouped = sorted_frame.groupby(list(groups), dropna=False, sort=True) if groups else [((), sorted_frame)]
    for key, group in grouped:
        first = group.iloc[0]
        last = group.iloc[-1]
        first_pred = _float_or_none(first.get("prediction"))
        last_pred = _float_or_none(last.get("prediction"))
        first_actual = _float_or_none(first.get("actual"))
        last_actual = _float_or_none(last.get("actual"))
        first_resid = _float_or_none(first.get("residual"))
        last_resid = _float_or_none(last.get("residual"))
        row = _group_row(key, groups)
        row.update(
            {
                "first_date": first.get("date"),
                "first_origin": first.get("origin"),
                "first_origin_pos": first.get("origin_pos"),
                "first_prediction": first_pred,
                "first_actual": first_actual,
                "first_residual": first_resid,
                "last_date": last.get("date"),
                "last_origin": last.get("origin"),
                "last_origin_pos": last.get("origin_pos"),
                "last_prediction": last_pred,
                "last_actual": last_actual,
                "last_residual": last_resid,
                "prediction_change": _difference_or_none(last_pred, first_pred),
                "actual_change": _difference_or_none(last_actual, first_actual),
                "residual_change": _difference_or_none(last_resid, first_resid),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {"kind": "first_vs_last_forecast", "version": 1}
    return out


def training_loss_trace(
    forecasts: Any,
    *,
    load_pickle: bool = False,
) -> pd.DataFrame:
    """Read saved model sidecars and return in-sample fit metrics by origin."""

    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    rows: list[dict[str, Any]] = []
    for context, stored, diagnostics in _iter_sidecar_diagnostics(table, load_pickle=load_pickle):
        metrics = _mapping_or_none(diagnostics.get("metrics"))
        if metrics is None:
            continue
        for metric, value in metrics.items():
            rows.append(
                {
                    "date": context.get("date"),
                    "origin": context.get("origin"),
                    "origin_pos": context.get("origin_pos"),
                    "horizon": context.get("horizon"),
                    "model": context.get("model"),
                    "model_spec": context.get("model_spec"),
                    "fit_step": context.get("fit_step"),
                    "metric": str(metric),
                    "value": _float_or_none(value),
                    "stored_metadata_path": stored.get("metadata_path"),
                    "stored_model_path": stored.get("model_path"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "date",
                "origin",
                "origin_pos",
                "horizon",
                "model",
                "model_spec",
                "fit_step",
                "metric",
                "value",
                "stored_metadata_path",
                "stored_model_path",
            ]
        )
    out.attrs["macroforecast_metadata_schema"] = {"kind": "training_loss_trace", "version": 1}
    return out


def rolling_training_loss(
    forecasts_or_trace: Any,
    *,
    metric: str = "rmse",
    window: int = 12,
    min_periods: int | None = None,
    group_by: Sequence[str] = DEFAULT_GROUPS,
    load_pickle: bool = False,
) -> pd.DataFrame:
    """Return a rolling trace of saved in-sample training metrics."""

    metric_value = str(metric)
    window_value = int(window)
    if window_value < 1:
        raise ValueError("window must be a positive integer")
    min_value = window_value if min_periods is None else int(min_periods)
    if min_value < 1:
        raise ValueError("min_periods must be a positive integer")
    trace = _coerce_training_loss_input(forecasts_or_trace, load_pickle=load_pickle)
    groups = _normalize_group_by(group_by, trace) if not trace.empty else tuple(str(column) for column in group_by)
    columns = list(groups) + ["date", "origin", "origin_pos", "metric", "n_window", f"rolling_{metric_value}"]
    if trace.empty:
        return _empty_frame(columns, kind="rolling_training_loss")
    work = trace.loc[trace["metric"].astype(str) == metric_value].copy()
    if work.empty:
        return _empty_frame(columns, kind="rolling_training_loss")
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.sort_values([*groups, "origin_pos", "date"] if groups else ["origin_pos", "date"])
    rows: list[dict[str, Any]] = []
    grouped = work.groupby(list(groups), dropna=False, sort=True) if groups else [((), work)]
    for key, group in grouped:
        values = group["value"].astype(float)
        rolled = values.rolling(window_value, min_periods=min_value).mean()
        counts = values.rolling(window_value, min_periods=min_value).count()
        for (_, row), score, n_window in zip(group.iterrows(), rolled, counts, strict=True):
            out_row = _group_row(key, groups)
            out_row.update(
                {
                    "date": row.get("date"),
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "metric": metric_value,
                    "n_window": None if pd.isna(n_window) else int(n_window),
                    f"rolling_{metric_value}": _float_or_none(score),
                }
            )
            rows.append(out_row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "rolling_training_loss",
        "version": 1,
        "metric": metric_value,
        "window": window_value,
        "min_periods": min_value,
    }
    return out


def dfm_idiosyncratic_acf(
    source: Any,
    *,
    max_lag: int = 12,
    load_pickle: bool = False,
) -> pd.DataFrame:
    """Return ACF diagnostics for DFM idiosyncratic residual series."""

    max_lag_value = int(max_lag)
    if max_lag_value < 0:
        raise ValueError("max_lag must be non-negative")
    columns = [
        "date",
        "origin",
        "origin_pos",
        "horizon",
        "model",
        "model_spec",
        "fit_step",
        "residual",
        "lag",
        "acf",
        "n",
    ]
    rows: list[dict[str, Any]] = []
    for context, diagnostics in _iter_diagnostics(source, load_pickle=load_pickle):
        residuals = _diagnostic_frame_or_series(diagnostics.get("residuals"))
        if residuals is None:
            continue
        frame = residuals.to_frame("residual") if isinstance(residuals, pd.Series) else residuals
        for residual_name in frame.columns:
            values = pd.to_numeric(frame[residual_name], errors="coerce").dropna()
            for lag in range(max_lag_value + 1):
                rows.append(
                    {
                        **context,
                        "residual": str(residual_name),
                        "lag": int(lag),
                        "acf": _acf_value(values, lag),
                        "n": int(values.shape[0]),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=columns)
    else:
        out = out.loc[:, [column for column in columns if column in out]]
    out.attrs["macroforecast_metadata_schema"] = {
        "kind": "dfm_idiosyncratic_acf",
        "version": 1,
        "max_lag": max_lag_value,
    }
    return out


def dfm_factor_stability(
    source: Any,
    *,
    load_pickle: bool = False,
) -> pd.DataFrame:
    """Summarize filtered DFM factors saved in fit diagnostics."""

    columns = [
        "date",
        "origin",
        "origin_pos",
        "horizon",
        "model",
        "model_spec",
        "fit_step",
        "factor",
        "n",
        "mean",
        "sd",
        "variance",
        "first",
        "last",
        "drift",
        "autocorr1",
    ]
    rows: list[dict[str, Any]] = []
    for context, diagnostics in _iter_diagnostics(source, load_pickle=load_pickle):
        factors = _diagnostic_frame_or_series(diagnostics.get("factors_filtered"))
        if factors is None:
            continue
        frame = factors.to_frame("factor_1") if isinstance(factors, pd.Series) else factors
        for factor_name in frame.columns:
            values = pd.to_numeric(frame[factor_name], errors="coerce").dropna().astype(float)
            if values.empty:
                continue
            first = _float_or_none(values.iloc[0])
            last = _float_or_none(values.iloc[-1])
            rows.append(
                {
                    **context,
                    "factor": str(factor_name),
                    "n": int(values.shape[0]),
                    "mean": _float_or_none(values.mean()),
                    "sd": _float_or_none(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "variance": _float_or_none(values.var(ddof=1)) if len(values) > 1 else 0.0,
                    "first": first,
                    "last": last,
                    "drift": _difference_or_none(last, first),
                    "autocorr1": _acf_value(values, 1),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=columns)
    else:
        out = out.loc[:, [column for column in columns if column in out]]
    out.attrs["macroforecast_metadata_schema"] = {"kind": "dfm_factor_stability", "version": 1}
    return out


def coefficient_trace(
    forecasts: Any,
    *,
    include_intercept: bool = True,
    load_pickle: bool = False,
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read saved fit sidecars and return coefficient paths over origins."""

    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    if models is not None:
        selected = {str(model) for model in models}
        table = table.loc[table["model"].astype(str).isin(selected)].copy()
    rows: list[dict[str, Any]] = []
    for context, stored, diagnostics in _iter_sidecar_diagnostics(
        table,
        load_pickle=load_pickle,
    ):
        coefficients = _coefficient_records(diagnostics.get("coefficients"))
        if include_intercept and "intercept" in diagnostics:
            coefficients.append({"feature": "intercept", "coefficient": _scalar_or_json(diagnostics["intercept"])})
        for item in coefficients:
            rows.append(
                {
                    "date": context.get("date"),
                    "origin": context.get("origin"),
                    "origin_pos": context.get("origin_pos"),
                    "horizon": context.get("horizon"),
                    "model": context.get("model"),
                    "model_spec": context.get("model_spec"),
                    "fit_step": context.get("fit_step"),
                    "feature": item.get("feature"),
                    "coefficient": item.get("coefficient"),
                    "component": item.get("component"),
                    "stored_metadata_path": stored.get("metadata_path"),
                    "stored_model_path": stored.get("model_path"),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "date",
                "origin",
                "origin_pos",
                "horizon",
                "model",
                "model_spec",
                "fit_step",
                "feature",
                "coefficient",
                "component",
                "stored_metadata_path",
                "stored_model_path",
            ]
        )
    out.attrs["macroforecast_metadata_schema"] = {"kind": "coefficient_trace", "version": 1}
    return out


def parameter_stability(
    forecasts: Any,
    *,
    include_intercept: bool = True,
    load_pickle: bool = False,
    group_by: Sequence[str] = ("model", "horizon", "feature"),
    models: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Summarize coefficient drift and sign changes over forecast origins."""

    trace = coefficient_trace(
        forecasts,
        include_intercept=include_intercept,
        load_pickle=load_pickle,
        models=models,
    )
    groups = _normalize_group_by(group_by, trace) if not trace.empty else tuple(str(column) for column in group_by)
    columns = list(groups) + [
        "n",
        "mean",
        "sd",
        "min",
        "max",
        "first",
        "last",
        "drift",
        "abs_drift",
        "sign_changes",
    ]
    if trace.empty:
        return _empty_frame(columns, kind="parameter_stability")
    work = trace.copy()
    work["coefficient"] = pd.to_numeric(work["coefficient"], errors="coerce")
    work = work.loc[work["coefficient"].notna()].sort_values([*groups, "origin_pos"])
    rows: list[dict[str, Any]] = []
    grouped = work.groupby(list(groups), dropna=False, sort=True) if groups else [((), work)]
    for key, group in grouped:
        values = group["coefficient"].astype(float)
        signs = np.sign(values.to_numpy(dtype=float))
        signs = signs[signs != 0]
        row = _group_row(key, groups)
        first = _float_or_none(values.iloc[0])
        last = _float_or_none(values.iloc[-1])
        drift = None if first is None or last is None else last - first
        row.update(
            {
                "n": int(values.shape[0]),
                "mean": _float_or_none(values.mean()),
                "sd": _float_or_none(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": _float_or_none(values.min()),
                "max": _float_or_none(values.max()),
                "first": first,
                "last": last,
                "drift": drift,
                "abs_drift": None if drift is None else abs(float(drift)),
                "sign_changes": int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {"kind": "parameter_stability", "version": 1}
    return out


def tuning_trace(forecasts: Any) -> pd.DataFrame:
    """Return one row per forecast row carrying parameter-selection metadata."""

    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        selection = _mapping_or_none(row.get("model_selection"))
        if selection is None:
            continue
        rows.append(
            {
                "date": row.get("date"),
                "origin": row.get("origin"),
                "origin_pos": row.get("origin_pos"),
                "horizon": row.get("horizon"),
                "model": row.get("model"),
                "model_spec": row.get("model_spec"),
                "method": selection.get("method"),
                "metric": selection.get("metric"),
                "window": selection.get("window"),
                "retuned": bool(selection.get("retuned", False)),
                "best_score": _float_or_none(selection.get("best_score")),
                "best_params": selection.get("best_params"),
                "n_trials": _int_or_none(selection.get("n_trials")),
                "n_successful": _int_or_none(selection.get("n_successful")),
                "n_failed": _int_or_none(selection.get("n_failed")),
                "policy": selection.get("policy"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "date",
                "origin",
                "origin_pos",
                "horizon",
                "model",
                "model_spec",
                "method",
                "metric",
                "window",
                "retuned",
                "best_score",
                "best_params",
                "n_trials",
                "n_successful",
                "n_failed",
                "policy",
            ]
        )
    out.attrs["macroforecast_metadata_schema"] = {"kind": "tuning_trace", "version": 1}
    return out


def tuning_objective_trace(forecasts: Any) -> pd.DataFrame:
    """Return best validation objective over origins for tuned models."""

    trace = tuning_trace(forecasts)
    columns = [
        "date",
        "origin",
        "origin_pos",
        "horizon",
        "model",
        "model_spec",
        "method",
        "metric",
        "window",
        "best_score",
        "retuned",
        "n_trials",
        "n_successful",
        "n_failed",
        "policy",
    ]
    if trace.empty:
        return _empty_frame(columns, kind="tuning_objective_trace")
    out = trace.loc[:, columns].copy()
    out.attrs["macroforecast_metadata_schema"] = {"kind": "tuning_objective_trace", "version": 1}
    return out


def hyperparameter_path(forecasts: Any) -> pd.DataFrame:
    """Return long-form selected hyperparameters by origin."""

    trace = tuning_trace(forecasts)
    rows: list[dict[str, Any]] = []
    for _, row in trace.iterrows():
        params = _mapping_or_none(row.get("best_params"))
        if params is None:
            continue
        for name, value in params.items():
            rows.append(
                {
                    "date": row.get("date"),
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": row.get("horizon"),
                    "model": row.get("model"),
                    "model_spec": row.get("model_spec"),
                    "method": row.get("method"),
                    "parameter": str(name),
                    "value": _json_scalar(value),
                    "numeric_value": _float_or_none(value),
                    "retuned": bool(row.get("retuned", False)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["date", "origin", "origin_pos", "horizon", "model", "model_spec", "method", "parameter", "value", "numeric_value", "retuned"])
    out.attrs["macroforecast_metadata_schema"] = {"kind": "hyperparameter_path", "version": 1}
    return out


def tuning_score_distribution(
    forecasts: Any,
    *,
    group_by: Sequence[str] = ("model", "horizon", "method"),
) -> pd.DataFrame:
    """Summarize the distribution of selected validation scores over origins."""

    trace = tuning_trace(forecasts)
    groups = _normalize_group_by(group_by, trace) if not trace.empty else tuple(str(column) for column in group_by)
    columns = list(groups) + ["n", "mean", "sd", "min", "q25", "median", "q75", "max"]
    if trace.empty:
        return _empty_frame(columns, kind="tuning_score_distribution")
    work = trace.copy()
    work["best_score"] = pd.to_numeric(work["best_score"], errors="coerce")
    work = work.loc[work["best_score"].notna()]
    rows: list[dict[str, Any]] = []
    grouped = work.groupby(list(groups), dropna=False, sort=True) if groups else [((), work)]
    for key, group in grouped:
        values = group["best_score"].astype(float)
        row = _group_row(key, groups)
        row.update(
            {
                "n": int(values.shape[0]),
                "mean": _float_or_none(values.mean()),
                "sd": _float_or_none(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": _float_or_none(values.min()),
                "q25": _float_or_none(values.quantile(0.25)),
                "median": _float_or_none(values.quantile(0.5)),
                "q75": _float_or_none(values.quantile(0.75)),
                "max": _float_or_none(values.max()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {"kind": "tuning_score_distribution", "version": 1}
    return out


def ensemble_weights_over_time(
    forecasts: Any,
    *,
    unsupported: UnsupportedWeights = "skip",
) -> pd.DataFrame:
    """Reconstruct combination weights when the method has identifiable weights."""

    if unsupported not in {"skip", "nan", "raise"}:
        raise ValueError("unsupported must be one of 'skip', 'nan', or 'raise'")
    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    if table.empty or "combination" not in table:
        return _empty_weight_frame()
    base = table.loc[~_bool_series(table.get("combined"), len(table))].copy()
    combined = table.loc[_bool_series(table.get("combined"), len(table))].copy()
    rows: list[dict[str, Any]] = []
    for _, combined_row in combined.iterrows():
        spec = _mapping_or_none(combined_row.get("combination"))
        if spec is None:
            continue
        method = str(spec.get("method", ""))
        name = str(spec.get("name", combined_row.get("model")))
        selected_models = spec.get("models")
        group = base.loc[
            (base["horizon"] == combined_row.get("horizon"))
            & (base["date"] == combined_row.get("date"))
            & (base["origin_pos"] == combined_row.get("origin_pos"))
        ].copy()
        if selected_models is not None:
            selected = {str(model) for model in selected_models}
            group = group.loc[group["model"].astype(str).isin(selected)]
        if group.empty:
            continue
        weights = _combination_weights_for_row(
            base,
            combined_row,
            group,
            spec,
            unsupported=unsupported,
        )
        for model, weight in weights.items():
            rows.append(
                {
                    "date": combined_row.get("date"),
                    "origin": combined_row.get("origin"),
                    "origin_pos": combined_row.get("origin_pos"),
                    "horizon": combined_row.get("horizon"),
                    "combination": name,
                    "method": method,
                    "model": model,
                    "weight": weight,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_weight_frame()
    out.attrs["macroforecast_metadata_schema"] = {"kind": "ensemble_weights_over_time", "version": 1}
    return out


def ensemble_weight_concentration(forecasts: Any) -> pd.DataFrame:
    """Return concentration metrics for identifiable forecast-combination weights."""

    weights = ensemble_weights_over_time(forecasts, unsupported="skip")
    groups = ("combination", "method", "date", "origin_pos", "horizon")
    columns = list(groups) + ["n_members", "min_weight", "max_weight", "hhi", "effective_n", "entropy"]
    if weights.empty:
        return _empty_frame(columns, kind="ensemble_weight_concentration")
    rows: list[dict[str, Any]] = []
    for key, group in weights.groupby(list(groups), dropna=False, sort=True):
        values = pd.to_numeric(group["weight"], errors="coerce").dropna().astype(float)
        positive = values.loc[values > 0]
        hhi = float((values**2).sum()) if len(values) else None
        row = _group_row(key, groups)
        row.update(
            {
                "n_members": int(values.shape[0]),
                "min_weight": _float_or_none(values.min()) if len(values) else None,
                "max_weight": _float_or_none(values.max()) if len(values) else None,
                "hhi": hhi,
                "effective_n": None if not hhi or hhi <= 0 else float(1.0 / hhi),
                "entropy": _float_or_none(-(positive * np.log(positive)).sum()) if len(positive) else None,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows).loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {"kind": "ensemble_weight_concentration", "version": 1}
    return out


def ensemble_member_contribution(forecasts: Any) -> pd.DataFrame:
    """Return member-level weighted forecast contributions for combinations."""

    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    weights = ensemble_weights_over_time(table, unsupported="skip")
    columns = [
        "date",
        "origin",
        "origin_pos",
        "horizon",
        "combination",
        "method",
        "model",
        "weight",
        "member_prediction",
        "contribution",
        "combined_prediction",
    ]
    if weights.empty:
        return _empty_frame(columns, kind="ensemble_member_contribution")
    base = table.loc[~_bool_series(table.get("combined"), len(table))].copy()
    combined = table.loc[_bool_series(table.get("combined"), len(table))].copy()
    combined_lookup = {
        (row.get("date"), row.get("origin_pos"), row.get("horizon"), str(row.get("model"))): row.get("prediction")
        for _, row in combined.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for _, weight_row in weights.iterrows():
        member = base.loc[
            (base["date"] == weight_row["date"])
            & (base["origin_pos"] == weight_row["origin_pos"])
            & (base["horizon"] == weight_row["horizon"])
            & (base["model"].astype(str) == str(weight_row["model"]))
        ]
        if member.empty:
            continue
        member_prediction = _float_or_none(member.iloc[0]["prediction"])
        weight = _float_or_none(weight_row["weight"])
        combined_prediction = _float_or_none(
            combined_lookup.get(
                (
                    weight_row["date"],
                    weight_row["origin_pos"],
                    weight_row["horizon"],
                    str(weight_row["combination"]),
                )
            )
        )
        rows.append(
            {
                "date": weight_row["date"],
                "origin": member.iloc[0].get("origin"),
                "origin_pos": weight_row["origin_pos"],
                "horizon": weight_row["horizon"],
                "combination": weight_row["combination"],
                "method": weight_row["method"],
                "model": weight_row["model"],
                "weight": weight,
                "member_prediction": member_prediction,
                "contribution": None if weight is None or member_prediction is None else weight * member_prediction,
                "combined_prediction": combined_prediction,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=columns)
    else:
        out = out.loc[:, columns]
    out.attrs["macroforecast_metadata_schema"] = {"kind": "ensemble_member_contribution", "version": 1}
    return out


def stage_update_trace(forecasts: Any) -> pd.DataFrame:
    """Return stage update records saved by the forecasting runner."""

    _table, metadata = _coerce_forecast_input(forecasts)
    rows: list[dict[str, Any]] = []
    for record in metadata.get("stages", []) or []:
        if not isinstance(record, Mapping):
            continue
        rows.append(
            {
                "stage": record.get("stage"),
                "origin": record.get("origin"),
                "origin_pos": record.get("origin_pos"),
                "updated": bool(record.get("updated", False)),
                "fit_start": record.get("fit_start"),
                "fit_end": record.get("fit_end"),
                "test_start": record.get("test_start"),
                "test_end": record.get("test_end"),
                "metadata_keys": sorted(str(key) for key in (record.get("metadata") or {})),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "stage",
                "origin",
                "origin_pos",
                "updated",
                "fit_start",
                "fit_end",
                "test_start",
                "test_end",
                "metadata_keys",
            ]
        )
    out.attrs["macroforecast_metadata_schema"] = {"kind": "stage_update_trace", "version": 1}
    return out


def diagnose_forecasts(
    forecasts: Any,
    *,
    include_fitted: bool = True,
    include_residuals: bool = True,
    include_residual_acf: bool = False,
    include_residual_qq: bool = False,
    include_rolling_loss: bool = True,
    rolling_window: int = 12,
    rolling_metric: LossMetric = "rmse",
    include_forecast_scale: bool = False,
    levels: Any | None = None,
    scale_view: ScaleView = "both_overlay",
    back_transform: Callable[..., Any] | None = None,
    include_training_loss: bool = False,
    include_rolling_training_loss: bool = False,
    training_loss_metric: str = "rmse",
    include_first_vs_last: bool = False,
    include_dfm_idiosyncratic_acf: bool = False,
    include_dfm_factor_stability: bool = False,
    include_coefficients: bool = True,
    include_parameter_stability: bool = True,
    include_tuning: bool = True,
    include_tuning_objective: bool = True,
    include_hyperparameters: bool = True,
    include_tuning_scores: bool = True,
    include_ensemble_weights: bool = True,
    include_ensemble_concentration: bool = True,
    include_member_contribution: bool = False,
    include_stage_updates: bool = True,
    include_combined: bool = True,
) -> ForecastDiagnosticReport:
    """Run the standard forecast diagnostics on a ForecastResult or table."""

    table, base_metadata = _coerce_forecast_input(forecasts)
    overview = forecast_overview(ForecastResult(table, metadata=base_metadata))
    fitted = fitted_vs_actual(table, include_combined=include_combined) if include_fitted else None
    residuals = residual_report(table, include_combined=include_combined) if include_residuals else None
    residual_acf = residual_autocorrelation(table, include_combined=include_combined) if include_residual_acf else None
    qq = residual_qq(table, include_combined=include_combined) if include_residual_qq else None
    rolling = (
        rolling_loss(
            table,
            metric=rolling_metric,
            window=rolling_window,
            include_combined=include_combined,
        )
        if include_rolling_loss
        else None
    )
    scale = (
        forecast_scale_view(
            ForecastResult(table, metadata=base_metadata),
            levels=levels,
            view=scale_view,
            back_transform=back_transform,
            include_combined=include_combined,
        )
        if include_forecast_scale
        else None
    )
    train_loss = training_loss_trace(table) if include_training_loss else None
    rolling_train_loss = (
        rolling_training_loss(
            train_loss if train_loss is not None else table,
            metric=training_loss_metric,
            window=rolling_window,
        )
        if include_rolling_training_loss
        else None
    )
    first_last = first_vs_last_forecast(table, include_combined=include_combined) if include_first_vs_last else None
    dfm_acf = dfm_idiosyncratic_acf(table) if include_dfm_idiosyncratic_acf else None
    dfm_factors = dfm_factor_stability(table) if include_dfm_factor_stability else None
    coefficients = coefficient_trace(table) if include_coefficients else None
    stability = parameter_stability(table) if include_parameter_stability else None
    tuning = tuning_trace(table) if include_tuning else None
    objective = tuning_objective_trace(table) if include_tuning_objective else None
    hyperparams = hyperparameter_path(table) if include_hyperparameters else None
    tuning_scores = tuning_score_distribution(table) if include_tuning_scores else None
    weights = (
        ensemble_weights_over_time(table)
        if include_ensemble_weights
        else None
    )
    concentration = ensemble_weight_concentration(table) if include_ensemble_concentration else None
    contribution = ensemble_member_contribution(table) if include_member_contribution else None
    stages = stage_update_trace(ForecastResult(table, metadata=base_metadata)) if include_stage_updates else None
    stage = {
        "overview": {
            "n_forecasts": overview["n_forecasts"],
            "n_models": overview["n_models"],
            "combined_count": overview.get("combined_count", 0),
            "stored_model_count": overview.get("stored_model_count", 0),
            "selection_count": overview.get("selection_count", 0),
        },
        "options": {
            "include_fitted": bool(include_fitted),
            "include_residuals": bool(include_residuals),
            "include_residual_acf": bool(include_residual_acf),
            "include_residual_qq": bool(include_residual_qq),
            "include_rolling_loss": bool(include_rolling_loss),
            "rolling_window": int(rolling_window),
            "rolling_metric": rolling_metric,
            "include_forecast_scale": bool(include_forecast_scale),
            "scale_view": scale_view,
            "include_training_loss": bool(include_training_loss),
            "include_rolling_training_loss": bool(include_rolling_training_loss),
            "training_loss_metric": training_loss_metric,
            "include_first_vs_last": bool(include_first_vs_last),
            "include_dfm_idiosyncratic_acf": bool(include_dfm_idiosyncratic_acf),
            "include_dfm_factor_stability": bool(include_dfm_factor_stability),
            "include_coefficients": bool(include_coefficients),
            "include_parameter_stability": bool(include_parameter_stability),
            "include_tuning": bool(include_tuning),
            "include_tuning_objective": bool(include_tuning_objective),
            "include_hyperparameters": bool(include_hyperparameters),
            "include_tuning_scores": bool(include_tuning_scores),
            "include_ensemble_weights": bool(include_ensemble_weights),
            "include_ensemble_concentration": bool(include_ensemble_concentration),
            "include_member_contribution": bool(include_member_contribution),
            "include_stage_updates": bool(include_stage_updates),
            "include_combined": bool(include_combined),
        },
        "tables": {
            "fitted_rows": None if fitted is None else int(fitted.shape[0]),
            "residual_rows": None if residuals is None else int(residuals.shape[0]),
            "residual_acf_rows": None if residual_acf is None else int(residual_acf.shape[0]),
            "residual_qq_rows": None if qq is None else int(qq.shape[0]),
            "rolling_loss_rows": None if rolling is None else int(rolling.shape[0]),
            "forecast_scale_rows": None if scale is None else int(scale.shape[0]),
            "training_loss_rows": None if train_loss is None else int(train_loss.shape[0]),
            "rolling_training_loss_rows": None if rolling_train_loss is None else int(rolling_train_loss.shape[0]),
            "first_vs_last_rows": None if first_last is None else int(first_last.shape[0]),
            "dfm_idiosyncratic_acf_rows": None if dfm_acf is None else int(dfm_acf.shape[0]),
            "dfm_factor_stability_rows": None if dfm_factors is None else int(dfm_factors.shape[0]),
            "coefficient_rows": None if coefficients is None else int(coefficients.shape[0]),
            "parameter_stability_rows": None if stability is None else int(stability.shape[0]),
            "tuning_rows": None if tuning is None else int(tuning.shape[0]),
            "tuning_objective_rows": None if objective is None else int(objective.shape[0]),
            "hyperparameter_rows": None if hyperparams is None else int(hyperparams.shape[0]),
            "tuning_score_rows": None if tuning_scores is None else int(tuning_scores.shape[0]),
            "ensemble_weight_rows": None if weights is None else int(weights.shape[0]),
            "ensemble_concentration_rows": None if concentration is None else int(concentration.shape[0]),
            "member_contribution_rows": None if contribution is None else int(contribution.shape[0]),
            "stage_update_rows": None if stages is None else int(stages.shape[0]),
        },
    }
    metadata = attach_metadata(base_metadata, "forecast_analysis", stage)
    _attach_metadata(fitted, metadata)
    _attach_metadata(residuals, metadata)
    _attach_metadata(residual_acf, metadata)
    _attach_metadata(qq, metadata)
    _attach_metadata(rolling, metadata)
    _attach_metadata(scale, metadata)
    _attach_metadata(train_loss, metadata)
    _attach_metadata(rolling_train_loss, metadata)
    _attach_metadata(first_last, metadata)
    _attach_metadata(dfm_acf, metadata)
    _attach_metadata(dfm_factors, metadata)
    _attach_metadata(coefficients, metadata)
    _attach_metadata(stability, metadata)
    _attach_metadata(tuning, metadata)
    _attach_metadata(objective, metadata)
    _attach_metadata(hyperparams, metadata)
    _attach_metadata(tuning_scores, metadata)
    _attach_metadata(weights, metadata)
    _attach_metadata(concentration, metadata)
    _attach_metadata(contribution, metadata)
    _attach_metadata(stages, metadata)
    return ForecastDiagnosticReport(
        overview=overview,
        fitted=fitted,
        residuals=residuals,
        residual_acf=residual_acf,
        residual_qq=qq,
        rolling_loss=rolling,
        forecast_scale=scale,
        training_loss=train_loss,
        rolling_training_loss=rolling_train_loss,
        first_vs_last=first_last,
        dfm_idiosyncratic_acf=dfm_acf,
        dfm_factor_stability=dfm_factors,
        coefficients=coefficients,
        parameter_stability=stability,
        tuning=tuning,
        tuning_objective=objective,
        hyperparameters=hyperparams,
        tuning_scores=tuning_scores,
        ensemble_weights=weights,
        ensemble_concentration=concentration,
        member_contribution=contribution,
        stage_updates=stages,
        metadata=metadata,
    )


def custom_forecast_diagnostic(
    forecasts: Any,
    func: Callable[..., Any],
    *,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **params: Any,
) -> pd.DataFrame:
    """Run a user-supplied forecast diagnostic and attach macroforecast metadata."""

    table, base_metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    resolved_name = str(name or _callable_name(func) or "custom_forecast_diagnostic")
    call_metadata = dict(base_metadata)
    call_metadata.update(dict(metadata or {}))
    result = func(table.copy(), metadata=call_metadata, **params)
    output = _coerce_custom_table(result)
    overview = forecast_overview(ForecastResult(table, metadata=base_metadata))
    stage = {
        "name": resolved_name,
        "callable": _callable_name(func),
        "params": dict(params),
        "input": {
            "n_forecasts": int(overview["n_forecasts"]),
            "n_models": int(overview["n_models"]),
            "horizons": list(overview["horizons"]),
        },
        "output": {
            "rows": int(output.shape[0]),
            "columns": [str(column) for column in output.columns],
        },
        "user_metadata": dict(metadata or {}),
    }
    updated_metadata = attach_metadata(base_metadata, "custom_forecast_diagnostic", stage)
    output.attrs["macroforecast_metadata_schema"] = {
        "kind": "custom_forecast_diagnostic",
        "version": 1,
        "method": resolved_name,
        "columns": [str(column) for column in output.columns],
        "metadata": stage,
    }
    _attach_metadata(output, updated_metadata)
    return output


def _coerce_forecast_input(value: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(value, ForecastResult):
        return value.to_frame(), dict(value.metadata)
    if isinstance(value, pd.DataFrame):
        return value.copy(), dict(value.attrs.get("macroforecast_metadata", {}) or {})
    raise TypeError("forecasts must be a ForecastResult or pandas DataFrame")


def _validate_forecast_table(table: pd.DataFrame) -> None:
    missing = [column for column in FORECAST_REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(f"forecast table is missing required columns: {missing}")


def _empty_frame(columns: Sequence[str], *, kind: str) -> pd.DataFrame:
    frame = pd.DataFrame(columns=list(columns))
    frame.attrs["macroforecast_metadata_schema"] = {"kind": kind, "version": 1}
    return frame


def _empty_weight_frame() -> pd.DataFrame:
    return _empty_frame(
        [
            "date",
            "origin",
            "origin_pos",
            "horizon",
            "combination",
            "method",
            "model",
            "weight",
        ],
        kind="ensemble_weights_over_time",
    )


def _acf_value(series: pd.Series, lag: int) -> float | None:
    # Canonical biased sample autocorrelation (matches statsmodels.acf with
    # adjusted=False): a single global mean and the gamma_k / gamma_0 ratio.
    # The previous np.corrcoef of the two lagged sub-windows used separate
    # per-window means/variances and diverged from the standard ACF, most
    # noticeably on trending series.
    values = series.dropna().astype(float).to_numpy()
    n = len(values)
    if lag == 0:
        return 1.0 if n else None
    if n <= lag:
        return None
    centered = values - values.mean()
    gamma_0 = float(np.sum(centered * centered))
    if gamma_0 == 0:
        return None
    gamma_k = float(np.sum(centered[lag:] * centered[:-lag]))
    return gamma_k / gamma_0


def _normalize_group_by(group_by: Sequence[str], frame: pd.DataFrame) -> tuple[str, ...]:
    groups = tuple(str(column) for column in group_by)
    missing = [column for column in groups if column not in frame.columns]
    if missing:
        raise ValueError(f"group_by columns are not in the forecast table: {missing}")
    return groups


def _group_row(key: Any, groups: Sequence[str]) -> dict[str, Any]:
    if not groups:
        return {}
    if len(groups) == 1:
        values = (key,)
    else:
        values = key if isinstance(key, tuple) else (key,)
    return {group: value for group, value in zip(groups, values, strict=False)}


def _loss_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "mse" or metric == "rmse":
        return frame["squared_error"].astype(float)
    if metric == "mae":
        return frame["abs_error"].astype(float)
    if metric == "bias":
        return frame["residual"].astype(float)
    raise ValueError(f"unsupported rolling loss metric {metric!r}")


def _normalize_loss_metric(metric: str) -> str:
    key = str(metric).lower()
    aliases = {"mean_squared_error": "mse", "root_mean_squared_error": "rmse", "mean_absolute_error": "mae"}
    key = aliases.get(key, key)
    if key not in {"mse", "rmse", "mae", "bias"}:
        raise ValueError("metric must be one of 'mse', 'rmse', 'mae', or 'bias'")
    return key


def _combination_weights_for_row(
    base: pd.DataFrame,
    combined_row: pd.Series,
    group: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    unsupported: UnsupportedWeights,
) -> dict[str, float | None]:
    method = str(spec.get("method", ""))
    models = [str(model) for model in group["model"]]
    if method == "mean":
        return {model: 1.0 / len(models) for model in models}
    if method in {"inverse_mspe", "dmspe"}:
        return _inverse_mspe_weights(base, combined_row, group, spec)
    if method == "best_n":
        return _best_n_weights(base, combined_row, group, spec)
    if unsupported == "raise":
        raise ValueError(f"combination method {method!r} does not have identifiable weights")
    if unsupported == "nan":
        return {model: None for model in models}
    return {}


def _historical_wide(
    base: pd.DataFrame,
    combined_row: pd.Series,
    group: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    models = [str(model) for model in group["model"]]
    horizon = combined_row.get("horizon")
    origin_pos = combined_row.get("origin_pos")
    history = base.loc[(base["horizon"] == horizon) & (base["origin_pos"] <= origin_pos)].copy()
    history = history.loc[history["model"].astype(str).isin(models)]
    wide = history.pivot_table(
        index=["date", "origin_pos"],
        columns="model",
        values="prediction",
        aggfunc="first",
    ).sort_index()
    actual = (
        history.drop_duplicates(["date", "origin_pos"])
        .set_index(["date", "origin_pos"])["actual"]
        .reindex(wide.index)
    )
    return wide.loc[:, [column for column in models if column in wide]], actual


def _inverse_mspe_weights(
    base: pd.DataFrame,
    combined_row: pd.Series,
    group: pd.DataFrame,
    spec: Mapping[str, Any],
) -> dict[str, float]:
    wide, actual = _historical_wide(base, combined_row, group)
    if wide.empty:
        return {}
    discount = float((spec.get("params") or {}).get("discount", 1.0))
    min_weight = float((spec.get("params") or {}).get("min_weight", 1e-12))
    errors = wide.sub(actual, axis=0) ** 2
    running = pd.Series(0.0, index=wide.columns, dtype=float)
    weights = pd.Series(1.0 / len(wide.columns), index=wide.columns, dtype=float)
    target_key = (combined_row.get("date"), combined_row.get("origin_pos"))
    for step, (key, current) in enumerate(errors.iterrows()):
        if step == 0 or float(running.sum()) <= 0:
            weights = pd.Series(1.0 / len(wide.columns), index=wide.columns, dtype=float)
        else:
            inv = 1.0 / running.clip(lower=min_weight)
            weights = inv / inv.sum()
        if key == target_key:
            break
        running = discount * running + current.fillna(running.mean() if running.notna().any() else 0.0)
    return {str(model): float(weight) for model, weight in weights.items()}


def _best_n_weights(
    base: pd.DataFrame,
    combined_row: pd.Series,
    group: pd.DataFrame,
    spec: Mapping[str, Any],
) -> dict[str, float]:
    wide, actual = _historical_wide(base, combined_row, group)
    if wide.empty:
        return {}
    n = int((spec.get("params") or {}).get("n", 3))
    target_key = (combined_row.get("date"), combined_row.get("origin_pos"))
    errors = wide.sub(actual, axis=0).pow(2)
    historical = errors.loc[errors.index < target_key]
    if historical.empty:
        selected = list(wide.columns[:n])
    else:
        mspe = historical.mean(axis=0).fillna(float("inf")).sort_values()
        selected = list(mspe.index[:n])
    return {str(model): (1.0 / len(selected) if model in selected else 0.0) for model in wide.columns}


def _coerce_level_series(levels: Any, *, target: str | None) -> pd.Series:
    if isinstance(levels, pd.Series):
        series = levels.copy()
        if target is not None:
            series = series.rename(target)
    elif isinstance(levels, pd.DataFrame):
        frame = levels.copy()
        if target is not None:
            if target not in frame:
                raise ValueError(f"target column {target!r} is not in levels")
            series = frame[target]
        else:
            numeric = frame.select_dtypes("number")
            if numeric.shape[1] != 1:
                raise ValueError("levels DataFrame must have exactly one numeric column when target is not supplied")
            series = numeric.iloc[:, 0]
    else:
        raise TypeError("levels must be a pandas Series or DataFrame")
    series = pd.to_numeric(series, errors="coerce")
    series.index = pd.to_datetime(series.index, errors="ignore")
    return series.sort_index()


def _metadata_target_transform(metadata: Mapping[str, Any]) -> str | None:
    keys = ("target_transform", "transform")
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return str(value)
    for section in ("forecast_policy", "features", "run"):
        nested = metadata.get(section)
        if isinstance(nested, Mapping):
            for key in keys:
                value = nested.get(key)
                if value is not None:
                    return str(value)
    return None


def _forecast_row_base(row: pd.Series) -> dict[str, Any]:
    return {
        "date": row.get("date"),
        "origin": row.get("origin"),
        "origin_pos": row.get("origin_pos"),
        "horizon": row.get("horizon"),
        "forecast_policy": row.get("forecast_policy"),
        "target": row.get("target"),
        "model": row.get("model"),
        "model_spec": row.get("model_spec"),
        "combined": row.get("combined"),
    }


def _residual_value(prediction: Any, actual: Any) -> float | None:
    pred = _float_or_none(prediction)
    obs = _float_or_none(actual)
    if pred is None or obs is None:
        return None
    return float(obs - pred)


def _back_transformed_values(
    row: pd.Series,
    transform: str,
    *,
    levels: pd.Series | None,
    back_transform: Callable[..., Any] | None,
) -> tuple[float | None, float | None, bool]:
    if back_transform is not None:
        return _call_back_transform(row, levels=levels, func=back_transform)
    transform_key = str(transform).lower()
    prediction = _float_or_none(row.get("prediction"))
    actual = _float_or_none(row.get("actual"))
    if transform_key == "level":
        return prediction, actual, True
    if levels is None:
        return None, None, False
    origin_level = _level_at(levels, row.get("origin"))
    if origin_level is None:
        return None, _level_at(levels, row.get("date")), False

    # Path-average / direct-average targets store the MEAN one-period transform
    # over `horizon` steps, so the level at t+h needs the horizon factor (the
    # one-step inversion silently drops it). For change and log_growth the path
    # telescopes exactly to the endpoint; simple growth and level-value averages
    # have no exact pointwise level inverse and are reported unavailable.
    policy = str(row.get("forecast_policy") or "").lower()
    is_average = policy in {"path_average", "direct_average"} or transform_key.startswith(
        "average_"
    )
    if is_average:
        step_key = (
            transform_key[len("average_"):]
            if transform_key.startswith("average_")
            else transform_key
        )
        try:
            horizon = int(row.get("horizon") or 1)
        except (TypeError, ValueError):
            horizon = 1
        if step_key not in {"change", "diff", "difference", "log_growth", "log_diff"}:
            # growth (arithmetic mean of simple returns) and value (mean of
            # levels) do not determine x[t+h]; do not report a misleading level.
            return None, None, False
        pred_level = _path_average_back_transform(origin_level, prediction, step_key, horizon)
        actual_level = _level_at(levels, row.get("date"))
        if actual_level is None:
            actual_level = _path_average_back_transform(origin_level, actual, step_key, horizon)
        return pred_level, actual_level, pred_level is not None or actual_level is not None

    actual_level = _level_at(levels, row.get("date"))
    pred_level = _one_step_back_transform(origin_level, prediction, transform_key)
    if actual_level is None:
        actual_level = _one_step_back_transform(origin_level, actual, transform_key)
    return pred_level, actual_level, pred_level is not None or actual_level is not None


def _call_back_transform(
    row: pd.Series,
    *,
    levels: pd.Series | None,
    func: Callable[..., Any],
) -> tuple[float | None, float | None, bool]:
    try:
        result = func(row=row.copy(), levels=levels)
    except TypeError:
        try:
            result = func(row.copy())
        except TypeError:
            result = None
    if isinstance(result, Mapping):
        pred = result.get("prediction", result.get("predicted", result.get("forecast")))
        actual = result.get("actual", result.get("observed"))
        return _float_or_none(pred), _float_or_none(actual), True
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)) and len(result) >= 2:
        return _float_or_none(result[0]), _float_or_none(result[1]), True
    prediction = _float_or_none(row.get("prediction"))
    actual = _float_or_none(row.get("actual"))
    try:
        pred = func(prediction)
        obs = func(actual)
    except Exception:  # noqa: BLE001 - custom back transforms are best-effort diagnostics.
        return None, None, False
    return _float_or_none(pred), _float_or_none(obs), True


def _one_step_back_transform(origin_level: float, value: float | None, transform: str) -> float | None:
    if value is None:
        return None
    if transform in {"change", "diff", "difference"}:
        return float(origin_level + value)
    if transform in {"growth", "pct_change", "percent_change"}:
        return float(origin_level * (1.0 + value))
    if transform in {"log_growth", "log_diff"}:
        return float(origin_level * np.exp(value))
    return None


def _path_average_back_transform(
    origin_level: float, avg_value: float | None, step_transform: str, horizon: int
) -> float | None:
    """Reconstruct x[t+h] from the mean one-period transform over ``horizon`` steps.

    The sum of one-period transforms equals ``horizon * mean``; for change and
    log_growth this telescopes exactly to the endpoint level.
    """
    if avg_value is None:
        return None
    total = float(horizon) * float(avg_value)
    if step_transform in {"change", "diff", "difference"}:
        return float(origin_level + total)
    if step_transform in {"log_growth", "log_diff"}:
        return float(origin_level * np.exp(total))
    return None


def _level_at(levels: pd.Series, label: Any) -> float | None:
    if label is None or pd.isna(label):
        return None
    try:
        value = levels.loc[pd.Timestamp(label)]
    except (KeyError, TypeError, ValueError):
        try:
            value = levels.loc[label]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, pd.Series):
        value = value.iloc[-1] if len(value) else None
    return _float_or_none(value)


def _difference_or_none(left: Any, right: Any) -> float | None:
    left_value = _float_or_none(left)
    right_value = _float_or_none(right)
    if left_value is None or right_value is None:
        return None
    return float(left_value - right_value)


def _coerce_training_loss_input(value: Any, *, load_pickle: bool) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame) and {"metric", "value"}.issubset(value.columns):
        out = value.copy()
        out.attrs["macroforecast_metadata_schema"] = {
            "kind": "training_loss_trace",
            "version": 1,
        }
        return out
    return training_loss_trace(value, load_pickle=load_pickle)


def _iter_sidecar_diagnostics(
    table: pd.DataFrame,
    *,
    load_pickle: bool,
) -> Iterable[tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any]]]:
    for _, row in table.iterrows():
        base_context = _forecast_row_base(row)
        stored = _mapping_or_none(row.get("stored_model"))
        if stored is None:
            continue
        step_records = _stored_model_records(stored)
        for step, step_stored in step_records:
            sidecar = _load_stored_model_sidecar(step_stored, load_pickle=load_pickle)
            if sidecar is None:
                continue
            diagnostics = _sidecar_diagnostics(sidecar)
            if not diagnostics:
                continue
            context = dict(base_context)
            context["fit_step"] = step
            yield context, step_stored, diagnostics


def _stored_model_records(stored: Mapping[str, Any]) -> list[tuple[str | None, Mapping[str, Any]]]:
    steps = stored.get("steps")
    if isinstance(steps, Mapping):
        records: list[tuple[str | None, Mapping[str, Any]]] = []
        for step, item in steps.items():
            if isinstance(item, Mapping):
                records.append((str(step), item))
        return records
    return [(None, stored)]


def _iter_diagnostics(
    source: Any,
    *,
    load_pickle: bool,
) -> Iterable[tuple[dict[str, Any], Mapping[str, Any]]]:
    if hasattr(source, "diagnostics"):
        diagnostics = getattr(source, "diagnostics")
        if isinstance(diagnostics, Mapping):
            context = {
                "date": None,
                "origin": None,
                "origin_pos": None,
                "horizon": None,
                "model": getattr(source, "model", None),
                "model_spec": getattr(source, "model", None),
                "fit_step": None,
            }
            yield context, diagnostics
        return
    table, _metadata = _coerce_forecast_input(source)
    _validate_forecast_table(table)
    for context, _stored, diagnostics in _iter_sidecar_diagnostics(table, load_pickle=load_pickle):
        yield context, diagnostics


def _diagnostic_frame_or_series(value: Any) -> pd.DataFrame | pd.Series | None:
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.copy()
    if isinstance(value, Mapping):
        if "columns" in value and "data" in value:
            data = value.get("data")
            index = value.get("index")
            try:
                frame = pd.DataFrame(data)
                if index is not None and len(index) == len(frame):
                    frame.index = pd.to_datetime(index, errors="ignore")
                columns = value.get("columns")
                if columns is not None:
                    frame = frame.loc[:, [column for column in columns if column in frame]]
                return frame
            except Exception:  # noqa: BLE001 - diagnostic coercion is best-effort.
                return None
        if "data" in value and "index" in value:
            data = value.get("data")
            index = value.get("index")
            if isinstance(data, list):
                series = pd.Series(data, name=value.get("name"))
                if index is not None and len(index) == len(series):
                    series.index = pd.to_datetime(index, errors="ignore")
                return series
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return pd.Series(list(value))
    return None


def _load_stored_model_sidecar(stored: Mapping[str, Any], *, load_pickle: bool) -> Mapping[str, Any] | None:
    metadata_path = stored.get("metadata_path")
    if metadata_path:
        path = Path(str(metadata_path))
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return None
    if not load_pickle:
        return None
    model_path = stored.get("model_path")
    if not model_path:
        return None
    try:
        from macroforecast.models import load_fit

        fit = load_fit(model_path)
    except Exception:  # noqa: BLE001 - optional fallback only.
        return None
    if hasattr(fit, "to_metadata"):
        return {"fit": fit.to_metadata()}
    return None


def _sidecar_diagnostics(sidecar: Mapping[str, Any]) -> Mapping[str, Any]:
    fit = sidecar.get("fit")
    if isinstance(fit, Mapping):
        nested = fit.get("fit")
        if isinstance(nested, Mapping):
            diagnostics = nested.get("diagnostics")
            if isinstance(diagnostics, Mapping):
                return diagnostics
    diagnostics = sidecar.get("diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _coefficient_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "index" in value and "data" in value:
            index = list(value.get("index") or [])
            data = value.get("data")
            if isinstance(data, list):
                return [
                    {"feature": str(feature), "coefficient": _float_or_json(coef)}
                    for feature, coef in zip(index, data, strict=False)
                ]
            if isinstance(data, Mapping):
                rows: list[dict[str, Any]] = []
                for component, values in data.items():
                    if not isinstance(values, list):
                        continue
                    for feature, coef in zip(index, values, strict=False):
                        rows.append(
                            {
                                "feature": str(feature),
                                "component": str(component),
                                "coefficient": _float_or_json(coef),
                            }
                        )
                return rows
        return [
            {"feature": str(feature), "coefficient": _float_or_json(coef)}
            for feature, coef in value.items()
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            {"feature": f"x{position}", "coefficient": _float_or_json(coef)}
            for position, coef in enumerate(value)
        ]
    return []


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
    raise TypeError("custom forecast diagnostic must return a DataFrame, Series, mapping, or sequence")


def _callable_name(func: Any) -> str:
    return str(getattr(func, "__name__", func.__class__.__name__))


def _bool_series(value: Any, n: int) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value
    else:
        series = pd.Series([False] * n)
    return series.fillna(False).map(bool)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _selection_retuned(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value.get("retuned", False))


def _date_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


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


def _float_or_json(value: Any) -> Any:
    numeric = _float_or_none(value)
    return _json_scalar(value) if numeric is None else numeric


def _scalar_or_json(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return _float_or_json(value[0])
    return _float_or_json(value)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


__all__ = [
    "ForecastDiagnosticReport",
    "coefficient_trace",
    "custom_forecast_diagnostic",
    "dfm_factor_stability",
    "dfm_idiosyncratic_acf",
    "diagnose_forecasts",
    "ensemble_member_contribution",
    "ensemble_weight_concentration",
    "ensemble_weights_over_time",
    "first_vs_last_forecast",
    "fitted_vs_actual",
    "forecast_overview",
    "forecast_scale_view",
    "hyperparameter_path",
    "parameter_stability",
    "residual_autocorrelation",
    "residual_qq",
    "residual_report",
    "rolling_loss",
    "rolling_training_loss",
    "select_forecast_origins",
    "stage_update_trace",
    "training_loss_trace",
    "tuning_objective_trace",
    "tuning_score_distribution",
    "tuning_trace",
]
