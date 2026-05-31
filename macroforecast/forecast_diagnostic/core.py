from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from macroforecast.data import attach_metadata
from macroforecast.forecasting import ForecastResult


LossMetric = Literal["mse", "rmse", "mae", "bias"]
UnsupportedWeights = Literal["skip", "nan", "raise"]

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
    rolling_loss: pd.DataFrame | None = None
    coefficients: pd.DataFrame | None = None
    tuning: pd.DataFrame | None = None
    ensemble_weights: pd.DataFrame | None = None
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
        if self.rolling_loss is not None:
            out["rolling_loss"] = self.rolling_loss.to_dict(orient="records")
        if self.coefficients is not None:
            out["coefficients"] = self.coefficients.to_dict(orient="records")
        if self.tuning is not None:
            out["tuning"] = self.tuning.to_dict(orient="records")
        if self.ensemble_weights is not None:
            out["ensemble_weights"] = self.ensemble_weights.to_dict(orient="records")
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
    selection = table.get("selection", pd.Series([None] * len(table)))
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
                "residual_autocorr1": _float_or_none(values.autocorr(lag=1)) if len(values) > 2 else None,
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


def coefficient_trace(
    forecasts: Any,
    *,
    include_intercept: bool = True,
    load_pickle: bool = False,
) -> pd.DataFrame:
    """Read saved fit sidecars and return coefficient paths over origins."""

    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    rows: list[dict[str, Any]] = []
    for _, forecast_row in table.iterrows():
        stored = _mapping_or_none(forecast_row.get("stored_model"))
        if stored is None:
            continue
        sidecar = _load_stored_model_sidecar(stored, load_pickle=load_pickle)
        if sidecar is None:
            continue
        diagnostics = _sidecar_diagnostics(sidecar)
        coefficients = _coefficient_records(diagnostics.get("coefficients"))
        if include_intercept and "intercept" in diagnostics:
            coefficients.append({"feature": "intercept", "coefficient": _scalar_or_json(diagnostics["intercept"])})
        for item in coefficients:
            rows.append(
                {
                    "date": forecast_row.get("date"),
                    "origin": forecast_row.get("origin"),
                    "origin_pos": forecast_row.get("origin_pos"),
                    "horizon": forecast_row.get("horizon"),
                    "model": forecast_row.get("model"),
                    "model_spec": forecast_row.get("model_spec"),
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
                "feature",
                "coefficient",
                "component",
                "stored_metadata_path",
                "stored_model_path",
            ]
        )
    out.attrs["macroforecast_metadata_schema"] = {"kind": "coefficient_trace", "version": 1}
    return out


def tuning_trace(forecasts: Any) -> pd.DataFrame:
    """Return one row per forecast row carrying parameter-selection metadata."""

    table, _metadata = _coerce_forecast_input(forecasts)
    _validate_forecast_table(table)
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        selection = _mapping_or_none(row.get("selection"))
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
    include_rolling_loss: bool = True,
    rolling_window: int = 12,
    rolling_metric: LossMetric = "rmse",
    include_coefficients: bool = True,
    include_tuning: bool = True,
    include_ensemble_weights: bool = True,
    include_stage_updates: bool = True,
    include_combined: bool = True,
) -> ForecastDiagnosticReport:
    """Run the standard forecast diagnostics on a ForecastResult or table."""

    table, base_metadata = _coerce_forecast_input(forecasts)
    overview = forecast_overview(ForecastResult(table, metadata=base_metadata))
    fitted = fitted_vs_actual(table, include_combined=include_combined) if include_fitted else None
    residuals = residual_report(table, include_combined=include_combined) if include_residuals else None
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
    coefficients = coefficient_trace(table) if include_coefficients else None
    tuning = tuning_trace(table) if include_tuning else None
    weights = (
        ensemble_weights_over_time(table)
        if include_ensemble_weights
        else None
    )
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
            "include_rolling_loss": bool(include_rolling_loss),
            "rolling_window": int(rolling_window),
            "rolling_metric": rolling_metric,
            "include_coefficients": bool(include_coefficients),
            "include_tuning": bool(include_tuning),
            "include_ensemble_weights": bool(include_ensemble_weights),
            "include_stage_updates": bool(include_stage_updates),
            "include_combined": bool(include_combined),
        },
        "tables": {
            "fitted_rows": None if fitted is None else int(fitted.shape[0]),
            "residual_rows": None if residuals is None else int(residuals.shape[0]),
            "rolling_loss_rows": None if rolling is None else int(rolling.shape[0]),
            "coefficient_rows": None if coefficients is None else int(coefficients.shape[0]),
            "tuning_rows": None if tuning is None else int(tuning.shape[0]),
            "ensemble_weight_rows": None if weights is None else int(weights.shape[0]),
            "stage_update_rows": None if stages is None else int(stages.shape[0]),
        },
    }
    metadata = attach_metadata(base_metadata, "forecast_diagnostic", stage)
    _attach_metadata(fitted, metadata)
    _attach_metadata(residuals, metadata)
    _attach_metadata(rolling, metadata)
    _attach_metadata(coefficients, metadata)
    _attach_metadata(tuning, metadata)
    _attach_metadata(weights, metadata)
    _attach_metadata(stages, metadata)
    return ForecastDiagnosticReport(
        overview=overview,
        fitted=fitted,
        residuals=residuals,
        rolling_loss=rolling,
        coefficients=coefficients,
        tuning=tuning,
        ensemble_weights=weights,
        stage_updates=stages,
        metadata=metadata,
    )


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
    "diagnose_forecasts",
    "ensemble_weights_over_time",
    "fitted_vs_actual",
    "forecast_overview",
    "residual_report",
    "rolling_loss",
    "stage_update_trace",
    "tuning_trace",
]
