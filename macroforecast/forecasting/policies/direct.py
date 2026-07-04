"""The direct / direct_average forecast policy (Phase 3 of the runner
decomposition; body moved verbatim from the former inline branch of
``runner._fit_predict_origin``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from macroforecast.forecasting.policies.base import (
    _OriginRunConfig,
    _fit_one_model_at_origin,
)
from macroforecast.forecasting.selection_stage import (
    _allow_non_temporal_selection_splits,
    _availability_safe_selection_splits,
    _filter_xy_to_target_availability,
    _target_availability_window_fields,
)


def forecast_direct_origin(
    item: dict[str, Any],
    cfg: _OriginRunConfig,
) -> list[dict[str, Any]]:
    model_runs = cfg.model_runs

    X_fit = item["X_fit"]
    y_fit = item["y_fit"]
    X_selection = item.get("X_selection", X_fit)
    y_selection = item.get("y_selection", y_fit)
    X_test = item["X_test"]
    y_test = item["y_test"]
    if X_fit.empty or X_test.empty:
        return []

    records: list[dict[str, Any]] = []
    row = {
        **item["row"],
        "horizon": int(item.get("forecast_horizon", item["row"].get("horizon", 1))),
        "forecast_policy": item.get("forecast_policy", "direct"),
        "target_key": item.get("target_key"),
    }
    target_dates = _forecast_target_dates(
        X_test.index,
        base_index=item.get("base_index"),
        horizon=int(row["horizon"]),
    )
    target_step = int(row["horizon"])
    row.update(_target_availability_window_fields(item, target_step=target_step))
    X_fit, y_fit = _filter_xy_to_target_availability(
        X_fit,
        y_fit,
        item,
        target_step=target_step,
    )
    X_selection, y_selection = _filter_xy_to_target_availability(
        X_selection,
        y_selection,
        item,
        target_step=target_step,
    )
    selection_splits = _availability_safe_selection_splits(
        item,
        X_selection.index,
        target_step=target_step,
    )
    if X_fit.empty:
        return []
    for model_run in model_runs:
        model_spec = model_run.spec
        # This branch handles the direct / direct_average policies only (recursive and
        # path_average are dispatched earlier). Direct-capable iterated models (ar, far)
        # declare a ``direct`` flag in their default params; the shared skeleton sets
        # it so they do a one-shot projection onto the fresh one-period lag features
        # instead of a roll-forward that would persist a stale origin-h value. IC order
        # selection uses the same flag so n_lag is chosen for the direct projection,
        # not the autoregression.
        direct_capable = "direct" in getattr(model_spec, "default_params", {})
        outcome = _fit_one_model_at_origin(
            model_run,
            X_fit,
            y_fit,
            X_selection,
            y_selection,
            X_test,
            cfg,
            direct_capable_flag=direct_capable,
            cache_key=_model_cache_key(model_run.alias, row.get("target_key")),
            row=row,
            selection_splits=selection_splits,
            target_step=target_step,
            allow_non_temporal_splits=_allow_non_temporal_selection_splits(item),
        )
        fit = outcome.fit
        fit_params = outcome.fit_params
        selection_metadata = outcome.selection_metadata
        stored_model = outcome.stored_model
        pred = outcome.prediction
        variance_pred = _variance_series(fit, X_test=X_test, index=X_test.index)
        quantile_pred = _quantile_frame(fit, X_test=X_test, index=X_test.index)
        for date, value in pred.items():
            target_date = target_dates.loc[date]
            if pd.isna(target_date):
                continue
            actual: Any = (
                y_test.reindex([date]).iloc[0] if date in y_test.index else None
            )
            actual_value = None if actual is None or pd.isna(actual) else float(actual)
            variance_value = None
            if variance_pred is not None and date in variance_pred.index:
                variance_at_date = variance_pred.loc[date]
                variance_value = (
                    None if pd.isna(variance_at_date) else float(variance_at_date)
                )
            quantile_value = None
            if quantile_pred is not None and date in quantile_pred.index:
                quantile_row = quantile_pred.loc[date].dropna()
                quantile_value = {
                    str(level): float(quantile)
                    for level, quantile in quantile_row.items()
                }
            records.append(
                {
                    "date": target_date,
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": row.get("horizon"),
                    "forecast_policy": row.get("forecast_policy"),
                    "target_transform": item.get("target_transform"),
                    "target": item.get("target_name") or y_fit.name,
                    "model": model_run.alias,
                    "model_spec": model_spec.name,
                    "prediction": float(value),
                    "variance_prediction": variance_value,
                    "quantile_predictions": quantile_value,
                    "actual": actual_value,
                    "params": dict(fit_params),
                    "model_selection": selection_metadata,
                    "stored_model": stored_model,
                    "window": row,
                    "preprocessed": bool(item.get("preprocessed", False)),
                    "combined": False,
                    "combination": None,
                }
            )
    return records


# Bottom import for the same circularity reason as policies.base.
from macroforecast.forecasting.runner import (  # noqa: E402
    _forecast_target_dates,
    _model_cache_key,
    _quantile_frame,
    _variance_series,
)
