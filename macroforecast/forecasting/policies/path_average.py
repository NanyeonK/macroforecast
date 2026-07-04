"""The path_average forecast policy (Phase 3 of the runner decomposition;
body moved verbatim from ``runner._fit_predict_path_average_origin``).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from macroforecast.forecasting.policies.base import (
    _OriginRunConfig,
    _fit_one_model_at_origin,
)
from macroforecast.forecasting.selection_stage import (
    _align_feature_xy,
    _allow_non_temporal_selection_splits,
    _availability_safe_selection_splits,
    _filter_xy_to_target_availability,
    _selection_for_model,
    _target_availability_window_fields,
)


def forecast_path_average_origin(
    item: dict[str, Any],
    cfg: _OriginRunConfig,
) -> list[dict[str, Any]]:
    model_runs = cfg.model_runs
    selection = cfg.selection
    selection_policy = cfg.selection_policy
    save_models = cfg.save_models

    X_fit = pd.DataFrame(item["X_fit"])
    X_selection_base = pd.DataFrame(item.get("X_selection", X_fit))
    X_test = pd.DataFrame(item["X_test"])
    y_fit = pd.DataFrame(item["y_fit"])
    y_selection_base = pd.DataFrame(item.get("y_selection", y_fit))
    y_test = pd.DataFrame(item["y_test"])
    if X_fit.empty or X_test.empty:
        return []

    horizon = int(item.get("forecast_horizon", item["row"].get("horizon", 1)))
    if horizon < 1:
        raise ValueError("forecast horizon must be at least 1")
    step_columns = _path_step_columns(y_fit, horizon=horizon)
    row = {
        **item["row"],
        "horizon": horizon,
        "forecast_policy": "path_average",
        "target_key": item.get("target_key"),
        **_target_availability_window_fields(
            item,
            target_step=horizon,
            policy="path_step_specific",
        ),
    }
    row["target_availability_by_step"] = {
        str(step): _target_availability_window_fields(
            item,
            target_step=step,
            policy="row_pos_plus_step_lte_origin_pos",
        )
        for step in range(1, horizon + 1)
    }
    target_dates = _forecast_target_dates(
        X_test.index,
        base_index=item.get("base_index"),
        horizon=horizon,
    )
    records: list[dict[str, Any]] = []

    for model_run in model_runs:
        model_spec = model_run.spec
        selected, use_model_default_selection = _selection_for_model(selection, model_run)
        should_select = selected is not None or (
            use_model_default_selection and bool(model_spec.search_spaces)
        )
        # Each path step forecasts the one-period target s steps ahead, regressed on
        # the origin-available features -- a DIRECT s-step projection. Direct-capable
        # iterated models (ar, far) must therefore run with ``direct=True`` per step,
        # exactly as the direct policy does -- the shared skeleton injects the flag,
        # so the direct policy and path per-step projections flow through the same
        # code; otherwise the per-step fit would fall back to the legacy roll-forward
        # and persist a stale value (mild at h1, growing with the step). IC order
        # selection uses the same flag so n_lag is chosen for the direct projection.
        direct_capable = "direct" in getattr(model_spec, "default_params", {})
        predictions_by_step: dict[int, pd.Series] = {}
        stored_by_step: dict[str, Any] = {}
        selection_by_step: dict[str, Any] = {}
        params_by_step: dict[str, Any] = {}

        for step, column in enumerate(step_columns, start=1):
            y_fit_step = y_fit[column].rename(str(column))
            y_selection_step = y_selection_base[column].rename(str(column))
            X_fit_step, y_fit_aligned = _align_feature_xy(X_fit, y_fit_step)
            X_selection_step, y_selection_aligned = _align_feature_xy(
                X_selection_base,
                y_selection_step,
            )
            X_fit_step, y_fit_aligned = _filter_xy_to_target_availability(
                X_fit_step,
                y_fit_aligned,
                item,
                target_step=step,
            )
            X_selection_step, y_selection_aligned = _filter_xy_to_target_availability(
                X_selection_step,
                y_selection_aligned,
                item,
                target_step=step,
            )
            selection_splits = _availability_safe_selection_splits(
                item,
                X_selection_step.index,
                target_step=step,
            )
            if X_fit_step.empty:
                continue
            step_key = _model_cache_key(
                model_run.alias,
                f"{item.get('target_key', 'path')}_step{step}",
            )
            step_row = {**row, "target_key": f"{row.get('target_key')}_step{step}"}
            # Per-step select -> fit -> save -> predict runs through the shared
            # skeleton -- the same code the direct policy uses, so the per-step
            # direct=True projection and the IC-on-full-sample selection (which
            # preserves the horizon-1 direct==path invariant) can never drift
            # from the direct policy again. The step-specific cache key keeps
            # per-step params isolated; ``path_step`` is stamped onto both fresh
            # and reused selection metadata, and the degraded-selection warning
            # names the step.
            outcome = _fit_one_model_at_origin(
                model_run,
                X_fit_step,
                y_fit_aligned,
                X_selection_step,
                y_selection_aligned,
                X_test,
                cfg,
                direct_capable_flag=direct_capable,
                cache_key=step_key,
                row=step_row,
                selection_splits=selection_splits,
                target_step=step,
                allow_non_temporal_splits=_allow_non_temporal_selection_splits(item),
                degraded_alias=f"{model_run.alias} (path step {step})",
                retuned_metadata_extras={"path_step": step},
                reused_metadata_extras={"path_step": step},
            )
            stored_by_step[str(step)] = outcome.stored_model
            predictions_by_step[step] = outcome.prediction
            selection_by_step[str(step)] = outcome.selection_metadata
            params_by_step[str(step)] = dict(outcome.fit_params)

        if set(predictions_by_step) != set(range(1, horizon + 1)):
            continue

        prediction_frame = pd.concat(predictions_by_step, axis=1)
        actual_frame = y_test.reindex(columns=step_columns)
        for origin_label, path_values in prediction_frame.iterrows():
            target_date = target_dates.loc[origin_label]
            if pd.isna(target_date):
                continue
            actual_values = actual_frame.reindex([origin_label]).iloc[0]
            actual_value = (
                None
                if actual_values.isna().any()
                else float(actual_values.astype(float).mean())
            )
            records.append(
                {
                    "date": target_date,
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": horizon,
                    "forecast_policy": "path_average",
                    "target_transform": item.get("target_transform"),
                    "target": item.get("target_name"),
                    "model": model_run.alias,
                    "model_spec": model_spec.name,
                    "prediction": float(path_values.astype(float).mean()),
                    "variance_prediction": None,
                    "quantile_predictions": None,
                    "actual": actual_value,
                    "params": {"steps": params_by_step},
                    "model_selection": {
                        "policy": selection_policy.to_dict(),
                        "retuned": any(
                            bool(value and value.get("retuned"))
                            for value in selection_by_step.values()
                        ),
                        "steps": selection_by_step,
                    }
                    if should_select
                    else None,
                    "stored_model": {"steps": stored_by_step} if save_models else None,
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
    _path_step_columns,
)
