"""The panel-input forecast strategy (Phase 3 of the runner decomposition;
body moved verbatim from ``runner._fit_predict_panel_origin``).

Unlike the feature-matrix policies it does NOT run through
``_fit_one_model_at_origin``: its input contract is the canonical panel
(a ``DataBundle``), not an engineered X/y feature matrix, and it is invoked
by the panel runner (``runner._run_panel_models``) with per-run keyword
arguments rather than the per-origin ``(item, cfg)`` dispatch (plan section 4
risk note).
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from macroforecast.data import DataBundle
from macroforecast.model_selection import SearchSpec
from macroforecast.window import StagePolicy

from macroforecast.forecasting.model_resolution import (
    _ModelRun,
    _actual_model_params,
)
from macroforecast.forecasting.policy_config import ForecastPolicy


def forecast_panel_origin(
    item: dict[str, Any],
    *,
    fit_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    target: str,
    metadata: Mapping[str, Any],
    model_runs: list[_ModelRun],
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    selection_policy: StagePolicy,
    preprocessed: bool,
    save_models: bool,
    model_store: str | Path,
    forecast_policy: ForecastPolicy,
) -> list[dict[str, Any]]:
    if fit_panel.empty or test_panel.empty:
        return []
    row = item["row"]
    records: list[dict[str, Any]] = []
    for model_run in model_runs:
        model_spec = model_run.spec
        _validate_panel_selection(selection, model_run)
        best_params = _panel_fit_params(model_spec, target=target)
        fit_params = _actual_model_params(model_spec, best_params)
        fit = model_spec(DataBundle(fit_panel, dict(metadata)), **best_params)
        stored_model = (
            _store_model_fit(
                fit,
                root=model_store,
                alias=model_run.alias,
                model_spec=model_spec,
                row=row,
                params=fit_params,
                selection_metadata=None,
            )
            if save_models
            else None
        )
        if model_spec.name == "dfm_unrestricted_midas" and hasattr(fit.estimator, "predict_from_panel"):
            pred_input = _panel_prediction_input_without_test_target(
                fit_panel,
                test_panel,
                target=target,
                metadata=metadata,
            )
            pred = _prediction_series(
                fit.estimator.predict_from_panel(
                    pred_input,
                    metadata=metadata,
                    anchor_dates=test_panel.index,
                ),
                index=test_panel.index,
            )
        else:
            pred = _prediction_series(fit.predict(test_panel), index=test_panel.index)
        y_test = test_panel[target] if target in test_panel.columns else pd.Series(dtype=float)
        requested_horizon = int(row.get("horizon", 1))
        for date, value in pred.items():
            step_horizon = _panel_prediction_horizon(
                date,
                origin=row.get("origin"),
                base_index=test_panel.index,
                default=requested_horizon,
            )
            if step_horizon != requested_horizon:
                # test_panel spans origin+1 .. origin+horizon so panel models
                # that predict a full multi-step path can still be scored
                # internally, but only the row at the REQUESTED horizon is a
                # genuine forecast the caller asked for. Emitting the other
                # path steps here would create duplicate (origin, horizon)
                # keys once a multi-horizon request runs this same origin at
                # another horizon. See #423.
                continue
            actual: Any = (
                y_test.reindex([date]).iloc[0] if date in y_test.index else None
            )
            records.append(
                {
                    "date": date,
                    "origin": row.get("origin"),
                    "origin_pos": row.get("origin_pos"),
                    "horizon": step_horizon,
                    "forecast_policy": forecast_policy,
                    "target_transform": "level",
                    "target": target,
                    "model": model_run.alias,
                    "model_spec": model_spec.name,
                    "prediction": float(value),
                    "variance_prediction": None,
                    "quantile_predictions": None,
                    "actual": None if actual is None or pd.isna(actual) else float(actual),
                    "params": fit_params,
                    "model_selection": {
                        "policy": selection_policy.to_dict(),
                        "retuned": False,
                        "metadata": {
                            "note": "panel-input runner does not tune model parameters yet"
                        },
                    },
                    "stored_model": stored_model,
                    "window": row,
                    "preprocessed": bool(preprocessed),
                    "combined": False,
                    "combination": None,
                }
            )
    return records


# Bottom import for the same circularity reason as policies.base.
from macroforecast.forecasting.runner import (  # noqa: E402
    _panel_fit_params,
    _panel_prediction_horizon,
    _panel_prediction_input_without_test_target,
    _prediction_series,
    _store_model_fit,
    _validate_panel_selection,
)
