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

import numpy as np
import pandas as pd

from macroforecast.data import DataBundle
from macroforecast.model_selection import SearchSpec
from macroforecast.models import ModelSpec
from macroforecast.window import StagePolicy

from macroforecast.forecasting.model_resolution import (
    _ModelRun,
    _actual_model_params,
)
from macroforecast.forecasting.policies.base import (
    _prediction_series,
    _store_model_fit,
)
from macroforecast.forecasting.policy_config import ForecastPolicy
from macroforecast.forecasting.selection_stage import _selection_for_model


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
    requested_horizon = int(row.get("horizon", 1))
    for model_run in model_runs:
        model_spec = model_run.spec
        _validate_panel_selection(selection, model_run)
        best_params = _panel_fit_params(
            model_spec,
            target=target,
            forecast_policy=forecast_policy,
            horizon=requested_horizon,
        )
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


# ---------------------------------------------------------------------------
# Panel-input helpers (Phase 5; moved verbatim from runner): fit params from
# the model spec, the prediction input without the leaking test target, panel
# selection validation, and the origin-distance horizon label (#423).
# ---------------------------------------------------------------------------


def _panel_fit_params(
    model_spec: ModelSpec,
    *,
    target: str,
    forecast_policy: ForecastPolicy,
    horizon: int,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if "target" in model_spec.default_params and model_spec.params.get("target") is None:
        params["target"] = target
    if (
        model_spec.name == "var"
        and forecast_policy in {"direct", "direct_average"}
        and "direct" in model_spec.default_params
    ):
        params["direct"] = True
        params["direct_horizon"] = int(horizon)
    return params


def _panel_prediction_input_without_test_target(
    fit_panel: pd.DataFrame,
    test_panel: pd.DataFrame,
    *,
    target: str,
    metadata: Mapping[str, Any],
) -> DataBundle:
    """Return panel available at prediction time with test target values masked."""

    combined = pd.concat([fit_panel, test_panel], axis=0)
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index()
    if target in combined.columns:
        combined.loc[test_panel.index, target] = np.nan
    combined.attrs["macroforecast_metadata"] = dict(metadata)
    return DataBundle(combined, dict(metadata))


def _validate_panel_selection(
    selection: SearchSpec | Mapping[str, SearchSpec | None] | None,
    model_run: _ModelRun,
) -> None:
    selected, use_model_default_selection = _selection_for_model(selection, model_run)
    if selected is not None:
        raise ValueError(
            "panel-input forecasting does not tune model parameters yet; pass "
            f"model_selection={{'{model_run.alias}': None}} or model_selection=None"
        )
    if isinstance(selection, Mapping) and (
        model_run.alias in selection or model_run.spec.name in selection
    ):
        return
    if use_model_default_selection and model_run.spec.search_spaces:
        return



def _panel_prediction_horizon(
    date: Any,
    *,
    origin: Any,
    base_index: pd.Index,
    default: int,
) -> int:
    """Positional distance from the origin to a panel prediction date.

    ``base_index`` is the panel test window, which (as of #423) always
    excludes the origin itself -- ``WindowSpec.origins(exclude_origin=True)``
    starts the test slice at ``origin_pos + 1``. The horizon is therefore the
    plain positional difference, with no floor: an earlier ``max(1, ...)``
    floor papered over the window bug by forcing the origin's own row (which
    the buggy window used to include) to read as horizon 1 -- exactly how
    every row of a multi-step path ended up mislabeled horizon=1. With the
    window fix, distances are already >= 1 in normal operation; the floor is
    removed so a caller that passes an origin still inside ``base_index``
    (e.g. a unit test) gets back the true distance (0 for the origin's own
    row) instead of a silently-clamped value.
    """
    origin_index = base_index.insert(0, origin) if origin not in base_index else base_index
    positions = origin_index.get_indexer(pd.Index([origin, date]))
    if (positions < 0).any():
        return int(default)
    return int(positions[1] - positions[0])
