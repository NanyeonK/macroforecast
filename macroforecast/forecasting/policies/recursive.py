"""The recursive forecast policy (Phase 3 of the runner decomposition;
body moved verbatim from ``runner._fit_predict_recursive_origin``).
"""
from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from macroforecast.forecasting.policies.base import (
    _OriginRunConfig,
    _fit_one_model_at_origin,
    _model_cache_key,
    _prediction_series,
)
from macroforecast.forecasting.policy_config import FutureFeaturePolicy
from macroforecast.forecasting.selection_stage import (
    _allow_non_temporal_selection_splits,
)


def forecast_recursive_origin(
    item: dict[str, Any],
    cfg: _OriginRunConfig,
) -> list[dict[str, Any]]:
    model_runs = cfg.model_runs

    X_fit = item["X_fit"]
    y_fit = item["y_fit"]
    X_selection = item.get("X_selection", X_fit)
    y_selection = item.get("y_selection", y_fit)
    if X_fit.empty:
        return []

    target = str(item.get("target_name") or getattr(y_fit, "name", "target"))
    transform = str(item.get("target_transform", "level"))
    horizon = int(item.get("forecast_horizon", item["row"].get("horizon", 1)))
    base_index = pd.Index(item["base_index"])
    origin_pos = int(item["row"]["origin_pos"])
    target_pos = origin_pos + horizon
    if target_pos >= len(base_index):
        return []
    origin_label = base_index[origin_pos]
    target_label = base_index[target_pos]
    recursive_panel = pd.DataFrame(item["recursive_panel"]).copy()
    recursive_builder = item["recursive_builder"]
    future_policy = cast(FutureFeaturePolicy, item.get("future_feature_policy") or "target_lags")
    row = {
        **item["row"],
        "horizon": horizon,
        "forecast_policy": "recursive",
        "target_key": item.get("target_key"),
        "future_feature_policy": future_policy,
    }
    origin_level = _target_level_at(recursive_panel, target, origin_label)
    actual_level = _target_level_at(pd.DataFrame(item.get("actual_panel", recursive_panel)), target, target_label)
    records: list[dict[str, Any]] = []

    for model_run in model_runs:
        model_spec = model_run.spec
        # The shared skeleton covers selection -> fit -> save; the roll-forward
        # below produces the predictions itself, so no X_test is passed. The
        # recursive policy never injects ``direct=True`` (its whole point is the
        # iterated roll-forward) and routes IC-capable models through the
        # CV/validation-split branch exactly as it always has
        # (ic_selection_enabled=False). Model saving now happens at fit time,
        # before the roll-forward, instead of after it -- the saved content is
        # unchanged because predicting does not mutate the fit.
        outcome = _fit_one_model_at_origin(
            model_run,
            X_fit,
            y_fit,
            X_selection,
            y_selection,
            None,
            cfg,
            direct_capable_flag=False,
            cache_key=_model_cache_key(model_run.alias, row.get("target_key")),
            row=row,
            selection_splits=item.get("selection_splits"),
            target_step=horizon,
            allow_non_temporal_splits=_allow_non_temporal_selection_splits(item),
            ic_selection_enabled=False,
            retuned_metadata_extras={
                "recursive": True,
                "future_feature_policy": future_policy,
            },
        )
        fit = outcome.fit
        fit_params = outcome.fit_params
        selection_metadata = outcome.selection_metadata
        stored_model = outcome.stored_model
        working_panel = recursive_panel.copy()
        step_predictions: list[float] = []
        step_levels: list[float] = []
        current_level = origin_level
        for step in range(1, horizon + 1):
            feature_label = base_index[origin_pos + step - 1]
            step_features = _test_feature_builder(recursive_builder).transform(
                working_panel,
                index=pd.Index([feature_label]),
            )
            X_step = step_features.X
            pred = _prediction_series(fit.predict(X_step), index=X_step.index)
            step_value = float(pred.iloc[0])
            current_level = _recursive_next_level(
                current_level,
                step_value,
                transform=transform,
            )
            step_predictions.append(step_value)
            # An ill-conditioned fit (e.g. near-collinear level predictors) can
            # produce a non-finite recursive level. Stop the recursion and report
            # the forecast as missing rather than poisoning the working panel
            # (panel validation rejects inf) or feeding NaN into the next predict.
            if not np.isfinite(current_level):
                current_level = float("nan")
                step_levels.append(current_level)
                break
            step_levels.append(current_level)
            update_pos = origin_pos + step
            if update_pos < len(base_index) and base_index[update_pos] in working_panel.index:
                working_panel.loc[base_index[update_pos], target] = current_level
        final_prediction = _recursive_output_value(
            origin_level,
            current_level,
            transform=transform,
        )
        actual_value = _recursive_output_value(
            origin_level,
            actual_level,
            transform=transform,
        )
        records.append(
            {
                "date": target_label,
                "origin": row.get("origin"),
                "origin_pos": row.get("origin_pos"),
                "horizon": horizon,
                "forecast_policy": "recursive",
                "target_transform": transform,
                "target": target,
                "model": model_run.alias,
                "model_spec": model_spec.name,
                "prediction": float(final_prediction),
                "variance_prediction": None,
                "quantile_predictions": None,
                "actual": float(actual_value),
                "params": {
                    **dict(fit_params),
                    "recursive": {
                        "future_feature_policy": future_policy,
                        "step_predictions": step_predictions,
                        "step_levels": step_levels,
                    },
                },
                "model_selection": selection_metadata,
                "stored_model": stored_model,
                "window": row,
                "preprocessed": bool(item.get("preprocessed", False)),
                "combined": False,
                "combination": None,
            }
        )
    return records


# Bottom import for the same circularity reason as policies.base:
# _test_feature_builder is feature-window plumbing shared with the runner
# orchestration and still lives there.
from macroforecast.forecasting.runner import (  # noqa: E402
    _test_feature_builder,
)


# ---------------------------------------------------------------------------
# Recursive level math (Phase 5; moved verbatim from runner).
# ---------------------------------------------------------------------------


def _target_level_at(panel: pd.DataFrame, target: str, label: Any) -> float:
    if target not in panel.columns:
        raise ValueError(f"recursive target {target!r} is not present in the panel")
    if label not in panel.index:
        raise ValueError(f"recursive target date {label!r} is not present in the panel")
    value = panel.loc[label, target]
    if pd.isna(value):
        raise ValueError(f"recursive target {target!r} is missing at {label!r}")
    return float(value)


def _recursive_next_level(
    current_level: float,
    prediction: float,
    *,
    transform: str,
) -> float:
    if transform == "level":
        return float(prediction)
    if transform == "change":
        return float(current_level + prediction)
    if transform == "growth":
        return float(current_level * (1.0 + prediction))
    if transform == "log_growth":
        return float(current_level * np.exp(prediction))
    raise ValueError(
        "recursive forecasting supports target_transform level, change, growth, or log_growth"
    )


def _recursive_output_value(
    origin_level: float,
    final_level: float,
    *,
    transform: str,
) -> float:
    if transform == "level":
        return float(final_level)
    if transform == "change":
        return float(final_level - origin_level)
    if transform == "growth":
        if origin_level == 0:
            return float("nan")
        return float(final_level / origin_level - 1.0)
    if transform == "log_growth":
        if origin_level <= 0 or final_level <= 0:
            return float("nan")
        return float(np.log(final_level) - np.log(origin_level))
    raise ValueError(
        "recursive forecasting supports target_transform level, change, growth, or log_growth"
    )
