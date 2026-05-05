"""Issue #216 -- ``macroforecast.custom.register_model`` callables must dispatch
end-to-end through ``execute_recipe`` (validator + ``_build_l4_model``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast
from macroforecast import custom


_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
2_preprocessing:
  fixed_axes: {transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
3_feature_engineering:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_model
      type: step
      op: fit_model
      params: {family: my_constant_model, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_model, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse, point_metrics: [mse]}
"""


@pytest.fixture(autouse=True)
def _reset_registry():
    custom.clear_custom_models()
    yield
    custom.clear_custom_models()


def _constant_model(X_train, y_train, X_test, context):
    return float(y_train.mean())


def test_custom_model_passes_validator(tmp_path):
    custom.register_model("my_constant_model", _constant_model)
    result = macroforecast.run(_RECIPE, output_directory=tmp_path)
    assert all(cell.succeeded for cell in result.cells)


def test_custom_model_predictions_match_train_mean(tmp_path):
    custom.register_model("my_constant_model", _constant_model)
    result = macroforecast.run(_RECIPE, output_directory=tmp_path)
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    # Every prediction equals the running training mean -- since each origin
    # uses an expanding window, predictions are monotonically increasing.
    values = [forecasts[k] for k in sorted(forecasts.keys(), key=lambda x: x[3])]
    assert all(v <= values[i + 1] for i, v in enumerate(values[:-1]))


def test_custom_model_artifact_records_custom_framework(tmp_path):
    custom.register_model("my_constant_model", _constant_model)
    result = macroforecast.run(_RECIPE, output_directory=tmp_path)
    art = result.cells[0].runtime_result.artifacts["l4_model_artifacts_v1"]
    fitted = list(art.artifacts.values())[0]
    assert fitted.family == "my_constant_model"
    assert fitted.framework == "custom"


def test_unregistered_family_still_rejected(tmp_path):
    # No registration for ``totally_unknown_family``.
    bad = _RECIPE.replace("my_constant_model", "totally_unknown_family")
    with pytest.raises(Exception, match="unknown model family"):
        macroforecast.run(bad, output_directory=tmp_path)
