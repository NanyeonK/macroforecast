"""Issue #250 -- ``parallel_unit = oos_dates`` fans out the walk-forward
origin loop. Recipe with ``parallel_unit = oos_dates`` produces the same
forecasts as the sequential path."""
from __future__ import annotations

from pathlib import Path

import macrocast


_BASE = """
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible, parallel_unit: __PU__}
  leaf_config:
    n_workers_inner: 2
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
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
      params: {family: ridge, alpha: 0.1, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_model, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse}
"""


def test_parallel_unit_oos_dates_matches_serial(tmp_path):
    out_serial = tmp_path / "serial"
    out_parallel = tmp_path / "parallel"
    serial = macrocast.run(_BASE.replace("__PU__", "cells"), output_directory=out_serial)
    parallel = macrocast.run(_BASE.replace("__PU__", "oos_dates"), output_directory=out_parallel)
    fa = serial.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    fb = parallel.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert fa == fb


def test_parallel_unit_horizons_runs_without_error(tmp_path):
    """``horizons`` and ``targets`` map to the same per-origin fan-out
    when the L4 runtime produces a single horizon / target per fit_node."""

    out = tmp_path / "horizons"
    result = macrocast.run(_BASE.replace("__PU__", "horizons"), output_directory=out)
    assert result.cells[0].succeeded
