"""Issues #255 + #256 -- real Chow-Lin disaggregation + extra L4.5 views."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macrocast.core.runtime import _chow_lin_disaggregate


# ---------------------------------------------------------------------------
# #255 Chow-Lin
# ---------------------------------------------------------------------------

def test_chow_lin_recovers_indicator_relationship():
    """Generate a quarterly target = 2 * monthly indicator (averaged) +
    noise. Chow-Lin should disaggregate close to the true indicator
    series at monthly frequency."""

    rng = np.random.default_rng(0)
    monthly_idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    indicator_monthly = pd.Series(rng.normal(0, 1, 60).cumsum(), index=monthly_idx)
    # True monthly target = 2 * indicator + small noise.
    true_monthly = 2.0 * indicator_monthly + rng.normal(0, 0.1, 60)
    # Pass the quarterly observations at their native QE index.
    quarterly = true_monthly.resample("QE").mean()
    disagg = _chow_lin_disaggregate(quarterly, indicator_monthly)
    # Disaggregated monthly series should correlate strongly with the
    # true monthly target.
    valid = ~disagg.isna()
    correlation = float(disagg[valid].corr(true_monthly[valid]))
    assert correlation > 0.85


def test_chow_lin_handles_too_few_observations_gracefully():
    monthly_idx = pd.date_range("2018-01-01", periods=4, freq="MS")
    indicator = pd.Series([1.0, 2.0, 3.0, 4.0], index=monthly_idx)
    quarterly = pd.Series([np.nan, np.nan, np.nan, 2.5], index=monthly_idx)
    result = _chow_lin_disaggregate(quarterly, indicator)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# #256 L4.5 residual views
# ---------------------------------------------------------------------------

def test_residual_acf_view_recorded_when_axis_set(tmp_path):
    import macrocast

    recipe = """
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}
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
4_5_generator_diagnostics:
  enabled: true
  fixed_axes:
    fit_view: multi
"""
    result = macrocast.run(recipe, output_directory=tmp_path)
    diag = result.cells[0].runtime_result.artifacts["l4_5_diagnostic_v1"]
    assert "residual_acf" in diag.metadata
    assert "residual_qq" in diag.metadata
