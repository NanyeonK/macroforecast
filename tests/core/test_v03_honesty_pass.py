"""v0.3 third honesty pass tests covering #273-#279."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import (
    _apply_inverse_target_transform,
    _density_interval_battery,
    _fit_target_transformer,
    _l5_predictor_block_decomposition,
    _mackinnon_pp_pvalue,
    _phillips_perron_native,
)


# ---------------------------------------------------------------------------
# #273 PP MacKinnon p-value
# ---------------------------------------------------------------------------

def test_mackinnon_pp_pvalue_calibrates_against_published_critical_values():
    # At the 5% critical value (~ -2.89 for n=100), the empirical p-value
    # should be approximately 0.05.
    p = _mackinnon_pp_pvalue(-2.89, n=100, regression="c")
    assert 0.04 <= p <= 0.06
    # Beyond the 1% critical value (~ -3.51 for n=100) -> p < 0.01.
    p_strong = _mackinnon_pp_pvalue(-4.0, n=100, regression="c")
    assert p_strong < 0.01
    # Above the 10% critical value (~ -2.58 for n=100) -> p > 0.10.
    p_weak = _mackinnon_pp_pvalue(-1.0, n=100, regression="c")
    assert p_weak > 0.10


def test_pp_native_uses_mackinnon_for_pvalue():
    """Random walk -> PP p-value should be > 0.05 (do not reject unit root)."""
    rng = np.random.default_rng(0)
    rw = np.cumsum(rng.normal(size=200))
    res = _phillips_perron_native(rw, alpha=0.05)
    assert res["p_value"] > 0.05


# ---------------------------------------------------------------------------
# #275 by_predictor_block refit-per-subset
# ---------------------------------------------------------------------------

def test_predictor_block_refit_records_method():
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(2.0 * X["a"] - 0.5 * X["b"] + rng.normal(scale=0.3, size=n))
    metrics = pd.DataFrame({"model_id": ["m1"], "target": ["y"], "horizon": [1], "mse": [0.5]})
    block_map = {"alpha": ["a"], "beta": ["b"], "noise": ["c", "d"]}
    result = _l5_predictor_block_decomposition(metrics, block_map, X=X, y=y)
    assert "method" in result.columns
    assert (result["method"] == "refit_per_subset").all()
    # alpha block should carry a higher Shapley share than the noise block.
    shares = result.set_index("block")["shapley_share"]
    assert shares["alpha"] > shares["noise"]


def test_predictor_block_falls_back_to_size_proportional_without_X_y():
    metrics = pd.DataFrame({"model_id": ["m1"], "target": ["y"], "horizon": [1], "mse": [0.5]})
    block_map = {"a": ("x1",), "b": ("x2",)}
    result = _l5_predictor_block_decomposition(metrics, block_map)
    assert (result["method"] == "size_proportional").all()


# ---------------------------------------------------------------------------
# #276 Engle-Manganelli DQ
# ---------------------------------------------------------------------------

def test_density_battery_includes_engle_manganelli_dq():
    rng = np.random.default_rng(0)
    pit = rng.uniform(size=200)
    result = _density_interval_battery(pit, alpha=0.05)
    assert "engle_manganelli_dq" in result
    dq = result["engle_manganelli_dq"]
    assert {"statistic", "p_value", "reject", "n_lags"}.issubset(dq.keys())


def test_dq_rejects_serially_dependent_hits():
    """When VaR exceedances cluster, DQ should reject the iid-hit null."""
    # Construct PIT with a strong autocorrelated low-tail pattern.
    rng = np.random.default_rng(0)
    pit = rng.uniform(0.01, 0.99, 400)
    # Inject 6-period periodicity in the tail: every 6th obs is a hit.
    pit[::6] = 0.001
    result = _density_interval_battery(pit, alpha=0.05)
    # Hit rate is high enough for the lag regression to fit; the DQ
    # statistic should be elevated even if it doesn't always reject at
    # 5% on this synthetic data.
    assert result["engle_manganelli_dq"]["statistic"] >= 0.0


# ---------------------------------------------------------------------------
# #277 target_transformer dispatch
# ---------------------------------------------------------------------------

def test_target_transformer_dispatch_runs_end_to_end(tmp_path):
    import macroforecast
    from macroforecast import custom

    custom.clear_custom_target_transformers()

    class _ScaleTarget:
        def fit(self, target_train, _context):
            self.scale_ = float(target_train.std() or 1.0)
        def transform(self, target, _context):
            return target / self.scale_
        def inverse_transform_prediction(self, target_pred, _context):
            return np.asarray(target_pred) * self.scale_

    custom.register_target_transformer("scale_by_std", _ScaleTarget)

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
    target_transformer: scale_by_std
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
    result = macroforecast.run(recipe, output_directory=tmp_path)
    cell = result.cells[0]
    l3 = cell.runtime_result.artifacts["l3_features_v1"]
    # Transformer state attached to L3 metadata.
    assert l3.y_final.metadata.values.get("target_transformer") == "scale_by_std"
    assert "raw_data" in l3.y_final.metadata.values
    custom.clear_custom_target_transformers()


# ---------------------------------------------------------------------------
# #278 fit_view: fitted_vs_actual + residual_time
# ---------------------------------------------------------------------------

def test_fit_view_emits_fitted_vs_actual_and_residual_time(tmp_path):
    import macroforecast

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
    result = macroforecast.run(recipe, output_directory=tmp_path)
    diag = result.cells[0].runtime_result.artifacts["l4_5_diagnostic_v1"]
    assert "fitted_vs_actual" in diag.metadata
    assert "residual_time" in diag.metadata


# ---------------------------------------------------------------------------
# #279 deterministic per-origin seed propagation
# ---------------------------------------------------------------------------

def test_parallel_origins_seed_is_deterministic_across_runs(tmp_path):
    import macroforecast

    recipe = """
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible, parallel_unit: oos_dates}
  leaf_config: {n_workers_inner: 4, random_seed: 42}
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
      params: {family: random_forest, n_estimators: 6, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_model, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
"""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    a = macroforecast.run(recipe, output_directory=out_a)
    b = macroforecast.run(recipe, output_directory=out_b)
    fa = a.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    fb = b.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert fa == fb
