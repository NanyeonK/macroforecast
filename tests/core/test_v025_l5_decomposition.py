"""Issue #258 -- L5 decomposition / oos_period / aggregation tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.core.runtime import _l5_per_subperiod_metrics, _l5_predictor_block_decomposition


def _toy_per_origin():
    rng = np.random.default_rng(0)
    rows = []
    for origin in pd.date_range("2010-01-01", periods=24, freq="MS"):
        for model_id in ("m1", "m2"):
            err = rng.normal(scale=0.5)
            rows.append(
                {"model_id": model_id, "target": "y", "horizon": 1, "origin": origin,
                 "squared_error": err ** 2, "absolute_error": abs(err)}
            )
    return pd.DataFrame(rows)


def test_per_subperiod_splits_at_user_boundaries():
    per_origin = _toy_per_origin()
    boundaries = ["2010-12-01"]
    result = _l5_per_subperiod_metrics(per_origin, l4_models=None, boundaries=boundaries)
    assert "subperiod" in result.columns
    # Two boundaries -> two subperiods (sp_0 = before, sp_1 = after).
    assert set(result["subperiod"].unique()) == {"sp_0", "sp_1"}
    assert {"mse", "mae"}.issubset(result.columns)


def test_per_subperiod_full_oos_when_no_boundaries():
    per_origin = _toy_per_origin()
    result = _l5_per_subperiod_metrics(per_origin, l4_models=None, boundaries=[])
    assert (result["subperiod"] == "full_oos").all()


def test_predictor_block_decomposition_shares_sum_to_one_per_target_horizon():
    metrics = pd.DataFrame(
        {"model_id": ["m1"], "target": ["y"], "horizon": [1], "mse": [0.5]}
    )
    block_map = {"block_a": ("x1", "x2"), "block_b": ("x3",)}
    result = _l5_predictor_block_decomposition(metrics, block_map)
    assert {"block", "shapley_share", "block_mse_contribution"}.issubset(result.columns)
    # Shapley shares per (target, horizon) must sum to 1 (allocation
    # property of the normalised payoff function).
    for (_target, _horizon), group in result.groupby(["target", "horizon"]):
        assert abs(group["shapley_share"].sum() - 1.0) < 1e-6


def test_predictor_block_decomposition_handles_many_blocks():
    metrics = pd.DataFrame(
        {"model_id": ["m1"], "target": ["y"], "horizon": [1], "mse": [1.0]}
    )
    # 10 blocks -> falls back to size-proportional allocation.
    block_map = {f"block_{i}": (f"x_{i}",) for i in range(10)}
    result = _l5_predictor_block_decomposition(metrics, block_map)
    assert len(result) == 10
    # Equal-sized blocks -> equal shares.
    np.testing.assert_allclose(result["shapley_share"].values, np.full(10, 0.1), rtol=1e-9)


def test_per_horizon_then_mean_aggregation_via_full_l5_recipe(tmp_path):
    """Smoke test that the resolved_axes carry the decomposition_tables
    when ``aggregation_axis = per_horizon_then_mean`` is enabled."""

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
  fixed_axes:
    primary_metric: mse
    agg_horizon: per_horizon_then_mean
"""
    result = macroforecast.run(recipe, output_directory=tmp_path)
    eval_artifact = result.cells[0].runtime_result.artifacts["l5_evaluation_v1"]
    assert "decomposition_tables" in eval_artifact.l5_axis_resolved
    assert "per_horizon_then_mean" in eval_artifact.l5_axis_resolved["decomposition_tables"]
