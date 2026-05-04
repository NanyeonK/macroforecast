"""Issues #202 / #203 / #204 -- FRED-SD freq align + NodeGroupSweep + parallel_unit."""
from __future__ import annotations

import numpy as np
import pandas as pd

import macrocast


# ---------------------------------------------------------------------------
# #203 NodeGroupSweep
# ---------------------------------------------------------------------------

_NODE_GROUP_RECIPE = """
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
      x2: [0.1, 0.4, 0.2, 0.6, 0.3, 0.7, 0.5, 0.8, 0.4, 0.9]
2_preprocessing:
  fixed_axes: {transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
3_feature_engineering:
  sweep_groups:
    - id: pipeline_lag1
      nodes:
        - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
        - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
        - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
        - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
      sinks:
        l3_features_v1: {X_final: lag_x, y_final: y_h}
        l3_metadata_v1: auto
    - id: pipeline_lag2
      nodes:
        - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
        - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
        - {id: lag_x, type: step, op: lag, params: {n_lag: 2}, inputs: [src_X]}
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
  fixed_axes: {primary_metric: mse, point_metrics: [mse]}
"""


def test_sweep_groups_expand_into_two_cells(tmp_path):
    result = macrocast.run(_NODE_GROUP_RECIPE, output_directory=tmp_path)
    assert len(result.cells) == 2
    # Two cells should have different L3 sink hashes (different lag values).
    a, b = result.cells
    assert a.sink_hashes["l3_features_v1"] != b.sink_hashes["l3_features_v1"]


# ---------------------------------------------------------------------------
# #204 parallel_unit = models
# ---------------------------------------------------------------------------

_PARALLEL_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
    parallel_unit: models
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
    - id: fit_a
      type: step
      op: fit_model
      params: {family: ridge, alpha: 0.1, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - id: fit_b
      type: step
      op: fit_model
      params: {family: ols, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_a, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_a
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse, point_metrics: [mse]}
"""


def test_parallel_unit_models_runs_two_fit_nodes(tmp_path):
    result = macrocast.run(_PARALLEL_RECIPE, output_directory=tmp_path)
    cell = result.cells[0]
    forecasts = cell.runtime_result.artifacts["l4_forecasts_v1"].forecasts
    # Both ``fit_a`` and ``fit_b`` produced forecasts.
    model_ids = {key[0] for key in forecasts}
    assert {"fit_a", "fit_b"} <= model_ids


# ---------------------------------------------------------------------------
# #202 FRED-SD frequency alignment helper (unit-level)
# ---------------------------------------------------------------------------

def test_fred_sd_freq_align_step_backward_fills_quarterly():
    from macrocast.core.runtime import _apply_fred_sd_frequency_alignment
    from macrocast.core.types import L1DataDefinitionArtifact, Panel, PanelMetadata

    idx = pd.date_range("2010-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "monthly_a": np.arange(12, dtype=float),
            "quarterly_b": [1.0, np.nan, np.nan, 2.0, np.nan, np.nan, 3.0, np.nan, np.nan, 4.0, np.nan, np.nan],
        },
        index=idx,
    )
    panel = Panel(
        data=df,
        shape=df.shape,
        column_names=tuple(df.columns),
        index=df.index,
        metadata=PanelMetadata(values={"series_frequency": {"monthly_a": "monthly", "quarterly_b": "quarterly"}}),
    )
    artifact = L1DataDefinitionArtifact(
        custom_source_policy="official_only",
        dataset="fred_sd",
        frequency="monthly",
        vintage_policy="current_vintage",
        target_structure="single_target",
        target="monthly_a",
        targets=("monthly_a",),
        variable_universe="all_variables",
        target_geography_scope=None,
        predictor_geography_scope=None,
        sample_start_rule="max_balanced",
        sample_end_rule="latest_available",
        horizon_set="standard_md",
        target_horizons=(1,),
        regime_definition="none",
        raw_panel=panel,
        leaf_config={},
    )
    log: dict[str, Any] = {"steps": []}  # noqa: F821
    aligned = _apply_fred_sd_frequency_alignment(
        df.copy(),
        resolved={"sd_series_frequency_filter": "both", "quarterly_to_monthly_rule": "step_backward"},
        l1_artifact=artifact,
        cleaning_log=log,
    )
    assert aligned["quarterly_b"].notna().all()
    # Step-backward fills with the next available value, so 2010-01 = 1.0.
    assert aligned["quarterly_b"].iloc[0] == 1.0
