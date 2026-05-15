"""Cycle 15 M-1 -- manifest sample_start_resolved / sample_end_resolved reflect post-L2 window.

Bug (Cycle 14 K-3): sample_start_resolved / sample_end_resolved were read from
_l1_art.raw_panel.data.index (pre-window) rather than the post-L2 clean panel.
A user with sample_start_rule=fixed_date, sample_start_date="2020-01-01" on a
2018-2023 panel got manifest showing 2018-01-01, not 2020-01-01.

Fix (Cycle 15 M-1): prefer l2_clean_panel_v1.panel.data.index; fallback to
raw_panel only when L2 artifact is absent.

Tests:
1. fixed_date sample_start / end rules -> manifest records windowed dates, not raw panel dates.
2. Explicit guard: start != raw panel start (2018-01).
3. Explicit guard: end != raw panel end (2023-12).
"""
from __future__ import annotations

import json

import pandas as pd

import macroforecast as mf


# ---------------------------------------------------------------------------
# Recipe: 2018-01 to 2023-12 inline panel (72 months).
# sample_start_rule=fixed_date 2020-01-01, sample_end_rule=fixed_date 2022-12-01.
# Note: sample_start_rule / sample_end_rule belong in 1_data.fixed_axes.
# ---------------------------------------------------------------------------
_WINDOWED_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 1
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
    sample_start_rule: fixed_date
    sample_end_rule: fixed_date
  leaf_config:
    target: y
    target_horizons: [1]
    sample_start_date: "2020-01-01"
    sample_end_date: "2022-12-01"
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01,
             2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01,
             2019-01-01, 2019-02-01, 2019-03-01, 2019-04-01, 2019-05-01, 2019-06-01,
             2019-07-01, 2019-08-01, 2019-09-01, 2019-10-01, 2019-11-01, 2019-12-01,
             2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01, 2020-05-01, 2020-06-01,
             2020-07-01, 2020-08-01, 2020-09-01, 2020-10-01, 2020-11-01, 2020-12-01,
             2021-01-01, 2021-02-01, 2021-03-01, 2021-04-01, 2021-05-01, 2021-06-01,
             2021-07-01, 2021-08-01, 2021-09-01, 2021-10-01, 2021-11-01, 2021-12-01,
             2022-01-01, 2022-02-01, 2022-03-01, 2022-04-01, 2022-05-01, 2022-06-01,
             2022-07-01, 2022-08-01, 2022-09-01, 2022-10-01, 2022-11-01, 2022-12-01,
             2023-01-01, 2023-02-01, 2023-03-01, 2023-04-01, 2023-05-01, 2023-06-01,
             2023-07-01, 2023-08-01, 2023-09-01, 2023-10-01, 2023-11-01, 2023-12-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
          1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5,
          2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
          2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5,
          3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
          3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
      x1: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
           0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25,
           0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3,
           0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35,
           0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
           0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45]
2_preprocessing:
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
3_feature_engineering:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}
    - id: src_y
      type: source
      selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}
    - id: lag_x
      type: step
      op: lag
      params: {n_lag: 1}
      inputs: [src_X]
    - id: y_h
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_y]
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}
    - id: src_y
      type: source
      selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}
    - id: fit_model
      type: step
      op: fit_model
      params:
        family: ridge
        alpha: 0.1
        min_train_size: 4
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
      inputs: [src_X, src_y]
    - id: predict
      type: step
      op: predict
      inputs: [fit_model, src_X]
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse]
8_output:
  fixed_axes:
    export_format: json_csv
    saved_objects: [forecasts, metrics]
    artifact_granularity: per_cell
    naming_convention: descriptive
"""


def _run_windowed(tmp_path):
    """Run windowed recipe, return provenance sub-dict from manifest.json."""
    out = tmp_path / "out"
    r = mf.run(_WINDOWED_RECIPE, output_directory=str(out))
    assert r.cells[0].succeeded, f"run failed: {r.cells[0].error}"
    m = json.loads((out / "manifest.json").read_text())
    prov = m.get("provenance", m)
    return prov


def test_manifest_sample_window_resolved_post_l2(tmp_path):
    """Cycle 15 M-1: sample_start_resolved/sample_end_resolved must reflect post-L2 window.

    Raw panel spans 2018-01 to 2023-12.
    sample_start_rule=fixed_date 2020-01-01, sample_end_rule=fixed_date 2022-12-01.
    Manifest must record windowed dates (2020-01, 2022-12), NOT raw panel dates (2018-01, 2023-12).
    """
    prov = _run_windowed(tmp_path)
    start_recorded = pd.Timestamp(prov["sample_start_resolved"])
    end_recorded = pd.Timestamp(prov["sample_end_resolved"])
    # MUST match the explicit fixed_date rules, not raw panel boundaries
    assert start_recorded.year == 2020 and start_recorded.month == 1, (
        f"sample_start_resolved={start_recorded} should be 2020-01 (fixed_date rule), "
        "not 2018-01 (raw panel start) -- Cycle 15 M-1 regression"
    )
    assert end_recorded.year == 2022 and end_recorded.month == 12, (
        f"sample_end_resolved={end_recorded} should be 2022-12 (fixed_date rule), "
        "not 2023-12 (raw panel end) -- Cycle 15 M-1 regression"
    )


def test_manifest_sample_window_start_not_raw_panel_start(tmp_path):
    """Explicit guard: sample_start_resolved must NOT equal the raw panel start (2018-01).

    This test catches any regression where the raw_panel fallback fires incorrectly.
    """
    prov = _run_windowed(tmp_path)
    start_recorded = pd.Timestamp(prov["sample_start_resolved"])
    assert not (start_recorded.year == 2018 and start_recorded.month == 1), (
        f"sample_start_resolved={start_recorded} equals raw panel start (2018-01); "
        "post-L2 window not applied -- Cycle 15 M-1 regression"
    )


def test_manifest_sample_window_end_not_raw_panel_end(tmp_path):
    """Explicit guard: sample_end_resolved must NOT equal the raw panel end (2023-12).

    This test catches any regression where the raw_panel fallback fires incorrectly.
    """
    prov = _run_windowed(tmp_path)
    end_recorded = pd.Timestamp(prov["sample_end_resolved"])
    assert not (end_recorded.year == 2023 and end_recorded.month == 12), (
        f"sample_end_resolved={end_recorded} equals raw panel end (2023-12); "
        "post-L2 window not applied -- Cycle 15 M-1 regression"
    )
