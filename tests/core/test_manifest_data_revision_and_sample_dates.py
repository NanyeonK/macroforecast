"""Cycle 14 K-3 — manifest provenance captures data_revision_tag and resolved sample dates.

Tests:
1. sample_start_resolved and sample_end_resolved keys exist in manifest.json provenance.
2. Resolved sample dates are parseable as pd.Timestamp.
3. data_revision_tag key exists in manifest provenance.
4. sample_start_resolved matches the first panel date.

Uses custom_panel_only recipe to avoid FRED I/O.

Closes: Cycle 14 F-H5 (P1-8)
"""
from __future__ import annotations

import json
import pandas as pd
import macroforecast as mf


# Full DAG-form recipe matching the J-3 pattern for offline execution.
_INLINE_RECIPE = """
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
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01,
             2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01,
             2019-01-01, 2019-02-01, 2019-03-01, 2019-04-01, 2019-05-01, 2019-06-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
          1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
           0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
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


def _run_and_load_manifest(tmp_path):
    """Run recipe, return parsed manifest.json dict and its provenance sub-dict."""
    out = tmp_path / "out"
    r = mf.run(_INLINE_RECIPE, output_directory=str(out))
    assert r.cells[0].succeeded, f"run failed: {r.cells[0].error}"
    m = json.loads((out / "manifest.json").read_text())
    # provenance is at top-level in the mf.run() manifest format
    prov = m.get("provenance", m)
    return m, prov


def test_manifest_provenance_has_sample_start_resolved(tmp_path):
    """manifest.json provenance must contain sample_start_resolved key (K-3 fix)."""
    _, prov = _run_and_load_manifest(tmp_path)
    assert "sample_start_resolved" in prov, (
        f"sample_start_resolved missing from manifest provenance. Keys: {list(prov.keys())}"
    )


def test_manifest_provenance_has_sample_end_resolved(tmp_path):
    """manifest.json provenance must contain sample_end_resolved key (K-3 fix)."""
    _, prov = _run_and_load_manifest(tmp_path)
    assert "sample_end_resolved" in prov, (
        f"sample_end_resolved missing from manifest provenance. Keys: {list(prov.keys())}"
    )


def test_manifest_sample_dates_parseable_as_timestamp(tmp_path):
    """sample_start_resolved and sample_end_resolved must be parseable as pd.Timestamp."""
    _, prov = _run_and_load_manifest(tmp_path)
    start = prov.get("sample_start_resolved")
    end = prov.get("sample_end_resolved")
    assert start is not None, "sample_start_resolved is None — expected a date string"
    assert end is not None, "sample_end_resolved is None — expected a date string"
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    assert ts_start < ts_end, (
        f"sample_start_resolved ({ts_start}) must be before sample_end_resolved ({ts_end})"
    )


def test_manifest_sample_start_matches_panel_start(tmp_path):
    """sample_start_resolved must match the first date in the inline panel."""
    _, prov = _run_and_load_manifest(tmp_path)
    start = pd.Timestamp(prov["sample_start_resolved"])
    # The inline panel starts at 2018-01-01
    assert start.year == 2018, f"Expected start year 2018, got {start.year}"
    assert start.month == 1, f"Expected start month 1, got {start.month}"


def test_manifest_provenance_has_data_revision_tag(tmp_path):
    """manifest.json provenance must contain data_revision_tag key (may be empty for custom inline)."""
    _, prov = _run_and_load_manifest(tmp_path)
    assert "data_revision_tag" in prov, (
        f"data_revision_tag missing from manifest provenance. Keys: {list(prov.keys())}"
    )
    # For custom_panel_only, tag is expected to be "" (no FRED data_through)
    tag = prov["data_revision_tag"]
    assert isinstance(tag, str), f"data_revision_tag must be str, got {type(tag)}"
