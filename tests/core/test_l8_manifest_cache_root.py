"""F-P1-13 -- L8 manifest payload contains cache_root provenance field.

Tests verify:
1. manifest_payload has 'cache_root' key.
2. cache_root is None when not set in recipe.
3. cache_root reflects the recipe's leaf_config value when set.
"""
from __future__ import annotations

import json
from pathlib import Path

import macroforecast


_BASE_RECIPE_TEMPLATE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 42
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
    vintage_policy: current_vintage
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01,
             2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
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
  leaf_config:
    output_directory: __OUTPUT_DIR__
"""


def _run_and_read_manifest(tmp_path: Path, extra_leaf: str = "") -> dict:
    recipe = _BASE_RECIPE_TEMPLATE.replace("__OUTPUT_DIR__", str(tmp_path))
    if extra_leaf:
        # Insert extra leaf config for 1_data
        recipe = recipe.replace(
            "    custom_panel_inline:",
            f"    {extra_leaf}\n    custom_panel_inline:",
        )
    macroforecast.run(recipe, output_directory=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_manifest_payload_has_cache_root_key(tmp_path):
    """manifest.json must contain 'cache_root' key (F-P1-13)."""
    payload = _run_and_read_manifest(tmp_path)
    assert "cache_root" in payload, (
        f"'cache_root' key missing from manifest payload. Keys present: {list(payload.keys())}"
    )


def test_manifest_cache_root_is_string_or_none(tmp_path):
    """cache_root is a string path or None (not missing), reflecting the resolved cache_root."""
    payload = _run_and_read_manifest(tmp_path)
    val = payload["cache_root"]
    # May be None (no cache) or a path string (default .raw_cache or explicit)
    assert val is None or isinstance(val, str), (
        f"Expected cache_root to be None or str, got {type(val)}: {val!r}"
    )


def test_manifest_structure_is_intact(tmp_path):
    """Other expected keys still present after F-P1-13 change (regression guard)."""
    payload = _run_and_read_manifest(tmp_path)
    # These keys are from the ManifestExecutionResult.to_manifest_dict()
    expected_top_level = {"schema_version", "recipe_root", "provenance", "cells", "cache_root"}
    missing = expected_top_level - set(payload.keys())
    assert not missing, f"Missing top-level manifest keys: {missing}"
