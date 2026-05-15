"""Cycle 14 J-3 — test that l1_data_definition_v1 sink hash is path-independent.

F-19-26/27/30: cache_root (which defaults to {output_directory}/.raw_cache) was
included in the _stable_repr of L1DataDefinitionArtifact.leaf_config, causing the
l1_data_definition_v1 hash to differ across runs with different output directories.
Fix: strip cache_root from leaf_config before hashing.
"""
from __future__ import annotations

import json

import pytest
import macroforecast as mf


# Minimal offline recipe: custom_panel_only, 12-month panel, single ridge model.
# Mirrors pattern from test_execution.py / test_l8_manifest_cache_root.py.
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
      date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01, 2020-05-01, 2020-06-01,
             2020-07-01, 2020-08-01, 2020-09-01, 2020-10-01, 2020-11-01, 2020-12-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
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


def test_l1_sink_hash_path_independent(tmp_path):
    """l1_data_definition_v1 hash MUST be identical for two runs with different
    output_directory values but the same recipe content (J-3 regression guard)."""
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    r1 = mf.run(_INLINE_RECIPE, output_directory=str(out1))
    r2 = mf.run(_INLINE_RECIPE, output_directory=str(out2))

    assert r1.cells[0].succeeded, f"run1 failed: {r1.cells[0].error}"
    assert r2.cells[0].succeeded, f"run2 failed: {r2.cells[0].error}"

    h1 = r1.cells[0].sink_hashes["l1_data_definition_v1"]
    h2 = r2.cells[0].sink_hashes["l1_data_definition_v1"]
    assert h1 == h2, (
        f"l1_data_definition_v1 hash differs across output directories: "
        f"{h1!r} vs {h2!r}. Regression of Cycle 14 J-3 fix."
    )


def test_l1_sink_hash_replicate(tmp_path):
    """mf.replicate() must report sink_hashes_match=True when recipe is unchanged
    (J-3 smoking gun: previously sink_hashes_match=False due to cache_root in hash)."""
    out1 = tmp_path / "replicate_out1"
    r1 = mf.run(_INLINE_RECIPE, output_directory=str(out1))
    assert r1.cells[0].succeeded, f"initial run failed: {r1.cells[0].error}"

    manifest_path = str(out1 / "manifest.json")
    rep = mf.replicate(manifest_path)
    assert rep.sink_hashes_match, (
        "mf.replicate() reported sink_hashes_match=False for an unchanged recipe. "
        "Regression of Cycle 14 J-3 fix."
    )


def test_l1_hash_excludes_cache_root_directly():
    """Unit test: _stable_repr of L1DataDefinitionArtifact must not include
    cache_root in the hashed representation of leaf_config."""
    from macroforecast.core.execution import _stable_repr
    from macroforecast.core.types import L1DataDefinitionArtifact

    artifact = L1DataDefinitionArtifact(
        custom_source_policy="custom_panel_only",
        dataset=None,
        frequency="monthly",
        vintage_policy=None,
        target_structure="single_target",
        target="y",
        leaf_config={
            "target": "y",
            "cache_root": "/some/output/dir/.raw_cache",
            "target_horizons": [1],
        },
    )

    repr_dict = _stable_repr(artifact)

    # leaf_config in the stable repr must NOT contain cache_root
    leaf_repr = repr_dict.get("leaf_config", {})
    assert "cache_root" not in leaf_repr, (
        f"cache_root found in _stable_repr(leaf_config): {leaf_repr}. "
        "J-3 fix should have excluded it."
    )
    # Other leaf_config keys must still be present
    assert "target" in leaf_repr, "Non-path key 'target' should remain in leaf_config repr"
    assert "target_horizons" in leaf_repr, "Non-path key 'target_horizons' should remain"
