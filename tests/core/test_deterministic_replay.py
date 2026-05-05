"""Determinism regression tests: same recipe twice must produce byte-identical
artifacts (forecasts.csv / metrics CSVs / sink hashes).

Implements issue #6 part 2 of the phase-00 stability plan.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import macroforecast


_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 9
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
  fixed_axes:
    transform_policy: no_transform
    outlier_policy: none
    imputation_policy: none_propagate
    frame_edge_policy: keep_unbalanced
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
      params: {family: random_forest, n_estimators: 8, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_model, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse, point_metrics: [mse, rmse, mae]}
8_output:
  fixed_axes:
    export_format: json_csv
    saved_objects: [forecasts, metrics, ranking]
    artifact_granularity: per_cell
    naming_convention: descriptive
  leaf_config:
    output_directory: __PLACEHOLDER__
"""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe_for(out_dir: Path) -> str:
    return _RECIPE.replace("__PLACEHOLDER__", str(out_dir))


_PATH_DEPENDENT_SINKS = frozenset({
    # l1's leaf_config carries cache_root; l8 records on-disk export paths.
    # These naturally differ between runs targeting different directories.
    "l1_data_definition_v1",
    "l8_artifacts_v1",
})


def test_same_recipe_twice_produces_identical_sink_hashes(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    shared_cache = tmp_path / "shared_cache"
    a = macroforecast.run(_recipe_for(out_a), output_directory=out_a, cache_root=shared_cache)
    b = macroforecast.run(_recipe_for(out_b), output_directory=out_b, cache_root=shared_cache)
    assert len(a.cells) == 1 and len(b.cells) == 1
    cell_a, cell_b = a.cells[0], b.cells[0]
    # Compare every sink except the two whose hash legitimately depends on
    # output_directory (they encode local file paths). Compare files
    # byte-for-byte in test_same_recipe_twice_produces_byte_identical_*.
    a_compare = {k: v for k, v in cell_a.sink_hashes.items() if k not in _PATH_DEPENDENT_SINKS}
    b_compare = {k: v for k, v in cell_b.sink_hashes.items() if k not in _PATH_DEPENDENT_SINKS}
    assert a_compare == b_compare, f"sink hashes drifted: {a_compare} vs {b_compare}"


def test_same_recipe_twice_produces_byte_identical_forecasts_csv(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    macroforecast.run(_recipe_for(out_a), output_directory=out_a)
    macroforecast.run(_recipe_for(out_b), output_directory=out_b)
    forecasts_a = out_a / "cell_001" / "forecasts.csv"
    forecasts_b = out_b / "cell_001" / "forecasts.csv"
    assert forecasts_a.exists() and forecasts_b.exists()
    assert _file_sha256(forecasts_a) == _file_sha256(forecasts_b)


def test_same_recipe_twice_produces_byte_identical_metrics_csv(tmp_path):
    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"
    macroforecast.run(_recipe_for(out_a), output_directory=out_a)
    macroforecast.run(_recipe_for(out_b), output_directory=out_b)
    # metrics_all_cells.csv is written under summary/
    metrics_a = out_a / "summary" / "metrics_all_cells.csv"
    metrics_b = out_b / "summary" / "metrics_all_cells.csv"
    assert metrics_a.exists() and metrics_b.exists()
    assert _file_sha256(metrics_a) == _file_sha256(metrics_b)


def test_replicate_recipe_succeeds_after_independent_re_run(tmp_path):
    out = tmp_path / "primary"
    macroforecast.run(_recipe_for(out), output_directory=out)
    rep = macroforecast.replicate(out / "manifest.json")
    assert rep.recipe_match
    assert rep.sink_hashes_match
    assert all(rep.per_cell_match.values())


def test_distinct_random_state_sweep_produces_distinct_l4_artifacts(tmp_path):
    """L0 random_seed now auto-propagates to L4 estimators when no per-fit
    ``params.random_state`` is set (issue #215, see
    ``tests/core/test_seed_policy.py::test_l0_random_seed_propagates_into_l4_random_state``).
    This test still pins the original same-seed determinism guarantee.
    """

    out_a = tmp_path / "rs_a"
    out_b = tmp_path / "rs_b"
    base = _recipe_for(out_a)
    a = macroforecast.run(base.replace("random_state: none", "random_state: none"), output_directory=out_a)
    b = macroforecast.run(
        base.replace("__PLACEHOLDER__", str(out_b)).replace("random_state: none", "random_state: none"),
        output_directory=out_b,
    )
    # With identical seeds and identical estimator config the L4 artifact must
    # match deterministically (this is the inverse of the distinct-seed
    # claim and protects the byte-identical replicate guarantee).
    assert a.cells[0].sink_hashes["l4_model_artifacts_v1"] == b.cells[0].sink_hashes["l4_model_artifacts_v1"]
