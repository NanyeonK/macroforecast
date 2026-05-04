"""Issue #207 -- L8 ``artifact_granularity`` axis (``per_target`` /
``per_horizon`` / ``per_target_horizon`` / ``flat`` / ``per_cell``).
"""
from __future__ import annotations

from pathlib import Path

import macrocast


_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
    target_structure: multi_series_target
  leaf_config:
    targets: [y, z]
    target_horizons: [1, 2]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      z: [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
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
  fixed_axes: {primary_metric: mse, point_metrics: [mse]}
8_output:
  fixed_axes:
    export_format: json_csv
    saved_objects: [forecasts, metrics]
    artifact_granularity: __GRANULARITY__
    naming_convention: descriptive
  leaf_config:
    output_directory: __PLACEHOLDER__
"""


def _run(tmp_path: Path, granularity: str) -> Path:
    recipe = (
        _RECIPE.replace("__PLACEHOLDER__", str(tmp_path)).replace("__GRANULARITY__", granularity)
    )
    macrocast.run(recipe, output_directory=tmp_path)
    return tmp_path


def test_per_horizon_creates_h_subdirs(tmp_path):
    out = _run(tmp_path, "per_horizon")
    cell = out / "cell_001"
    h_dirs = sorted(p.name for p in cell.iterdir() if p.is_dir() and p.name.startswith("horizon="))
    # Custom panel produces a single horizon (h=1) by default; pin that the
    # split still creates the subdir layout rather than the flat path.
    assert h_dirs, f"expected horizon=* subdirs under {cell}, found {sorted(p.name for p in cell.iterdir())}"
    # And the flat forecasts.csv at cell_001 root is gone.
    assert not (cell / "forecasts.csv").exists()
    # Each subdir holds its own forecasts.csv.
    for h in h_dirs:
        assert (cell / h / "forecasts.csv").exists()


def test_per_target_creates_target_subdirs(tmp_path):
    out = _run(tmp_path, "per_target")
    cell = out / "cell_001"
    t_dirs = sorted(p.name for p in cell.iterdir() if p.is_dir() and p.name.startswith("target="))
    assert t_dirs, f"expected target=* subdirs under {cell}"
    assert not (cell / "forecasts.csv").exists()


def test_per_target_horizon_creates_two_level_layout(tmp_path):
    out = _run(tmp_path, "per_target_horizon")
    cell = out / "cell_001"
    t_dirs = [p for p in cell.iterdir() if p.is_dir() and p.name.startswith("target=")]
    assert t_dirs
    # Inside each target dir there is at least one horizon=* sub-dir with
    # the forecasts.csv.
    for tdir in t_dirs:
        h_subs = [p for p in tdir.iterdir() if p.is_dir() and p.name.startswith("horizon=")]
        assert h_subs, f"expected horizon=* under {tdir}"
        for hdir in h_subs:
            assert (hdir / "forecasts.csv").exists()


def test_per_cell_default_unchanged(tmp_path):
    out = _run(tmp_path, "per_cell")
    cell = out / "cell_001"
    assert (cell / "forecasts.csv").exists()
    # No granular subdirs.
    assert not [p for p in cell.iterdir() if p.is_dir() and p.name.startswith(("target=", "horizon="))]


def test_flat_writes_to_output_root(tmp_path):
    out = _run(tmp_path, "flat")
    # Forecasts at the output root, no cell_001 wrapper.
    assert (out / "forecasts.csv").exists()
