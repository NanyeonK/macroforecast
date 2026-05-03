from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import macrocast
from macrocast.core.execution import execute_recipe, replicate_recipe


_BASE_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: 7
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01, 2020-05-01, 2020-06-01, 2020-07-01, 2020-08-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      x1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      x2: [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0]
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
    - {id: lag_x, type: step, op: lag, params: {n_lag: __NLAG__}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_ridge
      type: step
      op: fit_model
      params: {family: ridge, alpha: 1.0, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict_ridge, type: step, op: predict, inputs: [fit_ridge, src_X]}
  sinks:
    l4_forecasts_v1: predict_ridge
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes:
    primary_metric: mse
    point_metrics: [mse, rmse, mae]
"""


def _make_recipe(*, n_lag_value: object) -> str:
    if isinstance(n_lag_value, list):
        marker = f"{{sweep: {n_lag_value}}}"
    else:
        marker = str(n_lag_value)
    return _BASE_RECIPE.replace("__NLAG__", marker)


def test_execute_recipe_single_cell_runs_full_pipeline():
    recipe = _make_recipe(n_lag_value=1)
    result = execute_recipe(recipe)

    assert len(result.cells) == 1
    cell = result.cells[0]
    assert cell.succeeded
    assert "l1_data_definition_v1" in cell.runtime_result.artifacts
    assert "l4_forecasts_v1" in cell.runtime_result.artifacts
    assert "l5_evaluation_v1" in cell.runtime_result.artifacts
    assert "l5_evaluation_v1" in cell.sink_hashes


def test_execute_recipe_param_sweep_expands_cells():
    recipe = _make_recipe(n_lag_value=[1, 2])
    result = execute_recipe(recipe)

    assert len(result.cells) == 2
    assert all(cell.succeeded for cell in result.cells)
    sweep_value_lookup = {cell.cell_id: cell.sweep_values for cell in result.cells}
    sweep_keys = next(iter(sweep_value_lookup.values())).keys()
    assert any("n_lag" in key for key in sweep_keys)
    assert sorted(value for cell in result.cells for value in cell.sweep_values.values()) == [1, 2]


def test_execute_recipe_fail_fast_raises():
    bad = _BASE_RECIPE.replace("custom_panel_only", "official_only").replace("__NLAG__", "1")
    # remove inline custom panel so official_only path triggers a NotImplementedError below the loader
    bad = bad.replace("    custom_panel_inline:", "    custom_panel_inline_OFF:")
    with pytest.raises((Exception,)):
        execute_recipe(bad)


def test_execute_recipe_continue_on_failure_captures_error():
    bad = _BASE_RECIPE.replace("failure_policy: fail_fast", "failure_policy: continue_on_failure").replace("__NLAG__", "1")
    bad = bad.replace("custom_source_policy: custom_panel_only", "custom_source_policy: official_only")
    bad = bad.replace("    custom_panel_inline:", "    custom_panel_inline_OFF:")
    result = execute_recipe(bad)
    assert len(result.cells) == 1
    assert not result.cells[0].succeeded
    assert result.cells[0].error
    assert result.failed and not result.succeeded


def test_replicate_recipe_bit_exact(tmp_path: Path):
    recipe = _make_recipe(n_lag_value=[1, 2])
    result = execute_recipe(recipe, output_directory=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()

    replication = replicate_recipe(manifest_path)
    assert replication.recipe_match
    assert replication.sink_hashes_match
    assert all(replication.per_cell_match.values())
    assert len(replication.per_cell_match) == 2


def test_top_level_run_and_replicate_aliases(tmp_path: Path):
    recipe = _make_recipe(n_lag_value=1)
    result = macrocast.run(recipe, output_directory=tmp_path)
    assert isinstance(result, macrocast.ManifestExecutionResult)

    replication = macrocast.replicate(tmp_path / "manifest.json")
    assert isinstance(replication, macrocast.ReplicationResult)
    assert replication.sink_hashes_match
