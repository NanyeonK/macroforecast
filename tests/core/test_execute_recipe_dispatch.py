"""Regression tests for the execute_recipe dispatch surface (closes #169).

Pins the contract that string-vs-path dispatch is explicit:

* ``execute_recipe(dict)`` -- already-parsed recipe.
* ``execute_recipe(Path(...))`` -- file path, parsed from disk.
* ``execute_recipe(yaml_string)`` -- inline YAML text.
* ``execute_recipe(path_string)`` -- back-compat: still loads from disk if
  the string names an existing file, but raises ``DeprecationWarning``.
* ``execute_recipe_file(path)`` -- explicit file dispatch (preferred).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

import macrocast
from macrocast.core.execution import execute_recipe, execute_recipe_file


_RECIPE_YAML = textwrap.dedent(
    """
    0_meta:
      fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}
      leaf_config: {random_seed: 0}
    1_data:
      fixed_axes:
        custom_source_policy: custom_panel_only
        frequency: monthly
        horizon_set: custom_list
      leaf_config:
        target: y
        target_horizons: [1]
        custom_panel_inline:
          date: [2020-01-01, 2020-02-01, 2020-03-01, 2020-04-01, 2020-05-01, 2020-06-01]
          y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
          x: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
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
        - id: fit
          type: step
          op: fit_model
          params: {family: ridge, alpha: 1.0, min_train_size: 2, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
          inputs: [src_X, src_y]
        - {id: predict, type: step, op: predict, inputs: [fit, src_X]}
      sinks:
        l4_forecasts_v1: predict
        l4_model_artifacts_v1: fit
        l4_training_metadata_v1: auto
    5_evaluation:
      fixed_axes: {primary_metric: mse}
    """
)


def test_inline_yaml_string_runs_without_warning(recwarn):
    result = execute_recipe(_RECIPE_YAML)
    assert result.cells[0].succeeded
    deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert not deprecation_warnings


def test_path_object_runs_without_warning(tmp_path, recwarn):
    target = tmp_path / "recipe.yaml"
    target.write_text(_RECIPE_YAML, encoding="utf-8")
    result = execute_recipe(target)
    assert result.cells[0].succeeded
    deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert not deprecation_warnings


def test_execute_recipe_file_helper(tmp_path, recwarn):
    target = tmp_path / "recipe.yaml"
    target.write_text(_RECIPE_YAML, encoding="utf-8")
    result = execute_recipe_file(target)
    assert result.cells[0].succeeded
    deprecation_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
    assert not deprecation_warnings


def test_string_path_emits_deprecation_warning(tmp_path):
    target = tmp_path / "recipe.yaml"
    target.write_text(_RECIPE_YAML, encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="passing a file path as `str`"):
        result = execute_recipe(str(target))
    assert result.cells[0].succeeded


def test_dict_recipe_runs(recwarn):
    import yaml

    recipe_root = yaml.safe_load(_RECIPE_YAML)
    result = execute_recipe(recipe_root)
    assert result.cells[0].succeeded


def test_top_level_run_file_alias(tmp_path):
    target = tmp_path / "recipe.yaml"
    target.write_text(_RECIPE_YAML, encoding="utf-8")
    result = macrocast.run_file(target)
    assert isinstance(result, macrocast.ManifestExecutionResult)
    assert result.cells[0].succeeded
