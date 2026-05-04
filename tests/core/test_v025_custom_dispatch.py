"""Issue #251 -- L2 preprocessor + L3 feature_block / combiner dispatch."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macrocast
from macrocast import custom


@pytest.fixture(autouse=True)
def _reset_registries():
    custom.clear_custom_extensions()
    yield
    custom.clear_custom_extensions()


_BASE_RECIPE = """
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
  leaf_config:
    custom_postprocessor: __PREP__
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
  fixed_axes: {primary_metric: mse}
"""


def test_custom_preprocessor_dispatched_via_leaf_config(tmp_path):
    """Issue #251 -- ``leaf_config.custom_postprocessor`` routes a
    user-registered preprocessor at the end of L2."""

    @custom.register_preprocessor("clip_x1_at_2")
    def _clip(X_train, _y_train, X_test, _context):
        clipped_train = X_train.copy()
        clipped_test = X_test.copy()
        if "x1" in clipped_train.columns:
            clipped_train["x1"] = clipped_train["x1"].clip(upper=2.0)
        if "x1" in clipped_test.columns:
            clipped_test["x1"] = clipped_test["x1"].clip(upper=2.0)
        return clipped_train, clipped_test

    recipe = _BASE_RECIPE.replace("__PREP__", "clip_x1_at_2")
    result = macrocast.run(recipe, output_directory=tmp_path)
    cell = result.cells[0]
    panel = cell.runtime_result.artifacts["l2_clean_panel_v1"].panel.data
    # x1 originally ranges 0.5 -> 5.0; after clipping at 2.0 the max is 2.0.
    assert panel["x1"].max() == 2.0


def test_custom_l3_feature_block_dispatched_when_op_matches():
    """Issue #251 -- a registered feature_block runs when ``_execute_l3_op``
    sees its name as the op."""

    from macrocast.core.runtime import _try_custom_l3_dispatch

    @custom.register_feature_block("double_it", block_kind="temporal")
    def _double(frame, _params):
        return frame * 2.0

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(8, 2)), columns=["a", "b"])
    result = _try_custom_l3_dispatch("double_it", [df], params={"block_kind": "temporal"})
    assert result is not None
    np.testing.assert_allclose(result.values, df.values * 2.0)


def test_custom_l3_dispatch_returns_none_when_unregistered():
    from macrocast.core.runtime import _try_custom_l3_dispatch

    result = _try_custom_l3_dispatch("nothing_registered_here", [pd.DataFrame()], params={})
    assert result is None


def test_custom_feature_combiner_dispatched():
    """A combiner is found when the op name matches a registered combiner."""

    from macrocast.core.runtime import _try_custom_l3_dispatch

    @custom.register_feature_combiner("merge_concat")
    def _concat(frame_or_inputs, _params):
        # Accept either a single frame or a list of frames.
        if isinstance(frame_or_inputs, list):
            return pd.concat(frame_or_inputs, axis=1)
        return frame_or_inputs

    a = pd.DataFrame({"x": [1.0, 2.0]})
    b = pd.DataFrame({"y": [3.0, 4.0]})
    result = _try_custom_l3_dispatch("merge_concat", [a, b], params={})
    assert result is not None
    assert {"x", "y"}.issubset(result.columns)
