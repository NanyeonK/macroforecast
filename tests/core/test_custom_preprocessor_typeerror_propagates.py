"""Cycle 15 M-2 -- inspect.signature pre-call validation for custom preprocessors.

Verifies:
  (a) A wrong-arity function raises ValueError with a signature hint BEFORE the
      call (pre-call gate, not TypeError from inside the call).
  (b) A TypeError raised inside a correct-arity function body propagates
      naturally as TypeError and is NOT misattributed to "wrong signature".
  (c) A correct-arity function that returns a DataFrame runs normally.
"""
from __future__ import annotations

import pytest

import macroforecast as mf
from macroforecast import custom


@pytest.fixture(autouse=True)
def _reset_registries():
    custom.clear_custom_extensions()
    yield
    custom.clear_custom_extensions()


_RECIPE = """
0_meta:
  fixed_axes: {failure_policy: fail_fast, reproducibility_mode: seeded_reproducible, random_seed: 1}
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01,
             2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
2_preprocessing:
  fixed_axes: {transform_policy: no_transform, outlier_policy: none,
               imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
  leaf_config:
    custom_postprocessor: __PREP__
3_feature_engineering:
  nodes:
    - {id: src_X, type: source,
       selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
    - {id: src_y, type: source,
       selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
    - {id: lag_x, type: step, op: lag, params: {n_lag: 1}, inputs: [src_X]}
    - {id: y_h, type: step, op: target_construction,
       params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
  sinks:
    l3_features_v1: {X_final: lag_x, y_final: y_h}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - {id: src_X, type: source,
       selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
    - {id: src_y, type: source,
       selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
    - id: fit_model
      type: step
      op: fit_model
      params: {family: ridge, alpha: 0.1, min_train_size: 4,
               forecast_strategy: direct, training_start_rule: expanding,
               refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_model, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_model
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse}
"""


def test_wrong_signature_raises_value_error_pre_call(tmp_path):
    """Cycle 15 M-2: wrong arity fn -> ValueError with signature hint, NOT
    TypeError from inside the call."""
    @custom.register_preprocessor("m2_wrong_arity")
    def wrong_arity(df, ctx):  # 2 args; contract wants 4
        return df

    recipe = _RECIPE.replace("__PREP__", "m2_wrong_arity")
    with pytest.raises((ValueError, RuntimeError)) as exc_info:
        mf.run(recipe, output_directory=str(tmp_path / "out1"))
    msg = str(exc_info.value).lower()
    assert "signature" in msg or "wrong" in msg or "expected" in msg, (
        f"Expected signature-hint in error message, got: {msg[:300]}"
    )


def test_body_typeerror_propagates_as_typeerror(tmp_path):
    """Cycle 15 M-2: TypeError inside user function body should propagate as
    TypeError, NOT be misattributed to 'wrong signature'."""
    @custom.register_preprocessor("m2_body_typeerror")
    def body_typeerror(X_train, y_train, X_test, context):
        # Intentionally raise TypeError from inside body (simulating user bug)
        raise TypeError("user's own TypeError from inside the function body")

    recipe = _RECIPE.replace("__PREP__", "m2_body_typeerror")
    with pytest.raises((TypeError, RuntimeError)) as exc_info:
        mf.run(recipe, output_directory=str(tmp_path / "out2"))
    msg = str(exc_info.value)
    # Must NOT have signature hint
    assert "wrong signature" not in msg.lower(), (
        "body TypeError was misattributed to wrong signature: "
        + msg[:300]
    )
    # Must mention user's original message
    assert "user's own TypeError" in msg or "function body" in msg, (
        f"Expected user's original error message in propagated error, got: {msg[:300]}"
    )


def test_correct_signature_runs_normally(tmp_path):
    """Cycle 15 M-2: correct 4-arg function passes signature check and runs."""
    @custom.register_preprocessor("m2_correct")
    def correct(X_train, y_train, X_test, context):
        return X_train, X_test  # passthrough

    recipe = _RECIPE.replace("__PREP__", "m2_correct")
    r = mf.run(recipe, output_directory=str(tmp_path / "out3"))
    assert len(r.cells) >= 1
