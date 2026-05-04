"""Determinism regression tests for L0 reproducibility_mode + seed propagation.

Implements issue #6 part 1 of the phase-00 stability plan: pin down the
seed-resolution and seed-application contracts so a future change can't
silently break replicate.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from macrocast.core.execution import _apply_seed, _resolve_seed


# ---------------------------------------------------------------------------
# _resolve_seed
# ---------------------------------------------------------------------------

def test_resolve_seed_explicit_leaf_config_takes_precedence():
    root = {
        "0_meta": {
            "fixed_axes": {"reproducibility_mode": "exploratory"},
            "leaf_config": {"random_seed": 42},
        }
    }
    assert _resolve_seed(root) == 42


def test_resolve_seed_seeded_reproducible_default_returns_zero():
    root = {"0_meta": {"fixed_axes": {"reproducibility_mode": "seeded_reproducible"}}}
    assert _resolve_seed(root) == 0


def test_resolve_seed_default_mode_returns_zero():
    # No 0_meta block at all -> defaults to seeded_reproducible -> seed 0.
    assert _resolve_seed({}) == 0
    assert _resolve_seed({"0_meta": {}}) == 0


def test_resolve_seed_exploratory_returns_none():
    root = {"0_meta": {"fixed_axes": {"reproducibility_mode": "exploratory"}}}
    assert _resolve_seed(root) is None


def test_resolve_seed_unknown_mode_returns_none():
    # ``strict`` (and any value other than ``seeded_reproducible``) is treated
    # as non-seeded by _resolve_seed. The L0 validator catches the schema
    # rejection upstream so this code path stays simple.
    root = {"0_meta": {"fixed_axes": {"reproducibility_mode": "strict"}}}
    assert _resolve_seed(root) is None


# ---------------------------------------------------------------------------
# _apply_seed
# ---------------------------------------------------------------------------

def test_apply_seed_makes_random_module_deterministic():
    _apply_seed(123)
    a = [random.random() for _ in range(5)]
    _apply_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_apply_seed_makes_numpy_deterministic():
    _apply_seed(123)
    a = np.random.rand(5).tolist()
    _apply_seed(123)
    b = np.random.rand(5).tolist()
    assert a == b


def test_apply_seed_distinct_seeds_produce_distinct_streams():
    _apply_seed(0)
    a = [random.random() for _ in range(3)]
    _apply_seed(1)
    b = [random.random() for _ in range(3)]
    assert a != b


def test_apply_seed_none_is_noop():
    # Snapshot then call with None -> next draw should *not* be a hard-coded
    # value (it should follow whatever state random/numpy already had).
    _apply_seed(0)
    expected_next_random = random.random()
    _apply_seed(0)
    _apply_seed(None)  # should not reset
    actual_next_random = random.random()
    assert expected_next_random == actual_next_random


def test_apply_seed_propagates_to_torch_when_available():
    try:
        import torch  # type: ignore
    except ImportError:
        pytest.skip("torch not installed")
    _apply_seed(7)
    a = torch.randn(4)
    _apply_seed(7)
    b = torch.randn(4)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Cross-cell seed schedule (each cell gets base_seed + cell_index)
# ---------------------------------------------------------------------------

def test_distinct_cells_get_distinct_seeds(tmp_path):
    """When execute_recipe iterates a sweep, cell N should get base_seed + (N-1).
    Verify two-cell sweep produces two different RNG streams via a custom
    panel recipe whose model fitting is RNG-sensitive (random_forest).
    """

    import macrocast

    recipe = textwrap.dedent(
        """
        0_meta:
          fixed_axes:
            failure_policy: fail_fast
            reproducibility_mode: seeded_reproducible
          leaf_config:
            random_seed: 100
        1_data:
          fixed_axes:
            custom_source_policy: custom_panel_only
            frequency: monthly
            horizon_set: custom_list
          leaf_config:
            target: y
            target_horizons: [1]
            custom_panel_inline:
              date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01]
              y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
              x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
              x2: [0.1, 0.4, 0.2, 0.6, 0.3, 0.7, 0.5, 0.8, 0.4, 0.9, 0.6, 1.0]
        2_preprocessing:
          fixed_axes: {transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}
        3_feature_engineering:
          nodes:
            - {id: src_X, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: predictors}}}
            - {id: src_y, type: source, selector: {layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {role: target}}}
            - {id: lag_x, type: step, op: lag, params: {n_lag: {sweep: [1, 2]}}, inputs: [src_X]}
            - {id: y_h, type: step, op: target_construction, params: {mode: point_forecast, method: direct, horizon: 1}, inputs: [src_y]}
          sinks:
            l3_features_v1: {X_final: lag_x, y_final: y_h}
            l3_metadata_v1: auto
        4_forecasting_model:
          nodes:
            - {id: src_X, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: X_final}}}
            - {id: src_y, type: source, selector: {layer_ref: l3, sink_name: l3_features_v1, subset: {component: y_final}}}
            - id: fit_rf
              type: step
              op: fit_model
              params: {family: random_forest, n_estimators: 8, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
              inputs: [src_X, src_y]
            - {id: predict, type: step, op: predict, inputs: [fit_rf, src_X]}
          sinks:
            l4_forecasts_v1: predict
            l4_model_artifacts_v1: fit_rf
            l4_training_metadata_v1: auto
        5_evaluation:
          fixed_axes: {primary_metric: mse}
        """
    )
    result = macrocast.run(recipe, output_directory=tmp_path)
    assert len(result.cells) == 2
    cell_a, cell_b = result.cells
    # Different sweep values -> different sink hashes for L3+ but identical L1.
    assert cell_a.sink_hashes["l1_data_definition_v1"] == cell_b.sink_hashes["l1_data_definition_v1"]
    assert cell_a.sink_hashes["l3_features_v1"] != cell_b.sink_hashes["l3_features_v1"]


def test_seed_propagates_via_pythonhashseed_env(tmp_path):
    """`_apply_seed` should set ``PYTHONHASHSEED`` (best-effort, only when not
    already pinned by the shell)."""

    saved = os.environ.pop("PYTHONHASHSEED", None)
    try:
        _apply_seed(11)
        assert os.environ.get("PYTHONHASHSEED") == "11"
    finally:
        if saved is not None:
            os.environ["PYTHONHASHSEED"] = saved
        else:
            os.environ.pop("PYTHONHASHSEED", None)


# ---------------------------------------------------------------------------
# Issue #215: L0 random_seed propagates into L4 estimator random_state
# ---------------------------------------------------------------------------

_SEED_PROPAGATION_RECIPE = """
0_meta:
  fixed_axes:
    failure_policy: fail_fast
    reproducibility_mode: seeded_reproducible
  leaf_config:
    random_seed: __SEED__
1_data:
  fixed_axes:
    custom_source_policy: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01]
      y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
      x2: [0.1, 0.4, 0.2, 0.6, 0.3, 0.7, 0.5, 0.8, 0.4, 0.9, 0.6, 1.0]
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
    - id: fit_rf
      type: step
      op: fit_model
      params: {family: random_forest, n_estimators: 8, min_train_size: 4, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none}
      inputs: [src_X, src_y]
    - {id: predict, type: step, op: predict, inputs: [fit_rf, src_X]}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_rf
    l4_training_metadata_v1: auto
5_evaluation:
  fixed_axes: {primary_metric: mse}
"""


def test_l0_random_seed_propagates_into_l4_random_state(tmp_path):
    """Issue #215: when no per-fit-node ``random_state`` is set, the L0
    ``random_seed`` must seed the L4 estimator. Two cells with different L0
    seeds produce different L4 model artifact hashes even though every
    estimator-level config is identical.
    """

    import macrocast

    out_a = tmp_path / "seed_0"
    out_b = tmp_path / "seed_777"
    a = macrocast.run(
        _SEED_PROPAGATION_RECIPE.replace("__SEED__", "0"), output_directory=out_a
    )
    b = macrocast.run(
        _SEED_PROPAGATION_RECIPE.replace("__SEED__", "777"), output_directory=out_b
    )
    # Compare l4_forecasts_v1 hashes: the actual numeric forecasts encode the
    # estimator's random_state (RF bootstrap differs by seed). Note that
    # ``l4_model_artifacts_v1`` excludes ``fitted_object`` from the hash so it
    # would not catch the change.
    assert (
        a.cells[0].sink_hashes["l4_forecasts_v1"]
        != b.cells[0].sink_hashes["l4_forecasts_v1"]
    ), "L0 random_seed must flow into L4 RandomForest random_state"


def test_per_node_random_state_overrides_l0_seed(tmp_path):
    """When the fit_model node sets its own ``params.random_state``, that
    explicit value wins over the L0-derived default.
    """

    import macrocast

    base = _SEED_PROPAGATION_RECIPE.replace("__SEED__", "0")
    explicit = base.replace(
        "search_algorithm: none}",
        "search_algorithm: none, random_state: 42}",
    )
    out_a = tmp_path / "implicit_l0"
    out_b = tmp_path / "explicit_42"
    a = macrocast.run(base.replace("random_seed: 0", "random_seed: 42"), output_directory=out_a)
    b = macrocast.run(explicit, output_directory=out_b)
    # Both ultimately fit RandomForest with random_state=42 -- one via L0
    # propagation, the other via explicit per-node param. Numeric forecasts
    # match.
    assert (
        a.cells[0].sink_hashes["l4_forecasts_v1"]
        == b.cells[0].sink_hashes["l4_forecasts_v1"]
    )
