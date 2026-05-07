"""End-to-end procedure-level coverage of the L4 search_algorithm dispatch
gate (paper16 Round 1 phase A).

Background. ``runtime.py`` has two dispatch gates that wrap

    params = _resolve_l4_tuning(params, X, y)

with a set-literal of recognised ``search_algorithm`` values -- one in the
sequential walk-forward path (``materialize_l4_minimal``) and one in the
parallel-models path (``_run_single_model``). Round 0 audit fixes #6 / #15
/ #6b / #12 added ``kfold / poos / aic / bic / block_cv`` to the
``SEARCH_ALGORITHMS`` enum and to ``_resolve_l4_tuning`` itself, but the
gate set-literal was not updated. Result: a recipe that names one of the
five new schemes loaded fine through the validator and ran without error,
but the resolver was never invoked -- the discriminative initial ``alpha``
flowed through untouched.

Each test below seeds an L4 fit with ``alpha = 99.0`` (a deliberately bad
value for the synthetic DGP) and ``search_algorithm`` set to one of the
five new schemes, runs the recipe through ``macroforecast.run``, and
asserts the post-fit ``alpha`` in the model artefact's ``fit_metadata``
moved off ``99.0``. On the pre-fix code each test fails (gate skips the
resolver, alpha stays at 99.0); on the fixed code each test passes.

The synthetic DGP is shared across all five schemes via ``_e2e_recipe``:
T = 60 monthly observations, two predictors x1 / x2, target y is a clean
linear trend so any working CV picks an alpha far below 99.
"""

from __future__ import annotations

import pytest

import macroforecast


_DATES = (
    [f"2010-{m:02d}-01" for m in range(1, 13)]
    + [f"2011-{m:02d}-01" for m in range(1, 13)]
    + [f"2012-{m:02d}-01" for m in range(1, 13)]
    + [f"2013-{m:02d}-01" for m in range(1, 13)]
    + [f"2014-{m:02d}-01" for m in range(1, 13)]
)
_Y_VALUES = list(range(1, 61))
_X1_VALUES = [v * 0.5 for v in range(1, 61)]
_X2_VALUES = [v * 0.3 for v in range(1, 61)]


def _e2e_recipe(scheme: str) -> str:
    """Return a minimal end-to-end recipe (L0..L4) that fits ridge with
    ``alpha=99.0`` and ``search_algorithm=<scheme>``. The recipe routes
    through the public ``macroforecast.run`` path so the dispatch gate at
    ``runtime.py:1275`` (sequential) is exercised."""

    return f"""
0_meta:
  fixed_axes: {{failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}}
  leaf_config: {{random_seed: 42}}
1_data:
  fixed_axes: {{custom_source_policy: custom_panel_only, frequency: monthly, horizon_set: custom_list}}
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: {_DATES}
      y: {_Y_VALUES}
      x1: {_X1_VALUES}
      x2: {_X2_VALUES}
2_preprocessing:
  fixed_axes: {{transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}}
3_feature_engineering:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: predictors}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: target}}}}}}
    - {{id: lag_x, type: step, op: lag, params: {{n_lag: 1}}, inputs: [src_X]}}
    - {{id: y_h, type: step, op: target_construction, params: {{mode: point_forecast, method: direct, horizon: 1}}, inputs: [src_y]}}
  sinks:
    l3_features_v1: {{X_final: lag_x, y_final: y_h}}
    l3_metadata_v1: auto
4_forecasting_model:
  leaf_config: {{cv_path_alphas: [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}}
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
    - id: fit
      type: step
      op: fit_model
      params: {{family: ridge, alpha: 99.0, search_algorithm: {scheme}, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, min_train_size: 24}}
      inputs: [src_X, src_y]
    - {{id: predict, type: step, op: predict, inputs: [fit, src_X]}}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit
    l4_training_metadata_v1: auto
"""


def _run_and_extract_alpha(recipe: str, tmp_path) -> float:
    """Execute the recipe through ``macroforecast.run`` and return the
    ``alpha`` recorded in the fitted model artefact's ``fit_metadata``."""

    result = macroforecast.run(recipe, output_directory=tmp_path)
    assert result.cells, "recipe should produce at least one cell"
    artifacts = result.cells[0].runtime_result.artifacts["l4_model_artifacts_v1"].artifacts
    assert "fit" in artifacts, "fit_model node should appear in l4_model_artifacts_v1"
    metadata = artifacts["fit"].fit_metadata
    assert "alpha" in metadata, f"alpha missing from fit_metadata: {metadata!r}"
    return float(metadata["alpha"])


@pytest.mark.parametrize("scheme", ["kfold", "poos", "aic", "bic", "block_cv"])
def test_l4_dispatch_gate_fires_for_new_cv_schemes(scheme, tmp_path):
    """paper16 Round 1 phase A: each of the five Round-0-added CV schemes
    must reach ``_resolve_l4_tuning`` through the public dispatch gate.

    The discriminative ``alpha=99.0`` would survive the fit untouched if
    the gate skipped the resolver (the pre-fix behaviour). On the synthetic
    DGP the candidate grid spans ``[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]``
    and any working CV moves ``alpha`` well below 50, so an unchanged
    ``alpha == 99.0`` is a clean smoking gun for a silent dispatch drop."""

    recipe = _e2e_recipe(scheme)
    out_dir = tmp_path / scheme
    resolved_alpha = _run_and_extract_alpha(recipe, out_dir)
    assert resolved_alpha != 99.0, (
        f"search_algorithm={scheme!r} produced alpha={resolved_alpha} -- the "
        "dispatch gate did not invoke _resolve_l4_tuning; the discriminative "
        "initial alpha (99.0) flowed through unchanged."
    )
    # And the resolved alpha is one of the candidate grid values.
    assert resolved_alpha in {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}, (
        f"resolved alpha={resolved_alpha} is not in the cv_path_alphas grid"
    )
    # On this near-noiseless linear DGP, any working CV should pick a small
    # alpha (less penalty -> closer to OLS fit).
    assert resolved_alpha < 50.0, (
        f"resolved alpha={resolved_alpha} is implausibly large for the "
        "near-deterministic linear DGP; CV likely degenerated."
    )
