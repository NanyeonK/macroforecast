"""Tests for the v0.25 small wins:

* #253 lasso_inclusion rolling-window mode
* #254 sampling Shapley for n > 8
* #260 PRE_DEFINED_BLOCKS canonical column membership
* #261 derived saved_objects defaults
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macroforecast.core.runtime import _derive_saved_objects, _lasso_inclusion_frame, _l7_transformation_attribution
from macroforecast.core.ops.l7_ops import PRE_DEFINED_BLOCKS
from macroforecast.core.types import L5EvaluationArtifact, ModelArtifact
from sklearn.linear_model import Lasso


# ---------------------------------------------------------------------------
# #253 lasso rolling-window inclusion
# ---------------------------------------------------------------------------

def _toy_lasso():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = pd.Series(2.0 * X["a"] + rng.normal(scale=0.3, size=n))
    fitted = Lasso(alpha=0.05, max_iter=20000).fit(X, y)
    return X, y, ModelArtifact(
        model_id="m", family="lasso", fitted_object=fitted,
        framework="sklearn", feature_names=tuple(X.columns),
    )


def test_lasso_inclusion_rolling_records_window_count():
    X, y, artifact = _toy_lasso()
    frame = _lasso_inclusion_frame(artifact, X=X, y=y, sampling="rolling", rolling_window=20, seed=0)
    assert "n_rolling_windows_run" in frame.columns
    assert frame["n_rolling_windows_run"].iloc[0] > 0
    # Importance is in [0, 1].
    assert frame["importance"].between(0, 1).all()


def test_lasso_inclusion_both_returns_two_columns():
    X, y, artifact = _toy_lasso()
    frame = _lasso_inclusion_frame(artifact, X=X, y=y, sampling="both", n_bootstraps=10, rolling_window=20, seed=0)
    assert {"importance", "rolling_inclusion"}.issubset(frame.columns)
    assert frame["rolling_inclusion"].between(0, 1).all()


# ---------------------------------------------------------------------------
# #254 sampling Shapley for n > 8
# ---------------------------------------------------------------------------

def _l5_with_n_pipelines(n: int) -> L5EvaluationArtifact:
    rng = np.random.default_rng(0)
    rows = [
        {"model_id": f"p{i}", "target": "y", "horizon": 1, "mse": float(rng.uniform(0.5, 2.0))}
        for i in range(n)
    ]
    return L5EvaluationArtifact(metrics_table=pd.DataFrame(rows), ranking_table=pd.DataFrame(rows), l5_axis_resolved={})


def test_sampling_shapley_runs_for_large_n():
    eval_artifact = _l5_with_n_pipelines(12)
    result = _l7_transformation_attribution(
        eval_artifact,
        params={
            "decomposition_method": "shapley_over_pipelines",
            "loss_function": "mse",
            "shapley_n_permutations": 200,
            "random_state": 0,
        },
    )
    # 12 pipelines -> sampling Shapley path -> all 12 contributions populated.
    assert len(result.summary_table) == 12


def test_explicit_sampling_shapley_method_for_small_n():
    """When the user explicitly requests sampling, the small-n exhaustive
    fast path is bypassed and the sampling estimator runs."""

    eval_artifact = _l5_with_n_pipelines(4)
    result = _l7_transformation_attribution(
        eval_artifact,
        params={
            "decomposition_method": "shapley_over_pipelines_sampled",
            "loss_function": "mse",
            "shapley_n_permutations": 500,
            "random_state": 0,
        },
    )
    assert len(result.summary_table) == 4


# ---------------------------------------------------------------------------
# #260 pre-defined block backfill
# ---------------------------------------------------------------------------

def test_mccracken_md_groups_have_eight_categories():
    md = PRE_DEFINED_BLOCKS["mccracken_ng_md_groups"]
    assert isinstance(md, dict)
    assert len(md) == 8
    assert "output_and_income" in md
    assert "INDPRO" in md["output_and_income"]


def test_mccracken_qd_groups_have_canonical_categories():
    qd = PRE_DEFINED_BLOCKS["mccracken_ng_qd_groups"]
    assert isinstance(qd, dict)
    assert "industrial_production" in qd
    assert "INDPRO" in qd["industrial_production"]


def test_fred_sd_states_block_has_50_states_plus_dc():
    states = PRE_DEFINED_BLOCKS["fred_sd_states"]
    assert len(states) == 51  # 50 states + DC
    assert "CA" in states and "DC" in states


def test_taylor_rule_block_includes_ffr():
    assert "FEDFUNDS" in PRE_DEFINED_BLOCKS["taylor_rule_block"]


def test_term_structure_block_spans_short_to_long_end():
    ts = set(PRE_DEFINED_BLOCKS["term_structure_block"])
    assert {"TB3MS", "GS10"} <= ts


# ---------------------------------------------------------------------------
# #261 derived saved_objects defaults
# ---------------------------------------------------------------------------

def test_saved_objects_derived_minimal_recipe():
    recipe = {
        "1_data": {},
        "2_preprocessing": {},
        "3_feature_engineering": {},
        "4_forecasting_model": {},
        "5_evaluation": {},
    }
    derived = _derive_saved_objects(recipe, upstream_artifacts={})
    assert {"forecasts", "raw_panel", "clean_panel", "cleaning_log",
            "feature_metadata", "model_artifacts", "training_metadata",
            "metrics", "ranking"} <= derived


def test_saved_objects_includes_regime_when_active():
    recipe = {
        "1_data": {"fixed_axes": {"regime_definition": "external_nber"}},
    }
    derived = _derive_saved_objects(recipe, upstream_artifacts={})
    assert "regime_metrics" in derived
    assert "regime_metadata" in derived


def test_saved_objects_includes_diagnostic_layer_entries():
    recipe = {
        "1_5_data_summary": {},
        "2_5_pre_post_preprocessing": {},
        "3_5_feature_diagnostics": {},
        "4_5_generator_diagnostics": {},
    }
    derived = _derive_saved_objects(recipe, upstream_artifacts={})
    assert {"diagnostics_l1_5", "diagnostics_l2_5", "diagnostics_l3_5", "diagnostics_l4_5"} <= derived
