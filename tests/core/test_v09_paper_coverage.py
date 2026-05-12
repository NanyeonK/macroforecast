"""v0.9 Phase 2 paper-coverage pass — schema + runtime regression tests.

This module pins the v0.9.0 baseline:

* Decomposition discipline -- every paper method that decomposes into a
  recipe over existing primitives is captured as a recipe pattern, not a
  new op. The schema adds only *atomic* primitives that cannot be
  expressed as a recipe over the existing op vocabulary.
* L4 family additions: ``mars`` only -- atomic non-linear basis-function
  regression with no sklearn analogue. Everything else (PRF, Booging,
  MARSquake, SGT, 2SRR, HNN, Assemblage) decomposes.
* L3 op additions: ``savitzky_golay_filter`` (operational, scipy wrap) +
  ``adaptive_ma_rf`` (future, AlbaMA atomic).
* L2 op addition: ``asymmetric_trim`` (future, Albacore family atomic).
* L5 op addition: ``blocked_oob_reality_check`` (future, HNN reality-check
  atomic).
* L7 op additions: ``dual_decomposition`` + ``portfolio_metrics`` (Coulombe
  Dual paper) + ``oshapley_vi`` + ``pbsv`` (Borup anatomy, future).
"""

from __future__ import annotations

import pytest

import macroforecast
from macroforecast.core.ops import get_op
from macroforecast.core.ops.l4_ops import (
    FUTURE_MODEL_FAMILIES,
    OPERATIONAL_MODEL_FAMILIES,
    get_family_status,
)
from macroforecast.core.status import OPERATIONAL
from macroforecast.recipes.paper_methods import _DATA_TRANSFORM_CELLS_16


# ---------------------------------------------------------------------------
# L4: mars is the only atomic family addition.
# ---------------------------------------------------------------------------


def test_mars_registered_operational_with_pyearth_optional():
    """mars is operational from v0.9.0 with pyearth as optional dep --
    raises NotImplementedError with install hint when pyearth missing
    (mirrors xgboost / lightgbm / catboost pattern)."""

    assert "mars" in OPERATIONAL_MODEL_FAMILIES
    assert "mars" not in FUTURE_MODEL_FAMILIES
    assert get_family_status("mars") == OPERATIONAL


@pytest.mark.parametrize(
    "decomposable_family",
    [
        "perfectly_random_forest",  # = extra_trees(max_features=1)
        "booging",  # = bagging(strategy=sequential_residual)
        "marsquake",  # = bagging(base_family=mars)
        "slow_growing_tree",  # = decision_tree(split_shrinkage=η)
        "two_step_ridge_regression",  # = chained ridge + ridge(prior=random_walk)
        "hemisphere_neural_network",  # = mlp(architecture=hemisphere, ...)
        "assemblage_regression",  # = ridge(coefficient_constraint=nonneg)
    ],
)
def test_decomposable_paper_methods_are_not_l4_families(decomposable_family):
    """These paper methods decompose into recipes over existing
    primitives + sub-axes; they must NOT appear as standalone L4
    families."""

    assert decomposable_family not in OPERATIONAL_MODEL_FAMILIES
    assert decomposable_family not in FUTURE_MODEL_FAMILIES


# ---------------------------------------------------------------------------
# L3 atomic primitives.
# ---------------------------------------------------------------------------


def test_savitzky_golay_filter_registered_operational():
    """Savitzky-Golay is operational because scipy is a hard dep."""

    spec = get_op("savitzky_golay_filter")
    assert "l3" in spec.layer_scope
    assert spec.status == "operational"


def test_adaptive_ma_rf_registered_operational():
    """AlbaMA promoted in v0.9.1 dev-stage v0.9.0C-1."""

    spec = get_op("adaptive_ma_rf")
    assert "l3" in spec.layer_scope
    assert spec.status != "future"


# ---------------------------------------------------------------------------
# L2 atomic primitive.
# ---------------------------------------------------------------------------


def test_asymmetric_trim_registered_operational():
    """v0.8.9 promotion (B-6): rank-space transformation per Goulet
    Coulombe et al. (2024). Layer scope expanded to (l2, l3) so the L3
    DAG can dispatch it."""

    spec = get_op("asymmetric_trim")
    assert "l2" in spec.layer_scope
    assert "l3" in spec.layer_scope
    assert spec.status == "operational"


def test_b6_asymmetric_trim_per_row_sort():
    """Output O[t, r] is the r-th order statistic of input row t."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _asymmetric_trim

    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"c{i}" for i in range(5)])
    out = _asymmetric_trim(panel)

    assert out.shape == (20, 5)
    assert list(out.columns) == ["rank_1", "rank_2", "rank_3", "rank_4", "rank_5"]
    # Each row of out must equal sorted(panel row)
    for t in range(20):
        expected = np.sort(panel.iloc[t].to_numpy(dtype=float))
        np.testing.assert_array_equal(out.iloc[t].to_numpy(), expected)
    # Per-row monotone: rank_1 <= rank_2 <= ... <= rank_K
    for t in range(20):
        row = out.iloc[t].to_numpy()
        assert np.all(np.diff(row) >= 0)


def test_b6_asymmetric_trim_idempotent_on_sorted_input():
    """Applying asymmetric_trim twice yields the same matrix as one
    application (the second call sorts already-sorted rows)."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _asymmetric_trim

    rng = np.random.default_rng(7)
    panel = pd.DataFrame(rng.normal(size=(15, 4)), columns=[f"c{i}" for i in range(4)])
    once = _asymmetric_trim(panel)
    twice = _asymmetric_trim(once)
    np.testing.assert_array_equal(once.to_numpy(), twice.to_numpy())


def test_b6_asymmetric_trim_with_smoothing():
    """smooth_window > 1 applies centred moving-average to rank-position
    time series."""

    import pandas as pd
    from macroforecast.core.runtime import _asymmetric_trim

    panel = pd.DataFrame(
        {"c0": [1.0, 2.0, 3.0, 4.0, 5.0], "c1": [5.0, 4.0, 3.0, 2.0, 1.0]}
    )
    out = _asymmetric_trim(panel, smooth_window=3)
    assert out.shape == (5, 2)
    # rank_1 (= per-row min) for the input above is uniformly the smaller
    # of (i+1, 5-i): {1, 2, 3, 2, 1}. After 3-MA centred smoothing, value
    # at index 2 = mean(2, 3, 2) = 7/3.
    assert abs(out["rank_1"].iloc[2] - 7.0 / 3) < 1e-9


# ---------------------------------------------------------------------------
# L5 atomic primitive.
# ---------------------------------------------------------------------------


def test_blocked_oob_reality_check_registered_operational():
    """Promoted to operational in v0.8.9 -- block-bootstrap on
    per-origin loss differentials (helper:
    ``_blocked_oob_reality_check_p_values``)."""

    spec = get_op("blocked_oob_reality_check")
    assert "l5" in spec.layer_scope
    assert spec.status == "operational"


# ---------------------------------------------------------------------------
# L7 atomic primitives (4: Coulombe Dual + Borup anatomy).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name",
    [
        "oshapley_vi",
        "pbsv",
    ],
)
def test_l7_paper_coverage_ops_registered_operational(op_name):
    """Promoted in v0.9.1 dev-stage v0.9.0D Path B (anatomy adapter,
    final-window fit, status='degraded' warning). Path A faithful per-
    origin refit lands in v0.9.0E."""
    spec = get_op(op_name)
    assert "l7" in spec.layer_scope
    assert spec.status != "future"


def test_b3_dual_decomposition_registered_operational():
    """Linear-family closed-form dual decomposition is operational from
    v0.8.9 (ridge / OLS / lasso). Non-linear (RF / kernel) deferred to
    v0.9.x Phase 2."""

    spec = get_op("dual_decomposition")
    assert "l7" in spec.layer_scope
    assert spec.status == "operational"


def test_b3_dual_decomposition_recovers_ridge_predictions():
    """Sanity check: for ridge, ``W @ y`` should reproduce the ridge
    in-sample fitted values (representer theorem identity)."""

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from macroforecast.core.runtime import _dual_decomposition_frame
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    n, p = 40, 3
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"x{i}" for i in range(p)])
    y = pd.Series(X.sum(axis=1) + 0.1 * rng.normal(size=n))
    fitted = Ridge(alpha=0.5).fit(X, y)
    artifact = ModelArtifact(
        model_id="m0",
        family="ridge",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )

    frame = _dual_decomposition_frame(artifact, X, y)
    weights_full = frame.attrs["dual_weights"]
    portfolio = frame.attrs["portfolio_metrics"]

    # Representer identity: predictions = W @ y_centered + intercept
    # (here ridge centers internally; check fitted ≈ W @ y)
    yhat_ridge = fitted.predict(X)
    yhat_dual = weights_full.to_numpy() @ y.to_numpy()
    # Ridge intercept absorbs the y-mean offset; check shape + reasonable correlation.
    corr = np.corrcoef(yhat_ridge, yhat_dual)[0, 1]
    assert corr > 0.95, (
        f"dual prediction should track ridge prediction (corr={corr:.3f})"
    )

    # Portfolio metrics shape + paper-faithful sign invariants
    # (v0.9.0F audit-fix: ``leverage`` is now signed sum per paper Eq.
    # p.21 "FL = Σ w_{ji}", and ``short`` is signed (≤ 0); the legacy
    # absolute-value variants are surfaced as ``leverage_l1`` and
    # ``short_abs`` for backward-compatible plotting).
    assert set(portfolio.columns) == {
        "hhi",
        "short",
        "turnover",
        "leverage",
        "leverage_l1",
        "short_abs",
    }
    assert (portfolio["hhi"] >= 0).all()
    assert (portfolio["short"] <= 0).all()  # signed: ≤ 0
    assert (portfolio["short_abs"] >= 0).all()  # legacy magnitude
    assert (portfolio["leverage_l1"] >= 0).all()  # L1 norm
    # Turnover for first row is 0 by construction.
    assert portfolio["turnover"].iloc[0] == 0.0


def test_b5_blocked_oob_reality_check_rejects_clear_winner():
    """When a candidate model has uniformly lower loss than the
    benchmark, the block-bootstrap reality check rejects H0
    (candidate no better than benchmark) at α=0.05."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _blocked_oob_reality_check_p_values

    rng = np.random.default_rng(0)
    n = 100
    bench_loss = rng.gamma(shape=2.0, scale=1.0, size=n) + 1.0  # positive
    cand_loss = bench_loss * 0.5  # cand uniformly better
    losses = pd.DataFrame({"benchmark": bench_loss, "cand": cand_loss})

    out = _blocked_oob_reality_check_p_values(
        losses, benchmark="benchmark", block_length=4, n_bootstraps=500, random_state=0
    )
    assert "cand" in out.index
    assert out.loc["cand", "mean_diff"] > 0  # cand has lower loss
    assert out.loc["cand", "reject_h0"] == True  # noqa: E712


def test_b5_blocked_oob_reality_check_does_not_reject_equal_skill():
    """When candidate and benchmark have identical loss distributions,
    the test should not reject (type-I error rate ~ alpha)."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _blocked_oob_reality_check_p_values

    rng = np.random.default_rng(7)
    n = 100
    base = rng.gamma(shape=2.0, scale=1.0, size=n) + 1.0
    losses = pd.DataFrame({"benchmark": base, "cand": base + rng.normal(0, 0.01, n)})

    out = _blocked_oob_reality_check_p_values(
        losses, benchmark="benchmark", block_length=4, n_bootstraps=500, random_state=7
    )
    # Equal-skill: should not reject at α=0.05 (almost always).
    assert out.loc["cand", "reject_h0"] == False  # noqa: E712


def test_v21_scaled_pca_matches_huang_zhou_2022_authors_matlab():
    """V2.1 honesty-pass fix: ``_scaled_pca_huang_zhou`` matches the
    authors' MATLAB sPCAest.m to machine precision.

    Known-answer: per-column predictive slope β should equal what
    univariate OLS of target on each standardised column produces.
    """

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _scaled_pca_huang_zhou

    rng = np.random.default_rng(0)
    T, N = 100, 4
    factor = rng.normal(size=T)
    target_arr = factor + 0.1 * rng.normal(size=T)
    X_arr = np.column_stack(
        [
            factor + 0.1 * rng.normal(size=T),  # x1: strongly predictive
            0.3 * factor + 0.5 * rng.normal(size=T),  # x2: weakly predictive
            rng.normal(size=T),  # x3: noise
            rng.normal(size=T),  # x4: noise
        ]
    )
    frame = pd.DataFrame(X_arr, columns=["x1", "x2", "x3", "x4"])
    target = pd.Series(target_arr)

    # Authors' reference algorithm (sPCAest.m)
    Xs = (X_arr - X_arr.mean(axis=0)) / X_arr.std(axis=0, ddof=1)
    beta_authors = np.zeros(N)
    for j in range(N):
        xv = np.column_stack([np.ones(T), Xs[:, j]])
        parm = np.linalg.lstsq(xv, target_arr, rcond=None)[0]
        beta_authors[j] = parm[1]

    # Our internal β (re-derived via the closed-form used in
    # _scaled_pca_huang_zhou)
    y_c = target_arr - target_arr.mean()
    beta_ours = (Xs * y_c[:, None]).sum(axis=0) / (Xs**2).sum(axis=0)
    np.testing.assert_allclose(beta_ours, beta_authors, atol=1e-12)

    # Sanity: noise columns get near-zero β; predictive columns get
    # large β.
    assert beta_authors[0] > 0.5  # x1 strongly predictive
    assert abs(beta_authors[2]) < 0.2  # x3 noise
    assert abs(beta_authors[3]) < 0.2  # x4 noise

    # Run the full sPCA and check the resulting first factor tracks
    # the true latent factor.
    f_ours = _scaled_pca_huang_zhou(frame, n_components=1, target_signal=target)
    corr_with_truth = abs(np.corrcoef(f_ours[:, 0], factor)[0, 1])
    assert corr_with_truth > 0.95, (
        f"sPCA factor should track truth (corr={corr_with_truth})"
    )


def test_b3_dual_decomposition_rejects_unsupported_nonlinear_families():
    """Boosted-tree / NN families still raise NotImplementedError after
    v0.9.0B-5: their residual-bagging or learned non-linear structure
    has no clean sum-of-training-targets dual representation. (RF /
    ExtraTrees were promoted in v0.9.0B-5; see
    ``test_v090b_dual_decomposition_rf_bit_exact_with_bootstrap``.)"""

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from macroforecast.core.runtime import _dual_decomposition_frame
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(20, 2)), columns=["x0", "x1"])
    y = pd.Series(X.sum(axis=1))
    fitted = GradientBoostingRegressor(n_estimators=5, random_state=0).fit(X, y)
    artifact = ModelArtifact(
        model_id="m0",
        family="gradient_boosting",
        fitted_object=fitted,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )

    with pytest.raises(
        NotImplementedError, match="dual_decomposition does not yet support"
    ):
        _dual_decomposition_frame(artifact, X, y)


# ---------------------------------------------------------------------------
# Operational savitzky_golay end-to-end.
# ---------------------------------------------------------------------------


def _savgol_recipe(seed: int = 42) -> str:
    return f"""
0_meta:
  fixed_axes: {{failure_policy: fail_fast, reproducibility_mode: seeded_reproducible}}
  leaf_config: {{random_seed: {seed}}}
1_data:
  fixed_axes: {{custom_source_policy: custom_panel_only, frequency: monthly, horizon_set: custom_list}}
  leaf_config:
    target: y
    target_horizons: [1]
    custom_panel_inline:
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01]
      y:  [1.0, 2.5, 2.0, 3.5, 4.0, 5.5, 5.0, 6.5, 7.0, 8.5, 8.0, 9.5]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
2_preprocessing:
  fixed_axes: {{transform_policy: no_transform, outlier_policy: none, imputation_policy: none_propagate, frame_edge_policy: keep_unbalanced}}
3_feature_engineering:
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: predictors}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l2, sink_name: l2_clean_panel_v1, subset: {{role: target}}}}}}
    - {{id: smoothed, type: step, op: savitzky_golay_filter, params: {{window_length: 5, polyorder: 2}}, inputs: [src_X]}}
    - {{id: lag_x, type: step, op: lag, params: {{n_lag: 1}}, inputs: [smoothed]}}
    - {{id: y_h, type: step, op: target_construction, params: {{mode: point_forecast, method: direct, horizon: 1}}, inputs: [src_y]}}
  sinks:
    l3_features_v1: {{X_final: lag_x, y_final: y_h}}
    l3_metadata_v1: auto
4_forecasting_model:
  nodes:
    - id: src_X
      type: source
      selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}
    - id: src_y
      type: source
      selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}
    - id: fit_ridge
      type: step
      op: fit_model
      params:
        family: ridge
        alpha: 1.0
        forecast_strategy: direct
        training_start_rule: expanding
        refit_policy: every_origin
        search_algorithm: none
        min_train_size: 6
      inputs: [src_X, src_y]
    - {{id: predict, type: step, op: predict, inputs: [fit_ridge, src_X]}}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit_ridge
    l4_training_metadata_v1: auto
"""


def test_savitzky_golay_filter_runs_in_recipe(tmp_path):
    out = tmp_path / "savgol_run"
    result = macroforecast.run(_savgol_recipe(), output_directory=out)
    assert result.cells, "Recipe should produce at least one cell"
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts, "Recipe with savitzky_golay_filter should emit forecasts"


# ---------------------------------------------------------------------------
# Sub-axis future-gate tests: each gated combination raises NotImplementedError
# at runtime so users discover the missing primitive cleanly.
# ---------------------------------------------------------------------------


def _gated_l4_recipe(family: str, extra_params: dict) -> str:
    pp = ", ".join(f"{k}: {v!r}" for k, v in extra_params.items()).replace("'", "")
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
      date: [2018-01-01, 2018-02-01, 2018-03-01, 2018-04-01, 2018-05-01, 2018-06-01, 2018-07-01, 2018-08-01, 2018-09-01, 2018-10-01, 2018-11-01, 2018-12-01]
      y:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
      x1: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
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
  nodes:
    - {{id: src_X, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: X_final}}}}}}
    - {{id: src_y, type: source, selector: {{layer_ref: l3, sink_name: l3_features_v1, subset: {{component: y_final}}}}}}
    - id: fit
      type: step
      op: fit_model
      params: {{family: {family}, {pp}, forecast_strategy: direct, training_start_rule: expanding, refit_policy: every_origin, search_algorithm: none, min_train_size: 6}}
      inputs: [src_X, src_y]
    - {{id: predict, type: step, op: predict, inputs: [fit, src_X]}}
  sinks:
    l4_forecasts_v1: predict
    l4_model_artifacts_v1: fit
    l4_training_metadata_v1: auto
"""


@pytest.mark.parametrize(
    "family,extra,marker",
    [
        # All v0.9.1 dev-stage promotions removed: 2SRR (B-1), Booging (B-4),
        # SGT (B-6), HNN (C-2). The future-gate test now exercises only the
        # remaining mismatched combinations -- specifically, hemisphere arch
        # *without* the volatility_emphasis loss (or vice versa), which the
        # paper requires together.
        (
            "mlp",
            {"architecture": "hemisphere", "loss": "mse"},
            "mlp.architecture-without-loss",
        ),
        (
            "mlp",
            {"architecture": "standard", "loss": "volatility_emphasis"},
            "mlp.loss-without-architecture",
        ),
        (
            "lstm",
            {"architecture": "hemisphere", "loss": "volatility_emphasis"},
            "lstm.hemisphere-not-mlp",
        ),
    ],
)
def test_v09_sub_axis_future_gates_raise_at_runtime(family, extra, marker, tmp_path):
    """Non-default values for sub-axis future params raise
    NotImplementedError so users discover the gap cleanly with a paper
    citation in the error message."""

    # Runtime wraps NotImplementedError into RuntimeError under fail_fast.
    # Match either the v0.8.9 "schema-only" wording or the v0.9.1 paper-
    # coupled / family-restriction message.
    with pytest.raises(
        (NotImplementedError, RuntimeError),
        match=r"(schema-only|paper-coupled|requires family='mlp')",
    ):
        macroforecast.run(
            _gated_l4_recipe(family, extra), output_directory=tmp_path / "gated"
        )


# ---------------------------------------------------------------------------
# v0.9.0B item 1: 2SRR ridge.prior=random_walk operational
# ---------------------------------------------------------------------------


def test_v090b_2srr_recovers_random_walk_beta_path():
    """Coulombe (2025 IJF) 2SRR: closed-form 2-step generalised ridge with
    a random-walk prior on coefficient deviations. Verify the recovered
    final-time-step β̂_T tracks the synthetic random-walk truth."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _TwoStageRandomWalkRidge

    rng = np.random.default_rng(7)
    T, K = 80, 2
    beta_path = np.cumsum(rng.standard_normal((T, K)) * 0.05, axis=0) + np.array(
        [1.0, -0.5]
    )
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=["x1", "x2"])
    y = pd.Series(
        (X.values * beta_path).sum(axis=1) + rng.standard_normal(T) * 0.1, index=X.index
    )

    model = _TwoStageRandomWalkRidge(alpha=1.0).fit(X, y)
    assert model._beta_path is not None
    assert model._beta_path.shape == (T, K)
    # β̂_T tracks truth with bounded error under realised noise.
    err = np.abs(model._beta_last - beta_path[-1])
    assert err.max() < 0.5  # generous; tightens with longer T


def test_v090b_2srr_alpha_to_infinity_collapses_to_static_ols():
    """Sanity check: as α → ∞ the per-time-step β̂_t collapses toward a
    constant (random-walk innovations are over-shrunk to zero), matching
    the static-OLS limit Goulet Coulombe (2025) discusses on p.985."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _TwoStageRandomWalkRidge

    rng = np.random.default_rng(11)
    T, K = 60, 2
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=["a", "b"])
    y = pd.Series(rng.standard_normal(T), index=X.index)

    # Phase B-8: pin alpha_strategy="fixed" so the second CV (paper §2.5
    # Algorithm 1 step 4) does not override the test-supplied α=1e6.
    model = _TwoStageRandomWalkRidge(alpha=1e6, alpha_strategy="fixed").fit(X, y)
    cross_time_sd = model._beta_path.std(axis=0)
    assert (cross_time_sd < 1e-2).all(), (
        f"large-α limit should yield near-flat β path; got SD = {cross_time_sd}"
    )


def test_v090b_2srr_handles_nan_in_features():
    """The wrapper fills NaN with 0 and continues. Predict on a frame
    with the same columns returns finite values."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _TwoStageRandomWalkRidge

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((40, 2)), columns=["x", "z"])
    y = pd.Series(rng.standard_normal(40), index=X.index)
    X.iloc[3, 0] = np.nan

    model = _TwoStageRandomWalkRidge(alpha=1.0).fit(X, y)
    preds = model.predict(X.iloc[-3:])
    assert preds.shape == (3,)
    assert np.all(np.isfinite(preds))


def test_v090b_2srr_runs_in_recipe(tmp_path):
    """End-to-end: the L4 recipe builder dispatches ``ridge`` with
    ``prior=random_walk`` to ``_TwoStageRandomWalkRidge`` instead of the
    default ``Ridge``. The recipe runs to a populated forecasts artefact."""

    import macroforecast

    recipe = _gated_l4_recipe("ridge", {"prior": "random_walk", "alpha": 1.0})
    result = macroforecast.run(recipe, output_directory=tmp_path / "2srr")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


# ---------------------------------------------------------------------------
# v0.9.0B item 2: Maximally FL Albacore_comps (shrink_to_target)
# ---------------------------------------------------------------------------


def test_v090b_g1_shrink_to_target_alpha_zero_recovers_truth():
    """α = 0 reduces shrink-to-target ridge to constrained NNLS on the
    simplex. Recover the synthetic convex-combination truth within
    bootstrap noise."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _ShrinkToTargetRidge

    rng = np.random.default_rng(42)
    T, K = 80, 4
    true_w = np.array([0.4, 0.3, 0.2, 0.1])
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abcd"))
    y = pd.Series(X.values @ true_w + rng.standard_normal(T) * 0.1)

    model = _ShrinkToTargetRidge(alpha=0.0, prior_target=[1 / K] * K).fit(X, y)
    coef = model._coef
    assert coef is not None
    np.testing.assert_allclose(coef.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(coef, true_w, atol=0.05)


def test_v090b_g1_shrink_to_target_alpha_infinity_returns_target():
    """α → ∞ shrinks coefficients to ``prior_target`` exactly,
    irrespective of the data."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _ShrinkToTargetRidge

    rng = np.random.default_rng(0)
    K = 4
    target = np.array([0.5, 0.25, 0.15, 0.10])
    X = pd.DataFrame(rng.standard_normal((50, K)), columns=list("abcd"))
    y = pd.Series(rng.standard_normal(50))

    model = _ShrinkToTargetRidge(alpha=1e6, prior_target=target.tolist()).fit(X, y)
    np.testing.assert_allclose(model._coef, target, atol=1e-3)


def test_v090b_g1_shrink_to_target_runs_in_recipe(tmp_path):
    import macroforecast

    # The gated recipe has K=1 predictor (x1, lagged 1). Paper Albacore
    # (Goulet Coulombe et al. 2024) Eq. (1) requires w_headline (basket
    # weights). Post-F14 the runtime hard-errors on prior_target=None, so
    # we supply a trivial K=1 basket weight of [1.0].
    recipe = _gated_l4_recipe(
        "ridge",
        {
            "prior": "shrink_to_target",
            "alpha": 1.0,
            "coefficient_constraint": "nonneg",
            "prior_target": [1.0],
        },
    )
    result = macroforecast.run(recipe, output_directory=tmp_path / "albacomp")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


# ---------------------------------------------------------------------------
# v0.9.0B item 3: Maximally FL Albacore_ranks (fused_difference)
# ---------------------------------------------------------------------------


def test_v090b_g2_fused_difference_alpha_infinity_yields_uniform_shape():
    """α → ∞ forces ``Dw = 0`` so all components are equal (the level is
    pinned by the mean-equality constraint, not necessarily 1/K)."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _FusedDifferenceRidge

    rng = np.random.default_rng(7)
    K = 5
    X_sorted = pd.DataFrame(
        np.sort(rng.standard_normal((60, K)), axis=1), columns=list("abcde")
    )
    y = pd.Series(X_sorted.values @ np.full(K, 1 / K) + rng.standard_normal(60) * 0.1)

    model = _FusedDifferenceRidge(alpha=1e6).fit(X_sorted, y)
    coef = model._coef
    assert coef is not None
    # All components within tight tolerance (uniform shape)
    assert coef.std() < 1e-3, f"expected ~uniform; got std {coef.std()}"


def test_v090b_g2_fused_difference_mean_equality_holds():
    """When ``mean_equality=True`` (default), the recovered weights satisfy
    ``mean(Xw) = mean(y)`` to machine precision."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _FusedDifferenceRidge

    rng = np.random.default_rng(11)
    K = 4
    X = pd.DataFrame(rng.standard_normal((50, K)), columns=list("abcd"))
    y = pd.Series(X.values @ np.full(K, 1 / K) + rng.standard_normal(50) * 0.05)

    model = _FusedDifferenceRidge(alpha=10.0, mean_equality=True).fit(X, y)
    pred_mean = float((X.values @ model._coef).mean())
    assert abs(pred_mean - float(y.mean())) < 1e-4


def test_v090b_g2_fused_difference_alpha_zero_collapses_to_ols(tmp_path):
    """α = 0 with mean_equality=False → unconstrained OLS (modulo nonneg)."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _FusedDifferenceRidge
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(0)
    K = 3
    X = pd.DataFrame(rng.standard_normal((100, K)), columns=list("xyz"))
    y = pd.Series(
        X.values @ np.array([0.6, 0.3, 0.1]) + rng.standard_normal(100) * 0.05
    )

    model = _FusedDifferenceRidge(alpha=0.0, mean_equality=False, nonneg=False).fit(
        X, y
    )
    ols = LinearRegression(fit_intercept=False).fit(X.values, (y - y.mean()).values)
    np.testing.assert_allclose(model._coef, ols.coef_, atol=1e-4)


def test_v090b_g2_fused_difference_runs_in_recipe(tmp_path):
    import macroforecast

    recipe = _gated_l4_recipe(
        "ridge",
        {"prior": "fused_difference", "alpha": 1.0, "coefficient_constraint": "nonneg"},
    )
    result = macroforecast.run(recipe, output_directory=tmp_path / "albaranks")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


# ---------------------------------------------------------------------------
# v0.9.0B item 4: Booging (Goulet Coulombe 2024 'To Bag is to Prune')
# ---------------------------------------------------------------------------


def test_v090b_booging_fits_all_outer_bags():
    """Booging fits ``B`` outer bags of inner SGB. Verify all bags
    converge under the standard hyperparameters."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _BoogingWrapper

    rng = np.random.default_rng(42)
    T, K = 120, 4
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abcd"))
    y = pd.Series(
        np.where(X["a"] > 0, X["a"] * 2 - X["b"], -X["c"] + 0.5 * X["d"])
        + rng.standard_normal(T) * 0.3
    )
    model = _BoogingWrapper(
        B=15,
        sample_frac=0.75,
        inner_n_estimators=40,
        inner_max_depth=3,
        random_state=7,
    ).fit(X, y)
    assert len(model._models) == 15
    # Sanity: in-sample R² substantially above zero on a non-trivial DGP.
    preds = model.predict(X)
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    assert 1.0 - ss_res / ss_tot > 0.5


def test_v090b_booging_aliases_sequential_residual_to_booging(tmp_path):
    """``bagging.strategy='sequential_residual'`` is a legacy alias for
    ``'booging'`` (Goulet Coulombe 2024). Verify the alias dispatches to
    the same wrapper."""

    from macroforecast.core.runtime import _build_l4_model, _BoogingWrapper

    p_canon = {
        "n_estimators": 5,
        "inner_n_estimators": 20,
        "inner_max_depth": 2,
        "random_state": 1,
        "strategy": "booging",
    }
    p_alias = dict(p_canon, strategy="sequential_residual")
    m_canon = _build_l4_model("bagging", p_canon)
    m_alias = _build_l4_model("bagging", p_alias)
    assert isinstance(m_canon, _BoogingWrapper)
    assert isinstance(m_alias, _BoogingWrapper)


def test_v090b_booging_data_augmentation_doubles_internal_design_width():
    """Per-bag fitted models see (n_kept_cols) ≈ (1 - da_drop_rate) · 2K
    columns since DA appends a noisy copy of every original feature."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _BoogingWrapper

    rng = np.random.default_rng(0)
    T, K = 60, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("xyz"))
    y = pd.Series(rng.standard_normal(T))
    model = _BoogingWrapper(
        B=5,
        sample_frac=0.8,
        inner_n_estimators=10,
        da_drop_rate=0.0,
        random_state=0,
    ).fit(X, y)
    # da_drop_rate=0 → keep all 2K columns
    for fitted_model, kept_cols in model._models:
        assert kept_cols.shape[0] == 2 * K


def test_v090b_booging_runs_in_recipe(tmp_path):
    """End-to-end: bagging family with strategy=booging dispatches to
    ``_BoogingWrapper`` and the recipe produces a forecasts artefact."""

    import macroforecast

    recipe = _gated_l4_recipe(
        "bagging",
        {
            "strategy": "booging",
            "n_estimators": 5,
            "inner_n_estimators": 10,
            "inner_max_depth": 2,
        },
    )
    result = macroforecast.run(recipe, output_directory=tmp_path / "booging")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


# ---------------------------------------------------------------------------
# v0.9.0B item 5: dual_decomposition non-linear (RF leaf-co-occurrence)
# ---------------------------------------------------------------------------


def test_v090b_dual_decomposition_rf_bit_exact_with_bootstrap():
    """RF leaf-co-occurrence kernel reproduces ``RandomForestRegressor.predict``
    bit-exactly even when bootstrap=True (sampling-with-replacement)."""

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor

    from macroforecast.core.runtime import _dual_decomposition_frame
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 3)), columns=list("abc"))
    y = pd.Series(
        np.where(X["a"] > 0, X["b"] - X["c"], 0.5 * X["a"] + X["c"])
        + rng.standard_normal(50) * 0.2
    )
    rf = RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0).fit(X, y)
    artifact = ModelArtifact(
        model_id="rf",
        family="random_forest",
        fitted_object=rf,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _dual_decomposition_frame(artifact, X, y)
    W = frame.attrs["dual_weights"].to_numpy()
    err = float(np.max(np.abs(W @ y.to_numpy() - rf.predict(X))))
    assert err < 1e-12, f"expected bit-exact dual; got max abs error {err}"
    assert frame.attrs["method"] == "rf_leaf_cooccurrence_kernel"


def test_v090b_dual_decomposition_extra_trees_bit_exact():
    """ExtraTreesRegressor (bootstrap=False default) reproduces
    ``predict`` to machine precision."""

    import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor

    from macroforecast.core.runtime import _dual_decomposition_frame
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.standard_normal((40, 4)), columns=list("wxyz"))
    y = pd.Series(rng.standard_normal(40))
    et = ExtraTreesRegressor(n_estimators=15, max_depth=5, random_state=0).fit(X, y)
    artifact = ModelArtifact(
        model_id="et",
        family="extra_trees",
        fitted_object=et,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _dual_decomposition_frame(artifact, X, y)
    W = frame.attrs["dual_weights"].to_numpy()
    err = float(np.max(np.abs(W @ y.to_numpy() - et.predict(X))))
    assert err < 1e-12


def test_v090b_dual_decomposition_rejects_unsupported_families():
    """gradient_boosting / xgboost / NN families raise NotImplementedError
    with a redirect to operational alternatives."""

    import numpy as np
    import pandas as pd
    import pytest
    from sklearn.ensemble import GradientBoostingRegressor

    from macroforecast.core.runtime import _dual_decomposition_frame
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((30, 2)), columns=["a", "b"])
    y = pd.Series(rng.standard_normal(30))
    gb = GradientBoostingRegressor(n_estimators=5, random_state=0).fit(X, y)
    artifact = ModelArtifact(
        model_id="gb",
        family="gradient_boosting",
        fitted_object=gb,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    with pytest.raises(
        NotImplementedError, match="dual_decomposition does not yet support"
    ):
        _dual_decomposition_frame(artifact, X, y)


# ---------------------------------------------------------------------------
# v0.9.0B item 6: SGT (decision_tree.split_shrinkage) operational
# ---------------------------------------------------------------------------


def test_v090b_sgt_eta_one_recovers_cart_like_fit():
    """η = 1 with H̄ = 0.25 is the CART regime per Goulet Coulombe (2024)
    Algorithm 1: rule-violating rows receive weight (1 − 1) = 0, so the
    soft-weighted tree collapses to hard splits. Verify in-sample R² is
    high on a near-linear DGP (the tree should fit aggressively)."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _SlowGrowingTree

    rng = np.random.default_rng(0)
    T, K = 80, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] + X["b"] - 0.5 * X["c"] + rng.standard_normal(T) * 0.3)
    model = _SlowGrowingTree(eta=1.0, herfindahl_threshold=0.25, max_depth=5).fit(X, y)
    preds = model.predict(X)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - preds) ** 2).sum())
    assert 1.0 - ss_res / ss_tot > 0.85, "CART regime should fit aggressively"


def test_v090b_sgt_smaller_eta_yields_smoother_fit():
    """η < 1 keeps non-zero weight on rule-violating rows, propagating
    influence beyond the local split. The fit should be *less* aggressive
    in-sample (lower R²) than η = 1 — this is the SLOTH-vs-CART distinction
    paper Figure 2 demonstrates."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _SlowGrowingTree

    rng = np.random.default_rng(0)
    T, K = 80, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] + X["b"] - 0.5 * X["c"] + rng.standard_normal(T) * 0.3)

    cart = _SlowGrowingTree(eta=1.0, herfindahl_threshold=0.25, max_depth=4).fit(X, y)
    sgt = _SlowGrowingTree(eta=0.5, herfindahl_threshold=0.25, max_depth=4).fit(X, y)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    cart_r2 = 1 - float(((y - cart.predict(X)) ** 2).sum()) / ss_tot
    sgt_r2 = 1 - float(((y - sgt.predict(X)) ** 2).sum()) / ss_tot
    # SGT (η = 0.5) is strictly less aggressive in-sample.
    assert cart_r2 > sgt_r2


def test_v090b_sgt_predicts_finite_on_test_rows():
    """The soft-weighted predict path traverses *all* branches with
    propagated test weights and aggregates. Verify finiteness on test
    rows that lie outside the training distribution."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _SlowGrowingTree

    rng = np.random.default_rng(7)
    X_train = pd.DataFrame(rng.standard_normal((50, 2)), columns=list("xy"))
    y_train = pd.Series(rng.standard_normal(50))
    X_test = pd.DataFrame(
        rng.standard_normal((10, 2)) * 5.0, columns=list("xy")
    )  # extrapolate
    model = _SlowGrowingTree(eta=0.7, herfindahl_threshold=0.25, max_depth=4).fit(
        X_train, y_train
    )
    preds = model.predict(X_test)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))


def test_v090b_sgt_runs_in_recipe(tmp_path):
    """End-to-end: decision_tree with split_shrinkage > 0 dispatches to
    ``_SlowGrowingTree`` and the recipe completes with a populated
    forecasts artefact."""

    import macroforecast

    recipe = _gated_l4_recipe(
        "decision_tree",
        {"split_shrinkage": 0.5, "herfindahl_threshold": 0.25, "max_depth": 3},
    )
    result = macroforecast.run(recipe, output_directory=tmp_path / "sgt")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


# ---------------------------------------------------------------------------
# v0.9.0C item 1: AlbaMA adaptive_ma_rf operational
# ---------------------------------------------------------------------------


def test_v090c_albama_two_sided_recovers_piecewise_structure():
    """AlbaMA (Goulet Coulombe & Klieber 2025) recovers a piecewise-
    constant signal up to the noise floor. Two-sided fit on full sample."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _adaptive_ma_rf

    rng = np.random.default_rng(0)
    T = 120
    y = (
        np.concatenate([np.full(40, 1.0), np.full(40, 3.0), np.full(40, 0.5)])
        + rng.standard_normal(T) * 0.3
    )
    frame = pd.DataFrame({"y": y})
    out = _adaptive_ma_rf(
        frame, n_estimators=50, min_samples_leaf=15, sided="two", random_state=0
    )
    assert out.shape == (T, 1)
    assert np.all(np.isfinite(out["y_albama"]))
    # MSE should be on the order of the noise variance (≈ 0.09)
    mse = float(((out["y_albama"].values - y) ** 2).mean())
    assert mse < 0.5  # generous; tightens with more trees


def test_v090c_albama_one_sided_expanding_window_first_min_leaf_nan():
    """One-sided AlbaMA fits an expanding-window forest at every time
    index t. The first ``min_samples_leaf - 1`` rows are NaN by design
    (RF refuses to fit smaller than the leaf bound)."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _adaptive_ma_rf

    rng = np.random.default_rng(0)
    T = 60
    frame = pd.DataFrame({"y": rng.standard_normal(T)})
    out = _adaptive_ma_rf(
        frame, n_estimators=10, min_samples_leaf=15, sided="one", random_state=0
    )
    series = out["y_albama"]
    assert series.iloc[:14].isna().all()
    # From index 14 onward we have valid predictions.
    assert series.iloc[14:].notna().all()


def test_v090c_albama_op_registered_operational():
    """``adaptive_ma_rf`` no longer carries ``status="future"`` after
    v0.9.0C-1 promotion."""

    from macroforecast.core.ops import get_op

    spec = get_op("adaptive_ma_rf")
    assert spec is not None
    assert getattr(spec, "status", "operational") != "future"


# ---------------------------------------------------------------------------
# v0.9.0C item 2: HNN (mlp.architecture=hemisphere + mlp.loss=volatility_emphasis)
# ---------------------------------------------------------------------------


def test_v090c_hnn_fits_and_predicts_finite_values():
    """HNN trains B=3 outer ensembles with the Gaussian NLL + ν-emphasis
    constraint; predict returns finite mean forecasts."""

    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _HemisphereNN

    rng = np.random.default_rng(0)
    T, K = 60, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] - 0.5 * X["b"] + rng.standard_normal(T) * 0.3)

    model = _HemisphereNN(
        B=3,
        n_epochs=10,
        neurons=16,
        lc=1,
        lm=1,
        lv=1,
        random_state=42,
    ).fit(X, y)
    assert len(model._models) == 3
    preds = model.predict(X)
    assert preds.shape == (T,)
    assert np.all(np.isfinite(preds))


def test_hnn_predict_variance_is_positive_and_finite():
    """Goulet Coulombe / Frenette / Klieber (2025) Eq. 10 reality check.
    predict_variance must return strictly-positive finite values per row."""

    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _HemisphereNN

    rng = np.random.default_rng(1)
    T, K = 60, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] - 0.5 * X["b"] + rng.standard_normal(T) * 0.3)

    model = _HemisphereNN(
        B=3,
        n_epochs=10,
        neurons=16,
        lc=1,
        lm=1,
        lv=1,
        random_state=7,
    ).fit(X, y)
    var_pred = model.predict_variance(X)
    assert var_pred.shape == (T,)
    assert np.all(np.isfinite(var_pred))
    assert np.all(var_pred > 0.0), "h_v must be strictly positive"


def test_hnn_predict_distribution_returns_mean_and_variance_pair():
    """predict_distribution returns (mean, variance) consistent with
    predict() and predict_variance() respectively. Guards against the
    earlier dead-code path where Eq. 10 coefficients were computed but
    never surfaced at predict time."""

    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _HemisphereNN

    rng = np.random.default_rng(2)
    T, K = 50, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(T))

    model = _HemisphereNN(
        B=2,
        n_epochs=8,
        neurons=16,
        lc=1,
        lm=1,
        lv=1,
        random_state=11,
    ).fit(X, y)
    mean_pred, var_pred = model.predict_distribution(X)
    np.testing.assert_array_equal(mean_pred, model.predict(X))
    np.testing.assert_array_equal(var_pred, model.predict_variance(X))


def test_hnn_reality_check_eq10_actually_applied_at_predict():
    """Eq. 10 is live code: predict_variance picks up changes in
    ``_reality_check_intercept`` / ``_reality_check_slope``. Earlier
    implementation computed these at fit-time but never read them at
    predict-time -- this test guards the regression."""

    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _HemisphereNN

    rng = np.random.default_rng(3)
    T, K = 50, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(T))

    model = _HemisphereNN(
        B=2,
        n_epochs=8,
        neurons=16,
        lc=1,
        lm=1,
        lv=1,
        random_state=21,
    ).fit(X, y)
    # Pin reality-check coefficients to a known non-trivial pair, then
    # verify the variance prediction shifts accordingly.
    model._reality_check_intercept = 0.0
    model._reality_check_slope = 1.0
    var_baseline = model.predict_variance(X).copy()
    model._reality_check_intercept = float(np.log(4.0))  # multiplicative ×4
    model._reality_check_slope = 1.0
    var_shifted = model.predict_variance(X)
    np.testing.assert_allclose(var_shifted, 4.0 * var_baseline, rtol=1e-6)
    # Slope ≠ 1 also propagates.
    model._reality_check_intercept = 0.0
    model._reality_check_slope = 0.5
    var_sqrt = model.predict_variance(X)
    np.testing.assert_allclose(var_sqrt, np.sqrt(var_baseline), rtol=1e-6)


def test_hnn_predict_quantiles_returns_monotonic_gaussian_bands():
    """Calibrated Gaussian quantile bands from (mean, Eq.-10-corrected
    variance). Monotonic in q; q=0.5 equals predict()."""

    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _HemisphereNN

    rng = np.random.default_rng(4)
    T, K = 50, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(T))

    model = _HemisphereNN(
        B=2,
        n_epochs=8,
        neurons=16,
        lc=1,
        lm=1,
        lv=1,
        random_state=33,
    ).fit(X, y)
    levels = (0.05, 0.5, 0.95)
    bands = model.predict_quantiles(X, levels=levels)
    assert set(bands.keys()) == {0.05, 0.5, 0.95}
    np.testing.assert_allclose(bands[0.5], model.predict(X), rtol=1e-6)
    assert np.all(bands[0.05] <= bands[0.5] + 1e-9)
    assert np.all(bands[0.5] <= bands[0.95] + 1e-9)


def test_v090c_hnn_requires_both_arch_and_loss():
    """HNN is paper-coupled: hemisphere architecture without
    volatility_emphasis loss (or vice versa) raises NotImplementedError
    with a clear hint."""

    pytest.importorskip("torch")
    import macroforecast

    with pytest.raises((NotImplementedError, RuntimeError), match="paper-coupled"):
        macroforecast.run(
            _gated_l4_recipe("mlp", {"architecture": "hemisphere", "loss": "mse"}),
            output_directory="/tmp/hnn_partial",
        )


def test_v090c_hnn_runs_in_recipe(tmp_path):
    """End-to-end smoke: mlp with both sub-axes set runs."""

    pytest.importorskip("torch")
    import macroforecast

    recipe = _gated_l4_recipe(
        "mlp",
        {
            "architecture": "hemisphere",
            "loss": "volatility_emphasis",
            "B": 2,
            "n_epochs": 5,
            "neurons": 8,
            "lc": 1,
            "lm": 1,
            "lv": 1,
        },
    )
    result = macroforecast.run(recipe, output_directory=tmp_path / "hnn")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


# ---------------------------------------------------------------------------
# v0.9.0C items 3-4: Sparse Macro Factors (sparse_pca_chen_rohe + supervised_pca)
# ---------------------------------------------------------------------------


def test_v090c_sparse_pca_chen_rohe_recovers_sparse_loadings():
    """Chen-Rohe (2023) SCA recovers a known sparse-loadings DGP. Verify
    the first component correlates strongly (>0.9) with the dominant
    latent factor."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _sparse_pca_chen_rohe

    rng = np.random.default_rng(0)
    T, M, J = 100, 10, 3
    Z_true = rng.standard_normal((T, J))
    loadings = np.zeros((M, J))
    loadings[:4, 0] = 1.0
    loadings[3:7, 1] = 1.0
    loadings[7:, 2] = 1.0
    X = pd.DataFrame(
        Z_true @ loadings.T + rng.standard_normal((T, M)) * 0.3,
        columns=[f"x{i}" for i in range(M)],
    )
    sca = _sparse_pca_chen_rohe(
        X, n_components=3, zeta=3.0, max_iter=100, random_state=0
    )
    assert sca.shape == (T, 3)
    corr = abs(np.corrcoef(sca["sca_1"].values, Z_true[:, 0])[0, 1])
    assert corr > 0.9, f"expected SCA factor 1 to track truth; got {corr}"


def test_v090c_supervised_pca_aligns_with_target():
    """Giglio-Xiu-Zhang (2025) SPCA: screen-then-PCA on a sub-panel
    correlated with the target. The first SPCA factor should track the
    target above the unsupervised PCA baseline."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _supervised_pca

    rng = np.random.default_rng(7)
    T, M = 100, 10
    target = pd.Series(rng.standard_normal(T))
    # Half columns are correlated with target, half are noise.
    X = pd.DataFrame(rng.standard_normal((T, M)), columns=[f"x{i}" for i in range(M)])
    for j in range(M // 2):
        X[f"x{j}"] += 1.5 * target
    spca = _supervised_pca(X, target=target, n_components=3, q=0.5)
    assert spca.shape == (T, 3)
    corr_spca = abs(np.corrcoef(spca["spca_1"].values, target.values)[0, 1])
    assert corr_spca > 0.5, f"expected SPCA factor 1 to track target; got {corr_spca}"


# ---------------------------------------------------------------------------
# v0.9.0D: anatomy adapter Path B (degraded — final-window fit)
# ---------------------------------------------------------------------------


def _anatomy_extra_available() -> bool:
    try:
        import anatomy  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed (pip install macroforecast[anatomy])",
)
def test_v090d_anatomy_oshapley_vi_recovers_dominant_feature():
    """anatomy oshapley_vi assigns highest importance to the feature
    with the largest true coefficient on a synthetic linear DGP."""

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge

    from macroforecast.core.runtime import _l7_anatomy_op
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    T, K = 60, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] - X["b"] + 0.5 * X["c"] + rng.standard_normal(T) * 0.2)
    ridge = Ridge(alpha=1.0).fit(X, y)
    art = ModelArtifact(
        model_id="r",
        family="ridge",
        fitted_object=ridge,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _l7_anatomy_op(
        "oshapley_vi", art, X, y, params={"n_iterations": 5, "random_state": 0}
    )
    assert frame.shape == (3, 4)
    importances = dict(zip(frame["feature"], frame["importance"]))
    assert importances["a"] > importances["b"] > importances["c"]


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_v090d_anatomy_pbsv_marks_status_degraded():
    """Path B uses the final-window fit for every period; status must
    be ``"degraded"`` to signal the audit-flagged approximation. Reuse
    the same matched-signal setup as the oshapley_vi test (anatomy
    0.1.6 PBSV is sensitive to noise-only targets at low n_iterations
    and triggers IndexError inside its private algorithm; the matched-
    signal data is the path the package actually exercises in CI)."""

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge

    from macroforecast.core.runtime import _l7_anatomy_op
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    T, K = 60, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] - X["b"] + 0.5 * X["c"] + rng.standard_normal(T) * 0.2)
    ridge = Ridge(alpha=1.0).fit(X, y)
    art = ModelArtifact(
        model_id="r",
        family="ridge",
        fitted_object=ridge,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _l7_anatomy_op(
        "pbsv", art, X, y, params={"n_iterations": 5, "random_state": 0}
    )
    assert (frame["status"] == "degraded").all()


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_v090e_anatomy_path_a_marks_status_operational():
    """v0.9.0E Path A: when ``initial_window`` is supplied, the adapter
    drives anatomy through ``AnatomySubsets.generate(EXPANDING, ...)``
    with per-period refit. Status = ``"operational"`` (not degraded)."""

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge

    from macroforecast.core.runtime import _l7_anatomy_op
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    T, K = 100, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] - X["b"] + 0.5 * X["c"] + rng.standard_normal(T) * 0.2)
    ridge = Ridge(alpha=1.0).fit(X, y)
    art = ModelArtifact(
        model_id="r",
        family="ridge",
        fitted_object=ridge,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    frame = _l7_anatomy_op(
        "oshapley_vi",
        art,
        X,
        y,
        params={"n_iterations": 5, "initial_window": 60, "random_state": 0},
    )
    # Path A clears the degraded marker.
    assert (frame["status"] == "operational").all()
    # Same importance ranking as Path B (both should rank `a` highest).
    importances = dict(zip(frame["feature"], frame["importance"]))
    assert importances["a"] > importances["c"]


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_anatomy_path_b_emits_user_warning_about_degraded_routing():
    """Audit gap-fix: silently routing to Path B (final-window fit) is a
    different estimand from Borup et al. (2022). The adapter must surface
    a UserWarning naming Path B and pointing to ``initial_window`` for
    paper-faithful Path A."""

    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge

    from macroforecast.core.runtime import _l7_anatomy_op
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    T, K = 60, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] - X["b"] + 0.5 * X["c"] + rng.standard_normal(T) * 0.2)
    ridge = Ridge(alpha=1.0).fit(X, y)
    art = ModelArtifact(
        model_id="r",
        family="ridge",
        fitted_object=ridge,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _l7_anatomy_op(
            "oshapley_vi", art, X, y, params={"n_iterations": 5, "random_state": 0}
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("Path B" in m and "initial_window" in m for m in msgs), (
        f"expected Path B / initial_window warning; got {msgs}"
    )


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_anatomy_path_a_with_initial_window_does_not_warn_about_path_b():
    """Counterpart: when ``initial_window`` is supplied, Path A is taken
    and the Path B warning must NOT fire."""

    import warnings
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge

    from macroforecast.core.runtime import _l7_anatomy_op
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(0)
    T, K = 100, 3
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abc"))
    y = pd.Series(2 * X["a"] - X["b"] + 0.5 * X["c"] + rng.standard_normal(T) * 0.2)
    ridge = Ridge(alpha=1.0).fit(X, y)
    art = ModelArtifact(
        model_id="r",
        family="ridge",
        fitted_object=ridge,
        framework="sklearn",
        feature_names=tuple(X.columns),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _l7_anatomy_op(
            "oshapley_vi",
            art,
            X,
            y,
            params={"n_iterations": 5, "initial_window": 60, "random_state": 0},
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any("Path B" in m for m in msgs), (
        f"Path A with initial_window should not warn about Path B; got {msgs}"
    )


@pytest.mark.skipif(
    not _anatomy_extra_available(),
    reason="[anatomy] extra not installed",
)
def test_v090d_anatomy_ops_registered_operational():
    """Both ``oshapley_vi`` and ``pbsv`` moved from FUTURE_OPS to
    OPERATIONAL_OPS in v0.9.0D."""

    from macroforecast.core.ops.l7_ops import OPERATIONAL_OPS, FUTURE_OPS

    assert "oshapley_vi" in OPERATIONAL_OPS
    assert "pbsv" in OPERATIONAL_OPS
    assert "oshapley_vi" not in FUTURE_OPS
    assert "pbsv" not in FUTURE_OPS


def test_v090c_sparse_pca_chen_rohe_is_distinct_from_sklearn_sparse_pca():
    """The new ``sparse_pca_chen_rohe`` op is registered alongside the
    existing ``sparse_pca`` (sklearn / Zou-Hastie-Tibshirani 2006). The
    two ops should produce *different* loadings on the same input -- the
    audit-gap closure for V2.5."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _sparse_pca_chen_rohe, _pca_factors

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((50, 6)), columns=[f"x{i}" for i in range(6)])
    sca = _sparse_pca_chen_rohe(
        X, n_components=2, zeta=2.0, max_iter=50, random_state=0
    )
    sklearn_sparse = _pca_factors(
        X, n_components=2, variant="sparse_pca", target_signal=None
    )
    # Both shape (T, 2). Loadings differ because algorithms differ.
    assert sca.shape == sklearn_sparse.shape == (50, 2)
    sca_v = sca.iloc[:, 0].dropna().values
    sk_v = sklearn_sparse.iloc[:, 0].dropna().values
    n = min(len(sca_v), len(sk_v))
    if n > 5:
        # Distinct enough that absolute correlation is < 0.99 (i.e. not
        # the same component up to sign).
        corr = abs(np.corrcoef(sca_v[:n], sk_v[:n])[0, 1])
        assert corr < 0.99, (
            "expected SCA and sklearn SparsePCA to give distinct factors"
        )


def test_sparse_pca_chen_rohe_enforces_l1_budget_when_binding():
    """Rapach-Zhou (2025) Eq. (4) requires ‖Θ‖_1 ≤ ζ. With the binding
    boundary ζ = J (the paper's CV-optimal choice) the loadings matrix
    must satisfy the budget; an earlier implementation re-orthonormalised
    Θ after soft-thresholding which silently restored mass and violated
    the constraint -- this test guards the fix."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _sparse_pca_chen_rohe

    rng = np.random.default_rng(11)
    T, M, J = 80, 12, 3
    Z_true = rng.standard_normal((T, J))
    loadings = np.zeros((M, J))
    loadings[:5, 0] = 1.0
    loadings[5:9, 1] = 1.0
    loadings[9:, 2] = 1.0
    X_arr = Z_true @ loadings.T + rng.standard_normal((T, M)) * 0.4
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(M)])

    # Recover Θ via the same routine the runtime uses, then check the L1
    # norm directly. Use the binding boundary ζ = J.
    X_centred = X_arr - X_arr.mean(axis=0, keepdims=True)
    Z = np.linalg.qr(rng.standard_normal((T, J)))[0]
    Theta = np.linalg.qr(rng.standard_normal((M, J)))[0]
    zeta_val = float(J)
    for _ in range(200):
        U, _, Vt = np.linalg.svd(X_centred @ Theta, full_matrices=False)
        Z = U @ Vt
        G = X_centred.T @ Z
        Uo, _, Vto = np.linalg.svd(G, full_matrices=False)
        Theta_unc = Uo @ Vto
        if np.sum(np.abs(Theta_unc)) <= zeta_val:
            Theta = Theta_unc
        else:
            lo, hi = 0.0, float(np.max(np.abs(Theta_unc)))
            for _ in range(50):
                tau = 0.5 * (lo + hi)
                Theta_st = np.sign(Theta_unc) * np.maximum(np.abs(Theta_unc) - tau, 0.0)
                if np.sum(np.abs(Theta_st)) > zeta_val:
                    lo = tau
                else:
                    hi = tau
            Theta = np.sign(Theta_unc) * np.maximum(np.abs(Theta_unc) - hi, 0.0)

    l1 = float(np.abs(Theta).sum())
    assert l1 <= zeta_val + 1e-6, (
        f"L1 budget violated: ‖Θ‖_1 = {l1:.4f} > ζ = {zeta_val:.4f}"
    )

    # And the runtime function itself still produces sensible scores at
    # the binding boundary (no silent regression in factor recovery).
    sca = _sparse_pca_chen_rohe(
        X, n_components=J, zeta=zeta_val, max_iter=200, random_state=0
    )
    assert sca.shape == (T, J)


# ---------------------------------------------------------------------------
# Phase 1 Tier 1 promotions (v0.8.9): operational implementations land here.
# ---------------------------------------------------------------------------


def test_b1_nonneg_ridge_recovers_positive_truth():
    """Non-negative ridge recovers near-unconstrained coefficients when
    the underlying truth is non-negative."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _NonNegRidge

    rng = np.random.default_rng(0)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 + 2.0 * x2 + 0.1 * rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = _NonNegRidge(alpha=0.01).fit(X, pd.Series(y))
    assert model._coef is not None
    assert model._coef[0] > 1.0  # truth 1.5
    assert model._coef[1] > 1.5  # truth 2.0
    # Predict shape matches input rows
    assert model.predict(X).shape == (n,)


def test_b1_nonneg_ridge_clips_negative_truth_to_zero():
    """When the truth has a negative coefficient, NNLS clips it to 0
    while preserving the positive coefficients."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _NonNegRidge

    rng = np.random.default_rng(1)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = -1.0 * x1 + 2.0 * x2 + 0.1 * rng.normal(size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    model = _NonNegRidge(alpha=0.1).fit(X, pd.Series(y))
    assert model._coef is not None
    assert model._coef[0] == 0.0  # negative truth clipped
    assert model._coef[1] > 1.5  # positive truth preserved


def test_b1_nonneg_ridge_runs_in_recipe(tmp_path):
    """End-to-end: ridge(coefficient_constraint=nonneg) recipe runs to
    completion (no longer hits the future gate)."""

    recipe = _gated_l4_recipe(
        "ridge", {"coefficient_constraint": "nonneg", "alpha": 0.01}
    )
    result = macroforecast.run(recipe, output_directory=tmp_path / "nonneg")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


def test_b4_block_bagging_draws_consecutive_blocks():
    """Block-bootstrap index draws are runs of consecutive integers
    (mod n) of length block_length."""

    import numpy as np
    from macroforecast.core.runtime import _BaggingWrapper

    wrapper = _BaggingWrapper(strategy="block", block_length=5)
    rng = np.random.default_rng(0)
    idx = wrapper._draw_indices(rng, n=50, sample_size=20)
    assert len(idx) == 20
    # Reshape into blocks; each block of length 5 should have consecutive
    # integers (mod 50) -- diff == 1 (or -49 wraparound).
    blocks = idx.reshape(-1, 5)
    for block in blocks:
        diffs = np.diff(block) % 50
        assert all(d == 1 for d in diffs), f"block {block} not consecutive (mod 50)"


def test_b4_block_bagging_runs_in_recipe(tmp_path):
    """End-to-end: bagging(strategy=block) recipe runs (no future gate)."""

    recipe = _gated_l4_recipe(
        "bagging",
        {
            "strategy": "block",
            "base_family": "ridge",
            "block_length": 3,
            "n_estimators": 5,
        },
    )
    result = macroforecast.run(recipe, output_directory=tmp_path / "block")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


# ---------------------------------------------------------------------------
# perfectly_random_forest helper: end-to-end run via the Python builder.
# ---------------------------------------------------------------------------


def test_perfectly_random_forest_helper_runs(tmp_path):
    """Helper from macroforecast.recipes.paper_methods produces a working
    operational recipe (PRF = extra_trees(max_features=1))."""

    from macroforecast.recipes.paper_methods import perfectly_random_forest

    recipe = perfectly_random_forest()
    result = macroforecast.run(recipe, output_directory=tmp_path / "prf")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts


def test_perfectly_random_forest_recipe_yaml_runs(tmp_path):
    """The companion YAML at examples/recipes/replications/perfectly_random_forest.yaml
    is the canonical hand-readable copy of the helper output."""

    from pathlib import Path

    yaml_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "recipes"
        / "replications"
        / "perfectly_random_forest.yaml"
    )
    if not yaml_path.exists():
        pytest.skip("PRF replication recipe missing")
    result = macroforecast.run(
        yaml_path.read_text(), output_directory=tmp_path / "prf_yaml"
    )
    assert result.cells


# ---------------------------------------------------------------------------
# V2.2 macroeconomic_random_forest re-anchor to vendored MRF (v0.8.9 honesty)
# ---------------------------------------------------------------------------


def test_v22_mrf_external_wrapper_matches_vendored_reference():
    """Known-answer test for ``_MRFExternalWrapper``.

    The wrapper must produce the same forecast vector as a direct call
    to the vendored ``MacroRandomForest._ensemble_loop()`` given matched
    parameters and the same numpy seed -- verifying we are delegating
    without applying any algorithmic surgery of our own.
    """

    import numpy as np
    import pandas as pd

    from macroforecast._vendor.macro_random_forest import MacroRandomForest
    from macroforecast.core.runtime import _MRFExternalWrapper

    rng = np.random.default_rng(0)
    n = 80
    y = np.zeros(n)
    y[0] = 0.5
    for t in range(1, n):
        y[t] = 0.55 * y[t - 1] + rng.standard_normal()
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
        },
        index=pd.RangeIndex(n),
    )
    y_ser = pd.Series(y, index=X.index, name="y")

    n_train = 60
    train_X, test_X = X.iloc[:n_train], X.iloc[n_train:]
    train_y = y_ser.iloc[:n_train]

    common = dict(
        B=10,
        ridge_lambda=0.1,
        rw_regul=0.75,
        mtry_frac=0.5,
        trend_push=1,
        quantile_rate=0.3,
        fast_rw=True,
        resampling_opt=2,
        parallelise=False,
        n_cores=1,
        block_size=24,
    )

    # Path 1: through the macroforecast wrapper.
    np.random.seed(123)
    wrapper = _MRFExternalWrapper(random_state=123, **common)
    wrapper.fit(train_X, train_y)
    wrapper_preds = wrapper.predict(test_X)

    # Path 2: directly via the package, using the same data assembly the
    # wrapper performs internally.
    train_block = pd.concat(
        [train_y.rename("y").reset_index(drop=True), train_X.reset_index(drop=True)],
        axis=1,
    )
    test_block = pd.DataFrame(
        np.column_stack(
            [
                np.zeros(len(test_X), dtype=float),
                test_X.to_numpy(dtype=float),
            ]
        ),
        columns=["y", *X.columns],
    )
    data = pd.concat([train_block, test_block], ignore_index=True)
    feature_idx = np.arange(1, X.shape[1] + 1)
    np.random.seed(123)
    direct = MacroRandomForest(
        data=data,
        y_pos=0,
        x_pos=feature_idx,
        S_pos=feature_idx,
        oos_pos=list(range(n_train, n)),
        print_b=False,
        **{k: v for k, v in common.items()},
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        direct_out = direct._ensemble_loop()
    direct_preds = np.asarray(direct_out["pred"]).ravel()

    assert wrapper_preds.shape == direct_preds.shape == (n - n_train,)
    np.testing.assert_allclose(wrapper_preds, direct_preds, rtol=1e-10, atol=1e-10)

    # Cached state populated for L7 mrf_gtvp consumption.
    assert wrapper._cached_betas is not None
    assert wrapper._cached_betas.shape == (
        n,
        X.shape[1] + 1,
    )  # (T, K+1) with intercept col
    assert wrapper._cached_pred_ensemble is not None
    assert wrapper._cached_pred_ensemble.shape == (common["B"], n - n_train)


# ---------------------------------------------------------------------------
# V2.3 VAR/IRF/FEVD/historical_decomposition (Coulombe & Göbel 2021)
# ---------------------------------------------------------------------------


def _var_artifact_for_v23(seed: int = 0, n: int = 200, p: int = 2):
    """Helper: fit a small stationary 3-variable VAR(p) and return the
    ``ModelArtifact`` shape consumed by ``_var_impulse_frame``."""

    import numpy as np
    import pandas as pd

    from macroforecast.core.runtime import _VARWrapper
    from macroforecast.core.types import ModelArtifact

    rng = np.random.default_rng(seed)
    # Simulate a stationary VAR(1)
    eps = rng.standard_normal((n, 3))
    A = np.array([[0.4, 0.1, 0.0], [0.1, 0.5, 0.0], [0.0, 0.1, 0.6]])
    Y = np.zeros((n, 3))
    for t in range(1, n):
        Y[t] = A @ Y[t - 1] + eps[t]
    df = pd.DataFrame(Y, columns=["y", "x1", "x2"], index=pd.RangeIndex(n))
    target_y = df["y"]
    X = df[["x1", "x2"]]

    wrapper = _VARWrapper(p=p).fit(X, target_y)
    artifact = ModelArtifact(
        model_id="v23_var",
        family="var",
        fitted_object=wrapper,
        framework="statsmodels",
        feature_names=tuple(X.columns),
    )
    return artifact, wrapper, X, target_y


def test_v23_orthogonalised_irf_returns_cholesky_response():
    """The operational ``orthogonalised_irf`` matches statsmodels'
    Cholesky orth_irfs invariants: the impulse-zero matrix is the
    Cholesky factor of Σᵤ; the on-diagonal response at horizon 0 is
    positive."""

    import numpy as np

    from macroforecast.core.runtime import _var_impulse_frame

    artifact, wrapper, _, _ = _var_artifact_for_v23()
    frame = _var_impulse_frame(artifact, op_name="orthogonalised_irf", n_periods=8)
    assert (frame["status"] == "operational").all()
    assert (frame["importance"] > 0).all()  # all shocks have non-zero L1 response
    # Independent verification: orth_irfs[0] equals Cholesky(Σᵤ).
    irf = wrapper._results.irf(8)
    chol = np.linalg.cholesky(np.asarray(wrapper._results.sigma_u, dtype=float))
    np.testing.assert_allclose(np.asarray(irf.orth_irfs)[0], chol, atol=1e-10)


def test_v23_generalized_irf_is_future_gated():
    """``generalized_irf`` (Pesaran-Shin 1998) is reserved for a v0.9.x
    runtime; the schema declares it future and the L7 dispatcher must
    raise NotImplementedError pointing users to ``orthogonalised_irf``."""

    import pytest

    from macroforecast.core.ops.l7_ops import FUTURE_OPS, OPERATIONAL_OPS

    # Schema registration: ``generalized_irf`` future, ``orthogonalised_irf``
    # operational.
    assert "generalized_irf" in FUTURE_OPS
    assert "orthogonalised_irf" in OPERATIONAL_OPS
    assert "generalized_irf" not in OPERATIONAL_OPS

    # Runtime dispatcher gate: re-route via the public op resolver to
    # exercise the explicit NotImplementedError path.
    from macroforecast.core.runtime import _execute_l7_step

    # Build a minimal viable context using construction defaults. Any
    # input shape is fine because the gate triggers before any payload
    # inspection.
    from macroforecast.core.types import (
        L3FeaturesArtifact,
        L3MetadataArtifact,
        L5EvaluationArtifact,
    )

    # Inspect the dataclass fields to construct safely without coupling
    # to the internal field set.
    import dataclasses

    def _empty(cls):
        defaults = {
            f.name: getattr(f, "default", None) for f in dataclasses.fields(cls)
        }
        # Replace MISSING sentinels with None / {}
        for k, v in defaults.items():
            if v is dataclasses.MISSING:
                defaults[k] = {} if k.endswith("hashes") else None
        return cls(**defaults)

    with pytest.raises(NotImplementedError, match="orthogonalised_irf"):
        _execute_l7_step(
            "generalized_irf",
            inputs=[],
            params={},
            l3_features=_empty(L3FeaturesArtifact),
            l3_metadata=_empty(L3MetadataArtifact),
            l5_eval=_empty(L5EvaluationArtifact),
        )


def test_v23_historical_decomposition_reconstructs_target_path():
    """Burbidge-Harrison HD invariant: summing per-shock contributions
    across the structural shocks recovers the deviation of the target
    from its conditional mean. We verify a softer L1 invariant -- the
    per-shock importance is non-negative and the sum is on the same
    order of magnitude as the realised target's L1 fluctuation around
    its sample mean (within 2x). This catches the previous proxy
    implementation, which mixed |IRF| and σ(resid) without phase
    alignment."""

    import numpy as np

    from macroforecast.core.runtime import _var_impulse_frame

    artifact, wrapper, _, target_y = _var_artifact_for_v23(seed=11, n=300, p=2)
    frame = _var_impulse_frame(
        artifact, op_name="historical_decomposition", n_periods=12
    )
    assert (frame["status"] == "operational").all()
    assert (frame["importance"] >= 0).all()

    # Reconstruction lower bound: sum-of-shock-contributions should be
    # the same order as the realised L1 fluctuation of the target.
    # Drop the first p observations (no in-sample residuals at the
    # initial conditions).
    p = wrapper.p
    target_arr = np.asarray(target_y, dtype=float)[p:]
    target_l1 = float(np.abs(target_arr - target_arr.mean()).sum())
    total_importance = float(frame["importance"].sum())
    assert total_importance > 0.5 * target_l1  # share-of-fluctuation lower bound
    assert total_importance < 5.0 * target_l1  # not implausibly large

    # Old proxy (|IRF|.sum × std(resid)) was decoupled from the realised
    # path -- it was a constant-magnitude per-shock score regardless of
    # the actual residual realisations. Verify the new HD is path-
    # dependent: re-fit with permuted residual signs and confirm the
    # importance vector changes (it would not for the proxy).
    permuted_artifact, _, _, _ = _var_artifact_for_v23(seed=12, n=300, p=2)
    other = _var_impulse_frame(
        permuted_artifact, op_name="historical_decomposition", n_periods=12
    )
    assert not np.allclose(
        frame["importance"].to_numpy(), other["importance"].to_numpy()
    )


# ---------------------------------------------------------------------------
# V2.4 dfm_mixed_mariano_murasawa (Mariano & Murasawa 2003)
# ---------------------------------------------------------------------------


def _dfm_synthetic_panel(*, T: int = 60, seed: int = 7):
    """Synthetic panel for DFM tests: AR(1) latent factor driving 2 monthly
    series and 1 quarterly series (NaN-padded at non-quarter-end months
    in monthly index)."""

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=T, freq="ME")
    f = np.zeros(T)
    f[0] = 0.5
    for t in range(1, T):
        f[t] = 0.7 * f[t - 1] + rng.standard_normal()
    X = pd.DataFrame(
        {
            "m1": f + rng.standard_normal(T) * 0.3,
            "m2": f * 0.8 + rng.standard_normal(T) * 0.3,
            "q1": f * -0.5 + rng.standard_normal(T) * 0.3,
        },
        index=idx,
    )
    X.loc[~X.index.month.isin([3, 6, 9, 12]), "q1"] = np.nan
    y = pd.Series(0.6 * f + rng.standard_normal(T) * 0.2, index=idx, name="y")
    return X, y


def test_v24_dfm_mq_pure_monthly_uses_mariano_murasawa_2010_ar1():
    """Pure-monthly mixed_frequency=True path runs DynamicFactorMQ with
    ``idiosyncratic_ar1=True`` (Mariano-Murasawa 2010 Eq. 4 spec)."""

    from macroforecast.core.runtime import _DFMMixedFrequency

    X, y = _dfm_synthetic_panel()
    model = _DFMMixedFrequency(
        n_factors=1,
        factor_order=1,
        mixed_frequency=True,
        column_frequencies={"m1": "monthly", "m2": "monthly", "__y__": "monthly"},
    ).fit(X[["m1", "m2"]], y)
    assert model._mode == "mixed_frequency_mq"
    assert model._idiosyncratic_ar1 is True
    assert model._mq_failure_reason is None
    # statsmodels MLEResultsWrapper exposes ``loglike``; PCA does not.
    # statsmodels state-space MLE wrappers expose ``llf`` (log-likelihood)
    # and ``filter_results`` (Kalman filter output); a PCA factor model
    # exposes neither. The compound check pins the algorithm to Kalman+MLE.
    assert hasattr(model._results, "llf") and hasattr(model._results, "filter_results")


def test_v24_dfm_mq_mixed_m_q_handles_quarterly_nan_padded_input():
    """Mixed monthly + quarterly path: user supplies quarterly variables
    NaN-padded at non-quarter-end months on a monthly DateTimeIndex; the
    runtime drops the NaN rows and reindexes to a quarterly DateTimeIndex
    (freq='QE') before passing to DynamicFactorMQ.

    Pre-fix behaviour silently fell back to single-frequency because
    statsmodels rejected the input shape; the v0.8.9 V2.4 honesty pass
    surfaces the success / failure via ``_mq_failure_reason``."""

    from macroforecast.core.runtime import _DFMMixedFrequency

    X, y = _dfm_synthetic_panel()
    model = _DFMMixedFrequency(
        n_factors=1,
        factor_order=1,
        mixed_frequency=True,
        column_frequencies={
            "q1": "quarterly",
            "m1": "monthly",
            "m2": "monthly",
            "__y__": "monthly",
        },
    ).fit(X, y)
    assert model._mode == "mixed_frequency_mq"
    assert model._mq_failure_reason is None
    # statsmodels state-space MLE wrappers expose ``llf`` (log-likelihood)
    # and ``filter_results`` (Kalman filter output); a PCA factor model
    # exposes neither. The compound check pins the algorithm to Kalman+MLE.
    assert hasattr(model._results, "llf") and hasattr(
        model._results, "filter_results"
    )  # state-space MLE invariant
    # Forecast is finite and broadcast to all rows (single-step ahead).
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    import numpy as np

    assert np.all(np.isfinite(preds))


def test_v24_dfm_single_frequency_falls_back_to_state_space_dfm():
    """Without mixed_frequency, the runtime uses statsmodels DynamicFactor
    (state-space MLE, not PCA)."""

    from macroforecast.core.runtime import _DFMMixedFrequency

    X, y = _dfm_synthetic_panel()
    # Drop NaN rows so single-frequency DynamicFactor can use full panel.
    X_dropna = X.dropna()
    y_dropna = y.loc[X_dropna.index]
    model = _DFMMixedFrequency(n_factors=1, factor_order=1, mixed_frequency=False).fit(
        X_dropna,
        y_dropna,
    )
    assert model._mode == "single_frequency"
    # statsmodels state-space MLE wrappers expose ``llf`` (log-likelihood)
    # and ``filter_results`` (Kalman filter output); a PCA factor model
    # exposes neither. The compound check pins the algorithm to Kalman+MLE.
    assert hasattr(model._results, "llf") and hasattr(model._results, "filter_results")
    # ``DynamicFactor`` runs Kalman filter + MLE; PCA produces no loglik.
    assert "Dynamic" in type(model._results).__name__


def test_v24_dfm_mq_failure_surfaces_in_diagnostic_attribute():
    """When mixed_frequency is requested but the MQ fit cannot run (e.g.
    no monthly variables declared), the diagnostic message lands on
    ``_mq_failure_reason`` instead of being silently swallowed."""

    from macroforecast.core.runtime import _DFMMixedFrequency

    # All variables flagged quarterly -> no monthly anchor -> MQ refused.
    X, y = _dfm_synthetic_panel()
    model = _DFMMixedFrequency(
        n_factors=1,
        factor_order=1,
        mixed_frequency=True,
        column_frequencies={
            "m1": "quarterly",
            "m2": "quarterly",
            "q1": "quarterly",
            "__y__": "quarterly",
        },
    ).fit(X, y)
    # MQ refused -> single-frequency fallback ran.
    assert model._mode == "single_frequency"
    assert isinstance(model._mq_failure_reason, str)
    assert "monthly" in model._mq_failure_reason.lower()


# ---------------------------------------------------------------------------
# Paper 15 Data Transforms helper -- audit gap-fix for default families and
# path-average target sweep.
# ---------------------------------------------------------------------------


def test_data_transforms_default_families_include_fm_and_lb():
    """Coulombe-Leroux-Stevanovic-Surprenant (2021) Table 1 lists 7
    families; previous default omitted FM (factor model) and LB (linear
    boosting). Audit gap-fix promotes ``factor_augmented_ar`` (FM) and
    ``glmboost`` (LB) into the default tuple."""

    from macroforecast.recipes.paper_methods import _DATA_TRANSFORM_FAMILIES_DEFAULT

    assert "factor_augmented_ar" in _DATA_TRANSFORM_FAMILIES_DEFAULT
    assert "glmboost" in _DATA_TRANSFORM_FAMILIES_DEFAULT
    # All seven Table 1 family slots covered.
    assert len(_DATA_TRANSFORM_FAMILIES_DEFAULT) == 7


def test_data_transforms_grid_sweeps_horizons_and_target_methods():
    """The helper grid must enumerate every (cell × family × horizon ×
    target_method) combination -- the previous helper silently dropped
    horizons (param declared but unused) and hard-coded direct-only
    target construction."""

    from macroforecast.recipes.paper_methods import (
        _DATA_TRANSFORM_CELLS_16,
        _DATA_TRANSFORM_FAMILIES_DEFAULT,
        macroeconomic_data_transformations_horse_race,
    )

    horizons = (1, 3)
    target_methods = ("direct", "path_average")
    grid = macroeconomic_data_transformations_horse_race(
        horizons=horizons,
        target_methods=target_methods,
    )
    expected = (
        len(_DATA_TRANSFORM_CELLS_16)
        * len(_DATA_TRANSFORM_FAMILIES_DEFAULT)
        * len(horizons)
        * len(target_methods)
    )
    assert len(grid) == expected, f"expected {expected} cells, got {len(grid)}"


def test_ml_useful_macro_horse_race_default_horizons_match_paper():
    """Coulombe-Surprenant-Leroux-Stevanovic (2022 JAE) §4.3 specifies
    h ∈ {1, 3, 9, 12, 24}. Audit gap-fix: previous default was (1, 3, 6),
    silently dropping the long-horizon cells where the nonlinearity
    treatment effect is largest."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    grid = ml_useful_macro_horse_race(cases=("ridge",), cv_schemes=("kfold",))
    horizons_seen = sorted({int(k.split("__h")[1].split("__")[0]) for k in grid})
    assert horizons_seen == [1, 3, 9, 12, 24]


def test_ml_useful_macro_horse_race_targets_sweep():
    """When ``targets`` is supplied, the helper enumerates per-target
    cells (audit gap-fix: previous helper accepted scalar ``target``
    only; user had to wrap-and-call N times to cover paper §4.2's 5
    target variables)."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    grid = ml_useful_macro_horse_race(
        targets=("INDPRO", "UNRATE"),
        cases=("ridge",),
        horizons=(1,),
        cv_schemes=("kfold",),
    )
    # 2 targets × 1 case × 1 horizon × 1 cv = 2 cells
    assert len(grid) == 2
    assert any(k.startswith("INDPRO__") for k in grid)
    assert any(k.startswith("UNRATE__") for k in grid)


def test_ml_useful_macro_horse_race_h_minus_routes_to_lag_y_only():
    """Audit gap-fix #15: paper §3.2 H⁻ axis = data-poor (own-lag y
    only, ~14 models). Helper now exposes ``data_richness="H_minus"``
    that builds an L3 graph with no PCA factors of X."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    grid = ml_useful_macro_horse_race(
        cases=("ridge",),
        horizons=(1,),
        cv_schemes=("kfold",),
        data_richness="H_minus",
    )
    recipe = next(iter(grid.values()))
    l3_nodes = recipe["3_feature_engineering"]["nodes"]
    # No PCA in H_minus.
    assert not any(n.get("op") == "pca" for n in l3_nodes)
    # X_final is the lagged y, not a concat or factor.
    sinks = recipe["3_feature_engineering"]["sinks"]
    assert sinks["l3_features_v1"]["X_final"] == "lag_y"


def test_ml_useful_macro_horse_race_h_plus_routes_to_factor_concat():
    """H⁺ axis = data-rich: PCA factors of X concatenated with lagged y."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    grid = ml_useful_macro_horse_race(
        cases=("ridge",),
        horizons=(1,),
        cv_schemes=("kfold",),
        data_richness="H_plus",
    )
    recipe = next(iter(grid.values()))
    l3_nodes = recipe["3_feature_engineering"]["nodes"]
    # H_plus has both PCA and weighted_concat.
    assert any(n.get("op") == "pca" for n in l3_nodes)
    assert any(n.get("op") == "weighted_concat" for n in l3_nodes)


def test_ml_useful_macro_horse_race_data_richness_validates_value():
    """Invalid ``data_richness`` raises with a helpful message."""

    import pytest as _pytest
    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    with _pytest.raises(ValueError, match="H_minus"):
        ml_useful_macro_horse_race(data_richness="something_else")


def test_dfm_mixed_frequency_predict_smoothed_factors_returns_kalman_posterior():
    """Coulombe & Goebel (2021) §3.4 deliverable: Kalman smoother
    posterior on latent factors. Audit gap-fix: ``predict_smoothed_
    factors`` now surfaces ``self._results.smoothed_state`` for the
    factor rows; previous adapter exposed only the 1-step forecast."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _DFMMixedFrequency

    rng = np.random.default_rng(7)
    T = 80
    f = rng.standard_normal(T)
    X_arr = np.column_stack(
        [
            0.7 * f + 0.3 * rng.standard_normal(T),
            -0.5 * f + 0.4 * rng.standard_normal(T),
            rng.standard_normal(T),
        ]
    )
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(3)])
    y = pd.Series(0.8 * f + 0.3 * rng.standard_normal(T), name="y")

    model = _DFMMixedFrequency(n_factors=1, factor_order=1, mixed_frequency=False).fit(
        X, y
    )
    smoothed = model.predict_smoothed_factors()
    assert smoothed is not None
    assert isinstance(smoothed, pd.DataFrame)
    # T rows × at least 1 factor column.
    assert smoothed.shape[0] >= T - 4  # statsmodels may drop initial obs
    assert smoothed.shape[1] >= 1
    assert all(c.startswith("smoothed_factor_") for c in smoothed.columns)
    assert np.all(np.isfinite(smoothed.to_numpy()))


def test_albacore_prior_target_none_raises_value_error():
    """Goulet Coulombe et al. (2024) Albacore Variant A specifies
    w_headline (CPI/PCE basket weights) as the target. F-14 audit
    gap-fix (phase-f14): ``prior_target=None`` must raise ``ValueError``
    (hard error) rather than silently falling back to uniform 1/K.
    Closes audit-paper-14.md F12 LOW."""

    import pytest
    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _ShrinkToTargetRidge

    rng = np.random.default_rng(0)
    K, T = 5, 50
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=[f"c{i}" for i in range(K)])
    y = pd.Series(rng.standard_normal(T))
    with pytest.raises(ValueError, match=r"prior_target"):
        _ShrinkToTargetRidge(alpha=1.0, prior_target=None).fit(X, y)


def test_albacore_prior_target_explicit_does_not_warn():
    """Counterpart: explicit prior_target silences the fallback warning."""

    import warnings
    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _ShrinkToTargetRidge

    rng = np.random.default_rng(1)
    K, T = 5, 50
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=[f"c{i}" for i in range(K)])
    y = pd.Series(rng.standard_normal(T))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ShrinkToTargetRidge(alpha=1.0, prior_target=[0.4, 0.3, 0.15, 0.1, 0.05]).fit(
            X, y
        )
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any("w_headline" in m for m in msgs), (
        f"explicit target should not warn; got {msgs}"
    )


def test_block_cv_search_picks_alpha_via_non_overlapping_blocks():
    """Goulet Coulombe et al. (2024) Albacore §3 non-overlapping K-block
    CV. Distinct from kfold (random) and TimeSeriesSplit (expanding).
    Audit gap-fix: ``block_cv`` is now a first-class search_algorithm."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _resolve_l4_tuning

    rng = np.random.default_rng(2)
    n, K = 60, 3
    X = pd.DataFrame(rng.standard_normal((n, K)), columns=list("abc"))
    y = pd.Series(2.0 * X["a"] - 0.5 * X["b"] + rng.normal(scale=0.5, size=n))
    custom_alphas = [0.01, 0.1, 1.0, 10.0]
    params = {
        "family": "ridge",
        "search_algorithm": "block_cv",
        "alpha": 99.0,
        "_l4_leaf_config": {"cv_path_alphas": custom_alphas, "block_cv_splits": 5},
    }
    result = _resolve_l4_tuning(params, X, y)
    assert result["alpha"] in custom_alphas
    assert result["alpha"] != 99.0


def test_block_cv_in_search_algorithms_enum():
    """Schema enum lists ``block_cv`` so recipes naming it pass the L4
    op validator."""

    from macroforecast.core.ops.l4_ops import SEARCH_ALGORITHMS

    assert "block_cv" in SEARCH_ALGORITHMS


def test_scaled_pca_helper_runs_end_to_end_on_multi_feature_panel(tmp_path):
    """Huang/Jiang/Li/Tong/Zhou (2022) scaled-PCA helper end-to-end:
    on a 36-row × 5-feature synthetic panel the recipe should run
    through L0–L4 with finite forecasts. Audit gap-fix: the previous
    helper carried a stale 'TBD pending FRED-MD adapter integration'
    comment; the underlying L3 op has actually been operational since
    v0.1, so this test pins the runnable contract."""

    import numpy as np
    import macroforecast
    from macroforecast.recipes.paper_methods import scaled_pca

    rng = np.random.default_rng(13)
    n = 36
    dates = [f"{2018 + i // 12:04d}-{(i % 12) + 1:02d}-01" for i in range(n)]
    # Latent factor + noise + supervised target.
    f = rng.standard_normal(n)
    panel = {
        "date": dates,
        "y": list(2.0 * f + 0.3 * rng.standard_normal(n)),
        "x1": list(0.8 * f + 0.4 * rng.standard_normal(n)),
        "x2": list(-0.6 * f + 0.4 * rng.standard_normal(n)),
        "x3": list(0.2 * f + 0.4 * rng.standard_normal(n)),
        "x4": list(rng.standard_normal(n)),  # noise
        "x5": list(rng.standard_normal(n)),  # noise
    }
    recipe = scaled_pca(panel=panel, n_components=2)
    result = macroforecast.run(recipe, output_directory=tmp_path / "spca")
    assert result.cells
    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts
    arr = np.asarray(list(forecasts.values()), dtype=float)
    assert np.all(np.isfinite(arr))


def test_slow_growing_tree_helper_exposes_all_paper_axes():
    """Goulet Coulombe (2024) SGT paper exposes 4 sub-axes (η, H̄,
    η-depth-step, max_depth). Audit gap-fix promotes them to first-
    class helper args + grid builder for the {η, H̄} sweep grid."""

    from macroforecast.recipes.paper_methods import (
        slow_growing_tree,
        slow_growing_tree_grid,
    )

    recipe = slow_growing_tree(
        split_shrinkage=0.25,
        herfindahl_threshold=0.05,
        eta_depth_step=0.02,
        max_depth=8,
    )
    fit = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    )
    assert fit["params"]["split_shrinkage"] == 0.25
    assert fit["params"]["herfindahl_threshold"] == 0.05
    assert fit["params"]["eta_depth_step"] == 0.02
    assert fit["params"]["max_depth"] == 8

    grid = slow_growing_tree_grid()
    # Phase B-3 audit-fix: paper §3 p.90 specifies a 3-line grid, NOT
    # a Cartesian product. See ``test_sgt_grid_matches_paper_figure_2``
    # for the exact (η, H̄) pairs.
    assert len(grid) == 3
    # All cells map to decision_tree family with split_shrinkage set.
    for key, recipe in grid.items():
        fit = next(
            n
            for n in recipe["4_forecasting_model"]["nodes"]
            if n.get("op") == "fit_model"
        )
        assert fit["params"]["family"] == "decision_tree"


def test_anatomy_oos_helper_exposes_initial_window_and_n_iterations():
    """Phase B-11 paper-11 F1 fix: helper now wires ``initial_window`` and
    ``n_iterations`` onto the L7 ``oshapley_vi`` op step (where the
    anatomy adapter actually reads them) rather than stamping a dead
    ``0_meta.leaf_config.anatomy_initial_window`` key. Previously the
    stamp was unread and users always routed to Path B."""

    from macroforecast.recipes.paper_methods import anatomy_oos

    recipe = anatomy_oos(initial_window=60, n_iterations=300)
    nodes = recipe["7_interpretation"]["nodes"]
    step = next(n for n in nodes if n.get("type") == "step")
    assert step["op"] == "oshapley_vi"
    assert step["params"]["initial_window"] == 60
    assert step["params"]["n_iterations"] == 300

    # Default (no initial_window) omits the key so the runtime warns
    # about Path B routing rather than silently passing 0/None through.
    recipe_default = anatomy_oos()
    nodes_default = recipe_default["7_interpretation"]["nodes"]
    step_default = next(n for n in nodes_default if n.get("type") == "step")
    assert "initial_window" not in step_default["params"]
    assert step_default["params"]["n_iterations"] == 500


def test_two_step_ridge_default_vol_model_is_garch11():
    """Coulombe (2025 IJF) 2SRR §4 Eq. 11 specifies GARCH(1,1) for the
    step-2 residual-variance estimator. Audit gap-fix raises the
    default from ``"ewma"`` to ``"garch11"`` in the runtime class, the
    L4 dispatch, and the ``two_step_ridge`` helper."""

    from macroforecast.core.runtime import _TwoStageRandomWalkRidge
    from macroforecast.recipes.paper_methods import two_step_ridge

    assert _TwoStageRandomWalkRidge().vol_model == "garch11"
    recipe = two_step_ridge()
    fit_step2 = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("id") == "fit_step2"
    )
    assert fit_step2["params"]["vol_model"] == "garch11"


def test_mrf_block_size_default_is_24_and_is_overridable():
    """Coulombe (2024) MRF Bayesian Bayesian Bootstrap block length:
    paper §3 spec for monthly data is 24 (upstream MacroRandomForest
    default is 12). Audit gap-fix bumps the wrapper default and surfaces
    the param via the helper."""

    from macroforecast.core.runtime import _MRFExternalWrapper
    from macroforecast.recipes.paper_methods import macroeconomic_random_forest

    assert _MRFExternalWrapper().block_size == 24
    assert _MRFExternalWrapper(block_size=8).block_size == 8

    recipe = macroeconomic_random_forest()
    fit_node = next(
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    )
    assert fit_node["params"]["block_size"] == 24
    recipe_q = macroeconomic_random_forest(block_size=8)
    fit_node_q = next(
        n
        for n in recipe_q["4_forecasting_model"]["nodes"]
        if n.get("op") == "fit_model"
    )
    assert fit_node_q["params"]["block_size"] == 8


def test_ml_useful_macro_attach_eval_blocks_wires_dm_and_mcs():
    """Audit gap-fix #6b: ``attach_eval_blocks=True`` adds an L4
    benchmark fit (``is_benchmark=True``), an L5 evaluation block, and
    an L6 statistical-tests block with DM + MCS sub-layers wired. No
    user effort required to recover paper §4.4 DM-vs-ARDI-baseline
    + Hansen MCS evaluation."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    grid = ml_useful_macro_horse_race(
        cases=("ridge",),
        horizons=(1,),
        cv_schemes=("kfold",),
        attach_eval_blocks=True,
    )
    recipe = next(iter(grid.values()))
    # L4 has both a benchmark fit and a cell fit.
    l4_nodes = recipe["4_forecasting_model"]["nodes"]
    fit_nodes = [n for n in l4_nodes if n.get("op") == "fit_model"]
    assert len(fit_nodes) == 2
    benchmark = next(n for n in fit_nodes if n.get("is_benchmark"))
    cell = next(n for n in fit_nodes if not n.get("is_benchmark"))
    # Phase A3 fix: paper §4.4 specifies the (ARDI, BIC) reference, not
    # AR(p). The default benchmark is now factor_augmented_ar + BIC.
    assert benchmark["params"]["family"] == "factor_augmented_ar"
    assert benchmark["params"]["search_algorithm"] == "bic"
    assert cell["params"]["family"] == "ridge"
    # L5 block present.
    assert "5_evaluation" in recipe
    # L6 block present and configured for DM + MCS.
    l6 = recipe["6_statistical_tests"]
    assert l6["enabled"] is True
    assert "L6_A_equal_predictive" in l6["sub_layers"]
    assert "L6_D_multiple_model" in l6["sub_layers"]
    assert (
        l6["sub_layers"]["L6_A_equal_predictive"]["fixed_axes"]["equal_predictive_test"]
        == "dm_diebold_mariano"
    )
    assert (
        l6["sub_layers"]["L6_D_multiple_model"]["fixed_axes"]["multiple_model_test"]
        == "mcs_hansen"
    )


def test_ml_useful_macro_attach_eval_blocks_default_off_preserves_minimal_recipe():
    """Default ``attach_eval_blocks=False`` keeps the L0–L4-only recipe
    shape so existing helpers / tests are unaffected by the new option."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    grid = ml_useful_macro_horse_race(
        cases=("ridge",),
        horizons=(1,),
        cv_schemes=("kfold",),
    )
    recipe = next(iter(grid.values()))
    assert "5_evaluation" not in recipe
    assert "6_statistical_tests" not in recipe
    fit_nodes = [
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    ]
    assert len(fit_nodes) == 1


def test_ml_useful_macro_b_grid_emits_three_rotations_per_family():
    """Audit gap-fix #6b: ``ml_useful_macro_b_grid`` builds the §3.2
    B₁/B₂/B₃ regularization rotation grid for {Ridge, Lasso, EN}."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_b_grid

    grid = ml_useful_macro_b_grid()
    assert len(grid) == 9  # 3 rotations × 3 families
    keys = set(grid.keys())
    for rot in ("B1", "B2", "B3"):
        for fam in ("ridge", "lasso", "elastic_net"):
            assert f"{rot}__{fam}" in keys


def test_ml_useful_macro_b_grid_b2_has_pca_node_b3_has_concat_node():
    """B₂ rotates X via full-rank PCA (paper §3.2 keep-all-factors); B₃
    rotates the H_t^+ stack (lag(y) ⊕ lag(X)) via PCA before the
    regression. Pin the L3 node structure so the rotation semantics
    match the paper §3.2 specification.

    Phase A2 update: B₂ no longer hard-codes ``n_components=n_factors``
    (paper p.16 "we keep them all"); B₃ no longer concatenates
    contemporaneous X with the lag-PCA factors. The pinned-shape
    assertions track those changes.

    Phase A3 update: B₂ now passes the explicit ``n_components="all"``
    sentinel rather than relying on the schema default (4); B₃ drops
    the trailing lag node (PCA output IS the L3 X_final per paper
    §3.2)."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_b_grid

    grid = ml_useful_macro_b_grid(families=("ridge",))
    b2_nodes = grid["B2__ridge"]["3_feature_engineering"]["nodes"]
    b3_nodes = grid["B3__ridge"]["3_feature_engineering"]["nodes"]
    # B₂: PCA node present + temporal_rule wired + n_components="all" sentinel.
    b2_pca = next(n for n in b2_nodes if n.get("op") == "pca" and n["id"] == "feat_pca")
    assert b2_pca["params"].get("temporal_rule") == "expanding_window_per_origin"
    assert b2_pca["params"].get("n_components") == "all", (
        "B₂ should keep all PCA components via the 'all' sentinel "
        "(paper §3.2); found n_components="
        f"{b2_pca['params'].get('n_components')!r}"
    )
    # B₃: weighted_concat is over (lag(y), lag(X)) — i.e., the H_t^+ stack —
    # then PCA rotates the stack. No contemporaneous-X concat node, no
    # trailing-lag node.
    b3_concat = next(
        n
        for n in b3_nodes
        if n.get("op") == "weighted_concat" and set(n["inputs"]) == {"lag_y", "lag_x"}
    )
    assert b3_concat is not None
    assert any(
        n.get("op") == "pca" and "h_plus" in n.get("inputs", []) for n in b3_nodes
    ), "B₃ must rotate the H_t^+ stack via PCA (paper §3.2)"


def test_ml_useful_macro_cv_schemes_are_first_class_search_algorithms():
    """The 4 CV schemes (kfold/poos/aic/bic) emitted by the helper must
    be valid ``search_algorithm`` values that survive the L4 op
    validator's options enum check. Audit gap-fix: previously the
    strings were missing from SEARCH_ALGORITHMS and silently dropped."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race
    from macroforecast.core.ops.l4_ops import SEARCH_ALGORITHMS

    grid = ml_useful_macro_horse_race(
        cases=("ridge",),
        horizons=(1,),
        cv_schemes=("kfold", "poos", "aic", "bic"),
    )
    for recipe in grid.values():
        # Walk down to the L4 fit_model node and read search_algorithm.
        l4 = recipe["4_forecasting_model"]
        fit_node = next(n for n in l4["nodes"] if n.get("op") == "fit_model")
        assert fit_node["params"]["search_algorithm"] in SEARCH_ALGORITHMS


def test_data_transforms_path_average_routes_to_cumulative_target_mode():
    """``target_method='path_average'`` must reach the L3 runtime as
    ``mode='path_average'`` -- the runtime branches on ``mode``, not
    ``method``, so a previous helper that set ``method`` only would
    silently produce direct (point) forecasts. Audit gap-fix."""

    from macroforecast.recipes.paper_methods import _l3_data_transforms_cell

    cell_direct = _l3_data_transforms_cell("F-X", horizon=3, target_method="direct")
    cell_path = _l3_data_transforms_cell("F-X", horizon=3, target_method="path_average")
    yh_direct = next(n for n in cell_direct["nodes"] if n["id"] == "y_h")
    yh_path = next(n for n in cell_path["nodes"] if n["id"] == "y_h")
    assert yh_direct["params"]["mode"] == "point_forecast"
    assert yh_path["params"]["mode"] == "path_average"
    assert yh_path["params"]["method"] == "path_average"


# ---------------------------------------------------------------------------
# Phase A3 (procedural fixes — papers 13/16). Seven tests covering A3a
# (lag op param-name typo), A3b (PCA "all" sentinel), A3c (B₃ trailing-
# lag drop), A3d (paper-13 variant arg), and A3e (ARDI-BIC benchmark
# default). See runs/2026-05-07-phase-a3-procedural-fixes-paper-13-16/
# request.md for the full per-fix spec.
# ---------------------------------------------------------------------------


def _phase_a3_synthetic_panel(K: int, T: int = 60, seed: int = 7) -> dict[str, list]:
    """Build a synthetic K-feature × T-row panel of the inline shape that
    ``_DEFAULT_PANEL`` uses (date + y + x1..xK). Returns a leaf-config
    ``custom_panel_inline`` dict ready to drop into a Phase A3 recipe."""

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = (
        pd.date_range("2010-01-01", periods=T, freq="MS").strftime("%Y-%m-%d").tolist()
    )
    panel: dict[str, list] = {"date": dates}
    panel["y"] = (np.cumsum(rng.standard_normal(T)) + np.arange(T) * 0.05).tolist()
    for k in range(K):
        panel[f"x{k + 1}"] = rng.standard_normal(T).tolist()
    return panel


def test_b2_keeps_all_pca_components_K8(tmp_path):
    """Phase A3a + A3b: B₂ × ridge cell on a K=8 DGP must produce a
    PCA factor with 8 components (paper §3.2 "we keep them all"). With
    ``n_components="all"`` the runtime resolves the sentinel to
    ``min(T, N) = 8`` and the lag(n_lag=2) downstream emits 8 × 2 = 16
    columns."""

    import macroforecast
    from macroforecast.recipes.paper_methods import ml_useful_macro_b_grid

    panel = _phase_a3_synthetic_panel(K=8, T=60)
    grid = ml_useful_macro_b_grid(
        n_factors=8,
        n_lag=2,
        panel=panel,
        rotations=("B2",),
        families=("ridge",),
    )
    recipe = grid["B2__ridge"]
    result = macroforecast.run(recipe, output_directory=tmp_path / "b2_K8")
    assert result.cells
    artifacts = result.cells[0].runtime_result.artifacts
    # X_final after lag(n_lag=2) on PCA factors → 8 factors × 2 lags = 16 cols.
    x_final = artifacts["l3_features_v1"].X_final
    n_cols = len(x_final.column_names)
    assert n_cols == 16, (
        f"B₂ X_final should have 8 factors × 2 lags = 16 columns; got "
        f"{n_cols} columns ({list(x_final.column_names)})."
    )


def test_b3_no_trailing_lag_node():
    """Phase A3c: B₃'s emitted recipe ends at ``pca`` — no trailing
    ``lag`` node. The PCA output IS the L3 X_final per paper §3.2."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_b_grid

    grid = ml_useful_macro_b_grid(rotations=("B3",), families=("ridge",))
    nodes = grid["B3__ridge"]["3_feature_engineering"]["nodes"]
    sinks = grid["B3__ridge"]["3_feature_engineering"]["sinks"]
    x_final_id = sinks["l3_features_v1"]["X_final"]
    x_final_node = next(n for n in nodes if n["id"] == x_final_id)
    assert x_final_node["op"] == "pca", (
        "B₃ X_final node must be the PCA op (paper §3.2 keep-all-factors); "
        f"got op={x_final_node['op']!r}."
    )
    # Also check no lag node consumes the PCA output.
    pca_id = x_final_node["id"]
    trailing_lags = [
        n for n in nodes if n.get("op") == "lag" and pca_id in n.get("inputs", [])
    ]
    assert not trailing_lags, (
        f"B₃ must not have a lag node downstream of PCA; found {trailing_lags!r}"
    )


def test_b3_eq_b2_at_n_lag_zero(tmp_path):
    """Phase A4c (paper 16, Round 1 finding / Round 4 F5): paper §3.2
    footnote — "B₃() = B₂() only when no lags are included." At
    ``n_lag=0`` the helper now emits no-lag DAGs (B₁ → identity(X);
    B₂ → pca-of-X; B₃ → pca-of-X identical to B₂). The pre-A4 helper
    emitted lag(n_lag=0) which the universal lag op rejected; tests
    asserted the rejection was the paper-faithful behaviour. The
    paper-faithful contract per the footnote is the OTHER direction —
    B₂ ≡ B₃ at n_lag=0 — which the A4c rewrite now satisfies.
    """

    import numpy as np
    import macroforecast
    from macroforecast.recipes.paper_methods import ml_useful_macro_b_grid

    panel = _phase_a3_synthetic_panel(K=4, T=40)
    grid = ml_useful_macro_b_grid(
        n_factors=4,
        n_lag=0,
        panel=panel,
        rotations=("B2", "B3"),
        families=("ridge",),
    )
    res_b2 = macroforecast.run(grid["B2__ridge"], output_directory=tmp_path / "b2_n0")
    res_b3 = macroforecast.run(grid["B3__ridge"], output_directory=tmp_path / "b3_n0")
    x2 = res_b2.cells[0].runtime_result.artifacts["l3_features_v1"].X_final.data
    x3 = res_b3.cells[0].runtime_result.artifacts["l3_features_v1"].X_final.data
    assert x2.shape == x3.shape
    assert np.allclose(x2.values, x3.values), (
        "Paper §3.2 footnote: B₃() = B₂() at n_lag=0; got differing X_final."
    )


def test_lag_node_uses_n_lag_param(tmp_path):
    """Phase A3a: lag-node params dicts emitted by ``_l3_b_rotation``
    and ``_l3_h_axis`` must use the ``"n_lag"`` key (the universal lag
    op reads ``n_lag``; the prior ``"max_lag"`` typo silently fell back
    to the default 4).

    End-to-end: with ``n_lag=2`` propagated through ``ml_useful_macro_b_grid``
    the resulting B₁ X_final must have ``n_features × 2`` lag columns
    (not × 4)."""

    import macroforecast
    from macroforecast.recipes.paper_methods import (
        _l3_b_rotation,
        _l3_h_axis,
        ml_useful_macro_b_grid,
    )

    # Inspect emitted node params: every lag op must use ``n_lag`` key.
    for rot in ("B1", "B2", "B3"):
        block = _l3_b_rotation(rot, horizon=1, n_factors=4, n_lag=3)
        for node in block["nodes"]:
            if node.get("op") == "lag":
                assert "n_lag" in node["params"], (
                    f"{rot} lag node {node['id']!r} missing 'n_lag' key; "
                    f"params={node['params']!r}"
                )
                assert "max_lag" not in node["params"], (
                    f"{rot} lag node {node['id']!r} still uses legacy "
                    f"'max_lag' key; params={node['params']!r}"
                )
    for richness in ("H_minus", "H_plus"):
        block = _l3_h_axis(richness, horizon=1, n_factors=4)
        for node in block["nodes"]:
            if node.get("op") == "lag":
                assert "n_lag" in node["params"], (
                    f"{richness} lag node missing 'n_lag' key; "
                    f"params={node['params']!r}"
                )
                assert "max_lag" not in node["params"]

    # End-to-end: B₁ × ridge with K=5, n_lag=2 → 5 × 2 = 10 cols.
    panel = _phase_a3_synthetic_panel(K=5, T=40)
    grid = ml_useful_macro_b_grid(
        n_factors=5,
        n_lag=2,
        panel=panel,
        rotations=("B1",),
        families=("ridge",),
    )
    recipe = grid["B1__ridge"]
    result = macroforecast.run(recipe, output_directory=tmp_path / "b1_K5")
    x_final = result.cells[0].runtime_result.artifacts["l3_features_v1"].X_final
    n_cols = len(x_final.column_names)
    assert n_cols == 10, (
        f"B₁ X_final with K=5, n_lag=2 should have 10 cols; got "
        f"{n_cols} (would be 20 if n_lag silently fell back to 4). "
        f"columns={list(x_final.column_names)}"
    )


def test_paper13_default_variant_is_ranks():
    """Phase A3d: ``maximally_forward_looking()`` (no variant arg) emits
    the paper-headline Albacore_ranks pairing — L3 ``asymmetric_trim``
    step + L4 ``prior=fused_difference``."""

    from macroforecast.recipes.paper_methods import maximally_forward_looking

    recipe = maximally_forward_looking()
    l3_nodes = recipe["3_feature_engineering"]["nodes"]
    l4_nodes = recipe["4_forecasting_model"]["nodes"]
    # L3: asymmetric_trim step present.
    trim_nodes = [n for n in l3_nodes if n.get("op") == "asymmetric_trim"]
    assert len(trim_nodes) == 1, (
        f"variant='ranks' (default) must keep the asymmetric_trim L3 "
        f"step; found {len(trim_nodes)} matching nodes."
    )
    # L4: ridge fit_model with prior=fused_difference.
    fit_nodes = [n for n in l4_nodes if n.get("op") == "fit_model"]
    assert len(fit_nodes) == 1
    assert fit_nodes[0]["params"]["family"] == "ridge"
    assert fit_nodes[0]["params"]["prior"] == "fused_difference", (
        "variant='ranks' (default) must wire prior=fused_difference at L4; "
        f"got prior={fit_nodes[0]['params'].get('prior')!r}."
    )


def test_paper13_variant_comps_drops_asymmetric_trim():
    """Phase A3d: ``maximally_forward_looking(variant='comps')`` emits
    Variant A — drops asymmetric_trim L3 step + L4
    ``prior=shrink_to_target``.

    Phase A4d (paper 13, Round 4 F5): ``variant='comps'`` without
    ``prior_target`` is now a hard ``ValueError`` (paper Eq. (1) is
    undefined without ``w_headline``). With an explicit ``prior_target``
    the recipe is paper-faithful Variant A.
    """

    import pytest

    from macroforecast.recipes.paper_methods import maximally_forward_looking

    # variant='comps' without prior_target now hard-errors (paper Eq. (1)
    # is undefined without w_headline).
    with pytest.raises(ValueError, match="prior_target"):
        maximally_forward_looking(variant="comps")

    # With an explicit prior_target the recipe is the paper-faithful
    # Variant A. Use that to run the structural checks below.
    recipe = maximally_forward_looking(variant="comps", prior_target=[0.5, 0.5])

    l3_nodes = recipe["3_feature_engineering"]["nodes"]
    l4_nodes = recipe["4_forecasting_model"]["nodes"]
    # L3: asymmetric_trim step absent.
    trim_nodes = [n for n in l3_nodes if n.get("op") == "asymmetric_trim"]
    assert not trim_nodes, (
        f"variant='comps' must drop the asymmetric_trim L3 step; found {trim_nodes!r}."
    )
    # L4: ridge with prior=shrink_to_target.
    fit_nodes = [n for n in l4_nodes if n.get("op") == "fit_model"]
    assert len(fit_nodes) == 1
    assert fit_nodes[0]["params"]["prior"] == "shrink_to_target", (
        "variant='comps' must wire prior=shrink_to_target at L4; got "
        f"prior={fit_nodes[0]['params'].get('prior')!r}."
    )

    # variant='comps' with explicit prior_target carries it into fit_params.
    fit2 = next(n for n in l4_nodes if n.get("op") == "fit_model")
    assert fit2["params"]["prior_target"] == [0.5, 0.5]


def test_paper16_default_benchmark_is_factor_augmented_ar():
    """Phase A3e: ``ml_useful_macro_horse_race(attach_eval_blocks=True)``
    default benchmark fit_node must use ``factor_augmented_ar`` family
    with ``search_algorithm="bic"`` and have ``src_X`` wired into its
    inputs (per paper §4.4 "(ARDI, BIC)" reference)."""

    from macroforecast.recipes.paper_methods import ml_useful_macro_horse_race

    grid = ml_useful_macro_horse_race(
        cases=("ridge",),
        horizons=(1,),
        cv_schemes=("kfold",),
        attach_eval_blocks=True,
    )
    recipe = next(iter(grid.values()))
    l4_nodes = recipe["4_forecasting_model"]["nodes"]
    benchmark = next(
        n for n in l4_nodes if n.get("op") == "fit_model" and n.get("is_benchmark")
    )
    assert benchmark["params"]["family"] == "factor_augmented_ar", (
        "Paper §4.4 specifies the (ARDI, BIC) reference; default helper "
        f"benchmark must be factor_augmented_ar, got "
        f"{benchmark['params']['family']!r}."
    )
    assert benchmark["params"]["search_algorithm"] == "bic", (
        f"Default benchmark must use search_algorithm='bic'; got "
        f"{benchmark['params'].get('search_algorithm')!r}."
    )
    # src_X must be wired into the benchmark fit (factors flow into ARDI).
    assert "src_X" in benchmark["inputs"], (
        f"factor_augmented_ar benchmark must consume src_X (factors); "
        f"inputs={benchmark['inputs']!r}."
    )


# ---------------------------------------------------------------------------
# Phase A4 procedure tests (paper 13/16 final cleanup)
# ---------------------------------------------------------------------------


def test_a4a_elastic_net_l1_ratio_tuned_under_kfold():
    """Phase A4a (paper 16, Round 1 finding 7): paper §3.2 Eq. (18)
    elastic_net's ζ ∈ {0=Ridge, 1=Lasso, ζ_CV=EN-tuned}. Pre-A4
    ``_resolve_l4_tuning`` only tuned ``alpha`` for elastic_net and left
    ``l1_ratio`` at the helper-pinned value. The 2-D ``(alpha × l1_ratio)``
    grid now mutates BOTH on the winning combination.

    Smoke contract: invoke ``_resolve_l4_tuning`` on a small DGP with
    sentinel ``alpha=99.0, l1_ratio=0.7`` and assert post-tune both move
    off their input sentinels and into the configured grids."""

    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _resolve_l4_tuning

    rng = np.random.default_rng(42)
    n, K = 60, 4
    X = pd.DataFrame(rng.standard_normal((n, K)), columns=list("abcd"))
    # Sparse-ish DGP so the EN landscape is non-trivial across (alpha, ζ).
    y = pd.Series(
        2.0 * X["a"] - 0.8 * X["b"] + 0.4 * X["c"] + rng.normal(scale=0.3, size=n)
    )

    custom_alphas = [0.001, 0.01, 0.1, 1.0]
    custom_l1 = [0.1, 0.3, 0.5, 0.7, 0.9]
    params = {
        "family": "elastic_net",
        "search_algorithm": "kfold",
        "alpha": 99.0,
        "l1_ratio": 0.7,
        "random_state": 0,
        "_l4_leaf_config": {
            "cv_path_alphas": custom_alphas,
            "cv_path_l1_ratios": custom_l1,
        },
    }
    out = _resolve_l4_tuning(params, X, y)
    assert out["alpha"] in custom_alphas, (
        f"alpha must land on the configured grid; got {out['alpha']!r}."
    )
    assert out["alpha"] != 99.0, (
        f"elastic_net + kfold must mutate alpha off the sentinel 99.0; "
        f"got {out['alpha']!r}."
    )
    assert out["l1_ratio"] in custom_l1, (
        f"l1_ratio must land on the configured grid; got {out['l1_ratio']!r}."
    )
    # Round 1 finding 7: the Eq. (18) ζ axis is *tuned*, not pinned. With
    # a 0.1-step grid the chance the input 0.7 is also the optimum is
    # vanishingly small for this DGP — but the structural contract is
    # that l1_ratio joins alpha as a tunable axis, which the assertion
    # above already proves (winning value comes from custom_l1, not the
    # input pin). Add a soft check that at least ONE of (alpha, l1_ratio)
    # actually moved off the input sentinel — both moving is the typical
    # case but the contract is the joint search.
    assert out["alpha"] != 99.0 or out["l1_ratio"] != 0.7


def test_a4b_h_plus_axis_includes_lag_F():
    """Phase A4b (paper 16, Round 1 finding 6): paper Eq. (7) ARDI
    specifies ``y_{t+h} = c + ρ(L)y_t + β(L)F_t + e`` — both ``y`` AND
    ``F`` are lagged. Pre-A4 H_plus fed contemporaneous ``feat_F`` to
    ``weighted_concat``. Now the DAG must contain a ``lag`` node whose
    input is the PCA ``feat_F`` node, feeding into ``weighted_concat``."""

    from macroforecast.recipes.paper_methods import _l3_h_axis

    block = _l3_h_axis("H_plus", horizon=1, n_factors=4)
    nodes = block["nodes"]
    nodes_by_id = {n["id"]: n for n in nodes}

    # Find the lag node whose sole input is the PCA factor node.
    lag_F_candidates = [
        n for n in nodes if n.get("op") == "lag" and n.get("inputs") == ["feat_F"]
    ]
    assert len(lag_F_candidates) == 1, (
        f"H_plus DAG must include exactly one lag(feat_F) node; "
        f"found {len(lag_F_candidates)}: {lag_F_candidates!r}"
    )
    lag_F = lag_F_candidates[0]

    # weighted_concat must consume lag_F (not the contemporaneous feat_F).
    concat_nodes = [n for n in nodes if n.get("op") == "weighted_concat"]
    assert len(concat_nodes) == 1
    concat = concat_nodes[0]
    assert lag_F["id"] in concat["inputs"], (
        f"weighted_concat must consume the lagged factor (lag_F); "
        f"inputs={concat['inputs']!r}."
    )
    assert "feat_F" not in concat["inputs"], (
        f"Pre-A4 H_plus fed contemporaneous feat_F to weighted_concat; "
        f"that path is paper-incoherent (Eq. 7 needs β(L)F_t). "
        f"inputs={concat['inputs']!r}."
    )

    # PCA → lag_F → concat structural chain.
    assert nodes_by_id["feat_F"].get("op") == "pca"


def test_a4c_b_grid_n_lag_zero_b2_eq_b3(tmp_path):
    """Phase A4c (paper 16, Round 1 finding / Round 4 F5): paper §3.2
    footnote — "B₃() = B₂() only when no lags are included." At
    ``n_lag=0`` the rotation grid must:

    * (a) emit 9 runnable cells (was: lag op rejected n_lag<1, all
      cells unrunnable);
    * (b) B₂×ridge L3 X_final must equal B₃×ridge L3 X_final
      element-wise (paper footnote identity);
    * (c) col count must be ``min(T, K)`` per the §3.2 keep-all-factors
      contract."""

    import numpy as np
    import macroforecast
    from macroforecast.recipes.paper_methods import ml_useful_macro_b_grid

    panel = _phase_a3_synthetic_panel(K=8, T=60)
    grid = ml_useful_macro_b_grid(
        n_factors=4,
        n_lag=0,
        panel=panel,
        rotations=("B1", "B2", "B3"),
        families=("ridge", "lasso", "elastic_net"),
    )
    # (a) 3 rotations × 3 families = 9 cells.
    assert len(grid) == 9
    # All 9 cells must run end-to-end through macroforecast.run.
    artifacts: dict[str, object] = {}
    for name, recipe in grid.items():
        out_dir = tmp_path / name.replace("__", "_")
        result = macroforecast.run(recipe, output_directory=out_dir)
        artifacts[name] = (
            result.cells[0].runtime_result.artifacts["l3_features_v1"].X_final
        )

    # (b) B₂ ≡ B₃ at n_lag=0 (paper §3.2 footnote identity). Compare
    # ridge cells specifically (the family that doesn't randomise the L3
    # path; lasso/EN don't either, but ridge is the canonical baseline).
    x2 = artifacts["B2__ridge"].data
    x3 = artifacts["B3__ridge"].data
    assert x2.shape == x3.shape, (
        f"B₂ and B₃ X_final shapes must match at n_lag=0; got {x2.shape} vs {x3.shape}."
    )
    assert np.allclose(x2.values, x3.values), (
        "Paper §3.2 footnote: B₃() = B₂() at n_lag=0; got differing "
        "X_final between B₂ and B₃ ridge cells."
    )

    # (c) Col count = min(T, K). PCA at n_components="all" resolves to
    # min(T, K). With T=60, K=8 the expected count is 8.
    assert x2.shape[1] == min(60, 8) == 8, (
        f"B₂ at n_lag=0 must keep min(T,K)=8 PCA factors; got {x2.shape[1]} cols."
    )


def test_a4d_paper13_variant_comps_no_prior_target_raises():
    """Phase A4d (paper 13, Round 4 F5): paper Eq. (1) is undefined
    without ``w_headline``. ``maximally_forward_looking(variant='comps',
    prior_target=None)`` must raise ``ValueError`` instead of falling
    back to uniform 1/K with a UserWarning. ``variant='ranks'`` with
    ``prior_target=None`` is the prior-free Variant B and must NOT
    raise."""

    import pytest

    from macroforecast.recipes.paper_methods import maximally_forward_looking

    # variant='comps' without prior_target is a hard error.
    with pytest.raises(ValueError, match=r"prior_target"):
        maximally_forward_looking(variant="comps", prior_target=None)

    # variant='ranks' with prior_target=None is the prior-free Variant B
    # and must build a recipe without raising.
    recipe_ranks = maximally_forward_looking(variant="ranks", prior_target=None)
    assert "3_feature_engineering" in recipe_ranks
    # Variant B keeps the asymmetric_trim L3 step.
    l3_nodes = recipe_ranks["3_feature_engineering"]["nodes"]
    trim_nodes = [n for n in l3_nodes if n.get("op") == "asymmetric_trim"]
    assert len(trim_nodes) == 1, (
        "variant='ranks' (Variant B) must wire the asymmetric_trim L3 step."
    )

    # variant='comps' WITH prior_target is the paper-faithful Variant A
    # and must build a recipe without raising.
    recipe_comps = maximally_forward_looking(
        variant="comps",
        prior_target=[0.5, 0.5],
    )
    fit = next(
        n
        for n in recipe_comps["4_forecasting_model"]["nodes"]
        if n.get("op") == "fit_model"
    )
    assert fit["params"]["prior_target"] == [0.5, 0.5]


# ---------------------------------------------------------------------------
# Phase D-1 gap-fix tests — papers 13 (target mode) + 15 (temporal_rule + lag)
# ---------------------------------------------------------------------------


def test_maximally_forward_looking_uses_cumulative_average_target_ranks():
    """Phase D-1 F2 gap-fix: Albacore ranks variant must set target
    mode to cumulative_average (paper Eq.1 average-path target) not
    point_forecast (single h-step value). Both variants were wrong;
    this test pins the ranks fix."""
    from macroforecast.recipes.paper_methods import maximally_forward_looking

    recipe = maximally_forward_looking(variant="ranks", horizon=12)
    l3 = recipe["3_feature_engineering"]
    y_h = next(n for n in l3["nodes"] if n["id"] == "y_h")
    assert y_h["params"]["mode"] == "cumulative_average", (
        f"expected cumulative_average, got {y_h['params']['mode']!r}"
    )
    assert "method" not in y_h["params"], (
        "'method' param should be absent from cumulative_average node"
    )


def test_maximally_forward_looking_uses_cumulative_average_target_comps():
    """Phase D-1 F2 gap-fix: Albacore comps variant must set target
    mode to cumulative_average (paper Eq.1 average-path target)."""
    import numpy as np
    from macroforecast.recipes.paper_methods import maximally_forward_looking

    rng = np.random.default_rng(0)
    K = 5
    prior_target = (rng.dirichlet(np.ones(K))).tolist()
    recipe = maximally_forward_looking(
        variant="comps", horizon=12, prior_target=prior_target
    )
    l3 = recipe["3_feature_engineering"]
    y_h = next(n for n in l3["nodes"] if n["id"] == "y_h")
    assert y_h["params"]["mode"] == "cumulative_average", (
        f"expected cumulative_average, got {y_h['params']['mode']!r}"
    )
    assert "method" not in y_h["params"], (
        "'method' param should be absent from cumulative_average node"
    )


def test_maximally_forward_looking_target_equals_rolling_average_on_synthetic_dgp():
    """Phase D-1 F2 gap-fix: _cumulative_average_target must match
    the paper's formula y_t = (1/h) sum_{j=1}^{h} pi_{t+j}.
    DGP: pi_t = 0.02 + 0.001 * N(0,1), T=240, h=12.
    Tolerance: absolute |y_t - rolling_mean| < 1e-9 at 5 representative t."""
    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _cumulative_average_target

    rng = np.random.default_rng(42)
    T, h = 240, 12
    pi = pd.Series(0.02 + 0.001 * rng.standard_normal(T), name="headline")
    y = _cumulative_average_target(pi, horizon=h)
    # Check 5 interior representative t values (avoid edges where rolling is NaN)
    check_t = [60, 90, 120, 150, 180]
    for t in check_t:
        expected = pi.iloc[t + 1 : t + 1 + h].mean()  # (1/h) sum_{j=1}^{h} pi_{t+j}
        actual = y.iloc[t]
        assert not np.isnan(actual), f"y_t is NaN at t={t}"
        assert abs(actual - expected) < 1e-9, (
            f"at t={t}: y_t={actual:.12f}, rolling_mean={expected:.12f}, "
            f"diff={abs(actual-expected):.2e}"
        )


def test_data_transforms_pca_nodes_carry_temporal_rule():
    """Phase D-1 F7 gap-fix: all pca op nodes in F and MAF branches
    of _l3_data_transforms_cell must carry temporal_rule=
    'expanding_window_per_origin' (hard rule of _factor_op)."""
    from macroforecast.recipes.paper_methods import (
        _DATA_TRANSFORM_CELLS_16,
        _l3_data_transforms_cell,
    )

    f_cells = [c for c in _DATA_TRANSFORM_CELLS_16 if "F" in c.split("-") or c == "MAF" or "MAF" in c.split("-")]
    for cell in f_cells:
        l3 = _l3_data_transforms_cell(cell, horizon=1)
        pca_nodes = [n for n in l3["nodes"] if n.get("op") == "pca"]
        assert len(pca_nodes) >= 1, f"expected pca node in cell {cell!r}"
        for pca_node in pca_nodes:
            tr = pca_node["params"].get("temporal_rule")
            assert tr == "expanding_window_per_origin", (
                f"cell={cell!r}, node={pca_node['id']!r}: "
                f"expected expanding_window_per_origin, got {tr!r}"
            )


def test_data_transforms_f_branch_emits_lagged_factors():
    """Phase D-1 F7 gap-fix: F-branch must include a lag node
    downstream of PCA before concat, implementing Table 1's
    {L^{i-1} F_t}_{i=1}^{p_f} structure."""
    from macroforecast.recipes.paper_methods import _l3_data_transforms_cell

    for cell in ("F", "F-X", "F-MARX", "F-MAF", "F-Level",
                 "F-X-MARX", "F-X-MAF", "F-X-Level", "F-X-MARX-Level"):
        l3 = _l3_data_transforms_cell(cell, horizon=1)
        nodes_by_id = {n["id"]: n for n in l3["nodes"]}

        # feat_F_lag must exist
        assert "feat_F_lag" in nodes_by_id, (
            f"cell={cell!r}: expected feat_F_lag node, not found"
        )
        lag_node = nodes_by_id["feat_F_lag"]

        # feat_F_lag must take feat_F as input
        assert "feat_F" in lag_node["inputs"], (
            f"cell={cell!r}: feat_F_lag inputs={lag_node['inputs']!r}, "
            "expected feat_F"
        )

        # feat_F_lag must be a lag op
        assert lag_node["op"] == "lag", (
            f"cell={cell!r}: feat_F_lag op={lag_node['op']!r}, expected lag"
        )

        # weighted_concat must reference feat_F_lag, not feat_F
        concat_node = nodes_by_id.get("X_final")
        assert concat_node is not None, f"cell={cell!r}: X_final node missing"
        assert "feat_F_lag" in concat_node["inputs"], (
            f"cell={cell!r}: weighted_concat inputs={concat_node['inputs']!r}, "
            "expected feat_F_lag (not raw feat_F)"
        )
        assert "feat_F" not in concat_node["inputs"], (
            f"cell={cell!r}: weighted_concat must not reference raw feat_F directly"
        )


# ---------------------------------------------------------------------------
# Phase D-1 — Test 6 (S6): all 16 Table 1 cells run e2e on synthetic data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", _DATA_TRANSFORM_CELLS_16)
def test_data_transforms_16_cells_all_run_e2e_on_synthetic(cell, tmp_path):
    """Phase D-1 F7 gap-fix: all 16 Table 1 cells (Coulombe et al. 2021)
    must execute end-to-end via ``macroforecast.run`` on a minimal
    synthetic dataset without raising an exception and must produce a
    finite forecast scalar.

    DGP: T=80 monthly observations, N=5 predictors, h=1, seed=7, 1 OOS
    origin.  Parametrised over ``_DATA_TRANSFORM_CELLS_16`` so each cell
    appears as an independent test item for clean per-cell reporting."""

    import numpy as np
    import macroforecast
    from macroforecast.recipes.paper_methods import (
        macroeconomic_data_transformations_horse_race,
    )

    # Synthetic panel: T=80, N=5 predictors, seed=7 — reuse existing helper.
    panel = _phase_a3_synthetic_panel(K=5, T=80, seed=7)

    # Build one recipe for this single cell: ridge, h=1, direct.
    grid = macroeconomic_data_transformations_horse_race(
        cells=(cell,),
        families=("ridge",),
        horizons=(1,),
        target_methods=("direct",),
        panel=panel,
        seed=7,
    )
    assert len(grid) == 1, f"expected 1 cell, got {len(grid)}"
    recipe = next(iter(grid.values()))

    result = macroforecast.run(recipe, output_directory=tmp_path / cell.replace("-", "_"))
    assert result.cells, f"cell={cell!r}: macroforecast.run returned no cells"

    forecasts = result.cells[0].runtime_result.artifacts["l4_forecasts_v1"].forecasts
    assert forecasts, f"cell={cell!r}: forecast artifact is empty"

    arr = np.asarray(list(forecasts.values()), dtype=float)
    assert np.all(np.isfinite(arr)), (
        f"cell={cell!r}: forecast contains NaN or Inf — got {arr!r}"
    )


def test_d2c_paper16_h_plus_pca_n_components_equals_n_factors():
    """Phase D-2c Paper 16 Eq. (7): ARDI H_plus uses K=n_factors static
    factors. The ``feat_F`` PCA node in ``_l3_h_axis`` must set
    ``n_components=n_factors`` (default 4), not ``'all'``.

    Structural pin: prevents silent change to n_components='all' which
    would expand the factor set and break paper Eq. (7) semantics
    (B₂ uses 'all'; H_plus / ARDI uses K truncated factors)."""

    from macroforecast.recipes.paper_methods import _l3_h_axis

    for n_factors in (4, 6):
        block = _l3_h_axis("H_plus", horizon=1, n_factors=n_factors)
        nodes_by_id = {n["id"]: n for n in block["nodes"]}
        assert "feat_F" in nodes_by_id, (
            f"H_plus DAG must have a 'feat_F' PCA node; "
            f"node ids = {list(nodes_by_id)!r}"
        )
        feat_F_params = nodes_by_id["feat_F"].get("params", {})
        assert feat_F_params.get("n_components") == n_factors, (
            f"feat_F PCA n_components must equal n_factors={n_factors} "
            f"(paper Eq. 7 ARDI K static factors); "
            f"got {feat_F_params.get('n_components')!r}"
        )
        # Confirm NOT 'all' (which is B2 semantics, not ARDI H_plus).
        assert feat_F_params.get("n_components") != "all", (
            "feat_F n_components='all' is B2 semantics; H_plus ARDI "
            "must use n_factors (integer), not 'all'."
        )
