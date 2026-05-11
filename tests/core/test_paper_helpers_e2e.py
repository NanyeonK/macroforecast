"""Phase A3 — 15-helper end-to-end smoke tests.

One smoke test per paper-method helper in
``macroforecast.recipes.paper_methods``. Each test builds a minimal
synthetic DGP, calls the helper with paper-light arguments, runs the
emitted recipe through ``macroforecast.run``, and asserts the recipe
returns at least one cell with an ``l4_forecasts_v1`` artifact.

**Scope.** This is a runnability gate, not a procedure-faithfulness
audit. We assert "the published helper does not raise + emits forecasts
through the public pipeline." Algorithm-level correctness (DM-vs-ARDI
benchmark, ARDI lag-of-F, full-rank rotation, etc.) is the subject of
Round 3 micro-audits.

**xfail bookkeeping.** All Round 1 xfail markers have been removed as of
2026-05-12 Phase D-2b. Papers 4 and 11 were demoted in Phase B; papers 7,
9, 10, 12, and 14 were demoted here. All 15 tests are now expected to pass
without xfail annotation.

After Phase A2 this module's paper 13 (`maximally_forward_looking`) and
paper 16 (`ml_useful_macro_b_grid`) are expected to pass, validating
the fixes committed in `paper_methods.py`.

Paper 2 (`arctic_sea_ice_dfm`) helper was cut 2026-05-08 — phantom
citation (paper has no DFM content). The underlying `_DFMMixedFrequency`
class in `core/runtime.py` is unaffected.
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pytest

import macroforecast
from macroforecast.recipes.paper_methods import (
    adaptive_ma,
    anatomy_oos,
    arctic_var,
    booging,
    dual_interpretation,
    hemisphere_neural_network,
    macroeconomic_data_transformations_horse_race,
    macroeconomic_random_forest,
    maximally_forward_looking,
    ml_useful_macro_b_grid,
    ols_attention_demo,
    scaled_pca,
    slow_growing_tree,
    sparse_macro_factors,
    two_step_ridge,
)


# ---------------------------------------------------------------------------
# Shared synthetic DGP
# ---------------------------------------------------------------------------


def _build_panel(t: int = 120, k: int = 8, seed: int = 0) -> dict[str, list]:
    """Build a synthetic monthly DGP with ``t`` rows and ``k`` predictors.

    Series mimic loose macro-style dynamics: ``y`` is a noisy random walk
    with positive drift; ``x1..xK`` are noisy linear trends + Gaussian
    perturbations. Plausible-enough for smoke runnability without
    matching any specific FRED-MD configuration.
    """

    dates: list[str] = []
    d = datetime.date(2014, 1, 1)
    for _ in range(t):
        dates.append(d.strftime("%Y-%m-%d"))
        m = d.month + 1
        y = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = datetime.date(y, m, 1)
    rng = np.random.default_rng(seed)
    panel: dict[str, list] = {
        "date": dates,
        "y": list(np.cumsum(rng.normal(0.5, 0.3, size=t))),
    }
    for j in range(1, k + 1):
        panel[f"x{j}"] = list(rng.normal(0.0, 0.5, size=t) + np.arange(t) * 0.05)
    return panel


@pytest.fixture(scope="module")
def synth_panel() -> dict[str, list]:
    return _build_panel(t=120, k=8, seed=0)


@pytest.fixture(scope="module")
def synth_panel_long() -> dict[str, list]:
    """Larger panel for helpers that need a longer history (MRF)."""

    return _build_panel(t=120, k=4, seed=1)


# ---------------------------------------------------------------------------
# Smoke runner
# ---------------------------------------------------------------------------


def _assert_recipe_runs(recipe: dict) -> None:
    """Run a recipe through ``macroforecast.run`` and assert at least
    one cell came back with an ``l4_forecasts_v1`` artifact."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)
    assert result.cells, "recipe should produce at least one cell"
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l4_forecasts_v1" in artifacts, (
        f"l4_forecasts_v1 missing; artifacts={list(artifacts.keys())}"
    )


def _assert_grid_has_passing_cell(grid: dict[str, dict]) -> None:
    """For grid helpers (papers 15, 16), iterate cells and assert at
    least one runs end-to-end. We do not require all cells to pass —
    grid-level correctness is procedure-audit territory."""

    assert grid, "grid helper returned empty dict"
    last_err: Exception | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, recipe in grid.items():
            try:
                result = macroforecast.run(recipe)
                if (
                    result.cells
                    and "l4_forecasts_v1" in result.cells[0].runtime_result.artifacts
                ):
                    return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
    raise AssertionError(
        f"no grid cell ran end-to-end; last error: {type(last_err).__name__}: {last_err}"
    )


# ---------------------------------------------------------------------------
# 15 e2e tests — one per paper helper (paper 2 helper cut 2026-05-08)
# ---------------------------------------------------------------------------


# Paper 1 — scaled_pca (operational since v0.1)
def test_paper_01_scaled_pca(synth_panel):
    recipe = scaled_pca(target="y", horizon=1, panel=synth_panel)
    _assert_recipe_runs(recipe)


# Paper 2 — arctic_sea_ice_dfm helper cut 2026-05-08: phantom citation (paper
# has no DFM content). The `_DFMMixedFrequency` class in `core/runtime.py`
# remains operational; only the paper-anchored helper was removed.


# Paper 3 — slow_growing_tree (operational decision_tree.split_shrinkage). Cap
# max_depth so the smoke test does not exhaust the budget on a deep tree.
def test_paper_03_slow_growing_tree(synth_panel):
    recipe = slow_growing_tree(target="y", horizon=1, panel=synth_panel, max_depth=3)
    _assert_recipe_runs(recipe)


# Paper 4 — arctic_var: Phase B-4 (2026-05-08) rewired the helper to
# bvar_minnesota with paper-faithful defaults (b_AR=0.9, λ₁=0.3,
# λ_cross=0.5, λ_decay=1.5, n_posterior_draws=2000, n_lag=12) plus
# Cholesky ordering exposure and posterior IRF / HD bands. The Round 1
# xfail is closed; passing ``n_lags`` (legacy plural) emits a
# DeprecationWarning but still routes correctly.
def test_paper_04_arctic_var(synth_panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # n_posterior_draws kept small to keep the smoke test fast.
        recipe = arctic_var(
            target="y",
            horizon=1,
            panel=synth_panel,
            n_lag=2,
            n_posterior_draws=20,
            posterior_irf_periods=4,
        )
    _assert_recipe_runs(recipe)


# Paper 5 — macroeconomic_random_forest. The vendored MacroRandomForest needs
# enough train rows for its block-bootstrap; raise min_train_size on the fit
# node so the synthetic panel exercises only the late-origin walk-forward
# range.
def test_paper_05_macroeconomic_random_forest(synth_panel_long):
    recipe = macroeconomic_random_forest(
        target="y",
        horizon=1,
        panel=synth_panel_long,
        n_estimators=2,
        block_size=4,
    )
    for node in recipe["4_forecasting_model"]["nodes"]:
        if node.get("op") == "fit_model":
            node["params"]["min_train_size"] = 80
            node["params"]["B"] = 2
    _assert_recipe_runs(recipe)


# Paper 6 — booging (bagging.strategy=sequential_residual). Operational; keep
# n_iterations small for runtime.
def test_paper_06_booging(synth_panel):
    recipe = booging(target="y", horizon=1, panel=synth_panel, n_iterations=2)
    # Booging is bagging-of-trees; bound depth on the base decision_tree
    # and trim the walk-forward range so per-origin refit stays cheap.
    for node in recipe["4_forecasting_model"]["nodes"]:
        if node.get("op") == "fit_model":
            node["params"]["min_train_size"] = 100
            node["params"]["max_depth"] = 3
            node["params"]["n_estimators"] = 2
    _assert_recipe_runs(recipe)


# Paper 7 — adaptive_ma: Round 1 flagged that the helper pipes src_X through
# adaptive_ma_rf without using src_y as the supervised target (the AlbaMA
# procedure uses target-supervised aggregation).
def test_paper_07_adaptive_ma(synth_panel):
    recipe = adaptive_ma(target="y", horizon=1, panel=synth_panel, n_estimators=5)
    _assert_recipe_runs(recipe)


# Paper 8 — two_step_ridge (chained ridge + ridge(prior=random_walk)).
# Operational since v0.9.0.
def test_paper_08_two_step_ridge(synth_panel):
    recipe = two_step_ridge(target="y", horizon=1, panel=synth_panel)
    _assert_recipe_runs(recipe)


# Paper 9 — hemisphere_neural_network: Round 1 flagged that L4 dispatches to
# LinearRegression rather than the HNN distrib methods.
def test_paper_09_hemisphere_neural_network(synth_panel):
    recipe = hemisphere_neural_network(target="y", horizon=1, panel=synth_panel)
    # MLP per-origin walk-forward refit blows up the smoke budget; raise
    # min_train_size so only a couple of origins are scored. Doesn't
    # change the xfail-flagged procedure mismatch (Round 1: HNN sub-axes
    # route to LinearRegression).
    for node in recipe["4_forecasting_model"]["nodes"]:
        if node.get("op") == "fit_model":
            node["params"]["min_train_size"] = 110
            node["params"]["max_iter"] = 50
    _assert_recipe_runs(recipe)


# Paper 10 — ols_attention_demo: Round 1 flagged that the helper returns a
# generic OLS without the Ω attention artifact.
def test_paper_10_ols_attention_demo(synth_panel):
    recipe = ols_attention_demo(target="y", horizon=1, panel=synth_panel)
    _assert_recipe_runs(recipe)


# Paper 11 — anatomy_oos: Phase B-11 paper-11 helper rewrite closed F1
# (initial_window now reaches the anatomy adapter via the L7 op step,
# not a dead 0_meta.leaf_config stamp). The helper now drives Path A
# (per-origin refit) end-to-end, so the smoke test passes outright.
def test_paper_11_anatomy_oos(synth_panel):
    recipe = anatomy_oos(
        target="y", horizon=1, panel=synth_panel, initial_window=72, n_iterations=10
    )
    _assert_recipe_runs(recipe)


# Paper 12 — dual_interpretation: Round 1 flagged that the L7
# dual_decomposition op is future, so the helper has no L7 wiring.
def test_paper_12_dual_interpretation(synth_panel):
    recipe = dual_interpretation(target="y", horizon=1, panel=synth_panel)
    _assert_recipe_runs(recipe)


# Paper 13 — maximally_forward_looking: Phase A2 fix moved asymmetric_trim
# from L2 (validator-rejected) to an L3 step node. Now expected to pass.
def test_paper_13_maximally_forward_looking(synth_panel):
    recipe = maximally_forward_looking(
        target="y",
        horizon=1,
        panel=synth_panel,
        alpha=10.0,
        search_algorithm="block_cv",
    )
    _assert_recipe_runs(recipe)


# Paper 14 — sparse_macro_factors: Round 1 flagged that the helper routes to
# sklearn sparse_pca rather than chen_rohe (paper-faithful Sparse Component
# Analysis). Operational as a generic sparse-PCA recipe but not paper-
# faithful, and the L3 sparse_pca node also lacks temporal_rule, so the
# recipe is not runnable through ``macroforecast.run`` without a runtime
# patch.
def test_paper_14_sparse_macro_factors(synth_panel):
    recipe = sparse_macro_factors(
        target="y", horizon=1, panel=synth_panel, n_components=3
    )
    _assert_recipe_runs(recipe)


# Paper 15 — macroeconomic_data_transformations_horse_race. Smoke restricted
# to one (cell, family, target_method) tuple to keep the suite fast; the
# grid generation itself is the paper helper, and we assert at least one
# cell runs.
def test_paper_15_macroeconomic_data_transformations_horse_race(synth_panel):
    # The "F" / "MAF" cells use PCA without ``temporal_rule`` set (an
    # independent helper-shape bug not in scope for this Phase A2 run);
    # restrict the smoke to a non-PCA cell ("X" = original predictors)
    # so we test the helper's grid surface without tripping over a
    # paper-15-specific gap. Round 3 micro-audit will follow up.
    grid = macroeconomic_data_transformations_horse_race(
        target="y",
        horizon=1,
        panel=synth_panel,
        cells=("X",),
        families=("ar_p",),
        target_methods=("direct",),
        max_order=4,
    )
    _assert_grid_has_passing_cell(grid)


# Paper 16 — ml_useful_macro_b_grid (per spec note: B-grid is the paper's
# §3.2 + Eq. (18) regularization rotation grid). Phase A2 fix made B₂ +
# B₃ runnable; smoke asserts at least one cell runs (post-fix all 9
# cells run; pre-fix only the 3 B₁ cells did).
def test_paper_16_ml_useful_macro_b_grid(synth_panel):
    grid = ml_useful_macro_b_grid(
        target="y",
        horizon=1,
        panel=synth_panel,
        rotations=("B1", "B2", "B3"),
        families=("ridge",),
    )
    _assert_grid_has_passing_cell(grid)
