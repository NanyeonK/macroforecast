"""Phase B-8 paper-8 (Goulet Coulombe 2025 IJF -- "Time-Varying
Parameters as Ridge", 2SRR) helper-rewrite tests.

Round 1 audit identified two findings on this paper:

* **F3 (HIGH-grade Medium)** -- Algorithm 1 step 4 ("Use solution
  (17) to **rerun CV** and get β̂_2, the final estimator") was not
  implemented; the runtime reused the user-supplied step-1 λ for
  step 2 with no re-CV. Paper §2.5 footnote 4 + §2.4.1 explicitly
  justify the second λ-CV because heterogeneous variance changes
  the effective regularization.
* **F5 (Medium)** -- helper exposed ``alpha_step1, alpha_step2,
  vol_model`` but the runtime class only consumed one ``alpha``.
  ``alpha_step1`` was wired into a separate, unused ``fit_step1``
  ridge node whose output was discarded.

The five tests below close F3 + F5. Reference: Goulet Coulombe
(2025) "Time-Varying Parameters as Ridge", IJF doi:10.1016/
j.ijforecast.2024.08.006 §2.5 Algorithm 1 step 4.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast
from macroforecast.core.runtime import _TwoStageRandomWalkRidge
from macroforecast.recipes.paper_methods import two_step_ridge


# ----------------------------------------------------------------------
# Heteroskedastic-residual TVP DGP fixture
# ----------------------------------------------------------------------


def _build_tvp_xy(T: int = 80, K: int = 4, seed: int = 0):
    """Random-walk-coefficient TVP regression with time-varying noise.

    Mirrors the paper's §4 simulation regime: β_t evolves as a
    random walk, residual σ_t is sinusoidal in time. Returns
    ``(X_df, y_series)`` ready for ``_TwoStageRandomWalkRidge.fit``.
    """

    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, (T, K))
    beta = np.cumsum(rng.normal(0.0, 0.05, (K,)))
    betas_path = np.zeros((T, K))
    b = beta.copy()
    for t in range(T):
        b = b + rng.normal(0.0, 0.05, K)
        betas_path[t] = b
    y = (X * betas_path).sum(axis=1)
    sigma_t = 0.1 + 0.5 * np.abs(np.sin(np.linspace(0.0, 6.28, T)))
    y = y + rng.normal(0.0, sigma_t, T)
    X_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(K)])
    return X_df, pd.Series(y)


def _step2_node(recipe: dict) -> dict:
    """Return the ``fit_step2`` (ridge prior=random_walk) L4 node."""

    nodes = recipe["4_forecasting_model"]["nodes"]
    return next(n for n in nodes if n.get("id") == "fit_step2")


# ----------------------------------------------------------------------
# Test 1 -- helper default routes second CV (paper §2.5 step 4)
# ----------------------------------------------------------------------


def test_2srr_helper_default_alpha_strategy_second_cv():
    """Phase B-8 F3: paper §2.5 step 4 says rerun CV is "crucial"
    after the warm-start. Helper default must therefore route
    ``alpha_strategy="second_cv"`` into the fit_step2 params."""

    sig = inspect.signature(two_step_ridge)
    assert "alpha_strategy" in sig.parameters
    assert sig.parameters["alpha_strategy"].default == "second_cv"
    assert "cv_folds" in sig.parameters
    assert sig.parameters["cv_folds"].default == 5

    recipe = two_step_ridge()
    step2 = _step2_node(recipe)
    assert step2["params"]["alpha_strategy"] == "second_cv"
    assert step2["params"]["cv_folds"] == 5


# ----------------------------------------------------------------------
# Test 2 -- second CV picks alpha from the grid + improves MSE
# ----------------------------------------------------------------------


def test_2srr_second_cv_picks_alpha_from_grid():
    """Phase B-8 F3 procedure-level: with heteroskedastic-residual
    TVP DGP, the second-CV run picks a λ from the grid and produces
    held-out MSE no worse than the un-tuned (alpha=1.0) baseline."""

    X_df, y = _build_tvp_xy(T=80, K=4, seed=0)
    grid = [0.01, 0.1, 1.0, 10.0, 100.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tuned = _TwoStageRandomWalkRidge(
            alpha=1.0,
            alpha_strategy="second_cv",
            alpha_grid=grid,
            cv_folds=5,
            vol_model="ewma",
            random_state=0,
        ).fit(X_df, y)
        baseline = _TwoStageRandomWalkRidge(
            alpha=1.0,
            alpha_strategy="fixed",
            vol_model="ewma",
            random_state=0,
        ).fit(X_df, y)

    # Tuned λ must come from the user-supplied grid.
    assert any(abs(tuned.tuned_alpha_ - float(g)) < 1e-9 for g in grid), (
        f"tuned_alpha_={tuned.tuned_alpha_} not in grid={grid}"
    )

    # In-sample MSE comparison: the CV-tuned λ minimises held-out
    # MSE and should not under-fit relative to a fixed λ=1.0 baseline
    # on the same training sample.
    pred_tuned = tuned.predict(X_df)
    pred_baseline = baseline.predict(X_df)
    mse_tuned = float(np.mean((y.to_numpy() - pred_tuned) ** 2))
    mse_baseline = float(np.mean((y.to_numpy() - pred_baseline) ** 2))
    # CV minimum is bounded by the fixed-λ=1.0 case (which is in the
    # grid), so the tuned MSE must be ≤ the baseline.
    assert mse_tuned <= mse_baseline + 1e-9


# ----------------------------------------------------------------------
# Test 3 -- alpha_strategy="fixed" preserves the user-supplied λ
# ----------------------------------------------------------------------


def test_2srr_alpha_strategy_fixed_uses_user_alpha():
    """Phase B-8 F3 escape hatch: ``alpha_strategy="fixed"`` must
    bypass the second CV and keep the user-supplied λ as the
    tuned/effective λ. Lets users pin λ when they want to."""

    X_df, y = _build_tvp_xy(T=60, K=3, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = _TwoStageRandomWalkRidge(
            alpha=42.0,
            alpha_strategy="fixed",
            vol_model="ewma",
            random_state=0,
        ).fit(X_df, y)
    assert m.tuned_alpha_ == pytest.approx(42.0)


# ----------------------------------------------------------------------
# Test 4 -- helper signature regression: alpha_step1 dropped
# ----------------------------------------------------------------------


def test_2srr_helper_drops_dead_alpha_step1():
    """Phase B-8 F5: the ``alpha_step1`` param fed a separate,
    unused ``fit_step1`` ridge node whose output was discarded
    (paper-faithful 2SRR uses ONE λ across step 1 + step 2, picked
    by the second CV). The helper signature must drop ``alpha_step1``
    and the L4 DAG must no longer contain a ``fit_step1`` node."""

    sig = inspect.signature(two_step_ridge)
    assert "alpha_step1" not in sig.parameters

    recipe = two_step_ridge()
    node_ids = {n.get("id") for n in recipe["4_forecasting_model"]["nodes"]}
    assert "fit_step1" not in node_ids
    # Single fit_model node = fit_step2 (the heterogeneous-Ω solve).
    fit_nodes = [
        n for n in recipe["4_forecasting_model"]["nodes"] if n.get("op") == "fit_model"
    ]
    assert len(fit_nodes) == 1
    assert fit_nodes[0]["id"] == "fit_step2"


# ----------------------------------------------------------------------
# Test 5 -- e2e via macroforecast.run with default helper args
# ----------------------------------------------------------------------


def test_2srr_e2e_runs_with_second_cv():
    """Phase B-8 e2e gate: the helper with default args (second CV
    enabled) must run end-to-end through ``macroforecast.run`` and
    return at least one cell with an ``l4_forecasts_v1`` artifact."""

    import datetime

    rng = np.random.default_rng(0)
    T, K = 96, 4
    dates: list[str] = []
    d = datetime.date(2014, 1, 1)
    for _ in range(T):
        dates.append(d.strftime("%Y-%m-%d"))
        m = d.month + 1
        y_yr = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = datetime.date(y_yr, m, 1)
    panel: dict[str, list] = {
        "date": dates,
        "y": list(np.cumsum(rng.normal(0.5, 0.3, size=T))),
    }
    for j in range(1, K + 1):
        panel[f"x{j}"] = list(rng.normal(0.0, 0.5, size=T) + np.arange(T) * 0.05)

    # Use a short alpha_grid + cv_folds=3 to keep the smoke fast.
    recipe = two_step_ridge(
        target="y",
        horizon=1,
        panel=panel,
        alpha_grid=[0.1, 1.0, 10.0],
        cv_folds=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)
    assert result.cells, "recipe should produce at least one cell"
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l4_forecasts_v1" in artifacts, (
        f"l4_forecasts_v1 missing; artifacts={list(artifacts.keys())}"
    )
