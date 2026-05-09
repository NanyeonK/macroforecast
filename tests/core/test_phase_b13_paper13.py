"""Phase B-13 paper-13 (Goulet Coulombe / Klieber / Barrette / Goebel 2024
"Maximally Forward-Looking Core Inflation" / Albacore family) tests.

Round 5 audit on top of v0.9.0F flagged two implementation gaps that left
the paper's headline §2 / §3 reproduction inexact from the helper:

* **F3 (Medium)** -- paper §2 "Implementation Details" (p.10) explicitly:
  *"We use the CVXR package in R, which provides fast solutions for
  linear convex programming problems."* macroforecast used scipy SLSQP.
  Both ``_ShrinkToTargetRidge`` (Variant A) and ``_FusedDifferenceRidge``
  (Variant B) now solve the same convex QP via cvxpy + OSQP — the
  paper-stated convex backend.

* **F8 (Medium)** -- paper §3 (p.11) explicitly: *"the order statistics
  time series are smoothed using the 3 months moving average."* The
  helper ``maximally_forward_looking()`` previously hard-coded
  ``smooth_window: 0`` (no smoothing), so the paper-published
  Albacore_ranks pipeline diverged from the helper-built recipe by an
  un-smoothed order-statistic input.

The six tests below close F3 + F8.

Reference: Goulet Coulombe / Klieber / Barrette / Goebel (2024)
"Maximally Forward-Looking Core Inflation" technical report; §2
Implementation Details (CVXR) + §3 (3-month MA on order statistics).
"""

from __future__ import annotations

import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast
from macroforecast.recipes.paper_methods import maximally_forward_looking


# ---------------------------------------------------------------------
# DGP fixture: linear-Gaussian K-component panel with simplex weights
# ---------------------------------------------------------------------


def _build_panel(T: int = 120, K: int = 8, seed: int = 0) -> dict[str, list]:
    """CPI-like DGP: y is a convex combination of K component series.

    Same date-construction style as Phase B-11/B-12/B-14/B-15 sibling
    tests so the walk-forward harness lines up.
    """

    rng = np.random.default_rng(seed)
    dates: list[str] = []
    d = datetime.date(2010, 1, 1)
    for _ in range(T):
        dates.append(d.strftime("%Y-%m-%d"))
        m = d.month + 1
        y_yr = d.year + (m - 1) // 12
        m = ((m - 1) % 12) + 1
        d = datetime.date(y_yr, m, 1)

    X = rng.normal(0.0, 1.0, size=(T, K))
    # Simplex weights (matches Albacore Variant A semantics).
    w = np.array([0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04])[:K]
    w = w / w.sum()
    noise = rng.normal(0.0, 0.1, size=T)
    y = X @ w + noise

    panel: dict[str, list] = {"date": dates, "y": y.tolist()}
    for j in range(K):
        panel[f"x{j + 1}"] = X[:, j].tolist()
    return panel


# ---------------------------------------------------------------------
# Test 1 -- F3: cvxpy solver actually invoked for Variant A
# ---------------------------------------------------------------------


def test_albacore_variant_a_uses_cvxpy_osqp_solver(monkeypatch):
    """Phase B-13 F3: ``_ShrinkToTargetRidge.fit()`` must route through
    cvxpy (paper §2 CVXR backend), not scipy SLSQP. We monkey-patch
    ``cvxpy.Problem.solve`` to count invocations and confirm the cvxpy
    code path executed at least once during the fit."""

    import cvxpy as cp

    from macroforecast.core.runtime import _ShrinkToTargetRidge

    call_count = {"n": 0}
    original_solve = cp.Problem.solve

    def counting_solve(self, *args, **kwargs):
        call_count["n"] += 1
        return original_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", counting_solve)

    rng = np.random.default_rng(0)
    K = 4
    X = pd.DataFrame(rng.standard_normal((40, K)), columns=list("abcd"))
    y = pd.Series(
        X.values @ np.array([0.4, 0.3, 0.2, 0.1]) + rng.standard_normal(40) * 0.1
    )

    model = _ShrinkToTargetRidge(
        alpha=1.0,
        prior_target=[0.25] * K,
        simplex=True,
        nonneg=True,
    ).fit(X, y)

    assert call_count["n"] >= 1, (
        "cvxpy.Problem.solve should be invoked at least once by "
        "_ShrinkToTargetRidge.fit (paper §2 CVXR backend)."
    )
    assert model._coef is not None
    # Variant A simplex constraint: sum(w) == 1 to OSQP precision.
    np.testing.assert_allclose(float(model._coef.sum()), 1.0, atol=1e-4)


# ---------------------------------------------------------------------
# Test 2 -- F3: cvxpy solver actually invoked for Variant B
# ---------------------------------------------------------------------


def test_albacore_variant_b_uses_cvxpy_osqp_solver(monkeypatch):
    """Phase B-13 F3: ``_FusedDifferenceRidge.fit()`` must route through
    cvxpy + OSQP (paper §2 CVXR backend), not scipy SLSQP."""

    import cvxpy as cp

    from macroforecast.core.runtime import _FusedDifferenceRidge

    call_count = {"n": 0}
    original_solve = cp.Problem.solve

    def counting_solve(self, *args, **kwargs):
        call_count["n"] += 1
        return original_solve(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", counting_solve)

    rng = np.random.default_rng(7)
    K = 5
    X = pd.DataFrame(
        np.sort(rng.standard_normal((50, K)), axis=1),
        columns=[f"rank_{i + 1}" for i in range(K)],
    )
    y = pd.Series(X.values @ np.full(K, 1 / K) + rng.standard_normal(50) * 0.1)

    model = _FusedDifferenceRidge(alpha=10.0, mean_equality=True).fit(X, y)

    assert call_count["n"] >= 1, (
        "cvxpy.Problem.solve should be invoked at least once by "
        "_FusedDifferenceRidge.fit (paper §2 CVXR backend)."
    )
    assert model._coef is not None


# ---------------------------------------------------------------------
# Test 3 -- F3 limit cases: cvxpy result matches paper Eq. (1) limits
# ---------------------------------------------------------------------


def test_albacore_variant_a_cvxpy_alpha_zero_recovers_nnls():
    """Phase B-13 F3 limit: α = 0 in Variant A reduces the objective to
    pure NNLS / OLS (within nonneg + simplex). The cvxpy / OSQP solution
    must recover the true coefficient vector when the data is generated
    from those weights."""

    rng = np.random.default_rng(42)
    T, K = 80, 4
    true_w = np.array([0.4, 0.3, 0.2, 0.1])
    X = pd.DataFrame(rng.standard_normal((T, K)), columns=list("abcd"))
    y = pd.Series(X.values @ true_w + rng.standard_normal(T) * 0.05)

    from macroforecast.core.runtime import _ShrinkToTargetRidge

    model = _ShrinkToTargetRidge(
        alpha=0.0,
        prior_target=[1 / K] * K,
        simplex=True,
        nonneg=True,
    ).fit(X, y)
    coef = model._coef
    assert coef is not None
    # Simplex constraint holds.
    np.testing.assert_allclose(float(coef.sum()), 1.0, atol=1e-4)
    # NNLS / OLS recovers true_w within sample noise.
    np.testing.assert_allclose(coef, true_w, atol=0.05)


def test_albacore_variant_a_cvxpy_alpha_infinity_returns_target():
    """Phase B-13 F3 limit: α → ∞ in Variant A must collapse the
    coefficient vector onto ``prior_target`` exactly (paper Eq. 1
    large-α limit), independent of the data."""

    rng = np.random.default_rng(0)
    K = 4
    target = np.array([0.5, 0.25, 0.15, 0.10])
    X = pd.DataFrame(rng.standard_normal((50, K)), columns=list("abcd"))
    y = pd.Series(rng.standard_normal(50))

    from macroforecast.core.runtime import _ShrinkToTargetRidge

    model = _ShrinkToTargetRidge(
        alpha=1e6,
        prior_target=target.tolist(),
        simplex=True,
        nonneg=True,
    ).fit(X, y)
    np.testing.assert_allclose(model._coef, target, atol=1e-3)


# ---------------------------------------------------------------------
# Test 4 -- F8: helper default smooth_window == 3
# ---------------------------------------------------------------------


def test_albacore_helper_default_smooth_window_3():
    """Phase B-13 F8: ``maximally_forward_looking()`` (variant='ranks',
    default) must build an L3 ``asymmetric_trim`` step whose
    ``smooth_window`` param is 3 — paper §3 explicit default ("the order
    statistics time series are smoothed using the 3 months moving
    average")."""

    recipe = maximally_forward_looking(
        target="y",
        horizon=1,
        variant="ranks",
    )
    l3 = recipe["3_feature_engineering"]
    trim_nodes = [n for n in l3["nodes"] if n.get("op") == "asymmetric_trim"]
    assert len(trim_nodes) == 1, (
        f"variant='ranks' must include exactly one asymmetric_trim node; "
        f"got {len(trim_nodes)}"
    )
    params = trim_nodes[0].get("params", {})
    assert params.get("smooth_window") == 3, (
        f"helper default smooth_window must be 3 (paper §3 3-month MA); "
        f"got {params.get('smooth_window')!r}"
    )


# ---------------------------------------------------------------------
# Test 5 -- F3: cvxpy non-optimal status raises informative error
# ---------------------------------------------------------------------


def test_albacore_cvxpy_optimal_status_required(monkeypatch):
    """Phase B-13 F3: paper requires solver convergence. When cvxpy
    returns a non-optimal status, ``_ShrinkToTargetRidge.fit()`` must
    raise an informative ``RuntimeError`` rather than silently returning
    a non-converged weight vector."""

    import cvxpy as cp

    from macroforecast.core.runtime import _ShrinkToTargetRidge

    original_solve = cp.Problem.solve

    def failing_solve(self, *args, **kwargs):
        result = original_solve(self, *args, **kwargs)
        # Force an infeasible-style status post-hoc to simulate solver
        # non-convergence; cvxpy stores the status on the instance.
        self._status = "infeasible"
        return result

    monkeypatch.setattr(cp.Problem, "solve", failing_solve)

    rng = np.random.default_rng(3)
    K = 4
    X = pd.DataFrame(rng.standard_normal((30, K)), columns=list("abcd"))
    y = pd.Series(rng.standard_normal(30))

    with pytest.raises(RuntimeError, match="cvxpy did not converge"):
        _ShrinkToTargetRidge(
            alpha=1.0,
            prior_target=[0.25] * K,
            simplex=True,
            nonneg=True,
        ).fit(X, y)


# ---------------------------------------------------------------------
# Test 6 -- F3 + F8 e2e: full helper recipe runs via macroforecast.run
# ---------------------------------------------------------------------


def test_albacore_e2e_runs_via_macroforecast_run():
    """Phase B-13 F3 + F8 e2e: ``maximally_forward_looking()`` (default
    'ranks' variant) must produce a forecast end-to-end on a synthetic
    CPI-like DGP via ``macroforecast.run``. Default ``smooth_window=3``
    + cvxpy + OSQP backend stay live throughout the walk-forward."""

    panel = _build_panel(T=120, K=8, seed=0)
    recipe = maximally_forward_looking(
        target="y",
        horizon=1,
        panel=panel,
        alpha=10.0,
        search_algorithm="block_cv",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = macroforecast.run(recipe)

    assert result.cells, "Albacore_ranks recipe must produce at least one cell"
    artifacts = result.cells[0].runtime_result.artifacts
    assert "l4_forecasts_v1" in artifacts, (
        f"l4_forecasts_v1 missing; artifacts={list(artifacts.keys())}"
    )
    forecasts = artifacts["l4_forecasts_v1"].forecasts
    assert forecasts, "Albacore_ranks must produce at least one forecast"
