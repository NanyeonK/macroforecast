"""Cost-budget smoke tests for models with historically pathological defaults.

2026-07-04 (WP4): six models exceeded 150s on a 140-obs policy-matrix scan
(`.dev-notes/policy_matrix_scan.py`) using their out-of-the-box defaults --
`bvar_minnesota`/`bvar_normal_inverse_wishart` (MCMC `iter`/`burnin`),
`macro_random_forest` (per-node GTVP-ridge forest tree count `B`), `lgb_plus`
(sequential `lgb.train` calls, `n_ensemble` * `n_steps`), `mars` (the
"standard" search preset's `max_degree=2` x `max_terms=30` corner), and
`restricted_midas` (`least_squares` `maxiter`). All five now have cheapened
defaults and/or trimmed "standard" search-space presets; the deep/
paper-faithful settings remain reachable via explicit params or the "wide"
preset (see CHANGELOG and docs/reference/models.md).

These are single-fit smoke pins on a small toy panel, NOT a replication of the
full policy-matrix scan (that lives in `.dev-notes/policy_matrix_scan.py` and
is run by hand, not by CI). They exist to catch a future default value
silently regressing back toward a pathological cost. Budgets are generous
(observed costs are normally well under half the budget) to avoid flakiness
under shared/contended CI hardware.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

T = TypeVar("T")


def _toy_panel(n: int = 140) -> pd.DataFrame:
    rng = np.random.default_rng(20260704)
    factors = np.cumsum(rng.normal(size=(n, 2)) * 0.3, axis=0)
    loadings = rng.normal(size=(6, 2))
    predictors = factors @ loadings.T + rng.normal(size=(n, 6)) * 0.2
    y = 0.02 * factors[:, 0] - 0.015 * factors[:, 1] + rng.normal(size=n) * 0.05
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    cols = {f"x{i}": predictors[:, i] for i in range(6)}
    cols["Y"] = y
    return pd.DataFrame(cols, index=idx)


def _within_budget(label: str, budget_seconds: float, fn: Callable[[], T]) -> T:
    start = time.time()
    result = fn()
    elapsed = time.time() - start
    assert elapsed < budget_seconds, (
        f"{label} default fit took {elapsed:.1f}s, budget is {budget_seconds:.0f}s "
        "-- a default value may have regressed back toward a pathological cost"
    )
    return result


def _assert_finite_predictions(pred: object) -> None:
    values = np.asarray(pred, dtype=float).reshape(-1)
    assert len(values) > 0
    assert np.isfinite(values).all()


@pytest.mark.slow
def test_bvar_minnesota_default_cost_budget() -> None:
    panel = _toy_panel()
    fit = _within_budget(
        "bvar_minnesota",
        60.0,
        lambda: mf.models.bvar_minnesota(panel, target="Y", n_lag=1),
    )
    _assert_finite_predictions(fit.predict(panel.iloc[-4:]))


@pytest.mark.slow
def test_bvar_normal_inverse_wishart_default_cost_budget() -> None:
    panel = _toy_panel()
    fit = _within_budget(
        "bvar_normal_inverse_wishart",
        60.0,
        lambda: mf.models.bvar_normal_inverse_wishart(panel, target="Y", n_lag=1),
    )
    _assert_finite_predictions(fit.predict(panel.iloc[-4:]))


@pytest.mark.slow
def test_favar_default_cost_budget() -> None:
    panel = _toy_panel()
    X = panel.drop(columns=["Y"])
    y = panel["Y"]
    fit = _within_budget("favar", 20.0, lambda: mf.models.favar(X, y))
    _assert_finite_predictions(fit.predict(X.iloc[-4:]))


@pytest.mark.slow
def test_macro_random_forest_default_cost_budget() -> None:
    panel = _toy_panel()
    X = panel.drop(columns=["Y"])
    y = panel["Y"]
    fit = _within_budget(
        "macro_random_forest", 20.0, lambda: mf.models.macro_random_forest(X, y)
    )
    _assert_finite_predictions(fit.predict(X.iloc[-4:]))


@pytest.mark.slow
def test_lgb_plus_default_cost_budget() -> None:
    panel = _toy_panel()
    X = panel.drop(columns=["Y"])
    y = panel["Y"]
    fit = _within_budget("lgb_plus", 30.0, lambda: mf.models.lgb_plus(X, y))
    _assert_finite_predictions(fit.predict(X.iloc[-4:]))


@pytest.mark.slow
def test_mars_standard_preset_worst_combo_cost_budget() -> None:
    # Pin the cost of the specific corner (max_degree=2, max_terms=30) that
    # used to be part of the "standard" search preset before it was dropped
    # for cost reasons -- it must stay affordable in isolation since it is
    # still reachable explicitly (and via the "wide" preset).
    panel = _toy_panel()
    X = panel.drop(columns=["Y"])
    y = panel["Y"]
    fit = _within_budget(
        "mars(max_degree=2, max_terms=30)",
        20.0,
        lambda: mf.models.mars(X, y, max_degree=2, max_terms=30),
    )
    _assert_finite_predictions(fit.predict(X.iloc[-4:]))


@pytest.mark.slow
def test_restricted_midas_default_cost_budget() -> None:
    rng = np.random.default_rng(20260704)
    n = 140
    idx = pd.date_range("1995-01-01", periods=n, freq="MS")
    cols = {}
    for base in ("x0", "x1", "x2"):
        for lag in range(0, 4):
            cols[f"{base}_lag{lag}"] = rng.normal(size=n)
    X = pd.DataFrame(cols, index=idx)
    y = pd.Series(rng.normal(size=n), index=idx, name="Y")
    fit = _within_budget(
        "restricted_midas",
        20.0,
        lambda: mf.models.restricted_midas(X, y, weighting="beta", polynomial_order=3),
    )
    _assert_finite_predictions(fit.predict(X.iloc[-4:]))
