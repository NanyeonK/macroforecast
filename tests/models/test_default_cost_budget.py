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

2026-07-05 (WP-B4): `hemisphere_nn` and `density_hnn` (torch-backed, WP-A4's
Tier-1 zero-anchor closeout) did not finish a single policy-matrix scan
worker call (all 4 policies) even at a 900s timeout with their out-of-the-box
defaults. Root cause: `model_selection=None` still runs a 20-trial random
search over the model's "standard" search-space preset (same mechanism as
`mars` above), so the untuned `max_epochs`/`patience` defaults and the
"standard" preset's `neurons`/`n_estimators`/`prior_estimators` corners were
each paid 20 times per origin per policy. `hemisphere_nn`: `n_estimators`
100->20, `max_epochs` 100->40, `patience` 15->8; "standard" preset `neurons`
(32, 64)->(16, 32), `n_estimators` (5, 10)->(3, 5). `density_hnn`:
`n_estimators` 100->20, `prior_estimators` 50->10, `max_epochs` 100->40,
`patience` 15->8; "standard" preset `neurons` (32, 64)->(16, 32),
`n_estimators` (5, 10)->(3, 5), `prior_estimators` (3, 5)->(2, 3) (`neurons`
default stays 400 -- the paper-faithful Aionx width is never the scan's cost
driver since search always overrides `neurons`). Measured scan totals (4
policies, thread-pinned): `hemisphere_nn` 360.1s -> 90.2s; `density_hnn`
240.6s -> 66.3s -- both now comfortably under the 150s bar. Fewer bag members
trades statistical quality for cost: `density_hnn`'s per-row predicted
variance already has a wide right tail from bagged-member disagreement (the
WP-A4 anchor finding, `tests/models/anchors/test_hnn_anchors.py`), and fewer
members widens that tail further (more sampling noise in the
cross-member-disagreement variance estimate). Old defaults remain reachable
by passing them explicitly, or via the "wide" preset for search-driven runs
(see CHANGELOG and docs/reference/models.md).

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


@pytest.mark.slow
def test_hemisphere_nn_default_cost_budget() -> None:
    pytest.importorskip("torch")
    panel = _toy_panel()
    X = panel.drop(columns=["Y"])
    y = panel["Y"]
    fit = _within_budget(
        "hemisphere_nn", 20.0, lambda: mf.models.hemisphere_nn(X, y)
    )
    _assert_finite_predictions(fit.predict(X.iloc[-4:]))


@pytest.mark.slow
def test_density_hnn_default_cost_budget() -> None:
    pytest.importorskip("torch")
    panel = _toy_panel()
    X = panel.drop(columns=["Y"])
    y = panel["Y"]
    fit = _within_budget("density_hnn", 30.0, lambda: mf.models.density_hnn(X, y))
    _assert_finite_predictions(fit.predict(X.iloc[-4:]))
