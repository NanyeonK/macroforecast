"""Independent correctness anchors for ``assemblage_regression`` /
``supervised_aggregation`` (and their shared estimator,
``SupervisedAggregationRegressor``).

WP-V2. Per ``.dev-notes/anchor_coverage/matrix.csv``, both models were
"effectively untested"/"never fit in any test at all" -- the whole
assemblage family (``assemblage_regression``, ``supervised_aggregation``,
``component_aggregation``, ``rank_aggregation``) routes through the single
``SupervisedAggregationRegressor.fit()`` (confirmed by reading
``macroforecast/models/assemblage.py``), which solves

    minimize_b  ||y - X b||^2 + alpha * penalty(b)
    subject to  b >= 0            (optional, ``nonneg``)
                sum(b) == 1       (optional, ``simplex``)
                mean(X b) == mean(y)   (optional, ``mean_match``)

via ``scipy.optimize.minimize(..., method="SLSQP")``. This is a small,
well-understood constrained-QP family with closed-form solutions in the
cases exercised below (no inequality constraint active), so instead of a
smoke test this file fits fixtures with KNOWN, hand-computable optimal
weights and checks the solver recovers them.

Anchors (all analytic, no Monte Carlo):

1. ``test_unconstrained_ridge_matches_closed_form``: ``nonneg=False`` (no
   inequality constraint at all) -- plain ridge, closed form
   ``b = (X'X + alpha*I)^-1 X'y``.

2. ``test_target_shrinkage_penalty_matches_closed_form``: unconstrained,
   ``penalty="target_shrinkage"`` -- closed form
   ``b = (X'X + alpha*I)^-1 (X'y + alpha*target)``.

3. ``test_simplex_ridge_matches_closed_form_equality_constrained_solution``
   and ``test_mean_match_ridge_matches_closed_form_equality_constrained_solution``:
   a single linear equality constraint (``sum(b)=1`` or ``mean(Xb)=mean(y)``)
   plus ridge is an equality-constrained QP solvable exactly via the
   bordered KKT linear system; fixtures are constructed (and verified, not
   assumed) to keep the solution's ``nonneg`` constraint non-binding, so the
   SLSQP solve should land exactly on that closed form.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.models.assemblage import SupervisedAggregationRegressor


def _fixture(n_obs: int = 80, n_features: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=1.0, scale=1.0, size=(n_obs, n_features))
    true_weights = np.array([0.3, 0.4, 0.1, 0.2])
    y = x @ true_weights + 0.05 * rng.normal(size=n_obs)
    return x, y


def _fit_regressor(x, y, **kwargs) -> SupervisedAggregationRegressor:
    frame = pd.DataFrame(x, columns=[f"c{i}" for i in range(x.shape[1])])
    est = SupervisedAggregationRegressor(**kwargs)
    est.fit(frame, pd.Series(y))
    return est


# ---------------------------------------------------------------------------
# Anchor 1: unconstrained ridge.
# ---------------------------------------------------------------------------


@pytest.mark.reference
def test_unconstrained_ridge_matches_closed_form():
    x, y = _fixture()
    alpha = 2.0
    est = _fit_regressor(
        x, y, alpha=alpha, nonneg=False, simplex=False, mean_match=False,
        penalty="ridge", penalty_scale="none", fit_intercept=False,
    )
    n_features = x.shape[1]
    closed_form = np.linalg.solve(x.T @ x + alpha * np.eye(n_features), x.T @ y)
    np.testing.assert_allclose(est.coef_, closed_form, rtol=1e-5, atol=1e-6)


@pytest.mark.reference
def test_target_shrinkage_penalty_matches_closed_form():
    x, y = _fixture(seed=1)
    alpha = 1.5
    n_features = x.shape[1]
    target = np.array([0.1, 0.1, 0.1, 0.1])
    est = _fit_regressor(
        x, y, alpha=alpha, nonneg=False, simplex=False, mean_match=False,
        penalty="target_shrinkage", penalty_scale="none", fit_intercept=False,
        reference_weights=dict(zip([f"c{i}" for i in range(n_features)], target)),
    )
    closed_form = np.linalg.solve(
        x.T @ x + alpha * np.eye(n_features), x.T @ y + alpha * target
    )
    np.testing.assert_allclose(est.coef_, closed_form, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Anchors 2-3: single-equality-constrained ridge (nonneg non-binding).
# ---------------------------------------------------------------------------


def _bordered_kkt_solution(x: np.ndarray, y: np.ndarray, *, alpha: float, c: np.ndarray, d: float):
    """Exact solution of: minimize ||y-Xb||^2 + alpha*||b||^2 s.t. c'b = d.

    KKT system: [X'X + alpha*I, c; c', 0] [b; mu] = [X'y; d].
    """
    n_features = x.shape[1]
    a = x.T @ x + alpha * np.eye(n_features)
    top = np.column_stack([a, c.reshape(-1, 1)])
    bottom = np.concatenate([c, [0.0]]).reshape(1, -1)
    lhs = np.vstack([top, bottom])
    rhs = np.concatenate([x.T @ y, [d]])
    solution = np.linalg.solve(lhs, rhs)
    return solution[:n_features]


@pytest.mark.reference
def test_simplex_ridge_matches_closed_form_equality_constrained_solution():
    x, y = _fixture(seed=2)
    alpha = 0.5
    n_features = x.shape[1]
    c = np.ones(n_features)
    closed_form = _bordered_kkt_solution(x, y, alpha=alpha, c=c, d=1.0)
    # This fixture/alpha must keep the nonneg constraint non-binding, or the
    # analytic (unconstrained-inequality) formula above is the wrong target.
    assert np.all(closed_form > 0.0), closed_form

    est = _fit_regressor(
        x, y, alpha=alpha, nonneg=True, simplex=True, mean_match=False,
        penalty="ridge", penalty_scale="none", fit_intercept=False,
    )
    assert np.all(est.coef_ > 1e-9), est.coef_  # confirms nonneg indeed didn't bind
    np.testing.assert_allclose(est.coef_, closed_form, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(np.sum(est.coef_), 1.0, rtol=0, atol=1e-6)


@pytest.mark.reference
def test_mean_match_ridge_matches_closed_form_equality_constrained_solution():
    x, y = _fixture(seed=3)
    alpha = 0.5
    c = x.mean(axis=0)
    d = float(y.mean())
    closed_form = _bordered_kkt_solution(x, y, alpha=alpha, c=c, d=d)
    assert np.all(closed_form > 0.0), closed_form

    est = _fit_regressor(
        x, y, alpha=alpha, nonneg=True, simplex=False, mean_match=True,
        penalty="ridge", penalty_scale="none", fit_intercept=False,
    )
    assert np.all(est.coef_ > 1e-9), est.coef_
    np.testing.assert_allclose(est.coef_, closed_form, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(x.mean(axis=0) @ est.coef_), d, rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# Public entry points: confirm they route to the analytically-verified
# estimator without altering its solution (thin end-to-end check).
# ---------------------------------------------------------------------------


@pytest.mark.reference
def test_assemblage_regression_public_entrypoint_matches_direct_estimator_fit():
    x, y = _fixture(seed=4)
    frame = pd.DataFrame(x, columns=[f"c{i}" for i in range(x.shape[1])])
    fit = mf.models.assemblage_regression(frame, pd.Series(y), alpha=0.5)
    direct = _fit_regressor(
        x, y, alpha=0.5, nonneg=True, simplex=True, mean_match=False,
        penalty="ridge", penalty_scale="feature_std", fit_intercept=False,
    )
    np.testing.assert_allclose(
        fit.estimator.coef_, direct.coef_, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(np.sum(fit.estimator.coef_), 1.0, rtol=0, atol=1e-6)


@pytest.mark.reference
def test_supervised_aggregation_public_entrypoint_matches_direct_estimator_fit():
    x, y = _fixture(seed=5)
    frame = pd.DataFrame(x, columns=[f"c{i}" for i in range(x.shape[1])])
    fit = mf.models.supervised_aggregation(frame, pd.Series(y), alpha=0.5, nonneg=False)
    direct = _fit_regressor(
        x, y, alpha=0.5, nonneg=False, simplex=False, mean_match=False,
        penalty="ridge", penalty_scale="feature_std", fit_intercept=False,
    )
    np.testing.assert_allclose(
        fit.estimator.coef_, direct.coef_, rtol=1e-6, atol=1e-8
    )
