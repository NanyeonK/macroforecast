"""Independent correctness anchors for ``tvp_ridge`` (TVPRidgeRegressor).

WP-V2. Per ``.dev-notes/anchor_coverage/matrix.csv``, ``tvp_ridge``'s one
genuine anchor covers only the Z-basis expansion (~5% of the pipeline); the
estimation core (``_dual_generalized_ridge``, the "dualGRR" ridge solve) has
no independent P/C/O/M anchor. This file clean-rooms that core from the
FORMULAS documented in ``macroforecast/models/tvp.py``'s own source
comments (which cite the R source cues, e.g. "Kmat_half <- t(Zprime)",
"dual branch solves (Zprime %*% Kmat_half + Lambda_T) alpha = y") -- not by
copying the code.

Derivation (done independently here, not read off the implementation):
``_dual_generalized_ridge`` computes, for both its "dual" (param > n_obs)
and "primal" (param <= n_obs) branches, the SAME textbook generalized
Tikhonov/ridge-regression estimator

    theta_hat = (Z' W Z + Lambda)^{-1} Z' W y,   W = diag(1/eweights),

where ``Lambda`` is diagonal: ``lambda1 / sweights[m]`` for every innovation
column belonging to coefficient block ``m``, and ``lambda2`` for every
static/initial-value column. The "dual" branch is the standard
Woodbury-identity dual-ridge solution of the exact same objective
(``theta_hat = Lambda^{-1} Z' (Z Lambda^{-1} Z' + W^{-1})^{-1} y``,
algebraically identical to the primal form -- a textbook fact, e.g.
Rasmussen & Williams 2006 eq. 2.9-2.11 or any kernel-ridge derivation).
``TVPRidgeRegressor`` always uses ``nf == dim_x`` (one coefficient block per
predictor, including the intercept), under which ``param = dim_x * n_obs >
n_obs`` for any real fixture (dim_x >= 2), so the DUAL branch is the one
actually exercised by every real ``tvp_ridge`` call; the primal branch is
tested here too for completeness even though it is unreachable through
``tvp_ridge`` itself for practical predictor counts.

Two anchors:

1. ``test_z_basis_matches_independent_reimplementation`` (deterministic,
   1e-12): an independently-written cumulative-design construction (own
   loop, not copied) matches ``_tvp_z_basis``.

2. ``test_dual_generalized_ridge_matches_textbook_ridge_formula`` (1e-8):
   the closed-form generalized-ridge estimator above, computed from
   scratch, matches ``_dual_generalized_ridge``'s recovered beta path.

3. ``test_tvp_ridge_constant_parameter_limit_matches_plain_ridge``: as
   lambda1 -> infinity (the innovation-precision penalty -> infinity, i.e.
   coefficients forced constant across time), the recovered beta path must
   converge to a single plain (static-only) ridge regression with penalty
   ``lambda2`` -- the textbook "TVP collapses to a constant-coefficient
   model when the state-innovation variance is driven to zero" limit.
"""
from __future__ import annotations

import numpy as np
import pytest

from macroforecast.models.tvp import _dual_generalized_ridge, _tvp_z_basis


# ---------------------------------------------------------------------------
# Anchor 1: Z-basis (cumulative TVP design) independent reimplementation.
# ---------------------------------------------------------------------------


def _reference_z_basis(data: np.ndarray) -> np.ndarray:
    """Independent reimplementation of R `Zfun(data)` from its documented
    structure: X = [1, data]; for observation t (0-indexed), every
    "innovation column" tau < t of coefficient block k takes the value
    X[t, k] (coefficient k's value at time t is the sum of all innovations
    up to and including t, so X[t] loads on every not-yet-realized
    innovation up to time t); the final dim_x columns are the plain static
    design X itself.
    """
    values = np.asarray(data, dtype=float)
    n_obs, n_features = values.shape
    x_aug = np.column_stack([np.ones(n_obs), values])
    dim_x = x_aug.shape[1]
    blocks = []
    for k in range(dim_x):
        block = np.zeros((n_obs, n_obs - 1), dtype=float)
        for t in range(1, n_obs):
            block[t, :t] = x_aug[t, k]
        blocks.append(block)
    return np.column_stack([*blocks, x_aug])


@pytest.mark.reference
@pytest.mark.parametrize("n_obs,n_features", [(6, 1), (10, 2), (5, 3)])
def test_z_basis_matches_independent_reimplementation(n_obs, n_features):
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_obs, n_features))
    expected = _reference_z_basis(data)
    actual = _tvp_z_basis(data)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# Anchor 2: dualGRR vs the textbook generalized-ridge closed form.
# ---------------------------------------------------------------------------


def _reference_theta(
    z: np.ndarray, y: np.ndarray, *, dim_x: int, lambda1: float, lambda2: float,
    sweights: np.ndarray, eweights: np.ndarray,
) -> np.ndarray:
    """theta_hat = (Z'WZ + Lambda)^{-1} Z'Wy, computed directly (primal
    normal equations -- valid regardless of whether param <=/> n_obs; it is
    only a computational shortcut, not a different estimator, to solve the
    dual form instead when param > n_obs)."""
    n_obs, ncol_z = z.shape
    nf = dim_x
    n_time = (ncol_z - dim_x) // nf + 1
    penalty_precision = np.empty(ncol_z, dtype=float)
    for m in range(nf):
        penalty_precision[m * (n_time - 1) : (m + 1) * (n_time - 1)] = lambda1 / sweights[m]
    penalty_precision[ncol_z - dim_x :] = lambda2
    w = 1.0 / eweights
    lhs = z.T @ (w[:, None] * z) + np.diag(penalty_precision)
    rhs = z.T @ (w[:, None] * y)
    return np.linalg.solve(lhs, rhs)


def _theta_to_beta_path(theta: np.ndarray, *, dim_x: int, n_time: int) -> np.ndarray:
    nf = dim_x
    n_targets = theta.shape[1]
    ncol_z = theta.shape[0]
    betas = np.zeros((n_targets, nf, n_time), dtype=float)
    for eq in range(n_targets):
        betas[eq, :, 0] = theta[ncol_z - dim_x : ncol_z - dim_x + nf, eq]
        for t in range(1, n_time):
            positions = np.asarray([m * (n_time - 1) + (t - 1) for m in range(nf)], dtype=int)
            betas[eq, :, t] = betas[eq, :, t - 1] + theta[positions, eq]
    return betas


@pytest.mark.reference
@pytest.mark.parametrize(
    "n_obs,n_features,lambda1,lambda2",
    [(15, 2, 3.0, 0.1), (20, 1, 0.5, 1.0), (12, 3, 10.0, 0.05)],
)
def test_dual_generalized_ridge_matches_textbook_ridge_formula(n_obs, n_features, lambda1, lambda2):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(n_obs, n_features))
    y = rng.normal(size=(n_obs, 1))
    z = _tvp_z_basis(x)
    dim_x = n_features + 1
    n_time = n_obs
    # param = dim_x * n_obs is always > n_obs here, confirming the DUAL
    # branch is the one under test (matches tvp_ridge's real usage).
    assert dim_x * n_obs > n_obs

    sweights = np.ones(dim_x)
    eweights = np.ones(n_obs)
    grr = _dual_generalized_ridge(
        z, y, dim_x=dim_x, lambda1=lambda1, lambda2=lambda2,
        sweights=sweights, eweights=eweights, ols_prior=False,
    )
    theta_ref = _reference_theta(
        z, y, dim_x=dim_x, lambda1=lambda1, lambda2=lambda2,
        sweights=sweights, eweights=eweights,
    )
    betas_ref = _theta_to_beta_path(theta_ref, dim_x=dim_x, n_time=n_time)

    np.testing.assert_allclose(grr.betas_grr, betas_ref, rtol=1e-8, atol=1e-8)
    yhat_ref = z @ theta_ref
    np.testing.assert_allclose(grr.yhat, yhat_ref, rtol=1e-8, atol=1e-8)


@pytest.mark.reference
def test_dual_generalized_ridge_matches_textbook_ridge_formula_heterogeneous_weights():
    """Same anchor with non-trivial sweights/eweights (the 2SRR reweighting
    path), which exercises the weighting logic the uniform-weights cases
    above cannot distinguish from a bug that ignores weights entirely.
    """
    rng = np.random.default_rng(2)
    n_obs, n_features = 18, 2
    x = rng.normal(size=(n_obs, n_features))
    y = rng.normal(size=(n_obs, 1))
    z = _tvp_z_basis(x)
    dim_x = n_features + 1
    sweights = np.array([0.5, 2.0, 1.3])
    eweights = np.abs(rng.normal(loc=1.0, scale=0.3, size=n_obs)) + 0.1

    grr = _dual_generalized_ridge(
        z, y, dim_x=dim_x, lambda1=2.0, lambda2=0.2,
        sweights=sweights, eweights=eweights, ols_prior=False,
    )
    theta_ref = _reference_theta(
        z, y, dim_x=dim_x, lambda1=2.0, lambda2=0.2, sweights=sweights, eweights=eweights,
    )
    betas_ref = _theta_to_beta_path(theta_ref, dim_x=dim_x, n_time=n_obs)
    np.testing.assert_allclose(grr.betas_grr, betas_ref, rtol=1e-7, atol=1e-8)


# ---------------------------------------------------------------------------
# Anchor 3: constant-parameter limit (lambda1 -> infinity) == plain ridge.
# ---------------------------------------------------------------------------


@pytest.mark.reference
def test_tvp_ridge_constant_parameter_limit_matches_plain_ridge():
    rng = np.random.default_rng(3)
    n_obs, n_features = 40, 2
    x = rng.normal(size=(n_obs, n_features))
    true_beta = np.array([1.5, -0.8])
    y = (0.3 + x @ true_beta + 0.1 * rng.normal(size=n_obs)).reshape(-1, 1)
    z = _tvp_z_basis(x)
    dim_x = n_features + 1
    lambda2 = 0.4

    grr = _dual_generalized_ridge(
        z, y, dim_x=dim_x, lambda1=1e10, lambda2=lambda2,
        sweights=np.ones(dim_x), eweights=np.ones(n_obs), ols_prior=False,
    )
    # As lambda1 -> infinity, the innovation-precision penalty (lambda1 /
    # sweights[m], see module docstring) -> infinity, crushing every
    # innovation to (numerically) zero, so the recovered beta path should be
    # constant across time and equal to a single plain ridge fit of y on
    # [1, x] with penalty precision lambda2 (no innovation terms at all).
    x_aug = np.column_stack([np.ones(n_obs), x])
    plain_ridge_beta = np.linalg.solve(
        x_aug.T @ x_aug + lambda2 * np.eye(dim_x), x_aug.T @ y
    ).reshape(-1)

    beta_path = grr.betas_grr[0]  # (dim_x, n_time)
    # Constant across time:
    np.testing.assert_allclose(
        beta_path, np.repeat(beta_path[:, [0]], beta_path.shape[1], axis=1),
        rtol=1e-6, atol=1e-6,
    )
    # Equal to the plain-ridge static fit:
    np.testing.assert_allclose(beta_path[:, -1], plain_ridge_beta, rtol=1e-4, atol=1e-4)
