"""Independent correctness anchors for ``favar``.

WP-V2. Per ``.dev-notes/anchor_coverage/summary.md``, ``favar`` was the
"largest body of novel hand-rolled math in the package" (BGM/BBE factor
identification, conjugate loading Gibbs draws, OLS-SVD rotation) verified
only as "doesn't crash, finite output" -- no known-factor DGP oracle existed,
unlike its sibling ``far``.

Two anchors:

1. ``test_favar_extr_pc_matches_independent_pca_reference`` (deterministic):
   ``_favar_extr_pc`` is, per its own source comment, a port of
   ``FAVAR/R/ExtrPC.R`` -- factors/loadings from an eigendecomposition of
   ``X'X``. This test computes the SAME factors via a genuinely different
   numerical route (SVD of ``X`` directly, not eigendecomposition of
   ``X'X``) and checks they agree up to the unavoidable per-factor sign
   ambiguity of any eigen/singular decomposition.

2. ``test_favar_near_noiseless_dgp_oracle_recovers_forecast`` (Monte Carlo /
   tight bound): data generated from a known low-rank rotating state (same
   construction style as ``tests/forecasting/test_far_policy_oracle.py``,
   which anchors ``far``) plus a small (1%) idiosyncratic noise term, so the
   h-ahead target is very close to an exact linear function of the
   origin-time state. An EXACTLY noiseless version of this DGP was tried
   first and triggered two separate, already-diagnosed pathologies (BGM
   factor-purge collapse on exactly-rank-deficient data, and a Gibbs-sampler
   prior-fragility bug now xfailed in
   ``test_bvar_minnesota_niw_anchors.py``) rather than exercising the thing
   this test is meant to anchor -- see ``_rotating_state_dgp``'s docstring
   for the full ablation. With small noise and an explicit non-degenerate
   ``varprior``, recovery is tight (~<1% of the target's in-sample range)
   and clearly beats a naive last-value forecast.

A live R cross-check of the (fully deterministic, non-Bayesian) factor-
identification helpers -- ``_favar_extr_pc``/``_favar_facrot``/
``_favar_olssvd``/``_favar_bgm`` -- against ``FAVAR:::ExtrPC``/``facrot``/
``olssvd``/``BGM`` lives in ``tests/parity/test_favar_r_parity.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.models.timeseries import _favar_extr_pc


# ---------------------------------------------------------------------------
# Anchor 1: factor extraction vs an independently-computed SVD-based PCA.
# ---------------------------------------------------------------------------


def _svd_pca_reference(values: np.ndarray, n_factors: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA factors/loadings via direct SVD of X (not eigh of X'X).

    Reproduces the FAVAR/ExtrPC.R normalization independently: for
    ``X = U S V'``, ``lam = sqrt(N) * V[:, :K]`` and ``fac = X @ lam / N``
    equals ``S[:K] * U[:, :K] / sqrt(N)`` (since ``X @ V[:, i] = S[i] * U[:, i]``).
    Sign of each singular vector pair is arbitrary; that is handled by the
    caller.
    """
    n_obs, n_series = values.shape
    u, s, vt = np.linalg.svd(values, full_matrices=False)
    factors = (u[:, :n_factors] * s[:n_factors]) / np.sqrt(n_series)
    loadings = np.sqrt(n_series) * vt[:n_factors, :].T
    return factors, loadings


def _align_signs(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Flip each column of `candidate` to match `reference`'s sign convention.

    Eigen/singular decompositions fix each component only up to sign. Uses
    the sign of the largest-magnitude entry per column as the convention.
    """
    aligned = candidate.copy()
    for col in range(candidate.shape[1]):
        idx = np.argmax(np.abs(reference[:, col]))
        if np.sign(reference[idx, col]) != np.sign(candidate[idx, col]):
            aligned[:, col] *= -1.0
    return aligned


@pytest.mark.reference
def test_favar_extr_pc_matches_independent_pca_reference():
    rng = np.random.default_rng(2026)
    n_obs, n_series, n_factors = 150, 8, 3
    latent = rng.normal(size=(n_obs, n_factors))
    loadings_true = rng.normal(size=(n_series, n_factors))
    values = latent @ loadings_true.T + 0.05 * rng.normal(size=(n_obs, n_series))
    values = values - values.mean(axis=0)

    factors_impl, loadings_impl = _favar_extr_pc(values, n_factors)
    factors_ref, loadings_ref = _svd_pca_reference(values, n_factors)

    factors_ref_aligned = _align_signs(factors_impl, factors_ref)
    loadings_ref_aligned = _align_signs(loadings_impl, loadings_ref)

    np.testing.assert_allclose(factors_impl, factors_ref_aligned, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(loadings_impl, loadings_ref_aligned, rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Anchor 2: noiseless known-factor DGP oracle through the full favar() pipeline.
# ---------------------------------------------------------------------------


def _rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _rotating_state_dgp(n_obs: int = 260, seed: int = 11, idio_noise: float = 0.01):
    """A bounded (non-decaying) 2-D rotating state, exactly forecastable in
    principle: state[t+1] = A @ state[t] for a fixed rotation A. Eight
    predictors and the target both load linearly on the state, plus a SMALL
    (1% relative) idiosyncratic noise term.

    The noise is deliberately not exactly zero. Two independent, already-
    diagnosed pathologies make an EXACTLY noiseless version of this DGP a
    poor discriminator for this specific model (both confirmed by direct
    ablation, not assumed):

    1. With X exactly rank-deficient (no idiosyncratic noise), the BGM
       iterative factor-purge (``_favar_bgm``) can itself collapse to a
       rank-deficient factor solution (observed: a genuine second factor of
       variance ~1e-32 recovered as ~1e-16 noise) -- a property of the BGM
       identification scheme operating on exactly-collinear data, not a
       favar-specific bug (R FAVAR's own BGM.R has the identical fixed-point
       iteration and would plausibly do the same).
    2. Independently of (1), a near-singular residual covariance in the VAR-
       on-(factors, target) step interacts badly with favar()'s DEFAULT
       diffuse prior (``s0=0.0``) -- see the dedicated xfail regression test
       ``test_niw_default_s0_is_numerically_stable_on_near_singular_residual_covariance``
       in ``test_bvar_minnesota_niw_anchors.py`` for the isolated root cause.
       This test therefore passes an explicit, non-degenerate ``varprior``
       (``s0=1.0``) to avoid re-discovering that already-diagnosed, separately
       xfailed bug on every run.

    With 1% idiosyncratic noise and an explicit non-degenerate varprior,
    neither pathology triggers and recovery is tight (verified empirically
    to ~0.2% of the target's in-sample range -- see the tolerance below).
    """
    a = _rotation(np.pi / 9)
    state: np.ndarray = np.zeros((n_obs, 2), dtype=float)
    state[0] = [1.0, 0.4]
    for t in range(1, n_obs):
        state[t] = a @ state[t - 1]
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(8, 2))
    predictors = state @ loadings.T + idio_noise * rng.normal(size=(n_obs, 8))
    target = state @ np.array([0.8, -0.6]) + idio_noise * rng.normal(size=n_obs)
    idx = pd.date_range("1990-01-31", periods=n_obs, freq="ME")
    x = pd.DataFrame({f"S{i}": predictors[:, i] for i in range(8)}, index=idx)
    y = pd.Series(target, index=idx, name="Y")
    return x, y, a, state


@pytest.mark.reference
@pytest.mark.slow
def test_favar_near_noiseless_dgp_oracle_recovers_forecast():
    x, y, a, state = _rotating_state_dgp()
    train_x, train_y = x.iloc[:-1], y.iloc[:-1]

    fit = mf.models.favar(
        train_x,
        train_y,
        n_factors=2,
        n_lag=1,
        fctmethod="BGM",
        varprior={"b0": 0.0, "vb0": 0.0, "nu0": 0.0, "s0": 1.0},
        nburn=1500,
        nrep=3000,
        standardize=True,
        random_state=20260704,
    )
    pred = float(np.asarray(fit.predict(x.iloc[[-1]])).reshape(-1)[0])
    true_next = float(y.iloc[-1])
    naive_last = float(train_y.iloc[-1])

    # Bayesian loading + BVAR Gibbs steps still smooth even near-noiseless
    # data toward their (proper, non-degenerate) priors, so exact/machine-
    # precision recovery is not expected here the way it is for `far` (a
    # single deterministic PCA+OLS step, no Bayesian shrinkage anywhere).
    # 3% of the target's in-sample range is a tight-but-not-arbitrary band
    # -- empirically the actual error is <1% under this fixture -- and it is
    # far smaller than the naive-forecast error (checked below).
    target_range = float(train_y.max() - train_y.min())
    tolerance = 0.03 * target_range
    naive_error = abs(naive_last - true_next)

    assert abs(pred - true_next) < tolerance, (pred, true_next, tolerance)
    # Discriminating check: a correct FAVAR must beat the naive last-value
    # forecast by a wide margin on this rotating-state DGP (the naive
    # forecast ignores the known rotation entirely).
    assert abs(pred - true_next) < 0.3 * naive_error, (pred, true_next, naive_error)
