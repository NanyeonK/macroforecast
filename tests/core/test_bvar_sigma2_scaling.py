"""PR8: BVAR Minnesota σ² scaling fix — TDD test suite.

Karlsson (2013), Handbook of Economic Forecasting, Vol. 2B, Eq. 15.8-15.9.

The correct posterior mean per equation i is:
    beta_i = (V_i^{-1} + Z'Z/sigma2_i)^{-1} (V_i^{-1} m_i + Z'y_i/sigma2_i)

The buggy code before PR8 dropped the /sigma2_i divisor on both likelihood
terms, causing the data to be over-weighted relative to the prior by
a factor of sigma2_i.

The bug is most visible with high-variance data (sigma2 >> 1):
  - Buggy: ZtZ term dominates entirely because Vinv ~ 1/sigma2 but ZtZ
    has no 1/sigma2, making the prior invisible at any lambda1.
  - Correct: both ZtZ/sigma2 and Vinv ~ 1/sigma2 scale together, so
    the prior-to-data ratio is governed by lambda1 as intended.

Key diagnostic test: at lambda1=1.0 and b_AR=1.0 with high-variance data
(sigma2 ~ 50000), BVAR should show visible shrinkage of the own-lag-1
coefficient TOWARD 1.0 relative to OLS. The buggy code shows zero shrinkage.

These tests are written BEFORE the fix. The core failing tests are:
  - TestHighVarianceShrinkage: FAILS pre-fix (no shrinkage when sigma2 >> 1)
  - TestMonotoneConvergence: FAILS pre-fix (no monotone behavior under bug)
  - Other tests pass both before and after (regression guards).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _BayesianVAR


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_panel(T: int = 200, K: int = 3, seed: int = 42) -> pd.DataFrame:
    """Construct a (T, K) random-walk panel used across tests."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, K)).cumsum(axis=0)
    return pd.DataFrame(data, columns=[f"v{i}" for i in range(K)])


def _make_high_var_panel(T: int = 200, K: int = 3, seed: int = 42, scale: float = 50.0) -> pd.DataFrame:
    """Construct a (T, K) panel with sigma2 >> 1 to expose the sigma2 scaling bug.

    With the buggy code, BVAR at intermediate lambda1 is indistinguishable
    from OLS because Vinv ~ 1/sigma2 but ZtZ has no 1/sigma2 compensation.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((T, K)).cumsum(axis=0) * scale
    return pd.DataFrame(data, columns=[f"v{i}" for i in range(K)])


# ---------------------------------------------------------------------------
# Helper: OLS-VAR coefficient matrix for the target equation (eq 0)
# ---------------------------------------------------------------------------

def _ols_coefs_eq0(df: pd.DataFrame, p: int = 1) -> np.ndarray:
    """Return OLS-VAR lag coefficients for equation 0 (variable v0).

    Returns an array of shape (K,) corresponding to the K own-and-cross
    lag-1 coefficients in the design matrix order [v0_lag1, v1_lag1, ...].
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from statsmodels.tsa.vector_ar.var_model import VAR
        res = VAR(df).fit(maxlags=p, trend="c", ic=None)
    # res.coefs has shape (p, K, K): res.coefs[lag, eq, var]
    # coefs[0, 0, :] = lag-1 coefficients for equation 0 (v0)
    return res.coefs[0, 0, :]  # shape (K,)


# ---------------------------------------------------------------------------
# Test 0 — High-variance shrinkage (core bug diagnostic)
# ---------------------------------------------------------------------------

class TestHighVarianceShrinkage:
    """Core bug test: with high-variance data (sigma2 >> 1) and an
    intermediate lambda1 (=1.0), the BVAR should show visible shrinkage
    of the own-lag-1 coefficient toward the b_AR prior mean.

    With the BUGGY code: the Vinv terms scale as 1/sigma2 but ZtZ does
    not, so at lambda1=1 with sigma2=50000 the ZtZ term overwhelms
    Vinv by a factor of 50000, making the prior invisible. Result:
    BVAR ≡ OLS to machine precision.

    With the CORRECT code: ZtZ/sigma2 and Vinv/sigma2 scale together,
    so at lambda1=1 the prior contributes ~ 1/lambda1^2 of the precision.
    Result: own-lag-1 is shifted measurably toward b_AR=1.0.
    """

    def test_intermediate_lambda1_shows_prior_shrinkage_with_high_var_data(self):
        """At lambda1=0.1 and b_AR=1.0 with sigma2~50000, BVAR own-lag-1
        must differ from OLS by at least 0.001 (prior pulls toward 1.0).

        With the BUGGY code: BVAR ≡ OLS to < 1e-5 because the sigma2
        factor makes Vinv vanishingly small relative to ZtZ.
        With the FIXED code: shrinkage is ~0.006, well above 0.001.

        lambda1=0.1 is used (not 1.0) because at lambda1=0.1 the buggy
        vs fixed difference is most stark (2000x ratio in shrinkage).
        """
        df = _make_high_var_panel(T=300, K=3, seed=42, scale=50.0)
        target = df["v0"]
        features = df.drop(columns=["v0"])

        ols_coefs = _ols_coefs_eq0(df, p=1)
        ols_own = ols_coefs[0]  # v0 own-lag-1

        bvar = _BayesianVAR(
            p=1,
            prior="bvar_minnesota",
            lambda1=0.1,
            b_AR=1.0,
            n_draws=0,
        ).fit(features, target)

        assert bvar._results is not None
        bvar_own = bvar._results._B[0, 1]  # eq v0, own-lag-1

        # With the correct formula, the posterior must be pulled toward b_AR=1.0.
        # The OLS estimate should be ~0.937 and BVAR at lambda1=0.1 should
        # show visible shrinkage toward 1.0 (at least 0.001).
        assert abs(bvar_own - ols_own) >= 0.001, (
            f"BVAR own-lag-1={bvar_own:.6f} is NOT measurably shrunk from "
            f"OLS={ols_own:.6f} (diff={abs(bvar_own - ols_own):.2e} < 0.001). "
            "The sigma2 scaling bug is still present."
        )

    def test_shrinkage_direction_toward_b_AR(self):
        """The BVAR own-lag-1 must be pulled TOWARD b_AR=1.0 relative to OLS.

        At lambda1=0.1 with high-variance data:
        - BUGGY: BVAR ≡ OLS (difference ~ 3e-6, no direction).
        - FIXED: BVAR noticeably closer to 1.0 than OLS.

        Uses absolute tolerance 0.001 for the direction gap to be unambiguous.
        """
        df = _make_high_var_panel(T=300, K=3, seed=42, scale=50.0)
        target = df["v0"]
        features = df.drop(columns=["v0"])

        ols_coefs = _ols_coefs_eq0(df, p=1)
        ols_own = ols_coefs[0]

        bvar = _BayesianVAR(
            p=1,
            prior="bvar_minnesota",
            lambda1=0.1,
            b_AR=1.0,
            n_draws=0,
        ).fit(features, target)

        assert bvar._results is not None
        bvar_own = bvar._results._B[0, 1]

        # The BVAR posterior must be strictly closer to 1.0 than OLS,
        # by a margin of at least 0.001.
        dist_bvar = abs(bvar_own - 1.0)
        dist_ols = abs(ols_own - 1.0)
        assert dist_bvar < dist_ols - 0.001, (
            f"BVAR own-lag-1={bvar_own:.6f} is not closer to b_AR=1.0 "
            f"than OLS={ols_own:.6f} by margin 0.001. "
            f"dist_bvar={dist_bvar:.4f}, dist_ols={dist_ols:.4f}. "
            "sigma2 scaling bug may be present."
        )


# ---------------------------------------------------------------------------
# Test 1 — OLS convergence sanity (Chan's suggested test)
# ---------------------------------------------------------------------------

class TestOLSConvergence:
    """As λ₁ → ∞ (very weak prior) the BVAR posterior mean must converge
    to the OLS-VAR estimate.  With the buggy code this does NOT hold at
    the right scale because the σ² factor is missing on the likelihood
    side, making the effective prior weight scale with σ²_i."""

    def test_bvar_converges_to_ols_at_large_lambda1(self):
        """BVAR with λ₁ = 1000 should match OLS-VAR within 10 % rel. err
        for any coefficient larger than 0.05 in absolute value."""
        df = _make_panel()
        target = df["v0"]
        features = df.drop(columns=["v0"])

        ols_coefs = _ols_coefs_eq0(df, p=1)  # shape (K,)

        bvar = _BayesianVAR(
            p=1,
            prior="bvar_minnesota",
            lambda1=1000.0,
            n_draws=0,
        ).fit(features, target)

        # _results.B has shape (K, 1+K*p); row 0 is equation v0.
        # Columns: [intercept, v0_lag1, v1_lag1, v2_lag1]
        assert bvar._results is not None, "_results not populated"
        B_row0 = bvar._results._B[0, :]  # (1 + K*p,) = (4,)

        K = df.shape[1]
        for j in range(K):
            ols_c = ols_coefs[j]
            # Column index in B_row0: 1 + j (skip intercept at 0)
            bvar_c = B_row0[1 + j]
            if abs(ols_c) > 0.05:
                rel_err = abs(bvar_c - ols_c) / abs(ols_c)
                assert rel_err < 0.10, (
                    f"Eq. v0, coef j={j}: BVAR={bvar_c:.4f}, "
                    f"OLS={ols_c:.4f}, rel_err={rel_err:.3f} >= 0.10. "
                    "σ² scaling fix may be missing."
                )

    def test_bvar_posterior_mean_is_not_nan(self):
        """Posterior mean must contain no NaN after the fix."""
        df = _make_panel()
        target = df["v0"]
        features = df.drop(columns=["v0"])

        bvar = _BayesianVAR(
            p=1,
            prior="bvar_minnesota",
            lambda1=0.2,
            n_draws=0,
        ).fit(features, target)

        assert bvar._results is not None
        assert np.all(np.isfinite(bvar._results._B)), (
            "Posterior mean B contains NaN/Inf."
        )


# ---------------------------------------------------------------------------
# Test 2 — Prior dominance at tight λ₁ (regression guard)
# ---------------------------------------------------------------------------

class TestPriorDominance:
    """With very small λ₁ the prior must dominate; the own-lag-1
    coefficient of each equation should be pulled close to b_AR = 1.0.
    This was already correct before the fix (prior term correctly scaled);
    it serves as a non-regression guard."""

    def test_tight_prior_anchors_own_lag1_to_b_AR(self):
        df = _make_panel(T=200, K=3, seed=7)
        target = df["v0"]
        features = df.drop(columns=["v0"])

        bvar = _BayesianVAR(
            p=1,
            prior="bvar_minnesota",
            lambda1=0.001,
            b_AR=1.0,
            n_draws=0,
        ).fit(features, target)

        assert bvar._results is not None
        K = df.shape[1]
        for i in range(K):
            # Own-lag-1 for equation i is at B[i, 1+i] in the multi-eq layout.
            own_lag = bvar._results._B[i, 1 + i]
            assert abs(own_lag - 1.0) < 0.15, (
                f"Tight prior: eq {i} own-lag-1 = {own_lag:.4f}, expected ≈ 1.0"
            )

    def test_tight_prior_closer_to_bAR_than_loose_prior(self):
        """Tight λ₁ must put each own-lag-1 closer to 1 than loose λ₁."""
        df = _make_panel(T=200, K=3, seed=8)
        target = df["v0"]
        features = df.drop(columns=["v0"])

        loose = _BayesianVAR(
            p=1, prior="bvar_minnesota", lambda1=10.0, b_AR=1.0, n_draws=0,
        ).fit(features, target)
        tight = _BayesianVAR(
            p=1, prior="bvar_minnesota", lambda1=0.001, b_AR=1.0, n_draws=0,
        ).fit(features, target)

        assert loose._results is not None and tight._results is not None
        K = df.shape[1]
        for i in range(K):
            idx = 1 + i
            loose_own = loose._results._B[i, idx]
            tight_own = tight._results._B[i, idx]
            assert abs(tight_own - 1.0) <= abs(loose_own - 1.0) + 1e-6, (
                f"Eq {i}: tight {tight_own:.4f} not closer to 1 "
                f"than loose {loose_own:.4f}"
            )


# ---------------------------------------------------------------------------
# Test 3 — Monotone convergence to OLS as λ₁ increases
# ---------------------------------------------------------------------------

class TestMonotoneConvergence:
    """As lambda1 increases, BVAR coefficient distance from OLS should
    decrease monotonically.  With the buggy code and high-variance data,
    the posterior is already essentially OLS at all lambda1 values so
    the distance fails to decrease with increasing lambda1."""

    def test_coefficient_distance_to_ols_decreases_with_lambda1(self):
        """Max absolute coefficient error vs OLS should be non-increasing
        as lambda1 grows through [0.01, 0.1, 1, 10, 100, 500].

        Uses high-variance data to expose the bug: the buggy code shows
        near-zero variation across all lambda values (all errors ~ 0),
        violating the intended monotone prior-to-data trade-off.
        Post-fix the errors decrease monotonically from tight to loose."""
        df = _make_high_var_panel(T=300, K=3, seed=17, scale=30.0)
        target = df["v0"]
        features = df.drop(columns=["v0"])
        ols_coefs = _ols_coefs_eq0(df, p=1)  # shape (K,)

        lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 500.0]
        errors = []
        for lam in lambdas:
            bvar = _BayesianVAR(
                p=1, prior="bvar_minnesota", lambda1=lam, n_draws=0,
            ).fit(features, target)
            assert bvar._results is not None
            B_row0 = bvar._results._B[0, 1:]  # lag coefficients, skip intercept
            max_err = float(np.max(np.abs(B_row0 - ols_coefs)))
            errors.append(max_err)

        # Errors must be weakly decreasing (within a 20% tolerance to allow
        # for numerical noise at very small lambda).
        for k in range(len(errors) - 1):
            assert errors[k + 1] <= errors[k] * 1.20 + 1e-4, (
                f"Non-monotone at lambda1={lambdas[k+1]:.2f}: "
                f"err={errors[k+1]:.5f} > prev_err={errors[k]:.5f}. "
                f"All errors: {errors}"
            )

    def test_large_lambda1_posterior_closer_to_ols_than_small_lambda1(self):
        """lambda1=500 must produce coefficients strictly closer to OLS than
        lambda1=0.01 (coarse version of monotonicity).

        FAILS pre-fix with high-variance data because the buggy code puts
        BVAR at OLS for ALL lambda values, and lambda=0.01 also gives ~OLS.
        PASSES post-fix because small lambda shows strong prior shrinkage.
        """
        df = _make_high_var_panel(T=300, K=3, seed=19, scale=30.0)
        target = df["v0"]
        features = df.drop(columns=["v0"])
        ols_coefs = _ols_coefs_eq0(df, p=1)

        bvar_small = _BayesianVAR(
            p=1, prior="bvar_minnesota", lambda1=0.01, n_draws=0,
        ).fit(features, target)
        bvar_large = _BayesianVAR(
            p=1, prior="bvar_minnesota", lambda1=500.0, n_draws=0,
        ).fit(features, target)

        assert bvar_small._results is not None and bvar_large._results is not None
        err_small = float(np.max(np.abs(
            bvar_small._results._B[0, 1:] - ols_coefs
        )))
        err_large = float(np.max(np.abs(
            bvar_large._results._B[0, 1:] - ols_coefs
        )))
        assert err_large < err_small, (
            f"lambda1=500 err={err_large:.5f} is NOT less than lambda1=0.01 "
            f"err={err_small:.5f}. sigma2 scaling fix may be missing."
        )


# ---------------------------------------------------------------------------
# Test 4 — σ²-scaling of posterior covariance
# ---------------------------------------------------------------------------

class TestPosteriorCovarianceScaling:
    """The posterior covariance must scale correctly with σ²:
    cov_i = (V_i^{-1} + Z'Z/σ²_i)^{-1} · σ²_i.
    After the fix, the posterior covariance stored in
    ``_results.posterior_cov_per_eq[i]`` must be symmetric and PSD."""

    def test_posterior_cov_is_symmetric_and_psd(self):
        df = _make_panel(T=150, K=3, seed=55)
        target = df["v0"]
        features = df.drop(columns=["v0"])

        bvar = _BayesianVAR(
            p=1, prior="bvar_minnesota", lambda1=0.2, n_draws=0,
        ).fit(features, target)

        assert bvar._results is not None
        cov = bvar._results.posterior_cov_per_eq
        assert cov is not None
        K = bvar._results.endog.shape[1]
        for i in range(K):
            cov_i = cov[i]
            # Symmetry
            np.testing.assert_allclose(
                cov_i, cov_i.T, atol=1e-7,
                err_msg=f"posterior_cov_per_eq[{i}] is not symmetric"
            )
            # Positive semi-definite (all eigenvalues >= -tol)
            eigvals = np.linalg.eigvalsh(cov_i)
            assert np.all(eigvals >= -1e-7), (
                f"posterior_cov_per_eq[{i}] has negative eigenvalue: "
                f"{eigvals.min():.2e}"
            )

    def test_all_posterior_means_are_finite(self):
        """No NaN / Inf in any posterior mean coefficient after fix."""
        df = _make_panel(T=150, K=4, seed=99)
        target = df["v0"]
        features = df.drop(columns=["v0"])

        bvar = _BayesianVAR(
            p=1, prior="bvar_minnesota", lambda1=0.5, n_draws=0,
        ).fit(features, target)

        assert bvar._results is not None
        assert np.all(np.isfinite(bvar._results._B)), (
            "Posterior mean _B contains NaN or Inf after fix."
        )
        assert np.all(np.isfinite(bvar._results.sigma_u)), (
            "sigma_u contains NaN or Inf after fix."
        )
