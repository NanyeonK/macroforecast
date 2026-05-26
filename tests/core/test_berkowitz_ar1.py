"""Tests for Berkowitz (2001) LR_3 AR(1) component in _density_interval_battery.

Berkowitz, J. (2001). "Testing Density Forecasts, with Applications to Risk
Management." Journal of Business and Economic Statistics, 19(4), 465-474.

The LR_3 test (equation 6) uses df=3 and jointly tests H0: mu=0, sigma^2=1, rho=0
against the AR(1) alternative. This is distinct from the simpler LR_2 (df=2) test
which omits the serial-correlation component.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from macroforecast.core.runtime import _density_interval_battery


def _make_ar1_pit(rho: float, n: int, seed: int) -> np.ndarray:
    """Generate PIT values from a stationary AR(1) z-process.

    z_t = rho * z_{t-1} + sqrt(1 - rho^2) * eps_t,  eps_t ~ N(0,1)
    PIT = Phi(z_t)
    """
    rng = np.random.default_rng(seed)
    sigma_eps = np.sqrt(1 - rho**2)
    z = np.zeros(n)
    z[0] = rng.standard_normal()
    for t in range(1, n):
        z[t] = rho * z[t - 1] + sigma_eps * rng.standard_normal()
    return stats.norm.cdf(z)


def test_berkowitz_ar1_detects_serial_dependence() -> None:
    """AR(1) PIT with rho=0.6 should reject H0 under LR_3 (df=3).

    This is the core Berkowitz (2001) result: serial dependence in the PIT
    sequence cannot be detected by the simpler df=2 test but IS detected by
    the full LR_3 test.
    """
    pit_ar1 = _make_ar1_pit(rho=0.6, n=300, seed=42)
    result = _density_interval_battery(pit_ar1, alpha=0.05)
    berk = result["berkowitz"]

    chi2_crit = stats.chi2.ppf(0.95, df=3)
    assert berk["lr_statistic"] > chi2_crit, (
        f"AR(1) PIT should reject at 5%; "
        f"LR_3={berk['lr_statistic']:.3f}, critical={chi2_crit:.3f}"
    )
    assert berk["p_value"] < 0.05, (
        f"p_value should be < 0.05 for AR(1) PIT; got {berk['p_value']:.4f}"
    )
    assert berk["reject"] is True


def test_berkowitz_iid_normal_pit_accepts() -> None:
    """iid N(0,1) PIT should NOT reject H0 (good density forecast passes).

    Uses a lenient threshold (p > 0.10) because the test is stochastic.
    """
    rng = np.random.default_rng(100)
    n = 500
    z_iid = rng.standard_normal(n)
    pit_iid = stats.norm.cdf(z_iid)

    result = _density_interval_battery(pit_iid, alpha=0.05)
    berk = result["berkowitz"]

    assert berk["p_value"] > 0.10, (
        f"iid N(0,1) PIT should not reject; p_value={berk['p_value']:.4f}"
    )
    assert berk["reject"] is False


def test_berkowitz_rho_reported() -> None:
    """AR(1) coefficient rho and intercept mu must appear in the result dict."""
    rng = np.random.default_rng(0)
    pit = stats.norm.cdf(rng.standard_normal(100))
    result = _density_interval_battery(pit, alpha=0.05)
    berk = result["berkowitz"]

    assert "rho" in berk, (
        "berkowitz output must include AR(1) coefficient 'rho'"
    )
    assert "mu" in berk, (
        "berkowitz output must include AR(1) intercept 'mu'"
    )
    assert "sigma" in berk, (
        "berkowitz output must include noise std 'sigma'"
    )
    # df should be 3 for n >= 4
    assert berk.get("df") == 3, (
        f"berkowitz df should be 3 for n=100; got {berk.get('df')}"
    )


def test_berkowitz_df_is_3() -> None:
    """Explicitly verify that df=3 is used for normal-sized samples (n >= 4)."""
    rng = np.random.default_rng(7)
    pit = stats.norm.cdf(rng.standard_normal(200))
    result = _density_interval_battery(pit, alpha=0.05)
    berk = result["berkowitz"]

    assert berk.get("df") == 3, (
        f"Expected df=3 (LR_3), got df={berk.get('df')}"
    )


def test_berkowitz_known_case_lr3() -> None:
    """Numerical accuracy: LR_3 from _density_interval_battery matches hand computation.

    We generate z from AR(1) with rho=0.6 (seed=42, n=300) and compute the
    expected LR_3 value by the closed-form MLE formula from Berkowitz (2001).
    Tolerance is 0.5 (generous, to allow for floating-point differences in the
    first-observation treatment and sigma^2 biased/unbiased choice).
    """
    rng = np.random.default_rng(42)
    n = 300
    rho_true = 0.6
    sigma_eps = np.sqrt(1 - rho_true**2)
    z = np.zeros(n)
    z[0] = rng.standard_normal()
    for t in range(1, n):
        z[t] = rho_true * z[t - 1] + sigma_eps * rng.standard_normal()

    # Hand-compute MLE estimates
    z_lag = z[:-1]
    z_lead = z[1:]
    denom = float(np.dot(z_lag - z_lag.mean(), z_lag - z_lag.mean()))
    rho_hat = float(
        np.dot(z_lag - z_lag.mean(), z_lead - z_lead.mean()) / denom
    )
    mu_hat = float(z_lead.mean() - rho_hat * z_lag.mean())
    resid = z_lead - mu_hat - rho_hat * z_lag
    sigma2_hat = float(np.mean(resid**2))
    sigma_hat = float(np.sqrt(sigma2_hat))

    # Restricted (H0) and unrestricted (H1) log-likelihoods
    ll_h0 = float(stats.norm.logpdf(z, 0, 1).sum())
    ll_full = float(stats.norm.logpdf(z[0], 0, 1)) + float(
        stats.norm.logpdf(z_lead, mu_hat + rho_hat * z_lag, sigma_hat).sum()
    )
    lr3_expected = max(-2.0 * (ll_h0 - ll_full), 0.0)

    pit = stats.norm.cdf(z)
    result = _density_interval_battery(pit, alpha=0.05)
    berk = result["berkowitz"]

    assert abs(berk["lr_statistic"] - lr3_expected) < 0.5, (
        f"LR_3 mismatch: expected {lr3_expected:.4f}, "
        f"got {berk['lr_statistic']:.4f}"
    )


def test_berkowitz_rho_sign_correct() -> None:
    """Estimated rho should have the same sign as the true AR coefficient."""
    # Positive rho -> rho_hat should be positive
    pit_pos = _make_ar1_pit(rho=0.7, n=500, seed=13)
    result_pos = _density_interval_battery(pit_pos, alpha=0.05)
    assert result_pos["berkowitz"]["rho"] > 0, (
        "rho_hat should be positive for positively autocorrelated PIT"
    )

    # Near-zero rho -> rho_hat should be near zero
    pit_zero = _make_ar1_pit(rho=0.0, n=500, seed=14)
    result_zero = _density_interval_battery(pit_zero, alpha=0.05)
    assert abs(result_zero["berkowitz"]["rho"]) < 0.15, (
        f"rho_hat should be near 0 for iid PIT; got {result_zero['berkowitz']['rho']:.3f}"
    )
