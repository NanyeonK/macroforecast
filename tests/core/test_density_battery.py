"""PR1: Kupiec POF LR formula fix -- unit tests exposing and verifying the bug fix.

Reference: Kupiec (1995). "Techniques for Verifying the Accuracy of Risk Measurement
Models." Journal of Derivatives, 3(2), 73-84.

Correct formula: LR_uc = 2[x log(p_hat/alpha) + (n-x) log((1-p_hat)/(1-alpha))]

where:
  n     = total number of evaluation periods
  x     = number of violations (hits), hit_t = 1 if PIT_t < alpha
  p_hat = x / n (observed violation rate)
  alpha = nominal coverage level (e.g. 0.05 for 95% VaR)

LR_uc is ALWAYS >= 0 because it is twice the log-likelihood ratio of the
unrestricted (p = p_hat) vs restricted (p = alpha) binomial model.
"""
from __future__ import annotations

import numpy as np
import pytest

from macroforecast.core.runtime import _density_interval_battery


# ---------------------------------------------------------------------------
# Test 1: LR must be non-negative — exposes the current bug.
# Current (buggy) code returns LR ~ -27.15 for this case; correct is ~2.30.
# ---------------------------------------------------------------------------

def test_kupiec_pof_lr_is_positive() -> None:
    """Kupiec LR must be >= 0. The buggy formula produces a negative value."""
    rng = np.random.default_rng(0)
    # Construct PIT series with 7.5% hit rate at alpha=0.05.
    # 200 observations, 15 hits (p_hat = 0.075 > alpha = 0.05).
    n = 200
    pit = np.full(n, 0.10)   # all above alpha=0.05 by default
    hit_indices = rng.choice(n, size=15, replace=False)
    pit[hit_indices] = 0.02  # set 15 observations below alpha
    pit = rng.permutation(pit)

    result = _density_interval_battery(pit, alpha=0.05)

    lr = result["kupiec_pof"]["lr_statistic"]
    assert lr >= 0.0, (
        f"Kupiec LR must be >= 0 (chi-squared statistic), got {lr}. "
        "Bug: current formula computes entropy difference, not log-likelihood ratio."
    )
    # Additionally verify proximity to the correct analytical value ~2.297.
    assert abs(lr - 2.297) < 0.05, (
        f"Expected LR ~ 2.297 (n=200, x=15, alpha=0.05), got {lr}"
    )
    # p-value should be around 0.13 (fail to reject at 5% level).
    p_val = result["kupiec_pof"]["p_value"]
    assert 0.10 < p_val < 0.16, (
        f"Expected p-value ~ 0.13, got {p_val}"
    )
    assert result["kupiec_pof"]["reject"] is False, (
        "Should fail to reject H0 at alpha=0.05 when violation rate is 7.5%"
    )


# ---------------------------------------------------------------------------
# Test 2: Numerical accuracy against hand-computed value.
# Kupiec (1995) Table 1 region: n=252, alpha=0.01, x=3 (p_hat ~0.0119).
# ---------------------------------------------------------------------------

def test_kupiec_pof_lr_known_value() -> None:
    """LR matches hand-computed value to machine precision."""
    n, alpha_val, x = 252, 0.01, 3
    p_hat = x / n
    # Compute expected value analytically.
    expected_lr = 2.0 * (
        x * np.log(p_hat / alpha_val)
        + (n - x) * np.log((1.0 - p_hat) / (1.0 - alpha_val))
    )

    pit = np.full(n, 0.05)    # all above alpha=0.01
    pit[:x] = 0.005           # exactly x hits below alpha

    result = _density_interval_battery(pit, alpha=alpha_val)

    lr = result["kupiec_pof"]["lr_statistic"]
    assert abs(lr - expected_lr) < 1e-6, (
        f"Expected LR={expected_lr:.6f}, got {lr:.6f}"
    )
    # p_hat ~ 0.0119 vs alpha=0.01 is a small deviation — should not reject.
    assert result["kupiec_pof"]["p_value"] > 0.50, (
        f"Should fail to reject H0, got p={result['kupiec_pof']['p_value']}"
    )


# ---------------------------------------------------------------------------
# Test 3: Rejection at very high violation rate.
# n=200, alpha=0.01, x=10 (p_hat=0.05): LR should be large, p < 0.001.
# ---------------------------------------------------------------------------

def test_kupiec_pof_reject_high_violation_rate() -> None:
    """Strong rejection when violation rate far exceeds nominal alpha."""
    n, alpha_val, x = 200, 0.01, 10
    pit = np.full(n, 0.50)
    pit[:x] = 0.005   # 10 hits at alpha=0.01 -> p_hat = 0.05, 5x the nominal rate

    result = _density_interval_battery(pit, alpha=alpha_val)

    lr = result["kupiec_pof"]["lr_statistic"]
    assert lr > 10.0, (
        f"Expected LR > 10 for gross violation (p_hat=0.05 vs alpha=0.01), got {lr}"
    )
    assert result["kupiec_pof"]["p_value"] < 0.01, (
        f"Expected p < 0.01, got {result['kupiec_pof']['p_value']}"
    )
    assert result["kupiec_pof"]["reject"] is True, (
        "Should reject H0 at alpha=0.01 when violation rate is 5x nominal"
    )
