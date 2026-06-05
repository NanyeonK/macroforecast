"""Regression test for TS-BVAR-04.

The BVAR coefficient summary 'se' must report the posterior standard deviation
of each coefficient, not the Monte-Carlo standard error of the posterior mean
(posterior_sd / sqrt(n_draws)). The MCSE is exposed separately as 'mcse'.
"""
from __future__ import annotations

import numpy as np

from macroforecast.models.timeseries import _favar_bvar_draws


def test_bvar_summary_se_is_posterior_sd_not_mcse():
    rng = np.random.default_rng(0)
    # Small stationary 2-var VAR-ish panel.
    n = 200
    values = np.zeros((n, 2))
    for t in range(1, n):
        values[t, 0] = 0.5 * values[t - 1, 0] + rng.normal(scale=1.0)
        values[t, 1] = 0.4 * values[t - 1, 1] + rng.normal(scale=1.0)

    draws = _favar_bvar_draws(
        values, n_lag=1, prior="minnesota", b0=0.0, vb0=10.0, nu0=2.0, s0=1.0,
        kappa0=1.0, kappa1=0.5, n_iter=400, burnin=100, random_state=0,
    )
    coef = draws["coef_draws"]
    summary = draws["summary"]
    n_draws = coef.shape[0]

    expected_sd = coef.std(axis=0, ddof=1)
    np.testing.assert_allclose(summary["se"], expected_sd, rtol=1e-9, atol=1e-12)
    assert "mcse" in summary
    np.testing.assert_allclose(
        summary["mcse"], expected_sd / np.sqrt(n_draws), rtol=1e-9, atol=1e-12
    )
    # se must be materially larger than mcse for a non-trivial number of draws.
    assert np.nanmedian(summary["se"]) > 5.0 * np.nanmedian(summary["mcse"])
