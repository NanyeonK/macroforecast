"""Regression test for TS-BVAR-01.

The standalone bvar_minnesota / bvar_normal_inverse_wishart estimators fit a
no-intercept VAR. On a non-demeaned panel of stationary series with non-zero
unconditional means this forces the own-lag coefficient toward a near-unit root
and the multi-step forecast cannot revert to the true mean. After demeaning
inside the estimator (and adding the mean back in predict), a long-horizon
forecast of a stationary mean-reverting series must approach its unconditional
mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def _stationary_panel(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.zeros(n); z = np.zeros(n)
    y[0], z[0] = 100.0, 50.0
    for t in range(1, n):
        y[t] = 100.0 + 0.5 * (y[t - 1] - 100.0) + rng.normal(scale=1.0)
        z[t] = 50.0 + 0.4 * (z[t - 1] - 50.0) + rng.normal(scale=1.0)
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    return pd.DataFrame({"y": y, "z": z}, index=idx)


def test_bvar_minnesota_long_horizon_reverts_to_mean():
    panel = _stationary_panel()
    fit = mf.models.bvar_minnesota(
        panel, target="y", n_lag=1, kappa0=10.0, kappa1=0.5,
        s0=1.0, nu0=2.0, iter=400, burnin=100, random_state=0,
    )
    # Own-lag coefficient must recover the true AR(1) value ~0.5; a no-intercept
    # fit on the non-demeaned panel inflates it toward a near-unit root (~0.9) to
    # absorb the non-zero mean.
    own_lag = float(fit.diagnostics["coef_mean"].to_numpy()[0, 0])
    assert 0.3 < own_lag < 0.7, own_lag

    # Long-horizon forecast must approach the true unconditional mean (~100),
    # not stall at / drift away from the last observed value.
    horizon = 36
    future_idx = pd.date_range("2016-09-30", periods=horizon, freq="ME")
    pred = np.asarray(fit.predict(pd.DataFrame(index=future_idx))).reshape(-1)
    true_mean = float(panel["y"].mean())
    assert abs(pred[-1] - true_mean) < 1.5, (pred[-1], true_mean)
