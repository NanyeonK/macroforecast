"""ADF regression arg + Minnesota own-lag prior mean parameter."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf


def test_adf_regression_arg_default_and_override():
    rng = np.random.default_rng(0)
    y = pd.DataFrame({"x": np.arange(120) * 0.05 + rng.normal(scale=1.0, size=120)},
                     index=pd.date_range("2000-01-31", periods=120, freq="ME"))
    default = mf.data_analysis.stationarity_tests(y, test="adf")
    assert default["by_series"]["x"]["adf"]["regression"] == "c"
    ct = mf.data_analysis.stationarity_tests(y, test="adf", adf_regression="ct")
    assert ct["by_series"]["x"]["adf"]["regression"] == "ct"
    # ct includes a trend term -> statistic differs from the c-only spec.
    assert default["by_series"]["x"]["adf"]["statistic"] != ct["by_series"]["x"]["adf"]["statistic"]
    try:
        mf.data_analysis.stationarity_tests(y, test="adf", adf_regression="bad")
        assert False
    except ValueError:
        pass


def test_bvar_own_lag_prior_pulls_coefficient():
    rng = np.random.default_rng(1)
    n = 240
    y = np.zeros(n); z = np.zeros(n); y[0], z[0] = 100.0, 50.0
    for t in range(1, n):
        y[t] = 100 + 0.4 * (y[t - 1] - 100) + rng.normal()
        z[t] = 50 + 0.3 * (z[t - 1] - 50) + rng.normal()
    panel = pd.DataFrame({"y": y, "z": z}, index=pd.date_range("1990-01-31", periods=n, freq="ME"))
    common = dict(target="y", n_lag=1, kappa0=0.05, kappa1=0.5, s0=1.0, nu0=2.0,
                  iter=400, burnin=100, random_state=0)
    fit0 = mf.models.bvar_minnesota(panel, own_lag_prior_mean=0.0, **common)
    fit1 = mf.models.bvar_minnesota(panel, own_lag_prior_mean=1.0, **common)
    own0 = float(fit0.diagnostics["coef_mean"].to_numpy()[0, 0])
    own1 = float(fit1.diagnostics["coef_mean"].to_numpy()[0, 0])
    # A tight prior centred at 1.0 must pull the own-lag coefficient up vs prior 0.0.
    assert own1 > own0
