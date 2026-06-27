"""HAC long-run sd normalises every autocovariance by n.

Regression: the lag-k autocovariance used np.mean over the length-(n-k) lagged
product, dividing by n - k instead of n. That overweights higher lags relative
to gamma0 (which divides by n) and to the standard Newey-West / Bartlett
definition.
"""
import numpy as np

from macroforecast.metrics import _hac_long_run_sd


def _expected_lrv_sd(values, lags):
    d = values - values.mean()
    n = d.size
    g0 = np.mean(d**2)
    lrv = g0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        gk = np.dot(d[k:], d[:-k]) / n  # 1/n normalisation for every lag
        lrv += 2.0 * w * gk
    return float(np.sqrt(max(lrv, 0.0)))


def test_matches_one_over_n_newey_west():
    rng = np.random.default_rng(3)
    values = rng.normal(size=60)
    for lags in (1, 4, 8):
        got = _hac_long_run_sd(values, hac_lags=lags)
        assert abs(got - _expected_lrv_sd(values, lags)) < 1e-12, f"lags={lags}"


def test_differs_from_buggy_one_over_n_minus_k():
    # Confirm the fix actually changed behaviour: the old 1/(n-k) normalisation
    # gives a strictly different (larger-weighted) result for a positively
    # autocorrelated series.
    rng = np.random.default_rng(4)
    e = rng.normal(size=200)
    values = np.empty(200)
    values[0] = e[0]
    for t in range(1, 200):
        values[t] = 0.6 * values[t - 1] + e[t]  # AR(1), positive autocorrelation
    d = values - values.mean()
    n = d.size
    lags = 6
    buggy = d.var(ddof=0)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        buggy += 2.0 * w * np.mean(d[k:] * d[:-k])  # 1/(n-k)
    buggy_sd = np.sqrt(max(buggy, 0.0))
    got = _hac_long_run_sd(values, hac_lags=lags)
    assert abs(got - buggy_sd) > 1e-6
