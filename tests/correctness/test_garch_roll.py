"""Correctness tests for garch_roll rolling volatility/VaR backtest (rugarch::ugarchroll)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

import macroforecast as mf


def _garch_dgp(seed=0, n=600, omega=0.05, alpha=0.1, beta=0.85):
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    h = omega / (1.0 - alpha - beta)
    for t in range(n):
        h = omega + alpha * (r[t - 1] ** 2 if t > 0 else 0.0) + beta * h
        r[t] = np.sqrt(h) * rng.standard_normal()
    return pd.Series(r, index=pd.date_range("2000-01-01", periods=n, freq="D", name="date"))


def test_structure_and_lengths():
    res = mf.models.garch_roll(_garch_dgp(), forecast_length=100, refit_every=25)
    assert res["n_forecasts"] == 100
    for k in ("sigma", "var", "es", "realized", "violation"):
        assert len(res[k]) == 100
    assert np.all(res["sigma"] > 0)
    # lower-tail VaR is below the conditional mean (negative returns)
    assert np.mean(res["var"] < 0) > 0.9


def test_violation_indicator_matches_definition():
    res = mf.models.garch_roll(_garch_dgp(seed=1), forecast_length=80, refit_every=20)
    expected = res["realized"] < res["var"]
    np.testing.assert_array_equal(res["violation"], expected)
    assert res["coverage"]["n_violations"] == int(expected.sum())
    assert np.isclose(res["coverage"]["expected_violations"], 0.05 * 80)


def test_es_not_above_var():
    res = mf.models.garch_roll(_garch_dgp(seed=2), forecast_length=60, refit_every=30)
    # Expected Shortfall is at least as extreme (<=) as VaR in the lower tail
    assert np.all(res["es"] <= res["var"] + 1e-9)


def test_variants_and_validation():
    r = _garch_dgp(seed=3)
    for variant in ("garch11", "gjr_garch", "egarch"):
        res = mf.models.garch_roll(r, variant=variant, forecast_length=50, refit_every=25)
        assert res["variant"] == variant.lower()
    with pytest.raises(ValueError):
        mf.models.garch_roll(r, variant="not_a_model")
    with pytest.raises(ValueError):
        mf.models.garch_roll(r[:30])
