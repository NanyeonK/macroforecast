"""Correctness tests for news_impact_curve (Engle-Ng 1993)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

import macroforecast as mf


def _garch_returns(seed=0, n=800, omega=0.05, alpha=0.12, beta=0.82):
    # proper GARCH(1,1) recursion so the fitted ARCH coefficient is positive
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    h = omega / (1.0 - alpha - beta)
    for t in range(n):
        h = omega + alpha * (r[t - 1] ** 2 if t > 0 else 0.0) + beta * h
        r[t] = np.sqrt(h) * rng.standard_normal()
    return pd.Series(r, index=pd.date_range("2000-01-01", periods=n, freq="D", name="date"))


def test_symmetric_garch_curve_is_symmetric():
    nic = mf.models.news_impact_curve(mf.garch11(_garch_returns()))
    v = nic["variance"]
    assert np.allclose(v, v[::-1], rtol=1e-9)
    # minimum variance at zero shock
    assert abs(nic["shock"][np.argmin(v)]) < 1e-9
    assert nic["variant"] == "garch11"


def test_curve_increases_with_shock_magnitude():
    nic = mf.models.news_impact_curve(mf.garch11(_garch_returns()), span=3.0, n_points=101)
    shock, var = nic["shock"], nic["variance"]
    # variance at the largest |shock| exceeds variance at zero
    assert var[0] > var[len(var) // 2]
    assert var[-1] > var[len(var) // 2]


def test_asymmetric_models_supported():
    r = _garch_returns(seed=2)
    for fn, variant in [(mf.gjr_garch, "gjr"), (mf.egarch, "egarch"), (mf.tgarch, "tgarch")]:
        nic = mf.models.news_impact_curve(fn(r))
        assert nic["variant"] == variant
        assert nic["variance"].shape == nic["shock"].shape
        assert np.all(np.isfinite(nic["variance"]))


def test_reference_variance_and_explicit_shocks():
    nic = mf.models.news_impact_curve(
        mf.garch11(_garch_returns()), shocks=[-2.0, 0.0, 2.0], reference_variance=1.0
    )
    assert nic["reference_variance"] == 1.0
    np.testing.assert_allclose(nic["shock"], [-2.0, 0.0, 2.0])
    # h(eps) = omega + alpha*eps^2 + beta*1.0 ; symmetric endpoints equal
    assert np.isclose(nic["variance"][0], nic["variance"][2])
