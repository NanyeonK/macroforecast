"""Correctness tests for vcov_hc (sandwich::vcovHC HC0-HC3)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _hetero_data(n, rng):
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    e = rng.standard_normal(n) * (0.5 + np.abs(x1))
    y = 0.5 + 1.2 * x1 - 0.8 * x2 + e
    return pd.DataFrame({"x1": x1, "x2": x2}), y


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_matches_statsmodels(cov_type):
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(11)
    X, y = _hetero_data(180, rng)
    res = mf.data_analysis.vcov_hc(X, y, cov_type=cov_type)
    sm_res = sm.OLS(y, sm.add_constant(X.values)).fit(cov_type=cov_type)
    np.testing.assert_allclose(res["estimate"], sm_res.params, rtol=1e-10)
    np.testing.assert_allclose(res["std_error"], sm_res.bse, rtol=1e-8)


def test_hc_ordering_hc3_largest():
    # HC3 inflates more than HC2/HC1/HC0 under leverage.
    rng = np.random.default_rng(2)
    X, y = _hetero_data(120, rng)
    ses = {t: mf.data_analysis.vcov_hc(X, y, cov_type=t)["std_error"][1]
           for t in ("HC0", "HC1", "HC2", "HC3")}
    assert ses["HC3"] >= ses["HC2"] >= ses["HC0"]
    assert ses["HC1"] >= ses["HC0"]


def test_invalid_cov_type():
    rng = np.random.default_rng(0)
    X, y = _hetero_data(50, rng)
    with pytest.raises(ValueError):
        mf.data_analysis.vcov_hc(X, y, cov_type="HC9")
