"""ADD: genuine Giacomini-White (2006) conditional predictive ability test."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_gw_no_rejection_for_equal_ability():
    rng = np.random.default_rng(0)
    la = pd.Series(np.abs(rng.normal(size=400)))
    lb = pd.Series(np.abs(rng.normal(size=400)))      # same distribution -> equal ability
    res = mf.tests.giacomini_white_test(la, lb, horizon=1)
    assert res.metadata["df"] == 2                     # constant + lagged dL
    assert res.p_value > 0.05 and res.decision is False

def test_gw_rejects_systematic_difference():
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(size=400))
    la = pd.Series(base + 0.8)                          # a is systematically worse
    lb = pd.Series(base)
    res = mf.tests.giacomini_white_test(la, lb, horizon=1)
    assert res.p_value < 0.01 and res.decision is True
    assert res.alternative == "conditional_equal_predictive_ability"
