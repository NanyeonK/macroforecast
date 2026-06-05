"""ADD: first-class adf_test / kpss_test."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def _rw(n=200, seed=0):
    return pd.Series(np.cumsum(np.random.default_rng(seed).normal(size=n)))
def _wn(n=200, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(size=n))

def test_adf_test_unit_root_vs_stationary():
    rw = mf.data_analysis.adf_test(_rw())
    wn = mf.data_analysis.adf_test(_wn())
    assert rw["reject_unit_root"] is False   # random walk: cannot reject unit root
    assert wn["reject_unit_root"] is True    # white noise: reject unit root
    assert wn["regression"] == "c" and "p_value" in wn
    try:
        mf.data_analysis.adf_test(_wn(), regression="bad"); assert False
    except ValueError: pass

def test_kpss_test_level_vs_unit_root():
    wn = mf.data_analysis.kpss_test(_wn())
    rw = mf.data_analysis.kpss_test(_rw())
    assert wn["reject_stationarity"] is False  # white noise: stationary
    assert rw["reject_stationarity"] is True   # random walk: reject stationarity
