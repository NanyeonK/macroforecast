"""ADD: ndiffs / nsdiffs."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_ndiffs_random_walk_vs_stationary():
    rng = np.random.default_rng(0)
    rw = pd.Series(np.cumsum(rng.normal(size=200)))
    ar = np.zeros(300)
    for t in range(1, 300):
        ar[t] = 0.5 * ar[t-1] + rng.normal()
    for test in ("kpss", "adf"):
        assert mf.data_analysis.ndiffs(rw, test=test) >= 1     # RW needs differencing
        assert mf.data_analysis.ndiffs(pd.Series(ar), test=test) == 0

def test_nsdiffs_seasonal_vs_nonseasonal():
    rng = np.random.default_rng(1); t = np.arange(240)
    seasonal = pd.Series(5.0 * np.sin(2 * np.pi * t / 12) + rng.normal(scale=0.3, size=240))
    flat = pd.Series(rng.normal(size=240))
    assert mf.data_analysis.nsdiffs(seasonal, m=12) == 1       # strong seasonality
    assert mf.data_analysis.nsdiffs(flat, m=12) == 0
