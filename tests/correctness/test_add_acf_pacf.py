"""ADD/FIX: full acf / pacf vectors (stats::acf / pacf)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def _ar1(phi=0.7, n=2000, seed=0):
    rng=np.random.default_rng(seed); x=np.zeros(n)
    for t in range(1,n): x[t]=phi*x[t-1]+rng.normal()
    return pd.Series(x)

def test_acf_ar1_geometric_decay():
    tab=mf.data_analysis.acf(_ar1(0.7), nlags=10)
    assert list(tab.columns)==["lag","acf","lower","upper"]
    assert tab.loc[tab["lag"]==0,"acf"].iloc[0]==1.0
    assert abs(tab.loc[tab["lag"]==1,"acf"].iloc[0]-0.7) < 0.05      # acf[1] ~ phi
    assert abs(tab.loc[tab["lag"]==2,"acf"].iloc[0]-0.49) < 0.07     # acf[2] ~ phi^2

def test_pacf_ar1_cuts_off_after_lag1():
    tab=mf.data_analysis.pacf(_ar1(0.7, seed=1), nlags=10)
    assert abs(tab.loc[tab["lag"]==1,"pacf"].iloc[0]-0.7) < 0.05     # pacf[1] ~ phi
    # pacf[2..] ~ 0 (within band) for AR(1)
    later=tab[tab["lag"]>=2]
    assert (later["pacf"].abs() < 0.1).all()
