"""ADD/FIX: multivariate VAR residual tests (vars::serial.test/normality.test/arch.test)."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def _var(n=400, A=None, innov="normal", seed=0):
    rng=np.random.default_rng(seed); A=np.array([[0.5,0.1],[0.0,0.4]]) if A is None else A
    y=np.zeros((n,2))
    for t in range(1,n):
        if innov=="t": e=rng.standard_t(df=3,size=2)
        elif innov=="arch":
            e=rng.normal(size=2)*np.sqrt(0.2+0.7*(y[t-1]**2))
        else: e=rng.normal(size=2)
        y[t]=A@y[t-1]+e
    return pd.DataFrame(y, columns=["a","b"])

def test_var_serial_test_under_vs_correct_fit():
    A2=np.array([[0.5,0.1],[0.2,0.4]])
    # generate a VAR(2) process
    rng=np.random.default_rng(0); n=500; y=np.zeros((n,2))
    A1=np.array([[0.4,0.0],[0.0,0.3]]); B2=np.array([[0.3,0.0],[0.0,0.3]])
    for t in range(2,n): y[t]=A1@y[t-1]+B2@y[t-2]+rng.normal(size=2)
    df=pd.DataFrame(y,columns=["a","b"])
    under=mf.tests.var_serial_test(df, n_lag=1)   # under-fitted -> residual autocorr
    correct=mf.tests.var_serial_test(df, n_lag=2) # correct -> white
    assert under.decision is True
    assert correct.decision is False

def test_var_normality_test_normal_vs_t():
    assert mf.tests.var_normality_test(_var(innov="normal"), n_lag=1).decision is False
    assert mf.tests.var_normality_test(_var(innov="t", n=600), n_lag=1).decision is True

def test_var_arch_test_homoskedastic_vs_arch():
    assert mf.tests.var_arch_test(_var(innov="normal", n=600), n_lag=1, arch_lags=3).decision is False
    assert mf.tests.var_arch_test(_var(innov="arch", n=600), n_lag=1, arch_lags=3).decision is True
