"""ADD: first-class jarque_bera_test."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_jarque_bera_normal_vs_nonnormal():
    rng = np.random.default_rng(0)
    normal = mf.tests.jarque_bera_test(pd.Series(rng.normal(size=2000)))
    assert normal.decision is False          # ~normal -> do not reject
    heavy = mf.tests.jarque_bera_test(pd.Series(rng.standard_t(df=2, size=2000)))
    assert heavy.decision is True            # heavy tails -> reject normality
    assert normal.metadata["df"] == 2 and normal.alternative == "not_normal"
