"""ets(model='auto') selects an ETS spec by AICc instead of fixed SES."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_ets_auto_selects_trend_on_trending_series():
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    y = pd.Series(np.arange(60) * 0.7 + 10 + rng.normal(scale=0.3, size=60), index=idx)
    fit = mf.models.ets(y, model="auto")
    assert fit.metadata.get("selection") == "aicc_auto"
    # A clear linear trend must not be left as trendless SES.
    assert fit.metadata["trend"] is not None
    assert np.isfinite(fit.predict(pd.DataFrame(index=idx[:3]))).all()

def test_ets_default_unchanged_is_ses():
    idx = pd.date_range("2000-01-31", periods=40, freq="ME")
    y = pd.Series(np.ones(40) * 5.0, index=idx)
    fit = mf.models.ets(y)
    assert fit.metadata["trend"] is None and fit.metadata["seasonal"] is None
    assert "selection" not in fit.metadata
