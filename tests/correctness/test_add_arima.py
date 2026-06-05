"""ADD: arima + auto_arima."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

def test_arima_fits_and_forecasts():
    rng = np.random.default_rng(0)
    y = pd.Series(np.cumsum(rng.normal(size=120)),
                  index=pd.date_range("2000-01-31", periods=120, freq="ME"))
    fit = mf.models.arima(y, order=(1, 1, 1))
    pred = fit.predict(pd.DataFrame(index=y.index[:5]))
    assert np.asarray(pred).shape[0] == 5 and np.all(np.isfinite(np.asarray(pred)))

def test_auto_arima_selects_d_by_kpss():
    rng = np.random.default_rng(1)
    rw = pd.Series(np.cumsum(rng.normal(size=160)),
                   index=pd.date_range("2000-01-31", periods=160, freq="ME"))
    fit = mf.models.auto_arima(rw, max_p=2, max_q=2)
    # a random walk needs differencing -> d>=1
    assert fit.metadata["order"][1] >= 1
    assert fit.metadata["selection"]["d_kpss"] >= 1

    # a stationary AR(1) needs no differencing -> d==0
    x = np.zeros(200)
    for t in range(1, 200):
        x[t] = 0.5 * x[t-1] + rng.normal()
    ar = pd.Series(x, index=pd.date_range("2000-01-31", periods=200, freq="ME"))
    fit2 = mf.models.auto_arima(ar, max_p=2, max_q=2)
    assert fit2.metadata["order"][1] == 0
