"""The ADD models are first-class in the registry / forecasting pipeline."""
from __future__ import annotations
import numpy as np, pandas as pd
import macroforecast as mf

NEW = ["arima", "auto_arima", "gjr_garch", "tgarch"]

def test_new_models_registered_and_top_level_exported():
    registered = set(mf.models.list_model_specs()["name"])
    for m in NEW:
        assert m in registered, m
        assert mf.models.get_model(m) is not None
        assert hasattr(mf, m)                 # top-level lazy export (mf.arima, ...)

def test_arima_fits_via_registry():
    y = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=120)),
                  index=pd.date_range("2000-01-31", periods=120, freq="ME"))
    X = pd.DataFrame({"__o__": np.arange(120.0)}, index=y.index)
    fit = mf.models.get_model("arima", params={"order": (1, 1, 1)})(X, y)
    assert isinstance(fit, mf.models.ModelFit)
    assert np.isfinite(np.asarray(fit.predict(X.iloc[:3])).reshape(-1)).all()

def test_gjr_garch_fits_via_registry_like_egarch():
    r = pd.Series(np.random.default_rng(1).normal(scale=1.5, size=400),
                  index=pd.date_range("2000-01-31", periods=400, freq="ME"))
    X = pd.DataFrame({"ret": r.values}, index=r.index)
    for m in ("egarch", "gjr_garch", "tgarch"):
        fit = mf.models.get_model(m)(X, r)
        v = fit.estimator.predict_variance(horizon=2)
        assert v.shape == (2,) and np.all(np.isfinite(v)) and np.all(v > 0)
