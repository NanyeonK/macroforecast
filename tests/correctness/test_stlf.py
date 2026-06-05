"""Correctness tests for stlf (STL + forecast; forecast::stlf)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _seasonal_series(n=120, period=12, slope=0.1, amp=3.0, seed=0):
    t = np.arange(n)
    y = 10 + slope * t + amp * np.sin(2 * np.pi * t / period)
    y = y + np.random.default_rng(seed).standard_normal(n) * 0.3
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    return pd.Series(y, index=idx, name="y")


def test_stlf_preserves_seasonality_and_trend():
    s = _seasonal_series()
    fit = mf.stlf(s, period=12)
    X = pd.DataFrame(index=range(12))
    path = fit.predict(X).to_numpy()
    assert path.shape == (12,)
    assert np.all(np.isfinite(path))
    # one seasonal cycle ahead should vary (seasonality retained), not be flat
    assert path.std() > 0.5
    # trend continues upward: mean of next cycle exceeds last observed cycle mean
    assert path.mean() > s.to_numpy()[-12:].mean() - 1.0


def test_registry_and_public():
    from macroforecast.models.specs import MODEL_SPECS, get_model
    assert "stlf" in MODEL_SPECS
    assert get_model("stlf") is not None
    assert hasattr(mf, "stlf")


def test_no_seasonality_fallback_runs():
    # short / non-seasonal series -> reduces to seasonally-adjusted path, no crash
    s = pd.Series(np.linspace(0, 5, 30), index=pd.date_range("2000-01-31", periods=30, freq="ME", name="date"), name="y")
    fit = mf.stlf(s)
    path = fit.predict(pd.DataFrame(index=range(3))).to_numpy()
    assert path.shape == (3,) and np.all(np.isfinite(path))


def test_end_to_end_runner():
    s = _seasonal_series(n=96)
    panel = s.to_frame()
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=48),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )
    features = mf.feature_engineering.feature_spec(target="y", predictors=[], lags=None, target_lags=(0, 1))
    result = mf.forecasting.run(panel, "stlf", window=w, features=features, target="y", horizon=1, save_models=False)
    assert not result.to_frame().empty
