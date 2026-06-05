"""Baseline forecasters: naive / seasonal_naive / random_walk_drift (forecast::naive/snaive/rwf)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _series(vals):
    idx = pd.date_range("2000-01-31", periods=len(vals), freq="ME", name="date")
    return pd.Series(np.asarray(vals, dtype=float), index=idx, name="y")


def test_naive_carries_last_value():
    fit = mf.naive(_series([1, 2, 3, 4, 5]))
    X = pd.DataFrame(index=range(3))
    np.testing.assert_allclose(fit.predict(X).to_numpy(), [5.0, 5.0, 5.0])


def test_seasonal_naive_repeats_last_cycle():
    # period 4, last full cycle = [10, 20, 30, 40]
    fit = mf.seasonal_naive(_series([1, 2, 3, 4, 10, 20, 30, 40]), period=4)
    X = pd.DataFrame(index=range(6))
    # step1->10, step2->20, step3->30, step4->40, step5->10, step6->20
    np.testing.assert_allclose(fit.predict(X).to_numpy(), [10, 20, 30, 40, 10, 20])


def test_random_walk_drift_extrapolates_slope():
    # y = [2,4,6,8,10]; drift = (10-2)/(5-1) = 2; path = 10+2k
    fit = mf.random_walk_drift(_series([2, 4, 6, 8, 10]))
    X = pd.DataFrame(index=range(3))
    np.testing.assert_allclose(fit.predict(X).to_numpy(), [12.0, 14.0, 16.0])


def test_drift_zero_when_flat():
    fit = mf.random_walk_drift(_series([7, 7, 7, 7]))
    X = pd.DataFrame(index=range(2))
    np.testing.assert_allclose(fit.predict(X).to_numpy(), [7.0, 7.0])


def test_registry_and_get_model():
    from macroforecast.models.specs import MODEL_SPECS, get_model
    for name in ("naive", "seasonal_naive", "random_walk_drift"):
        assert name in MODEL_SPECS
        assert get_model(name) is not None


def test_end_to_end_runner_naive():
    n = 60
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    panel = pd.DataFrame({"y": np.linspace(1.0, 3.0, n)}, index=idx)
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(horizon=1, step=6),
    )
    features = mf.feature_engineering.feature_spec(
        target="y", predictors=[], lags=None, target_lags=(0, 1)
    )
    result = mf.forecasting.run(
        panel, "naive", window=w, features=features, target="y", horizon=1, save_models=False
    )
    table = result.to_frame()
    assert not table.empty
    assert set(table["model"]) == {"naive"}
    assert np.isfinite(table["prediction"]).all()
