"""SETAR / STAR regime-switching autoregressions: registered, run, and behave."""
import numpy as np
import pandas as pd
from macroforecast.models import get_model, macro_random_forest  # noqa: F401
import macroforecast.models as M


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-01", periods=n, freq="QS")
    y = pd.Series(np.cumsum(rng.normal(size=n)) * 0.1, index=idx, name="y")
    X = pd.DataFrame({f"y_lag{k}": y.shift(k) for k in (1, 2, 3)})
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep]


def test_setar_star_are_registered():
    for m in ("setar", "star"):
        sp = get_model(m)
        assert sp.family == "timeseries"
        assert sp.default_params["n_lag"] == 2


def test_setar_two_regime_prediction_varies_and_is_finite():
    setar = get_model("setar")
    X, y = _series()
    fit = setar(X, y, n_lag=2)
    pred = np.asarray(fit.predict(X.tail(8)), dtype=float)
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()
    assert np.std(pred) > 0  # not a single persisted constant


def test_star_transition_prediction_finite():
    star = get_model("star")
    X, y = _series(seed=1)
    fit = star(X, y, n_lag=2)
    pred = np.asarray(fit.predict(X.tail(8)), dtype=float)
    assert pred.shape == (8,)
    assert np.isfinite(pred).all()


def test_setar_degenerate_falls_back_to_mean():
    # feature matrix with no usable target lags -> fall back to the target mean
    setar = get_model("setar")
    X = pd.DataFrame({"z_lag1": [0.0, 1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0], name="y")
    pred = np.asarray(setar(X, y, n_lag=2).predict(X), dtype=float)
    assert np.allclose(pred, 1.0)
