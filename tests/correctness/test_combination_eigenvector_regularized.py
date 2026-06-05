"""Batch B combiners: eigenvector (PC) and regularized (ridge/lasso)."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _setup(n=300, seed=0, k=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    y = pd.Series(rng.standard_normal(n).cumsum(), index=idx, name="y")
    cols = {}
    for j in range(k):
        cols[f"M{j}"] = y + rng.standard_normal(n) * (0.3 + j)
    return pd.DataFrame(cols, index=idx), y


def test_eigenvector_runs_and_beats_worst_model():
    frame, y = _setup(k=3)
    ev = mf.forecasting.combine_eigenvector(frame, y, min_periods=20)
    sl = slice(60, None)
    rmse = lambda f: float(np.sqrt(np.nanmean((np.asarray(f)[sl] - y.to_numpy()[sl]) ** 2)))
    worst = max(rmse(frame[c].to_numpy()) for c in frame.columns)
    assert rmse(ev.to_numpy()) < worst
    assert np.all(np.isfinite(ev.iloc[60:].to_numpy()))


def test_regularized_ridge_and_lasso_run():
    frame, y = _setup(k=5)
    for penalty in ("ridge", "lasso"):
        out = mf.forecasting.combine_regularized(frame, y, penalty=penalty, alpha=0.5, min_periods=30)
        assert np.all(np.isfinite(out.iloc[60:].to_numpy()))


def test_regularized_invalid_penalty():
    frame, y = _setup()
    with pytest.raises(ValueError):
        mf.forecasting.combine_regularized(frame, y, penalty="elastic_banana", min_periods=20)


@pytest.mark.parametrize("combiner", ["combine_eigenvector", "combine_regularized"])
def test_leak_free(combiner):
    frame, y = _setup(seed=7, k=3)
    fn = getattr(mf.forecasting, combiner)
    base = fn(frame, y, min_periods=20)
    f2, y2 = frame.copy(), y.copy()
    f2.iloc[-5:] = 1e6
    y2.iloc[-5:] = -1e6
    poisoned = fn(f2, y2, min_periods=20)
    np.testing.assert_allclose(base.iloc[:-5].to_numpy(), poisoned.iloc[:-5].to_numpy(), rtol=1e-9, atol=1e-9)
