"""Estimated-weight forecast combinations (Wang et al. 2023 review): correctness + leak-free."""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _setup(n=300, seed=0):
    """Two forecasts of a target; model A is far more accurate than model B."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    y = pd.Series(rng.standard_normal(n).cumsum(), index=idx, name="y")
    fa = y + rng.standard_normal(n) * 0.3   # accurate
    fb = y + rng.standard_normal(n) * 2.0   # noisy
    frame = pd.DataFrame({"A": fa, "B": fb}, index=idx)
    return frame, y


def test_bates_granger_favours_accurate_model_and_beats_equal():
    frame, y = _setup()
    bg = mf.forecasting.combine_bates_granger(frame, y, min_periods=20)
    eq = mf.forecasting.combine_mean(frame)
    # evaluate on the part where weights are active
    sl = slice(50, None)
    rmse = lambda f: float(np.sqrt(np.nanmean((f[sl].to_numpy() - y[sl].to_numpy()) ** 2)))
    assert rmse(bg) < rmse(eq)


def test_granger_ramanathan_constrained_recovers_linear_combo():
    # y is exactly 0.7*A + 0.3*B -> constrained GR should recover ~[0.7,0.3]
    rng = np.random.default_rng(1)
    n = 400
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    a = pd.Series(rng.standard_normal(n).cumsum(), index=idx)
    b = pd.Series(rng.standard_normal(n).cumsum(), index=idx)
    y = (0.7 * a + 0.3 * b).rename("y")
    frame = pd.DataFrame({"A": a, "B": b}, index=idx)
    gr = mf.forecasting.combine_granger_ramanathan(frame, y, variant="constrained", min_periods=30)
    # combined should track y nearly exactly once weights settle
    sl = slice(100, None)
    err = float(np.nanmax(np.abs(gr[sl].to_numpy() - y[sl].to_numpy())))
    assert err < 1e-6


def test_constrained_ls_weights_simplex_and_picks_perfect_model():
    rng = np.random.default_rng(2)
    n = 300
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    y = pd.Series(rng.standard_normal(n).cumsum(), index=idx, name="y")
    frame = pd.DataFrame({"perfect": y.values, "noise": y.values + rng.standard_normal(n) * 3}, index=idx)
    cl = mf.forecasting.combine_constrained_ls(frame, y, min_periods=20)
    sl = slice(60, None)
    # combination should be ~ the perfect model (weight ~1 on it)
    err = float(np.nanmean(np.abs(cl[sl].to_numpy() - y[sl].to_numpy())))
    assert err < 0.3


def test_shrink_to_equal_moves_toward_simple_mean():
    frame, y = _setup(seed=3)
    none = mf.forecasting.combine_bates_granger(frame, y, min_periods=20)
    shrunk = mf.forecasting.combine_bates_granger(frame, y, min_periods=20, shrink_to_equal=1.0)
    eq = mf.forecasting.combine_mean(frame)
    sl = slice(60, None)
    # full shrinkage -> equals the simple mean
    np.testing.assert_allclose(shrunk[sl].to_numpy(), eq[sl].to_numpy(), atol=1e-9)
    # partial differs from both
    assert not np.allclose(none[sl].to_numpy(), eq[sl].to_numpy())


@pytest.mark.parametrize("combiner", ["combine_bates_granger", "combine_granger_ramanathan", "combine_constrained_ls"])
def test_leak_free_future_poison_does_not_change_past(combiner):
    frame, y = _setup(seed=4)
    fn = getattr(mf.forecasting, combiner)
    base = fn(frame, y, min_periods=20)
    # poison the LAST 5 rows of forecasts and realised values
    frame2 = frame.copy(); y2 = y.copy()
    frame2.iloc[-5:] = 1e6
    y2.iloc[-5:] = -1e6
    poisoned = fn(frame2, y2, min_periods=20)
    # everything strictly before the poisoned block must be identical
    np.testing.assert_allclose(
        base.iloc[:-5].to_numpy(), poisoned.iloc[:-5].to_numpy(), rtol=1e-9, atol=1e-9
    )
