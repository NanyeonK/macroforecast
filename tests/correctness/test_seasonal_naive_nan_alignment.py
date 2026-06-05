"""BUG-1: seasonal_naive must index by absolute season position, NaN-robust (forecast::snaive)."""
import numpy as np
import pandas as pd

import macroforecast as mf


def _predict(series, period, steps):
    fit = mf.seasonal_naive(series, period=period)
    return fit.predict(pd.DataFrame(index=range(steps))).to_numpy()


def test_nan_in_final_cycle_keeps_season_alignment():
    # period 4; NaN at the season-3 slot of the final cycle.
    s = pd.Series([10, 20, 30, 40, 11, 21, 31, 41, 12, 22, np.nan, 42])
    # R snaive: each season carries its last OBSERVED value.
    # slots: 0->12, 1->22, 2->31 (last observed season-2), 3->42
    # last position index 11 (slot 3); step k -> slot (11+k) % 4
    out = _predict(s, 4, 4)
    np.testing.assert_allclose(out, [12, 22, 31, 42])


def test_full_cycle_unchanged():
    s = pd.Series([10, 20, 30, 40, 11, 21, 31, 41, 12, 22, 32, 42])
    out = _predict(s, 4, 8)
    np.testing.assert_allclose(out, [12, 22, 32, 42, 12, 22, 32, 42])


def test_datetime_ragged_edge_calendar_anchored():
    # Realistic framework case: monthly DatetimeIndex with a missing latest month.
    # Season is anchored to the calendar, so a ragged edge does not shift seasons.
    idx = pd.date_range("2000-01-31", periods=23, freq="ME")  # Jan2000..Nov2001
    vals = [float(d.month) for d in idx]  # value == month number
    vals[-1] = np.nan  # missing latest observation (Nov 2001)
    s = pd.Series(vals, index=idx)
    out = _predict(s, 12, 4)
    # last observed = Oct2001 -> forecasts continue Nov, Dec, Jan, Feb = 11, 12, 1, 2
    np.testing.assert_allclose(out, [11, 12, 1, 2])


def test_period_not_dividing_length():
    s = pd.Series([1.0, 2, 3, 10, 20, 30, 40, 50, 60, 70])  # n=10, m=3
    # last slot = 9 % 3 = 0; slots: 0->(idx0,3,6,9)=70, 1->(1,4,7)=50, 2->(2,5,8)=60
    out = _predict(s, 3, 4)
    # step1 slot (9+1)%3=1 ->50 ; step2 slot2->60 ; step3 slot0->70 ; step4 slot1->50
    np.testing.assert_allclose(out, [50, 60, 70, 50])
