"""The panel-model horizon helper must return the true step count, not step+1.

Regression for an off-by-one: _panel_prediction_horizon returned
positions[1] - positions[0] + 1, so every panel prediction was tagged with a
horizon one larger than the real number of steps from the origin to the target
date. Horizon-keyed evaluation (e.g. selecting the h=1 rows) then silently used
the wrong records.
"""
import pandas as pd

from macroforecast.forecasting.runner import _panel_prediction_horizon


def test_horizon_when_origin_is_before_the_test_panel():
    # origin is the last training date; the test panel starts one step later.
    base = pd.date_range("2020-02-01", periods=4, freq="MS")  # origin + 1..4 months
    origin = pd.Timestamp("2020-01-01")
    for steps, date in enumerate(base, start=1):
        assert _panel_prediction_horizon(date, origin=origin, base_index=base, default=-1) == steps


def test_horizon_when_origin_is_in_the_test_panel():
    base = pd.date_range("2020-01-01", periods=4, freq="MS")
    origin = pd.Timestamp("2020-01-01")
    # origin itself is horizon 1 (clamped floor), then 1, 2, 3 steps ahead.
    expected = [1, 1, 2, 3]
    for exp, date in zip(expected, base):
        assert _panel_prediction_horizon(date, origin=origin, base_index=base, default=-1) == exp


def test_falls_back_to_default_when_date_not_locatable():
    base = pd.date_range("2020-02-01", periods=3, freq="MS")
    origin = pd.Timestamp("2020-01-01")
    missing = pd.Timestamp("1999-01-01")
    assert _panel_prediction_horizon(missing, origin=origin, base_index=base, default=7) == 7
