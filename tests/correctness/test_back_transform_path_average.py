"""F3/F4: correct level back-transform for path_average and direct_average targets.

The stored prediction/actual for these policies is the MEAN one-period transform
over `horizon` steps. The level at t+h therefore requires the horizon factor:
  change     -> x[t+h] = x[t] + h * mean_change          (telescoping, exact)
  log_growth -> x[t+h] = x[t] * exp(h * mean_log_growth)  (telescoping, exact)
  growth/value -> no exact pointwise level inverse -> unavailable.
"""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _levels():
    # x[t0]=100, x[t0+3]=112
    return pd.Series(
        [100.0, 103.0, 108.0, 112.0, 115.0],
        index=pd.date_range("2021-01-31", periods=5, freq="ME"),
        name="y",
    )


def _forecasts(transform, policy, pred, actual, horizon=3):
    idx = pd.date_range("2021-01-31", periods=5, freq="ME")
    return pd.DataFrame(
        {
            "date": [idx[3]],            # t0 + 3
            "origin": [idx[0]],          # t0
            "origin_pos": [0],
            "horizon": [horizon],
            "target": ["y"],
            "model": ["ols"],
            "forecast_policy": [policy],
            "target_transform": [transform],
            "prediction": [pred],
            "actual": [actual],
        }
    )


def _bt(scale):
    return scale.loc[scale["scale"] == "back_transformed"].iloc[0]


def test_path_average_change_uses_horizon_factor():
    levels = _levels()
    # mean one-period change = (112-100)/3 = 4
    fc = _forecasts("change", "path_average", pred=4.0, actual=4.0)
    scale = mf.forecast_analysis.forecast_scale_view(
        fc, levels=levels, target="y", view="back_transformed_only"
    )
    row = _bt(scale)
    assert row["back_transform_available"]
    # correct: 100 + 3*4 = 112  (buggy code gives 104)
    assert np.isclose(row["prediction"], 112.0)
    assert np.isclose(row["actual"], 112.0)


def test_path_average_log_growth_uses_horizon_factor():
    levels = _levels()
    g = (np.log(112.0) - np.log(100.0)) / 3.0  # mean one-period log-growth
    fc = _forecasts("log_growth", "path_average", pred=g, actual=g)
    scale = mf.forecast_analysis.forecast_scale_view(
        fc, levels=levels, target="y", view="back_transformed_only"
    )
    row = _bt(scale)
    assert row["back_transform_available"]
    assert np.isclose(row["prediction"], 112.0)


def test_direct_average_change_reconstructs_endpoint():
    levels = _levels()
    fc = _forecasts("average_change", "direct_average", pred=4.0, actual=4.0)
    scale = mf.forecast_analysis.forecast_scale_view(
        fc, levels=levels, target="y", view="back_transformed_only"
    )
    row = _bt(scale)
    assert row["back_transform_available"]
    assert np.isclose(row["prediction"], 112.0)
    # F4: prediction must NOT be silently dropped to None
    assert row["prediction"] is not None


def test_average_value_has_no_exact_inverse_marked_unavailable():
    levels = _levels()
    fc = _forecasts("average_value", "direct_average", pred=108.0, actual=108.0)
    scale = mf.forecast_analysis.forecast_scale_view(
        fc, levels=levels, target="y", view="back_transformed_only"
    )
    row = _bt(scale)
    # no exact endpoint level -> unavailable, and no misleading actual
    assert not row["back_transform_available"]
    assert row["prediction"] is None
    assert row["actual"] is None


def test_direct_change_unchanged_single_step_inverse():
    # A DIRECT (non-average) h-step change must still invert as origin + value.
    levels = _levels()
    fc = _forecasts("change", "direct", pred=12.0, actual=12.0)
    scale = mf.forecast_analysis.forecast_scale_view(
        fc, levels=levels, target="y", view="back_transformed_only"
    )
    row = _bt(scale)
    assert np.isclose(row["prediction"], 112.0)  # 100 + 12
