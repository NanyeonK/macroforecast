from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _target(n: int = 120) -> pd.Series:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    values = 0.2 + 0.03 * np.arange(n) + np.sin(np.arange(n) / 5.0)
    return pd.Series(values, index=idx, name="equity_premium")


def _predict_from_origin(y: pd.Series, origin: int, *, window: int | None = None) -> float:
    fit = mf.models.hist_mean(y.iloc[:origin], window=window)
    X_test = pd.DataFrame({"dummy": [0.0]}, index=[y.index[origin]])
    return float(fit.predict(X_test).iloc[0])


@pytest.mark.parametrize("origin", [18, 47, 96])
def test_hist_mean_expanding_matches_hand_computed_origin_mean(origin: int) -> None:
    y = _target()

    prediction = _predict_from_origin(y, origin)

    assert prediction == pytest.approx(float(y.iloc[:origin].mean()))


@pytest.mark.parametrize("origin", [60, 75, 119])
def test_hist_mean_window_matches_pandas_rolling_mean(origin: int) -> None:
    y = _target()
    rolling = y.rolling(window=60, min_periods=1).mean()

    prediction = _predict_from_origin(y, origin, window=60)

    assert prediction == pytest.approx(float(rolling.iloc[origin - 1]))


def test_hist_mean_is_horizon_invariant_for_same_fit_window() -> None:
    y = _target()
    fit = mf.models.hist_mean(y.iloc[:84])
    h1_index = pd.date_range(y.index[84], periods=1, freq="ME")
    h12_index = pd.date_range(y.index[84], periods=12, freq="ME")

    h1 = fit.predict(pd.DataFrame({"dummy": np.zeros(1)}, index=h1_index))
    h12 = fit.predict(pd.DataFrame({"dummy": np.zeros(12)}, index=h12_index))

    assert float(h1.iloc[0]) == pytest.approx(float(y.iloc[:84].mean()))
    assert np.allclose(h12.to_numpy(), h1.iloc[0])


def test_hist_mean_registry_exposes_target_kind_and_window_param() -> None:
    spec = mf.get_model("hist_mean")

    assert spec.input_kind == "target"
    assert spec.default_params == {"window": None}
    assert [parameter.name for parameter in spec.parameters] == ["window"]
    assert mf.hist_mean is mf.models.hist_mean
