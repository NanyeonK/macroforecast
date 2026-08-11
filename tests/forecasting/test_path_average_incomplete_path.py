"""A path forecast is the mean of every required step, or it is missing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

HORIZON = 3


class _ConstantFit:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.value, index=X.index, name="prediction")


def _fit_step_value(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    step_values: dict[int, float],
) -> _ConstantFit:
    del X
    name = str(y.name)
    for step, value in step_values.items():
        if name.endswith(f"_step{step}"):
            return _ConstantFit(value)
    raise AssertionError(f"path target column does not identify a step: {name!r}")


def _panel(n: int = 72) -> pd.DataFrame:
    index = pd.date_range("2000-01-01", periods=n, freq="MS", name="date")
    values = np.linspace(10.0, 20.0, len(index))
    return pd.DataFrame({"y": values, "x": values[::-1]}, index=index)


def _window(index: pd.DatetimeIndex) -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(
            first_origin=index[48],
            last_origin=index[50],
            horizon=HORIZON,
        ),
    )


def _run(step_values: dict[int, float]) -> pd.DataFrame:
    panel = _panel()
    model = mf.models.ModelSpec(
        name="path_step_value",
        family="test",
        fit_func=_fit_step_value,
        default_params={"step_values": step_values},
    )
    return mf.forecasting.run(
        panel,
        model,
        window=_window(panel.index),
        features=mf.feature_engineering.feature_spec(
            target="y",
            horizon=HORIZON,
            predictors=["x"],
            lags=(0,),
        ),
        target="y",
        horizon=HORIZON,
        forecast_policy="path_average",
        target_transform="change",
        model_selection={model.name: None},
        save_models=False,
    ).to_frame()


def test_complete_path_is_the_mean_of_every_step() -> None:
    table = _run({1: 1.0, 2: 2.0, 3: 3.0})
    assert not table.empty
    assert table["prediction"].notna().all()
    assert table["prediction"].eq(2.0).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_step_makes_the_path_forecast_missing(bad: float) -> None:
    with pytest.warns(
        RuntimeWarning,
        match=r"non-finite step forecasts.*steps \[2\]",
    ):
        table = _run({1: 1.0, 2: bad, 3: 3.0})
    assert not table.empty
    assert table["prediction"].isna().all()
    assert not table["prediction"].eq(2.0).any()
