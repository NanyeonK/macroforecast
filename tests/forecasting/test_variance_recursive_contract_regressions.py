"""Regression coverage for variance dispatch/alignment and recursive actuals."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting.policies.recursive import _optional_target_level_at


class _VarianceFit:
    def __init__(self, variance_method: Any) -> None:
        self._variance_method = variance_method

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index, name="prediction")

    def predict_variance(self, value=None, *, horizon=None):
        return self._variance_method(value, horizon=horizon)


def _variance_model(name: str, variance_method: Any) -> mf.models.ModelSpec:
    def fit(X: pd.DataFrame, y: pd.Series) -> _VarianceFit:
        del X, y
        return _VarianceFit(variance_method)

    return mf.models.ModelSpec(name=name, family="test", fit_func=fit)


def _variance_inputs():
    index = pd.date_range("2000-01-01", periods=72, freq="MS", name="date")
    frame = pd.DataFrame(
        {"y": np.linspace(0.0, 1.0, len(index)), "x": np.linspace(1.0, 0.0, len(index))},
        index=index,
    )
    features = mf.feature_engineering.feature_spec(
        target="y",
        predictors=["x"],
        lags=(0,),
    )
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(
            first_origin=index[48],
            last_origin=index[50],
            horizon=1,
        ),
    )
    return frame, features, window


def _run_variance(model: mf.models.ModelSpec) -> pd.DataFrame:
    frame, features, window = _variance_inputs()
    return mf.forecasting.run(
        frame,
        model,
        window=window,
        features=features,
        model_selection={model.name: None},
        save_models=False,
    ).to_frame()


def test_conditional_variance_internal_typeerror_is_not_a_protocol_switch() -> None:
    def variance(value=None, *, horizon=None):
        if isinstance(value, pd.DataFrame):
            raise TypeError("internal failure inside predict_variance")
        length = int(horizon if horizon is not None else value)
        return np.full(length, 11.0)

    with pytest.raises(TypeError, match="internal failure inside predict_variance"):
        _run_variance(_variance_model("internal_variance_failure", variance))


def test_conditional_variance_rejects_nonmatching_pandas_labels() -> None:
    def variance(value=None, *, horizon=None):
        del horizon
        assert isinstance(value, pd.DataFrame)
        wrong = pd.date_range("1980-01-31", periods=len(value), freq="ME")
        return pd.Series(7.0, index=wrong)

    with pytest.raises(ValueError, match="variance prediction index does not match X_test"):
        _run_variance(_variance_model("misindexed_variance", variance))


class _HorizonVarianceFit:
    def __init__(self, arguments: list[int]) -> None:
        self.arguments = arguments

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=X.index, name="prediction")

    def predict_variance(self, horizon: int = 1, /) -> pd.Series:
        assert not isinstance(horizon, pd.DataFrame)
        self.arguments.append(horizon)
        return pd.Series(
            np.full(horizon, 4.0),
            index=[f"h.{step}" for step in range(1, horizon + 1)],
        )


def test_horizon_variance_receives_int_and_remains_positional() -> None:
    arguments: list[int] = []

    def fit(X: pd.DataFrame, y: pd.Series) -> _HorizonVarianceFit:
        del X, y
        return _HorizonVarianceFit(arguments)

    model = mf.models.ModelSpec(name="horizon_variance", family="test", fit_func=fit)
    table = _run_variance(model)

    assert arguments and all(isinstance(argument, int) for argument in arguments)
    assert set(table["variance_prediction"]) == {4.0}


@pytest.mark.parametrize("use_range_index", [False, True])
def test_conditional_variance_accepts_owned_or_default_labels(use_range_index: bool) -> None:
    def variance(value=None, *, horizon=None):
        del horizon
        assert isinstance(value, pd.DataFrame)
        index = pd.RangeIndex(len(value)) if use_range_index else value.index
        return pd.Series(0.5, index=index)

    table = _run_variance(_variance_model("aligned_variance", variance))
    assert set(table["variance_prediction"]) == {0.5}


def test_conditional_variance_preserves_length_error() -> None:
    def variance(value=None, *, horizon=None):
        del value, horizon
        return np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="variance prediction length does not match X_test"):
        _run_variance(_variance_model("wrong_length_variance", variance))


def test_recursive_missing_target_date_actual_emits_unscored_forecast() -> None:
    index = pd.date_range("2000-01-01", periods=72, freq="MS", name="date")
    frame = pd.DataFrame(
        {
            "y": 10.0 + np.sin(np.arange(len(index)) / 4.0),
            "x": np.cos(np.arange(len(index)) / 5.0),
        },
        index=index,
    )
    frame.loc[index[-1], "y"] = np.nan
    window = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=8),
        test=mf.window.test_origins(
            first_origin=index[-4],
            last_origin=index[-4],
            horizon=3,
        ),
    )

    table = mf.forecasting.run(
        frame,
        "ols",
        window=window,
        target="y",
        horizon=3,
        forecast_policy="recursive",
        save_models=False,
    ).to_frame()

    assert not table.empty
    assert table["date"].eq(index[-1]).all()
    assert table["prediction"].notna().all()
    assert table["actual"].isna().all()
    for params in table["params"]:
        assert len(params["recursive"]["step_predictions"]) == 3
    assert table["window"].notna().all()


def test_optional_recursive_actual_lookup_keeps_shape_errors_strict() -> None:
    index = pd.date_range("2000-01-01", periods=2, freq="MS")
    panel = pd.DataFrame({"y": [1.0, np.nan]}, index=index)

    assert _optional_target_level_at(panel, "y", index[0]) == 1.0
    assert _optional_target_level_at(panel, "y", index[1]) is None
    with pytest.raises(ValueError, match="target 'z' is not present"):
        _optional_target_level_at(panel, "z", index[0])
    with pytest.raises(ValueError, match="target date .* is not present"):
        _optional_target_level_at(panel, "y", pd.Timestamp("1999-01-01"))
