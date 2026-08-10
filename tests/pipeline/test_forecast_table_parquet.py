"""The forecast table must be writable to Parquet.

A model with no parameters produces an empty ``params`` mapping, and Arrow types an
empty mapping as a struct with no child fields, which Parquet cannot represent -- so
``report.forecasts.to_parquet(...)`` raised for any run containing such a model
(``ols`` is one). Regression test for that, plus the round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)


def _report():
    idx = pd.date_range("2000-01-31", periods=90, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, 90)
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + rng.normal(0.0, 0.05, 90), "x1": x}, index=idx)
    bundle = mf.data.custom_dataset(frame, transform_codes={"y": 1, "x1": 1})
    features = mf.feature_engineering.feature_spec(
        target="y", predictors=["x1"], lags=1, target_lags=(0, 1)
    )
    window = mf.window.from_cutoffs(
        test_start=idx[60], horizon=1, embargo=0, val_method="expanding", val_min_train_size=24
    )
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y", transform="level")],
        horizons=[1],
        window=window,
        # `ols` carries no parameters, `ar` and `ridge` carry different ones -- so the
        # metadata columns hold both an empty mapping and two differing key sets.
        arms=[
            Arm("AR", model="ar", features=features),
            Arm("OLS", model="ols", features=features),
            Arm("RIDGE", model="ridge", features=features),
        ],
        evaluation=EvalSpec(benchmark="AR"),
        save_models=False,
    )
    return run_pipeline(spec)


def test_forecast_table_round_trips_through_parquet(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    report = _report()
    forecasts = report.forecasts

    path = tmp_path / "forecasts.parquet"
    forecasts.to_parquet(path)
    restored = pd.read_parquet(path)

    assert restored.shape == forecasts.shape
    np.testing.assert_allclose(
        restored["prediction"].astype(float),
        forecasts["prediction"].astype(float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        restored["actual"].astype(float), forecasts["actual"].astype(float), equal_nan=True
    )


def test_forecast_table_reports_no_params_as_none_not_empty_mapping() -> None:
    report = _report()
    forecasts = report.forecasts

    # the parameterless model reports None rather than {}
    ols_params = forecasts.loc[forecasts["arm"] == "OLS", "params"].iloc[0]
    assert ols_params is None

    # a parameterized model is untouched
    ar_params = forecasts.loc[forecasts["arm"] == "AR", "params"].iloc[0]
    assert isinstance(ar_params, dict) and ar_params

    # and no empty mapping survives anywhere in the metadata columns
    for column in ("params", "model_selection", "window", "model_spec"):
        if column in forecasts.columns:
            assert not any(
                isinstance(value, dict) and not value for value in forecasts[column]
            )
