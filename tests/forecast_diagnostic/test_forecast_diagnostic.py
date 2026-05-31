from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _panel(n: int = 54) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.1 * np.sin(np.arange(n) / 3.0),
            "x1": x,
            "x2": np.sin(np.arange(n) / 4.0),
        },
        index=idx,
    )


def _window() -> mf.window.WindowSpec:
    return mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=24),
        val=mf.window.val_last_block(size=6),
        test=mf.window.test_origins(horizon=1, step=6),
    )


def _features() -> mf.feature_engineering.FeatureSpec:
    return mf.feature_engineering.feature_spec(
        target="y",
        horizon=1,
        predictors=["x1", "x2"],
        lags=(0, 1),
    )


def test_diagnose_forecasts_reads_runner_outputs(tmp_path) -> None:
    result = mf.forecasting.run(
        _panel(),
        "ridge",
        window=_window(),
        features=_features(),
        selection=mf.selection.grid({"alpha": [0.01, 0.1]}),
        model_store=tmp_path / "trained_model",
    )

    report = mf.diagnose_forecasts(result, rolling_window=2)

    assert report.overview["n_forecasts"] == len(result.forecasts)
    assert report.fitted is not None
    assert {"residual", "abs_error", "squared_error"}.issubset(report.fitted.columns)
    assert report.residuals is not None
    assert report.residuals.loc[0, "n"] > 0
    assert report.rolling_loss is not None
    assert report.rolling_loss["rolling_rmse"].notna().any()
    assert report.coefficients is not None
    assert {"x1_lag0", "x1_lag1", "x2_lag0", "x2_lag1"}.intersection(
        set(report.coefficients["feature"])
    )
    assert report.tuning is not None
    assert set(report.tuning["method"]) == {"grid"}
    assert report.stage_updates is not None
    assert {"feature_engineering"}.issubset(set(report.stage_updates["stage"]))
    assert "forecast_diagnostic" in report.metadata
    assert report.fitted.attrs["macroforecast_metadata"] == report.metadata


def test_forecast_overview_counts_combination_and_uncertainty(tmp_path) -> None:
    result = mf.forecasting.run(
        _panel(),
        ["ridge", "quantile_regression_forest"],
        window=_window(),
        features=_features(),
        params={
            "ridge": {"alpha": 0.1},
            "quantile_regression_forest": {
                "n_estimators": 8,
                "min_samples_leaf": 2,
                "quantile_levels": (0.1, 0.5, 0.9),
                "random_state": 123,
            },
        },
        selection={"ridge": None, "quantile_regression_forest": None},
        combination="mean",
        model_store=tmp_path / "trained_model",
    )

    overview = mf.forecast_diagnostic.forecast_overview(result)

    assert overview["n_models"] == 3
    assert overview["combined_count"] > 0
    assert overview["stored_model_count"] > 0
    assert overview["quantile_prediction_count"] > 0


def test_ensemble_weights_over_time_reconstructs_equal_and_inverse_weights() -> None:
    result = mf.forecasting.run(
        _panel(),
        ["ols", "ridge"],
        window=_window(),
        features=_features(),
        params={"ridge": {"alpha": 0.1}},
        selection={"ols": None, "ridge": None},
        combination=[
            "mean",
            mf.forecasting.combination_spec("inverse_mspe", name="combined_dmspe"),
        ],
        save_models=False,
    )

    weights = mf.forecast_diagnostic.ensemble_weights_over_time(result)

    assert {"combined_mean", "combined_dmspe"}.issubset(set(weights["combination"]))
    grouped = weights.groupby(["combination", "date", "horizon"])["weight"].sum()
    assert np.allclose(grouped.dropna().to_numpy(dtype=float), 1.0)
    equal = weights.loc[weights["combination"] == "combined_mean"]
    assert set(equal["weight"]) == {0.5}


def test_stage_update_trace_is_empty_for_feature_set_input() -> None:
    with pytest.warns(UserWarning, match="feature engineering works best"):
        features = mf.feature_engineering.build_features(
            mf.data.spec(_panel(), target="y", horizons=[1], predictors=["x1", "x2"]),
            lags=(0, 1),
        )
    result = mf.forecasting.run(
        features,
        "ridge",
        window=_window(),
        params={"ridge": {"alpha": 0.1}},
        selection={"ridge": None},
        save_models=False,
    )

    trace = mf.forecast_diagnostic.stage_update_trace(result)

    assert trace.empty
    assert list(trace.columns) == [
        "stage",
        "origin",
        "origin_pos",
        "updated",
        "fit_start",
        "fit_end",
        "test_start",
        "test_end",
        "metadata_keys",
    ]


def test_custom_forecast_diagnostic_wraps_user_callable() -> None:
    forecasts = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-31", periods=4, freq="ME"),
            "origin": pd.date_range("2020-12-31", periods=4, freq="ME"),
            "origin_pos": [0, 1, 2, 3],
            "horizon": [1, 1, 1, 1],
            "model": ["ridge", "ridge", "lasso", "lasso"],
            "prediction": [1.0, 2.0, 1.5, 2.5],
            "actual": [1.2, 1.8, 1.4, 2.7],
        }
    )

    def signed_bias(table, *, metadata=None, group="model"):
        residual = table["actual"] - table["prediction"]
        return (
            pd.DataFrame({"model": table["model"], "residual": residual})
            .groupby(group, as_index=False)
            .agg(bias=("residual", "mean"))
        )

    out = mf.custom_forecast_diagnostic(
        forecasts,
        signed_bias,
        name="signed_bias",
        metadata={"sample": "toy"},
    )

    assert set(out["model"]) == {"ridge", "lasso"}
    assert out.attrs["macroforecast_metadata_schema"]["kind"] == "custom_forecast_diagnostic"
    assert out.attrs["macroforecast_metadata_schema"]["method"] == "signed_bias"
    assert "custom_forecast_diagnostic" in out.attrs["macroforecast_metadata"]
