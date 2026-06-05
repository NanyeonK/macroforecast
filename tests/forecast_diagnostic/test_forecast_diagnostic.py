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
        model_selection=mf.model_selection.grid({"alpha": [0.01, 0.1]}),
        model_store=tmp_path / "trained_model",
    )

    report = mf.forecast_analysis.diagnose_forecasts(
        result,
        rolling_window=2,
        include_residual_acf=True,
        include_residual_qq=True,
        include_forecast_scale=True,
        levels=_panel()["y"],
        include_training_loss=True,
        include_rolling_training_loss=True,
        include_first_vs_last=True,
    )

    assert report.overview["n_forecasts"] == len(result.forecasts)
    assert report.fitted is not None
    assert {"residual", "abs_error", "squared_error"}.issubset(report.fitted.columns)
    assert report.residuals is not None
    assert report.residuals.loc[0, "n"] > 0
    assert report.residual_acf is not None
    assert set(report.residual_acf["lag"]).issuperset({0, 1})
    assert report.residual_qq is not None
    assert {"sample_quantile", "normal_quantile"}.issubset(report.residual_qq.columns)
    assert report.rolling_loss is not None
    assert report.rolling_loss["rolling_rmse"].notna().any()
    assert report.forecast_scale is not None
    assert set(report.forecast_scale["scale"]) == {"transformed", "back_transformed"}
    assert report.training_loss is not None
    assert {"rmse", "mae", "mse"}.issubset(set(report.training_loss["metric"]))
    assert report.rolling_training_loss is not None
    assert report.rolling_training_loss["rolling_rmse"].notna().any()
    assert report.first_vs_last is not None
    assert {"first_prediction", "last_prediction", "prediction_change"}.issubset(report.first_vs_last.columns)
    assert report.coefficients is not None
    assert {"x1_lag0", "x1_lag1", "x2_lag0", "x2_lag1"}.intersection(
        set(report.coefficients["feature"])
    )
    assert report.parameter_stability is not None
    assert {"drift", "sign_changes"}.issubset(report.parameter_stability.columns)
    assert report.tuning is not None
    assert set(report.tuning["method"]) == {"grid"}
    assert report.tuning_objective is not None
    assert report.hyperparameters is not None
    assert set(report.hyperparameters["parameter"]) == {"alpha"}
    assert report.tuning_scores is not None
    assert report.stage_updates is not None
    assert {"feature_engineering"}.issubset(set(report.stage_updates["stage"]))
    assert "forecast_analysis" in report.metadata
    assert report.fitted.attrs["macroforecast_metadata"] == report.metadata
    assert mf.forecast_analysis.forecast_overview is mf.forecast_diagnostic.forecast_overview


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
        model_selection={"ridge": None, "quantile_regression_forest": None},
        combination="mean",
        model_store=tmp_path / "trained_model",
    )

    overview = mf.forecast_analysis.forecast_overview(result)

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
        model_selection={"ols": None, "ridge": None},
        combination=[
            "mean",
            mf.forecasting.combination_spec("inverse_mspe", name="combined_dmspe"),
        ],
        save_models=False,
    )

    weights = mf.forecast_analysis.ensemble_weights_over_time(result)
    concentration = mf.forecast_analysis.ensemble_weight_concentration(result)
    contribution = mf.forecast_analysis.ensemble_member_contribution(result)

    assert {"combined_mean", "combined_dmspe"}.issubset(set(weights["combination"]))
    grouped = weights.groupby(["combination", "date", "horizon"])["weight"].sum()
    assert np.allclose(grouped.dropna().to_numpy(dtype=float), 1.0)
    equal = weights.loc[weights["combination"] == "combined_mean"]
    assert set(equal["weight"]) == {0.5}
    assert {"hhi", "effective_n", "entropy"}.issubset(concentration.columns)
    assert {"member_prediction", "contribution", "combined_prediction"}.issubset(contribution.columns)


def test_residual_and_tuning_callable_views() -> None:
    result = mf.forecasting.run(
        _panel(),
        "ridge",
        window=_window(),
        features=_features(),
        model_selection=mf.model_selection.grid({"alpha": [0.01, 0.1]}),
        save_models=False,
    )

    acf = mf.forecast_analysis.residual_autocorrelation(result, max_lag=3)
    qq = mf.forecast_analysis.residual_qq(result, n_quantiles=5)
    objective = mf.forecast_analysis.tuning_objective_trace(result)
    params = mf.forecast_analysis.hyperparameter_path(result)
    scores = mf.forecast_analysis.tuning_score_distribution(result)

    assert set(acf["lag"]) == {0, 1, 2, 3}
    assert len(qq["probability"].unique()) == 5
    assert {
        "window",
        "n_trials",
        "n_successful",
        "n_failed",
        "policy",
    }.issubset(objective.columns)
    assert objective["best_score"].notna().all()
    assert set(params["parameter"]) == {"alpha"}
    assert {"mean", "median", "q75"}.issubset(scores.columns)


def test_forecast_origin_and_scale_views() -> None:
    forecasts = pd.DataFrame(
        {
            "date": pd.date_range("2021-02-28", periods=4, freq="ME"),
            "origin": pd.date_range("2021-01-31", periods=4, freq="ME"),
            "origin_pos": [0, 1, 2, 3],
            "horizon": [1, 1, 1, 1],
            "forecast_policy": ["direct"] * 4,
            "target_transform": ["change"] * 4,
            "target": ["y"] * 4,
            "model": ["ridge", "ridge", "lasso", "lasso"],
            "prediction": [0.1, 0.2, 0.0, -0.1],
            "actual": [0.0, 0.3, -0.2, -0.1],
        }
    )
    levels = pd.Series(
        [1.0, 1.0, 1.3, 1.1, 1.0],
        index=pd.date_range("2021-01-31", periods=5, freq="ME"),
        name="y",
    )

    selected = mf.forecast_analysis.select_forecast_origins(
        forecasts,
        view="every_n_origins",
        every_n=2,
    )
    scale = mf.forecast_analysis.forecast_scale_view(
        forecasts,
        levels=levels,
        target="y",
        view="both_overlay",
    )
    first_last = mf.forecast_analysis.first_vs_last_forecast(forecasts)

    assert set(selected["origin_pos"]) == {0, 2, 3}
    assert set(scale["scale"]) == {"transformed", "back_transformed"}
    assert scale.loc[scale["scale"] == "back_transformed", "back_transform_available"].all()
    assert {"first_prediction", "last_prediction", "prediction_change"}.issubset(first_last.columns)


def test_residual_report_autocorrelation_is_origin_ordered() -> None:
    forecasts = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-31", periods=4, freq="ME"),
            "origin": pd.date_range("2020-12-31", periods=4, freq="ME"),
            "origin_pos": [0, 1, 2, 3],
            "horizon": [1, 1, 1, 1],
            "model": ["ridge"] * 4,
            "prediction": [0.0, 0.0, 0.0, 0.0],
            "actual": [1.0, 2.0, 3.0, 4.0],
        }
    ).sample(frac=1.0, random_state=123)

    report = mf.forecast_analysis.residual_report(forecasts)
    acf = mf.forecast_analysis.residual_autocorrelation(forecasts, max_lag=1)
    lag1 = acf.loc[acf["lag"] == 1, "acf"].iloc[0]

    assert report.loc[0, "residual_autocorr1"] == pytest.approx(lag1)
    # Canonical biased ACF of residuals [1,2,3,4]: gamma_1/gamma_0 = 1.25/5.0.
    assert report.loc[0, "residual_autocorr1"] == pytest.approx(0.25)


def test_best_n_combination_weights_are_reconstructed_from_history() -> None:
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    base_rows = []
    predictions = {
        "a": [0.0, 0.1, 0.2],
        "b": [3.0, 0.2, 0.3],
        "c": [4.0, 5.0, 6.0],
    }
    actual = [0.0, 0.0, 0.0]
    for pos, date in enumerate(dates):
        for model, values in predictions.items():
            base_rows.append(
                {
                    "date": date,
                    "origin": date - pd.offsets.MonthEnd(1),
                    "origin_pos": pos,
                    "horizon": 1,
                    "model": model,
                    "prediction": values[pos],
                    "actual": actual[pos],
                    "combined": False,
                    "combination": None,
                }
            )
    combined_rows = []
    selected_by_origin = {
        0: ("a", "b"),
        1: ("a", "b"),
        2: ("a", "b"),
    }
    for pos, date in enumerate(dates):
        selected = selected_by_origin[pos]
        combined_rows.append(
            {
                "date": date,
                "origin": date - pd.offsets.MonthEnd(1),
                "origin_pos": pos,
                "horizon": 1,
                "model": "combo_best",
                "prediction": float(np.mean([predictions[name][pos] for name in selected])),
                "actual": actual[pos],
                "combined": True,
                "combination": {
                    "method": "best_n",
                    "name": "combo_best",
                    "models": ["a", "b", "c"],
                    "params": {"n": 2},
                },
            }
        )
    forecasts = pd.DataFrame([*base_rows, *combined_rows])

    weights = mf.forecast_analysis.ensemble_weights_over_time(forecasts)
    concentration = mf.forecast_analysis.ensemble_weight_concentration(forecasts)
    contributions = mf.forecast_analysis.ensemble_member_contribution(forecasts)

    third = weights.loc[weights["origin_pos"] == 2].set_index("model")["weight"].to_dict()
    assert third == {"a": 0.5, "b": 0.5, "c": 0.0}
    assert {"min_weight", "max_weight", "hhi", "effective_n"}.issubset(concentration.columns)
    summed = contributions.groupby(["date", "origin_pos", "horizon"])["contribution"].sum()
    expected = forecasts.loc[forecasts["combined"], ["date", "origin_pos", "horizon", "prediction"]].set_index(
        ["date", "origin_pos", "horizon"]
    )["prediction"]
    assert np.allclose(summed.sort_index(), expected.sort_index())


def test_dfm_diagnostics_accept_model_fit() -> None:
    idx = pd.date_range("2020-01-31", periods=8, freq="ME")
    fit = mf.ModelFit(
        estimator=object(),
        model="dfm_mixed_mariano_murasawa",
        diagnostics={
            "residuals": pd.Series(np.linspace(-0.2, 0.2, 8), index=idx, name="y"),
            "factors_filtered": pd.DataFrame(
                {
                    "factor_1": np.linspace(-1.0, 1.0, 8),
                    "factor_2": np.cos(np.arange(8)),
                },
                index=idx,
            ),
        },
    )

    acf = mf.forecast_analysis.dfm_idiosyncratic_acf(fit, max_lag=2)
    stability = mf.forecast_analysis.dfm_factor_stability(fit)

    assert set(acf["lag"]) == {0, 1, 2}
    assert set(stability["factor"]) == {"factor_1", "factor_2"}
    assert {"drift", "autocorr1"}.issubset(stability.columns)


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
        model_selection={"ridge": None},
        save_models=False,
    )

    trace = mf.forecast_analysis.stage_update_trace(result)

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

    out = mf.forecast_analysis.custom_forecast_diagnostic(
        forecasts,
        signed_bias,
        name="signed_bias",
        metadata={"sample": "toy"},
    )

    assert set(out["model"]) == {"ridge", "lasso"}
    assert out.attrs["macroforecast_metadata_schema"]["kind"] == "custom_forecast_diagnostic"
    assert out.attrs["macroforecast_metadata_schema"]["method"] == "signed_bias"
    assert "custom_forecast_diagnostic" in out.attrs["macroforecast_metadata"]
