from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def test_metrics_and_tests_are_separate_public_apis() -> None:
    assert mf.metrics.rmse([1.0], [1.0]) == 0.0
    assert not hasattr(mf, "rmse")
    assert not hasattr(mf, "dm_test")
    assert mf.evaluation.metrics is mf.metrics
    assert mf.evaluation.tests is mf.tests
    assert not hasattr(mf.evaluation, "rmse")
    assert not hasattr(mf.evaluation, "dm_test")


def test_point_relative_and_direction_metrics() -> None:
    actual = pd.Series([1.0, 2.0, 4.0], index=pd.date_range("2020-01-01", periods=3))
    model = pd.Series([1.0, 2.5, 3.5], index=actual.index)
    bench = pd.Series([0.5, 2.5, 5.0], index=actual.index)
    previous = pd.Series([0.5, 1.5, 3.0], index=actual.index)

    assert mf.metrics.mse(actual, model) == np.mean([0.0, 0.25, 0.25])
    assert mf.metrics.rmse(actual, model) == np.sqrt(mf.metrics.mse(actual, model))
    assert mf.metrics.mae(actual, model) == np.mean([0.0, 0.5, 0.5])
    assert mf.metrics.medae(actual, model) == 0.5
    assert np.isclose(mf.metrics.mape(actual, model), np.mean([0.0, 0.25, 0.125]) * 100.0)
    assert np.isfinite(mf.metrics.theil_u1(actual, model))
    assert np.isfinite(mf.metrics.theil_u2(actual, model, previous))
    assert np.isclose(mf.metrics.relative_mse(actual, model, bench), 0.5 / 1.5)
    assert np.isclose(mf.metrics.r2_oos(actual, model, bench), 1.0 - 0.5 / 1.5)
    assert np.isclose(mf.metrics.relative_mae(actual, model, bench), 1.0 / 2.0)
    assert np.isclose(mf.metrics.mse_reduction(actual, model, bench), (1.5 - 0.5) / 3.0)
    assert mf.metrics.success_ratio(actual, model, previous) == 1.0
    assert np.isfinite(mf.metrics.pesaran_timmermann_metric([1, -1, 2, -2], [0.8, -0.5, -0.1, -1.0]))


def test_distribution_metric_helpers_validate_inputs() -> None:
    assert mf.metrics.pinball_loss([1, 2], [0, 3], quantile=0.5) == 0.5
    assert mf.metrics.coverage_rate([1, 2], [0, 1], [1, 3]) == 1.0
    assert mf.metrics.interval_width([0, 1], [1, 3]) == 1.5
    assert np.isfinite(mf.metrics.gaussian_nll([1, 2], [1, 2], [0.5, 1.0]))
    assert np.isfinite(mf.metrics.log_score([1, 2], [1, 2], [0.5, 1.0]))
    assert np.isfinite(mf.metrics.crps([1, 2], [1, 2], [0.5, 1.0]))
    assert np.isfinite(mf.metrics.qlike([1, 2], [1.1, 2.1]))
    assert mf.metrics.interval_score([1, 2], [0, 1], [1, 3], alpha=0.1) == 1.5


def test_forecast_table_evaluation_scores_variance_quantiles_and_benchmark() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    forecasts = pd.DataFrame(
        {
            "date": [*dates, *dates],
            "model": ["a", "a", "a", "bench", "bench", "bench"],
            "horizon": [1, 1, 1, 1, 1, 1],
            "actual": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            "prediction": [1.0, 2.5, 3.5, 0.0, 2.0, 4.0],
            "previous_actual": [0.5, 1.5, 2.5, 0.5, 1.5, 2.5],
            "variance_prediction": [0.25, 0.5, 1.0, 0.25, 0.5, 1.0],
            "quantile_predictions": [
                {"0.1": 0.5, "0.5": 1.0, "0.9": 1.5},
                {"0.1": 1.5, "0.5": 2.0, "0.9": 2.5},
                {"0.1": 2.5, "0.5": 3.0, "0.9": 3.5},
                {"0.1": 0.5, "0.5": 1.0, "0.9": 1.5},
                {"0.1": 1.5, "0.5": 2.0, "0.9": 2.5},
                {"0.1": 2.5, "0.5": 3.0, "0.9": 3.5},
            ],
        }
    )

    out = mf.metrics.evaluate_forecasts(
        forecasts,
        metrics=("mse", "rmse", "mae", "relative_mse", "r2_oos", "qlike"),
        benchmark_model="bench",
    )
    row = out.loc[out["model"] == "a"].iloc[0]

    assert out.attrs["macroforecast_metadata_schema"]["kind"] == "forecast_metrics"
    assert out.attrs["macroforecast_metadata_schema"]["version"] == 1
    assert row["n"] == 3
    assert np.isclose(row["mse"], ((0.0**2 + 0.5**2 + 0.5**2) / 3.0))
    assert np.isclose(row["relative_mse"], row["mse"] / (2.0 / 3.0))
    assert np.isclose(row["r2_oos"], 1.0 - row["relative_mse"])
    assert row["variance_n"] == 3
    assert np.isfinite(row["gaussian_nll"])
    assert np.isfinite(row["crps"])
    assert np.isfinite(row["qlike"])
    assert row["quantile_n"] == 9
    assert "pinball_loss_q0_5" in out.columns
    assert row["coverage_q0_1_q0_9"] == 1.0
    assert row["interval_width_q0_1_q0_9"] == 1.0
    assert row["success_ratio"] == 1.0
    assert np.isfinite(row["theil_u2"])


def test_forecast_result_evaluate_method_and_ranking() -> None:
    result = mf.forecasting.ForecastResult(
        pd.DataFrame(
            {
                "model": ["a", "a", "b", "b"],
                "horizon": [1, 1, 1, 1],
                "actual": [1.0, 3.0, 1.0, 3.0],
                "prediction": [1.5, 2.5, 2.0, 2.0],
            }
        )
    )

    out = result.evaluate(by=("model", "horizon"))
    ranked = mf.metrics.rank_forecasts(out, metric="mae", by=("horizon",))

    assert out.loc[out["model"] == "a", "mae"].iloc[0] == 0.5
    assert ranked.loc[0, "model"] == "a"
    assert ranked.loc[0, "rank"] == 1.0
    assert ranked.attrs["macroforecast_metadata_schema"]["kind"] == "forecast_metric_ranking"


def test_get_metric_exposes_legacy_metric_names() -> None:
    assert mf.metrics.get_metric("msfe") is mf.metrics.mse
    assert mf.metrics.get_metric("validation_mse") is mf.metrics.mse
    assert mf.metrics.get_metric("relative_mse") is mf.metrics.relative_mse
    assert mf.metrics.get_metric("crps") is mf.metrics.crps
    assert mf.metrics.get_metric("qlike") is mf.metrics.qlike
    assert mf.metrics.get_metric("success_ratio") is mf.metrics.success_ratio
