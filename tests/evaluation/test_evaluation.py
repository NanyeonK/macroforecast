from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def test_distribution_metrics_and_forecast_evaluation() -> None:
    forecasts = pd.DataFrame(
        {
            "model": ["a", "a", "a"],
            "horizon": [1, 1, 1],
            "actual": [1.0, 2.0, 3.0],
            "prediction": [1.0, 2.5, 2.5],
            "variance_prediction": [0.25, 0.5, 1.0],
            "quantile_predictions": [
                {"0.1": 0.5, "0.5": 1.0, "0.9": 1.5},
                {"0.1": 1.5, "0.5": 2.0, "0.9": 2.5},
                {"0.1": 2.5, "0.5": 3.0, "0.9": 3.5},
            ],
        }
    )

    out = mf.evaluation.evaluate_forecasts(forecasts)

    assert out.loc[0, "model"] == "a"
    assert out.loc[0, "horizon"] == 1
    assert out.loc[0, "n"] == 3
    assert np.isclose(out.loc[0, "mse"], ((0.0**2 + 0.5**2 + 0.5**2) / 3.0))
    assert out.loc[0, "variance_n"] == 3
    assert np.isfinite(out.loc[0, "gaussian_nll"])
    assert out.loc[0, "quantile_n"] == 9
    assert "pinball_loss_q0_5" in out.columns
    assert out.loc[0, "coverage_q0_1_q0_9"] == 1.0
    assert out.loc[0, "interval_width_q0_1_q0_9"] == 1.0


def test_forecast_result_evaluate_method() -> None:
    result = mf.forecasting.ForecastResult(
        pd.DataFrame(
            {
                "model": ["a", "a"],
                "horizon": [1, 1],
                "actual": [1.0, 3.0],
                "prediction": [1.5, 2.5],
            }
        )
    )

    out = result.evaluate(by=("model",))

    assert out.loc[0, "model"] == "a"
    assert out.loc[0, "n"] == 2
    assert out.loc[0, "mae"] == 0.5


def test_distribution_metric_helpers_validate_inputs() -> None:
    assert mf.pinball_loss([1, 2], [0, 3], quantile=0.5) == 0.5
    assert mf.coverage_rate([1, 2], [0, 1], [1, 3]) == 1.0
    assert mf.interval_width([0, 1], [1, 3]) == 1.5
    assert np.isfinite(mf.gaussian_nll([1, 2], [1, 2], [0.5, 1.0]))

