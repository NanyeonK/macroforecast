from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _forecast_table() -> pd.DataFrame:
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    rows: list[dict[str, object]] = []
    for horizon in (1, 2):
        for date_pos, date in enumerate(dates):
            actual = 1.0 + date_pos + horizon
            for model, offset in {"model_a": 0.1, "model_b": 0.4, "bench": 0.8}.items():
                rows.append(
                    {
                        "date": date,
                        "model": model,
                        "horizon": horizon,
                        "target": "INDPRO",
                        "state": "expansion" if date_pos < 2 else "slowdown",
                        "actual": actual,
                        "prediction": actual + offset,
                        "combined": False,
                    }
                )
    return pd.DataFrame(rows)


def test_evaluation_report_scores_rankings_and_slices() -> None:
    table = _forecast_table()
    regimes = pd.Series(
        ["early", "early", "late", "late"],
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )

    report = mf.evaluate_report(
        table,
        metrics=("mse", "rmse", "mae", "bias", "relative_mse", "r2_oos"),
        benchmark_model="bench",
        oos_start="2020-02-01",
        regimes=regimes,
        time_frequency="Q",
        include_decomposition=True,
    )

    assert isinstance(report, mf.evaluation.EvaluationReport)
    assert {"model", "horizon", "mse", "rmse", "bias", "relative_mse", "r2_oos"}.issubset(
        report.scores.columns
    )
    assert report.ranking.loc[0, "model"] == "model_a"
    assert {"model", "horizon", "model_horizon_target", "model_horizon_state", "model_horizon_regime", "model_horizon_time"}.issubset(
        set(report.aggregations)
    )
    assert report.benchmark is not None
    assert set(report.benchmark["model"]) == {"model_a", "model_b"}
    assert report.regime is not None
    assert set(report.regime["regime"]) == {"early", "late"}
    assert report.decomposition is not None
    assert {"bias_squared", "residual_variance", "bias_share"}.issubset(report.decomposition.columns)
    assert "evaluation_report" in report.metadata
    assert report.metadata["evaluation_report"]["options"]["oos_start"] == "2020-02-01"
    assert report.scores.attrs["macroforecast_metadata"] == report.metadata
    assert report.to_dict()["metadata"]["evaluation_report"]["tables"]["scores"] == len(report.scores)
    assert "decomposition" in report.to_dict()


def test_evaluation_namespace_keeps_raw_metrics_and_tests_separate() -> None:
    assert mf.evaluation.metrics is mf.metrics
    assert mf.evaluation.tests is mf.tests
    assert mf.evaluation.evaluate_report is mf.evaluate_report
    assert not hasattr(mf.evaluation, "rmse")
    assert not hasattr(mf.evaluation, "dm_test")


def test_evaluation_report_accepts_recursive_and_combined_forecast_result() -> None:
    table = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=4, freq="ME").tolist() * 3,
            "model": ["ols"] * 4 + ["ridge"] * 4 + ["combined_mean"] * 4,
            "horizon": [2] * 12,
            "forecast_policy": ["recursive"] * 12,
            "target": ["y"] * 12,
            "actual": [1.0, 2.0, 3.0, 4.0] * 3,
            "prediction": [
                1.1,
                2.1,
                2.9,
                4.1,
                1.2,
                2.2,
                2.8,
                4.2,
                1.15,
                2.15,
                2.85,
                4.15,
            ],
            "combined": [False] * 8 + [True] * 4,
        }
    )
    result = mf.forecasting.ForecastResult(
        table,
        metadata={"forecast_policy": {"method": "recursive"}},
    )

    report = mf.evaluate_report(
        result,
        score_by=("model", "horizon", "forecast_policy"),
        include_combined=False,
    )

    assert set(report.scores["model"]) == {"ols", "ridge"}
    assert set(report.scores["forecast_policy"]) == {"recursive"}
    assert report.metadata["forecast_policy"]["method"] == "recursive"
    assert report.metadata["evaluation_report"]["options"]["include_combined"] is False


def test_aggregate_scores_requires_explicit_existing_columns() -> None:
    with pytest.raises(ValueError, match="not in the forecast table"):
        mf.evaluation.aggregate_scores(
            _forecast_table(),
            groupings={"bad": ("model", "missing_column")},
        )


def test_regime_scores_accepts_existing_column_name() -> None:
    table = _forecast_table()
    table["phase"] = np.where(table["date"].dt.month <= 2, "phase_1", "phase_2")

    out = mf.evaluation.regime_scores(table, regimes="phase", regime_column="regime")

    assert set(out["regime"]) == {"phase_1", "phase_2"}
    assert out.attrs["macroforecast_metadata_schema"]["kind"] == "forecast_regime_scores"


def test_filter_oos_period_and_error_decomposition_are_callable() -> None:
    table = _forecast_table()

    filtered = mf.evaluation.filter_oos_period(
        table,
        start="2020-02-01",
        end="2020-03-31",
    )
    decomposition = mf.evaluation.error_decomposition(filtered, by=("model", "horizon"))

    assert filtered["date"].min() == pd.Timestamp("2020-02-29")
    assert filtered["date"].max() == pd.Timestamp("2020-03-31")
    assert {"mse", "bias", "bias_squared", "residual_variance"}.issubset(decomposition.columns)
    assert decomposition.attrs["macroforecast_metadata_schema"]["kind"] == "forecast_error_decomposition"


def test_benchmark_comparison_validates_benchmark_presence() -> None:
    with pytest.raises(ValueError, match="benchmark_model"):
        mf.evaluation.benchmark_comparison(
            _forecast_table(),
            benchmark_model="missing",
        )
