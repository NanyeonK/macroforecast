from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf


def test_report_table_formats_values_and_renders_text() -> None:
    table = pd.DataFrame(
        {
            "model": ["ridge", "lasso"],
            "rmse": [0.12345, 0.23456],
            "r2": [0.101, 0.052],
        }
    )

    report = mf.reporting.report_table(
        table,
        columns=("model", "rmse", "r2"),
        rename={"model": "Model", "rmse": "RMSE", "r2": "R2 OOS"},
        sort_by="rmse",
        precision=2,
        percent_columns=("R2 OOS",),
        caption="Forecast accuracy",
        label="tab:accuracy",
        notes=("Lower RMSE is better.",),
    )

    assert isinstance(report, mf.reporting.ReportTable)
    assert report.data.loc[0, "RMSE"] == "0.12"
    assert report.data.loc[0, "R2 OOS"] == "10.10%"
    assert report.data.attrs["macroforecast_metadata_schema"]["kind"] == "report_table"
    assert report.data.attrs["macroforecast_metadata"]["source_shape"] == [2, 3]
    assert report.data.attrs["macroforecast_metadata"]["percent_columns"] == ["R2 OOS"]
    assert "\\caption{Forecast accuracy}" in report.to_latex()
    assert "<figcaption>Forecast accuracy</figcaption>" in report.to_html()
    assert "| Model | RMSE | R2 OOS |" in report.to_markdown()


def test_figure_data_and_report_bundle() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "model": ["ridge", "ridge", "lasso"],
            "loss": [0.2, 0.3, None],
            "unused": [1, 2, 3],
        }
    )

    figure = mf.reporting.figure_data(data, x="date", y="loss", group="model")
    bundle = mf.reporting.report_bundle(
        tables={"loss": mf.reporting.report_table(figure, precision=1)},
        figures={"loss_figure": figure},
        metadata={"study": "demo"},
    )
    rendered = mf.reporting.render_tables(bundle, format="markdown")

    assert figure.attrs["macroforecast_metadata_schema"]["kind"] == "figure_data"
    assert figure.attrs["macroforecast_metadata"]["source_shape"] == [3, 4]
    assert figure.attrs["macroforecast_metadata"]["x"] == "date"
    assert figure.attrs["macroforecast_metadata"]["y"] == ["loss"]
    assert figure.attrs["macroforecast_metadata"]["group"] == "model"
    assert list(figure.columns) == ["date", "loss", "model"]
    assert len(figure) == 2
    assert isinstance(bundle, mf.reporting.ReportBundle)
    assert bundle.metadata["n_tables"] == 1
    assert bundle.metadata["n_figures"] == 1
    assert "loss" in rendered
    assert bundle.to_dict()["metadata_schema"]["kind"] == "report_bundle"


def test_test_report_table_formats_forecast_tests_for_papers() -> None:
    loss_a = pd.Series([0.2, 0.3, 0.1, 0.4, 0.2, 0.3])
    loss_b = pd.Series([0.4, 0.5, 0.2, 0.5, 0.3, 0.6])
    tests = mf.tests.equal_predictive_tests(
        loss_a,
        loss_b,
        tests=("dm", "gw", "dmp", "hn"),
        error_a=np.sqrt(loss_a),
        error_b=np.sqrt(loss_b),
    )

    report = mf.reporting.test_report_table(
        tests,
        caption="Equal predictive ability tests",
        label="tab:equal_predictive_tests",
    )
    with_source = mf.reporting.test_report_table(tests, include_reference=True)
    provenance = mf.reporting.test_provenance_table(tests)

    assert list(report.data.columns) == ["Test", "Name", "Ref.", "Statistic", "p-value", "Reject", "N"]
    assert report.data.loc[0, "Test"] == "dm"
    assert report.data.loc[0, "Name"] == "Diebold-Mariano"
    assert report.data.loc[0, "Ref."] == "t"
    assert report.data.loc[0, "p-value"].endswith("***")
    assert report.data.loc[0, "Reject"] == "Yes"
    assert "\\caption{Equal predictive ability tests}" in report.to_latex()
    assert "Source" in with_source.data.columns
    assert with_source.data.loc[0, "Source"].startswith("R: forecast/R/DM2.R")
    assert provenance.data.loc[0, "R reference"] == "forecast/R/DM2.R::dm.test"
    assert "Partial alignment" in provenance.data.loc[0, "Alignment"]
    assert "No exact R comparator" in provenance.data.loc[1, "Alignment"]


def test_metric_report_tables_format_evaluation_reports_for_papers() -> None:
    rows: list[dict[str, object]] = []
    for horizon in (1, 2):
        for date_pos, date in enumerate(pd.date_range("2020-01-31", periods=4, freq="ME")):
            actual = 1.0 + date_pos + horizon
            for model, offset in {"model_a": 0.1, "model_b": 0.4, "bench": 0.8}.items():
                rows.append(
                    {
                        "date": date,
                        "model": model,
                        "horizon": horizon,
                        "actual": actual,
                        "prediction": actual + offset,
                    }
                )
    report = mf.evaluation.evaluate_report(
        pd.DataFrame(rows),
        metrics=("mse", "rmse", "mae", "relative_mse", "r2_oos"),
        benchmark_model="bench",
        include_decomposition=True,
    )

    scores = mf.reporting.metric_report_table(
        report,
        columns=("model", "horizon", "rmse", "r2_oos"),
        percent_columns=("r2_oos",),
        caption="Forecast accuracy",
    )
    benchmark = mf.reporting.metric_report_table(report, table="benchmark")
    bundle = mf.reporting.evaluation_report_tables(
        report,
        include=("scores", "ranking", "benchmark", "decomposition"),
        include_aggregations=True,
        percent_columns=("r2_oos",),
    )

    assert list(scores.data.columns) == ["Model", "H", "RMSE", "R2 OOS"]
    model_a_h1 = scores.data.loc[(scores.data["Model"] == "model_a") & (scores.data["H"] == "1")].iloc[0]
    assert model_a_h1["RMSE"] == "0.100"
    assert model_a_h1["R2 OOS"] == "98.438%"
    assert "\\caption{Forecast accuracy}" in scores.to_latex()
    assert "Relative MSE" in benchmark.data.columns
    assert {"scores", "ranking", "benchmark", "decomposition"}.issubset(bundle.tables)
    assert any(name.startswith("aggregation_") for name in bundle.tables)
    assert bundle.metadata["source_kind"] == "evaluation_report_tables"


def test_reporting_helpers_are_namespace_only() -> None:
    assert "reporting" in mf.__all__
    assert mf.reporting.report_table is not None
    assert mf.reporting.metric_report_table is not None
    assert mf.reporting.test_report_table is not None
    assert not hasattr(mf, "report_table")
    assert not hasattr(mf, "latex_table")
    assert not hasattr(mf, "metric_report_table")
    assert not hasattr(mf, "evaluation_report_tables")
    assert not hasattr(mf, "test_report_table")
    assert not hasattr(mf, "test_provenance_table")
