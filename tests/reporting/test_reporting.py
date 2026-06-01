from __future__ import annotations

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
    assert list(figure.columns) == ["date", "loss", "model"]
    assert len(figure) == 2
    assert isinstance(bundle, mf.reporting.ReportBundle)
    assert bundle.metadata["n_tables"] == 1
    assert bundle.metadata["n_figures"] == 1
    assert "loss" in rendered
    assert bundle.to_dict()["metadata_schema"]["kind"] == "report_bundle"


def test_reporting_exports_are_available_at_top_level() -> None:
    assert mf.report_table is mf.reporting.report_table
    assert mf.latex_table is mf.reporting.latex_table
