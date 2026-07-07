from __future__ import annotations

import numpy as np
import pytest
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


def test_reporting_presets_wrap_metric_and_test_tables() -> None:
    rows: list[dict[str, object]] = []
    for horizon in (1, 2):
        for date_pos, date in enumerate(pd.date_range("2021-01-31", periods=4, freq="ME")):
            actual = 2.0 + date_pos
            for model, offset in {"model_a": 0.1, "model_b": 0.3, "bench": 0.6}.items():
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
        metrics=("rmse", "mae", "r2_oos"),
        benchmark_model="bench",
    )
    tests = mf.tests.equal_predictive_tests(
        pd.Series([0.1, 0.2, 0.1, 0.2]),
        pd.Series([0.4, 0.5, 0.3, 0.4]),
        tests=("dm",),
    )

    accuracy = mf.reporting.accuracy_table(report)
    comparison = mf.reporting.model_comparison_table(report)
    test_table = mf.reporting.forecast_test_table(tests)

    assert list(accuracy.data.columns) == ["Model", "H", "RMSE", "MAE", "R2 OOS"]
    assert accuracy.caption == "Forecast accuracy"
    assert accuracy.metadata["source_kind"] == "accuracy_table"
    assert "Rank" in comparison.data.columns
    assert comparison.metadata["source_kind"] == "model_comparison_table"
    assert test_table.caption == "Forecast comparison tests"
    assert test_table.metadata["source_kind"] == "forecast_test_table"


def test_reporting_helpers_are_namespace_only() -> None:
    assert "reporting" in mf.__all__
    assert mf.reporting.report_table is not None
    assert mf.reporting.metric_report_table is not None
    assert mf.reporting.test_report_table is not None
    assert mf.reporting.accuracy_table is not None
    assert mf.reporting.model_comparison_table is not None
    assert mf.reporting.forecast_test_table is not None
    assert mf.reporting.paper_accuracy_table is not None
    assert mf.reporting.pairwise_test_table is not None
    assert not hasattr(mf, "report_table")
    assert not hasattr(mf, "latex_table")
    assert not hasattr(mf, "metric_report_table")
    assert not hasattr(mf, "evaluation_report_tables")
    assert not hasattr(mf, "test_report_table")
    assert not hasattr(mf, "test_provenance_table")
    assert not hasattr(mf, "accuracy_table")
    assert not hasattr(mf, "model_comparison_table")
    assert not hasattr(mf, "forecast_test_table")
    assert not hasattr(mf, "paper_accuracy_table")
    assert not hasattr(mf, "pairwise_test_table")


def _pairwise_master() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2020-01-31", periods=14, freq="ME")
    errors = {
        "AR": np.array([0.10, -0.22, 0.18, -0.12, 0.28, -0.16, 0.20, -0.24, 0.14, -0.10, 0.26, -0.18, 0.22, -0.14]),
        "RF": np.array([0.30, -0.34, 0.26, -0.20, 0.36, -0.28, 0.31, -0.30, 0.24, -0.22, 0.35, -0.32, 0.29, -0.25]),
        "Ridge": np.array([-0.16, 0.18, -0.12, 0.20, -0.22, 0.24, -0.18, 0.16, -0.14, 0.22, -0.20, 0.26, -0.17, 0.19]),
    }
    for origin, date in enumerate(dates):
        actual = 2.0 + 0.04 * origin + 0.2 * np.sin(origin / 3.0)
        for model, model_errors in errors.items():
            rows.append(
                {
                    "target": "y",
                    "horizon": 3,
                    "origin": origin,
                    "date": date,
                    "contender": model,
                    "prediction": actual + model_errors[origin],
                    "actual": actual,
                }
            )
    return pd.DataFrame(rows)


def test_pairwise_test_table_builds_square_dm_matrix_from_public_test() -> None:
    master = _pairwise_master()
    models = ["AR", "RF", "Ridge"]

    pvals = mf.reporting.pairwise_test_table(
        master,
        target="y",
        horizon=3,
        models=models,
        test_options={"hac_lags": 0},
    )
    stats = mf.reporting.pairwise_test_table(
        master,
        target="y",
        horizon=3,
        models=models,
        value="stat",
        test_options={"hac_lags": 0},
    )

    assert list(pvals.index) == models
    assert list(pvals.columns) == models
    assert pvals.shape == (3, 3)
    assert np.isnan(pvals.to_numpy().diagonal()).all()
    assert np.isclose(pvals.loc["AR", "RF"], pvals.loc["RF", "AR"])
    assert np.isclose(stats.loc["AR", "RF"], -stats.loc["RF", "AR"])

    cell = master.loc[(master["target"] == "y") & (master["horizon"] == 3)]
    wide = cell.pivot_table(index="origin", columns="contender", values="prediction", aggfunc="mean")
    actual = cell.groupby("origin")["actual"].first().reindex(wide.index)
    loss_ar = (wide["AR"].to_numpy(dtype=float) - actual.to_numpy(dtype=float)) ** 2
    loss_rf = (wide["RF"].to_numpy(dtype=float) - actual.to_numpy(dtype=float)) ** 2
    expected = mf.tests.dm_test(
        loss_ar,
        loss_rf,
        horizon=3,
        hac_lags=0,
        input_type="loss",
    )
    assert np.isclose(pvals.loc["AR", "RF"], expected.p_value)
    assert np.isclose(stats.loc["AR", "RF"], expected.statistic)


def test_pairwise_test_table_honors_test_options_and_renders_latex() -> None:
    master = _pairwise_master()

    default = mf.reporting.pairwise_test_table(
        master,
        target="y",
        horizon=3,
        value="stat",
    )
    fixed_lag = mf.reporting.pairwise_test_table(
        master,
        target="y",
        horizon=3,
        value="stat",
        test_options={"hac_lags": 0},
    )
    starred = mf.reporting.pairwise_test_table(
        {"forecasts": master},
        target="y",
        horizon=3,
        test_options={"hac_lags": 0},
        stars=True,
    )

    assert not np.isclose(default.loc["AR", "RF"], fixed_lag.loc["AR", "RF"])
    assert starred.loc["AR", "AR"] == ""
    assert isinstance(starred.loc["AR", "RF"], str)
    pytest.importorskip("jinja2")  # pandas>=2 DataFrame.to_latex delegates to Styler
    latex = starred.to_latex()
    assert "\\begin{tabular}" in latex
    assert "AR" in latex
    assert starred.attrs["macroforecast_metadata_schema"]["kind"] == "pairwise_test_table"


def _paper_report(
    *,
    targets: tuple[str, ...] = ("INDPRO",),
    include_significance: bool = True,
    include_mcs: bool = True,
    horizons: tuple[int, ...] = (1, 3),
):
    """A hand-built PipelineReport-shaped fixture: AR benchmark vs. RF, per target.

    relative_mse is chosen so rel-RMSE = sqrt(relative_mse) is exact to floating
    precision (0.81 -> 0.900, 0.64 -> 0.800), so precision=3 formatting is
    unambiguous.
    """

    accuracy_rows: list[dict[str, object]] = []
    significance_rows: list[dict[str, object]] = []
    mcs_rows: list[dict[str, object]] = []
    for one_target in targets:
        for horizon in horizons:
            accuracy_rows.append(
                {
                    "target": one_target, "horizon": horizon, "contender": "AR",
                    "rmse": 1.0, "relative_mse": 1.0, "r2_oos": 0.0,
                    "n_common": 100, "is_benchmark": True, "benchmark_present": True,
                }
            )
            rel_mse = 0.81 if horizon == 1 else 0.64
            accuracy_rows.append(
                {
                    "target": one_target, "horizon": horizon, "contender": "RF",
                    "rmse": rel_mse ** 0.5, "relative_mse": rel_mse,
                    "r2_oos": 1.0 - rel_mse,
                    "n_common": 100, "is_benchmark": False, "benchmark_present": True,
                }
            )
            significance_rows.append(
                {
                    "target": one_target, "horizon": horizon, "contender": "RF",
                    "dm_stat": -2.5 if horizon == 1 else -0.8,
                    "dm_p": 0.02 if horizon == 1 else 0.2,
                }
            )
            mcs_rows.append({"target": one_target, "horizon": horizon, "contender": "AR", "in_mcs": True})
            mcs_rows.append(
                {
                    "target": one_target, "horizon": horizon, "contender": "RF",
                    "in_mcs": horizon == 1,
                }
            )

    class _Report:
        accuracy = pd.DataFrame(accuracy_rows)
        significance = pd.DataFrame(significance_rows) if include_significance else pd.DataFrame()
        mcs = pd.DataFrame(mcs_rows) if include_mcs else pd.DataFrame()

    return _Report()


def test_paper_accuracy_table_joins_rel_rmse_stars_and_mcs_mark() -> None:
    report = _paper_report()

    table = mf.reporting.paper_accuracy_table(report)

    assert isinstance(table, mf.reporting.ReportTable)
    assert list(table.data.columns) == ["Model", "h1", "h3"]
    assert list(table.data["Model"]) == ["AR (benchmark)", "RF"]
    ar_row = table.data.loc[table.data["Model"] == "AR (benchmark)"].iloc[0]
    rf_row = table.data.loc[table.data["Model"] == "RF"].iloc[0]
    # Benchmark: rel-RMSE 1.000 by construction, never starred, MCS-marked.
    assert ar_row["h1"] == "1.000†"
    assert ar_row["h3"] == "1.000†"
    # RF: rel-RMSE = sqrt(relative_mse); h1 is significant (p=0.02 -> **) and in
    # the MCS; h3 is not significant (p=0.2) and not in the MCS.
    assert rf_row["h1"] == "0.900**†"
    assert rf_row["h3"] == "0.800"
    assert table.caption == "Forecast accuracy — INDPRO"
    assert table.metadata["source_kind"] == "paper_accuracy_table"
    assert table.metadata["benchmark"] == "AR"


def test_paper_accuracy_table_to_latex_renders():
    report = _paper_report()
    table = mf.reporting.paper_accuracy_table(report)

    latex = table.to_latex(booktabs=True)

    assert "\\toprule" in latex
    assert "\\bottomrule" in latex
    assert "RF" in latex
    assert "0.900" in latex


def test_paper_accuracy_table_benchmark_row_false_drops_benchmark() -> None:
    report = _paper_report()

    table = mf.reporting.paper_accuracy_table(report, benchmark_row=False)

    assert list(table.data["Model"]) == ["RF"]


def test_paper_accuracy_table_custom_metric_column() -> None:
    report = _paper_report()

    table = mf.reporting.paper_accuracy_table(report, metric="relative_mse")

    rf_row = table.data.loc[table.data["Model"] == "RF"].iloc[0]
    assert rf_row["h1"] == "0.810**†"


def test_paper_accuracy_table_multi_target_returns_dict() -> None:
    report = _paper_report(targets=("INDPRO", "UNRATE"))

    tables = mf.reporting.paper_accuracy_table(report)

    assert isinstance(tables, dict)
    assert set(tables) == {"INDPRO", "UNRATE"}
    assert all(isinstance(value, mf.reporting.ReportTable) for value in tables.values())

    single = mf.reporting.paper_accuracy_table(report, target="UNRATE")
    assert isinstance(single, mf.reporting.ReportTable)
    assert single.metadata["target"] == "UNRATE"


def test_paper_accuracy_table_missing_significance_and_mcs_frames() -> None:
    report = _paper_report(include_significance=False, include_mcs=False)

    table = mf.reporting.paper_accuracy_table(report)

    rf_row = table.data.loc[table.data["Model"] == "RF"].iloc[0]
    # No significance/MCS frames at all -> plain formatted numbers, no crash.
    assert rf_row["h1"] == "0.900"
    assert rf_row["h3"] == "0.800"


def test_paper_accuracy_table_selects_subsample_window() -> None:
    report = _paper_report()

    class _Report:
        accuracy = pd.concat(
            [
                report.accuracy.assign(subsample="full"),
                report.accuracy.assign(subsample="ex_covid", relative_mse=lambda x: x["relative_mse"] * 1.21),
            ],
            ignore_index=True,
        )
        significance = pd.concat(
            [
                report.significance.assign(subsample="full"),
                report.significance.assign(subsample="ex_covid"),
            ],
            ignore_index=True,
        )
        mcs = pd.concat(
            [
                report.mcs.assign(subsample="full"),
                report.mcs.assign(subsample="ex_covid"),
            ],
            ignore_index=True,
        )

    default_table = mf.reporting.paper_accuracy_table(_Report())
    covid_table = mf.reporting.paper_accuracy_table(_Report(), subsample="ex_covid")

    assert default_table.metadata["subsample"] == "full"
    assert covid_table.metadata["subsample"] == "ex_covid"
    rf_row = covid_table.data.loc[covid_table.data["Model"] == "RF"].iloc[0]
    assert rf_row["h1"].startswith("0.990")


def test_paper_accuracy_table_missing_benchmark_edge_case() -> None:
    # The named benchmark never appears as a contender for this target (e.g. it
    # failed to run / was never included): accuracy has RF only, and
    # relative_mse could not be computed against anything.
    accuracy_frame = pd.DataFrame(
        [
            {
                "target": "INDPRO", "horizon": 1, "contender": "RF",
                "rmse": 0.9, "relative_mse": float("nan"), "r2_oos": float("nan"),
                "n_common": 100, "is_benchmark": False, "benchmark_present": False,
            }
        ]
    )

    class _Report:
        accuracy = accuracy_frame
        significance = pd.DataFrame()
        mcs = pd.DataFrame()

    table = mf.reporting.paper_accuracy_table(_Report())

    assert list(table.data["Model"]) == ["RF"]
    assert table.data.loc[0, "h1"] == "--"
    assert table.metadata["benchmark"] is None


def test_paper_accuracy_table_single_horizon_edge_case() -> None:
    report = _paper_report(horizons=(1,))

    table = mf.reporting.paper_accuracy_table(report)

    assert list(table.data.columns) == ["Model", "h1"]
    assert len(table.data) == 2
    assert isinstance(table.data, pd.DataFrame)
