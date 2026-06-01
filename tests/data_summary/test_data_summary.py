import pandas as pd
import pytest
import numpy as np

import macroforecast as mf


def _panel() -> pd.DataFrame:
    return mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "y": [1.0, 2.0, None, 4.0],
                "x": [10.0, 11.0, 12.0, 13.0],
            }
        ),
        date="date",
        metadata={"dataset": "custom", "source_family": "unit_test"},
    )


def test_summarize_data_returns_single_panel_report() -> None:
    bundle = mf.data.DataBundle(_panel(), {"dataset": "custom", "source_family": "unit_test"})

    report = mf.data_summary.summarize_data(bundle, include_correlation=True)

    assert isinstance(report, mf.data_summary.DataSummaryReport)
    assert report.overview["n_rows"] == 4
    assert report.coverage.loc["y", "n_obs"] == 3
    assert report.missing.loc["y", "longest_missing_run"] == 1
    assert report.univariate.loc["x", "n_missing"] == 0
    assert report.correlation is not None
    assert report.outliers is None
    assert report.stationarity is None
    assert report.metadata["dataset"] == "custom"
    assert report.metadata["data_summary"]["include_correlation"] is True
    assert report.metadata["data_summary"]["panel"]["n_rows"] == 4
    assert report.metadata["data_summary"]["outputs"]["stationarity"] is False
    assert report.to_dict()["overview"]["n_columns"] == 2


def test_data_summary_accepts_preprocessed_data() -> None:
    processed = mf.preprocessing.reprocess(
        _panel(),
        transform="none",
        outliers="none",
        impute="mean",
        frame="keep",
    )

    coverage = mf.data_summary.sample_coverage(processed)
    report = mf.data_summary.summarize_data(processed)

    assert coverage.loc["y", "n_missing"] == 0
    assert report.metadata["data_summary"]["input"]["has_preprocessing"] is True


def test_data_summary_small_helpers_return_compact_outputs() -> None:
    panel = _panel()

    snapshot = mf.data_summary.panel_snapshot(panel)
    counts = mf.data_summary.observation_counts(panel)
    rates = mf.data_summary.missing_rates(panel)

    assert snapshot == {
        "n_rows": 4,
        "n_columns": 2,
        "start": "2020-01-01",
        "end": "2020-04-01",
        "missing_values": 1,
        "frequency": "MS",
    }
    assert counts.to_dict() == {"y": 3, "x": 4}
    assert rates.loc["y"] == pytest.approx(0.25)
    assert rates.loc["x"] == 0.0


def test_summary_metric_and_correlation_validation() -> None:
    with pytest.raises(ValueError, match="unknown summary metric"):
        mf.data_summary.univariate_summary(_panel(), metrics=["bad_metric"])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="method must be one of"):
        mf.data_summary.correlation_matrix(_panel(), method="bad")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="min_periods"):
        mf.data_summary.correlation_matrix(_panel(), min_periods=0)


def test_outlier_summary_flags_iqr_and_zscore() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=8, freq="MS"),
                "x": [1.0, 1.1, 1.2, 1.1, 1.0, 1.2, 1.1, 10.0],
            }
        ),
        date="date",
    )

    outliers = mf.data_summary.outlier_summary(panel, method="multi", iqr_threshold=3.0, zscore_threshold=2.0)

    assert outliers.loc["x", "iqr_outlier_count"] == 1
    assert outliers.loc["x", "zscore_outlier_count"] == 1


def test_outlier_summary_validates_thresholds_and_matches_preprocessing_zscore() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "x": [1.0, 1.0, 10.0],
            }
        ),
        date="date",
    )

    with pytest.raises(ValueError, match="iqr_threshold"):
        mf.data_summary.outlier_summary(panel, iqr_threshold=0)
    with pytest.raises(ValueError, match="zscore_threshold"):
        mf.data_summary.outlier_summary(panel, method="zscore", zscore_threshold=-1)

    summary = mf.data_summary.outlier_summary(panel, method="zscore", zscore_threshold=1.3)

    assert summary.loc["x", "zscore_outlier_count"] == 1


def test_summarize_data_can_include_outliers_and_stationarity() -> None:
    rng = np.random.default_rng(0)
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2010-01-01", periods=80, freq="MS"),
                "y": rng.normal(size=80),
            }
        ),
        date="date",
    )

    report = mf.data_summary.summarize_data(
        panel,
        include_outliers=True,
        include_stationarity=True,
        stationarity_test="adf",
    )

    assert report.outliers is not None
    assert report.stationarity is not None
    assert report.metadata["data_summary"]["include_outliers"] is True
    assert report.metadata["data_summary"]["stationarity_test"] == "adf"
    assert report.stationarity["by_series"]["y"]["adf"]["reject_unit_root"] is True
    assert "stationarity" in report.to_dict()


def test_stationarity_scope_uses_data_spec_targets() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2010-01-01", periods=20, freq="MS"),
                "y": np.arange(20, dtype=float),
                "x": np.arange(20, dtype=float),
            }
        ),
        date="date",
    )
    spec = mf.data.spec(panel, target="y", horizons=[1])

    results = mf.data_summary.stationarity_tests(spec, test="adf", scope="target_only")

    assert set(results["by_series"]) == {"y"}


def test_stationarity_scope_requires_valid_targets_for_target_scopes() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2010-01-01", periods=20, freq="MS"),
                "y": np.arange(20, dtype=float),
                "x": np.arange(20, dtype=float),
            }
        ),
        date="date",
    )

    with pytest.raises(ValueError, match="requires target"):
        mf.data_summary.stationarity_tests(panel, test="adf", scope="target_only")
    with pytest.raises(ValueError, match="not in the panel"):
        mf.data_summary.stationarity_tests(panel, test="adf", scope="target_only", target="missing")
    with pytest.raises(ValueError, match="alpha"):
        mf.data_summary.stationarity_tests(panel, test="adf", alpha=1.0)
