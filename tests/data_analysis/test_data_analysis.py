from __future__ import annotations

import pandas as pd
import pytest

from macroforecast.data_analysis import (
    DataAnalysisReport,
    cleaning_effect_summary,
    compare_panels,
    correlation_shift,
    analyze_data,
    distribution_shift,
    missing_shift,
)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    raw = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x1": [10.0, None, 30.0, 80.0],
            "x2": [5.0, 6.0, 7.0, 8.0],
        },
        index=index,
    )
    clean = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x1": [10.0, 40.0, 30.0, 40.0],
            "x2": [5.0, 6.0, 7.0, 8.0],
        },
        index=index,
    )
    return raw, clean


def test_compare_panels_reports_shape_missing_and_changed_cells():
    raw, clean = _panels()

    out = compare_panels(raw, clean)

    assert out["raw_shape"] == (4, 3)
    assert out["clean_shape"] == (4, 3)
    assert out["raw_missing_total"] == 1
    assert out["clean_missing_total"] == 0
    assert out["common_columns"] == ["y", "x1", "x2"]
    assert out["changed_cell_count"] == 2


def test_missing_shift_returns_per_column_dataframe():
    raw, clean = _panels()

    out = missing_shift(raw, clean)

    assert list(out.index) == ["y", "x1", "x2"]
    assert out.loc["x1", "raw_missing"] == 1
    assert out.loc["x1", "clean_missing"] == 0
    assert out.loc["x1", "delta_missing"] == -1


def test_distribution_shift_computes_requested_metrics():
    raw, clean = _panels()

    out = distribution_shift(raw, clean, metrics=("mean_change", "sd_change", "ks_statistic"))

    assert {"mean_change", "sd_change", "ks_statistic"} <= set(out.columns)
    assert out.loc["x1", "mean_change"] == pytest.approx(-10.0)
    assert out.loc["x1", "ks_statistic"] is not None


def test_distribution_shift_rejects_unknown_metric():
    raw, clean = _panels()

    with pytest.raises(ValueError, match="unknown distribution metric"):
        distribution_shift(raw, clean, metrics=("bad_metric",))  # type: ignore[list-item]


def test_correlation_shift_returns_clean_minus_raw_matrix():
    raw, clean = _panels()

    out = correlation_shift(raw, clean, fill_value=0.0)

    assert list(out.index) == ["y", "x1", "x2"]
    assert list(out.columns) == ["y", "x1", "x2"]
    assert out.loc["y", "y"] == pytest.approx(0.0)


def test_cleaning_effect_summary_normalizes_preprocessing_metadata():
    out = cleaning_effect_summary(
        cleaning_metadata={
            "n_imputed_cells": 1,
            "n_outliers_flagged": 2,
            "n_truncated_obs": 3,
            "transform_map_applied": {"x1": 5},
            "cleaning_log": {"impute": "mean"},
        }
    )

    assert out["n_imputed_cells"] == 1
    assert out["n_outliers_flagged"] == 2
    assert out["n_truncated_obs"] == 3
    assert out["transform_map_applied"] == {"x1": 5}
    assert out["cleaning_log"] == {"impute": "mean"}


def test_analyze_data_returns_report_with_optional_correlation():
    raw, clean = _panels()
    raw.attrs["macroforecast_metadata"] = {"dataset": "unit", "stage": "raw"}
    clean.attrs["macroforecast_metadata"] = {"dataset": "unit", "stage": "preprocessed"}

    report = analyze_data(
        raw,
        clean,
        include_correlation=True,
        n_imputed_cells=1,
        transform_map_applied={"x1": 1},
    )

    assert isinstance(report, DataAnalysisReport)
    assert report.comparison["clean_missing_total"] == 0
    assert report.missing_shift.loc["x1", "delta_missing"] == -1
    assert "mean_change" in report.distribution_shift.columns
    assert report.correlation_shift is not None
    assert report.cleaning_effect_summary["n_imputed_cells"] == 1
    assert report.metadata["dataset"] == "unit"
    assert report.metadata["stage"] == "preprocessed"
    assert report.metadata["data_analysis"]["before"]["missing_values"] == 1
    assert report.metadata["data_analysis"]["after"]["missing_values"] == 0
    assert report.metadata["data_analysis"]["common"]["changed_cells"] == 2
    assert report.metadata["data_analysis"]["options"]["include_correlation"] is True
    assert report.metadata["data_analysis"]["effects"]["n_transform_codes"] == 1
    assert report.to_dict()["cleaning_effect_summary"]["transform_map_applied"] == {"x1": 1}
    assert report.to_dict()["metadata"]["data_analysis"]["after"]["missing_values"] == 0


def test_functions_require_pandas_dataframes():
    raw, clean = _panels()

    with pytest.raises(TypeError, match="raw must be a pandas DataFrame"):
        compare_panels({"x": [1]}, clean)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="clean must be a pandas DataFrame"):
        compare_panels(raw, {"x": [1]})  # type: ignore[arg-type]
