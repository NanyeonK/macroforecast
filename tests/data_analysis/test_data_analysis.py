from __future__ import annotations

import pandas as pd
import pytest
from scipy.stats import ks_2samp

from macroforecast.data_analysis import (
    DataAnalysisReport,
    changed_cell_count,
    changed_cell_summary,
    changed_cells,
    cleaning_effect_summary,
    compare_panels,
    correlation_shift,
    analyze_data,
    distribution_shift,
    missing_shift,
    panel_snapshots,
)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    index.name = "date"
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


def test_changed_cell_helpers_return_mask_count_and_summary():
    raw, clean = _panels()

    mask = changed_cells(raw, clean)
    summary = changed_cell_summary(raw, clean)

    assert mask.loc[raw.index[1], "x1"]
    assert mask.loc[raw.index[3], "x1"]
    assert not mask.loc[raw.index[0], "y"]
    assert changed_cell_count(raw, clean) == 2
    assert summary == {
        "common_rows": 4,
        "common_columns": 3,
        "common_cells": 12,
        "changed_cells": 2,
        "changed_cell_rate": pytest.approx(2 / 12),
        "tolerance": 0.0,
    }


def test_panel_snapshots_return_before_after_compact_views():
    raw, clean = _panels()

    snapshots = panel_snapshots(raw, clean)

    assert snapshots["before"]["missing_values"] == 1
    assert snapshots["after"]["missing_values"] == 0
    assert snapshots["before"]["n_columns"] == 3


def test_missing_shift_returns_per_column_dataframe():
    raw, clean = _panels()

    out = missing_shift(raw, clean)

    assert list(out.index) == ["y", "x1", "x2"]
    assert out.loc["x1", "raw_missing"] == 1
    assert out.loc["x1", "clean_missing"] == 0
    assert out.loc["x1", "delta_missing"] == -1
    assert out.loc["x1", "column_status"] == "common"


def test_missing_shift_marks_raw_only_and_clean_only_columns():
    raw, clean = _panels()
    raw = raw.assign(raw_only=1.0)
    clean = clean.assign(clean_only=2.0)

    out = missing_shift(raw, clean)

    assert out.loc["raw_only", "column_status"] == "raw_only"
    assert out.loc["clean_only", "column_status"] == "clean_only"


def test_distribution_shift_computes_requested_metrics():
    raw, clean = _panels()

    out = distribution_shift(raw, clean, metrics=("mean_change", "sd_change", "ks_statistic"))

    assert {"mean_change", "sd_change", "ks_statistic"} <= set(out.columns)
    assert out.loc["x1", "mean_change"] == pytest.approx(-10.0)
    assert out.loc["x1", "ks_statistic"] is not None
    assert out.loc["x1", "sample"] == "common_index"


def test_distribution_shift_uses_common_index_by_default():
    index = pd.date_range("2020-01-01", periods=3, freq="MS")
    index.name = "date"
    raw = pd.DataFrame({"x": [1.0, 100.0, 200.0]}, index=index)
    clean = pd.DataFrame({"x": [110.0, 210.0]}, index=index[1:])

    common = distribution_shift(raw, clean, metrics=("mean_change",))
    full = distribution_shift(raw, clean, metrics=("mean_change",), sample="full")

    assert common.loc["x", "sample_n"] == 2
    assert common.loc["x", "mean_change"] == pytest.approx(10.0)
    assert full.loc["x", "sample_n"] == 3
    assert full.loc["x", "mean_change"] == pytest.approx(179.0 / 3.0)


def test_distribution_shift_rejects_unknown_metric():
    raw, clean = _panels()

    with pytest.raises(ValueError, match="unknown distribution metric"):
        distribution_shift(raw, clean, metrics=("bad_metric",))  # type: ignore[list-item]


def test_distribution_shift_ks_statistic_matches_scipy():
    index = pd.date_range("2020-01-01", periods=4, freq="MS", name="date")
    raw = pd.DataFrame({"x": [1.0, 2.0, 3.0, 5.0]}, index=index)
    clean = pd.DataFrame({"x": [1.5, 2.5, 4.0, 6.0]}, index=index)

    out = distribution_shift(raw, clean, metrics=("ks_statistic",))

    assert out.loc["x", "ks_statistic"] == pytest.approx(
        ks_2samp(raw["x"], clean["x"]).statistic
    )


def test_correlation_shift_returns_clean_minus_raw_matrix():
    raw, clean = _panels()

    out = correlation_shift(raw, clean, fill_value=0.0)

    assert list(out.index) == ["y", "x1", "x2"]
    assert list(out.columns) == ["y", "x1", "x2"]
    assert out.loc["y", "y"] == pytest.approx(0.0)


def test_correlation_shift_uses_common_index_by_default():
    index = pd.date_range("2020-01-01", periods=4, freq="MS")
    index.name = "date"
    raw = pd.DataFrame({"a": [100.0, 1.0, 2.0, 3.0], "b": [0.0, 3.0, 2.0, 1.0]}, index=index)
    clean = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}, index=index[1:])

    out = correlation_shift(raw, clean)
    expected = clean.corr() - raw.loc[index[1:]].corr()

    pd.testing.assert_frame_equal(out, expected)


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
    assert report.metadata["data_analysis"]["options"]["sample"] == "common_index"
    assert report.metadata["data_analysis"]["effects"]["n_transform_codes"] == 1
    assert report.to_dict()["cleaning_effect_summary"]["transform_map_applied"] == {"x1": 1}
    assert report.to_dict()["metadata"]["data_analysis"]["after"]["missing_values"] == 0


def test_analyze_data_reads_preprocessing_metadata_from_clean_panel():
    raw, clean = _panels()
    clean.attrs["macroforecast_metadata"] = {
        "preprocessing": {
            "transform_state": {"x1": {"tcode": 5}},
            "steps": [
                {"step": "transform", "method": "official", "applied": {"x1": 5}},
                {"step": "tcode_lag", "method": "drop", "rows_removed": 1},
                {"step": "outliers", "method": "iqr", "missing_added": 2},
                {"step": "impute", "method": "em_factor", "missing_filled": 3},
                {"step": "frame", "method": "truncate", "input_shape": (4, 3), "output_shape": (3, 3)},
            ],
        }
    }

    report = analyze_data(raw, clean)

    assert report.cleaning_effect_summary["n_imputed_cells"] == 3
    assert report.cleaning_effect_summary["n_outliers_flagged"] == 2
    assert report.cleaning_effect_summary["n_truncated_obs"] == 2
    assert report.cleaning_effect_summary["transform_map_applied"] == {"x1": 5}
    assert report.cleaning_effect_summary["column_metadata"] == {"x1": {"tcode": 5}}
    assert report.metadata["data_analysis"]["effects"]["has_cleaning_log"] is True


def test_functions_require_pandas_dataframes():
    raw, clean = _panels()

    with pytest.raises(TypeError, match="raw must be a pandas DataFrame"):
        compare_panels({"x": [1]}, clean)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="clean must be a pandas DataFrame"):
        compare_panels(raw, {"x": [1]})  # type: ignore[arg-type]


def test_functions_reject_duplicate_index_and_negative_tolerance():
    raw, clean = _panels()
    duplicated = raw.copy()
    duplicated.index = [raw.index[0], *raw.index[:3]]
    duplicated.index.name = "date"

    with pytest.raises(ValueError, match="duplicate index"):
        compare_panels(duplicated, clean)
    with pytest.raises(ValueError, match="non-negative"):
        compare_panels(raw, clean, tolerance=-1.0)


def test_functions_reject_noncanonical_panel_contract():
    raw, clean = _panels()
    raw_bad = raw.copy()
    raw_bad.index.name = None

    with pytest.raises(ValueError, match="canonical panel contract"):
        compare_panels(raw_bad, clean)


def test_distribution_shift_common_index_requires_overlap():
    raw_index = pd.date_range("2020-01-01", periods=2, freq="MS", name="date")
    clean_index = pd.date_range("2021-01-01", periods=2, freq="MS", name="date")
    raw = pd.DataFrame({"x": [1.0, 2.0]}, index=raw_index)
    clean = pd.DataFrame({"x": [3.0, 4.0]}, index=clean_index)

    with pytest.raises(ValueError, match="common date"):
        distribution_shift(raw, clean)
