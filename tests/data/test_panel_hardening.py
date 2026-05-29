from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def test_as_panel_rejects_invalid_dates_by_default() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "not-a-date"], "x": [1.0, 2.0]})

    with pytest.raises(ValueError, match="invalid or missing date values"):
        mf.data.as_panel(frame, date="date")


def test_as_panel_can_permissively_report_dropped_dates_and_numeric_coercion() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "bad-date", "2020-03-01"], "x": ["1.0", "bad-number", "bad-number"]})

    panel = mf.data.as_panel(frame, date="date", strict=False)

    assert list(panel.index.strftime("%Y-%m-%d")) == ["2020-01-01", "2020-03-01"]
    report = panel.attrs["macroforecast_panel_report"]
    assert report["invalid_date_rows_dropped"] == 1
    assert report["numeric_coercion"]["coerced_cells"] == 1


def test_as_panel_rejects_non_numeric_values_by_default() -> None:
    frame = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "x": ["1.0", "bad-number"]})

    with pytest.raises(ValueError, match="non-numeric panel values"):
        mf.data.as_panel(frame, date="date")


def test_validate_panel_rejects_infinite_values() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=2, freq="MS"), "x": [1.0, np.inf]})

    with pytest.raises(ValueError, match="infinite values"):
        mf.data.as_panel(frame, date="date")


def test_spec_predictors_all_expands_to_non_target_columns() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "target": [1.0, 2.0, 3.0],
                "x": [4.0, 5.0, 6.0],
            }
        ),
        date="date",
    )

    data_spec = mf.data.spec(panel, target="target")

    assert data_spec.predictors == ("x",)
    assert data_spec.metadata["data_spec"]["predictors"] == ["x"]


def test_spec_rejects_target_leakage_in_explicit_predictors() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=3, freq="MS"),
                "target": [1.0, 2.0, 3.0],
                "x": [4.0, 5.0, 6.0],
            }
        ),
        date="date",
    )

    with pytest.raises(ValueError, match="must not include target"):
        mf.data.spec(panel, target="target", predictors=["target", "x"])


def test_load_custom_csv_preserves_first_pass_panel_report_when_permissive(tmp_path) -> None:
    path = tmp_path / "custom.csv"
    path.write_text(
        "date,x\n"
        "2020-01-01,1.0\n"
        "bad-date,2.0\n"
        "2020-03-01,bad-number\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", strict=False)

    assert bundle.metadata["panel"]["invalid_date_rows_dropped"] == 1
    assert bundle.metadata["panel"]["numeric_coercion"]["coerced_cells"] == 1
