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


def test_spec_allows_target_only_design() -> None:
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

    data_spec = mf.data.spec(panel, target="target", predictors=[])

    assert data_spec.predictors == ()
    assert list(data_spec.panel.columns) == ["target"]
    assert data_spec.metadata["data_spec"]["predictors"] == []


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


def test_as_panel_duplicate_dates_in_long_format_names_pivot_fix() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"],
            "country": ["US", "CA", "US", "CA"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    with pytest.raises(ValueError, match="pivot to wide"):
        mf.data.as_panel(frame, date="date")


def test_load_custom_csv_custom_na_values_pass_strict_mode(tmp_path) -> None:
    path = tmp_path / "custom_na.csv"
    path.write_text(
        "date,x\n"
        "2020-01-01,1.0\n"
        "2020-02-01,MISSING\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", na_values=["MISSING"])

    assert np.isnan(bundle.panel.loc[pd.Timestamp("2020-02-01"), "x"])


def test_load_custom_csv_dayfirst_dates_parse_when_requested(tmp_path) -> None:
    path = tmp_path / "dayfirst.csv"
    path.write_text(
        "date,x\n"
        "01/02/2020,1.0\n"
        "02/03/2020,2.0\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", dayfirst=True)

    assert list(bundle.panel.index) == [
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-03-02"),
    ]


def test_load_custom_csv_date_format_parses_when_requested(tmp_path) -> None:
    path = tmp_path / "date_format.csv"
    path.write_text(
        "date,x\n"
        "20200131,1.0\n"
        "20200229,2.0\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(path, date="date", date_format="%Y%m%d")

    assert list(bundle.panel.index) == [
        pd.Timestamp("2020-01-31"),
        pd.Timestamp("2020-02-29"),
    ]


def test_set_frequencies_marks_existing_panel_as_mixed() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "monthly": [1.0, 2.0, 3.0, 4.0],
                "quarterly": [10.0, np.nan, np.nan, 20.0],
            }
        ),
        date="date",
    )

    bundle = mf.data.set_frequencies(
        panel,
        {"monthly": "monthly", "quarterly": "quarterly"},
        metadata={"dataset": "custom_mixed"},
    )

    assert bundle.metadata["frequency"] == "mixed"
    assert bundle.metadata["native_frequency_by_column"] == {
        "monthly": "monthly",
        "quarterly": "quarterly",
    }
    assert bundle.metadata["native_frequency_counts"] == {"monthly": 1, "quarterly": 1}
    assert mf.data.panel_info(bundle)["frequency"] == "mixed"


def test_load_custom_csv_accepts_column_frequency_contract(tmp_path) -> None:
    path = tmp_path / "mixed.csv"
    path.write_text(
        "date,m,q\n"
        "2020-01-01,1,10\n"
        "2020-02-01,2,\n"
        "2020-03-01,3,\n"
        "2020-04-01,4,20\n",
        encoding="utf-8",
    )

    bundle = mf.data.load_custom_csv(
        path,
        date="date",
        frequency_by_column={"m": "m", "q": "q"},
    )

    assert bundle.metadata["frequency"] == "mixed"
    assert bundle.metadata["native_frequency_by_column"] == {"m": "monthly", "q": "quarterly"}
    assert bundle.panel["q"].isna().sum() == 2


def test_set_frequencies_rejects_incomplete_frequency_map() -> None:
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
                "x": [1.0, 2.0],
                "y": [3.0, 4.0],
            }
        ),
        date="date",
    )

    with pytest.raises(ValueError, match="must include every panel column"):
        mf.data.set_frequencies(panel, {"x": "monthly"})
