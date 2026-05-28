from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import macroforecast as mf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _bundle(frame: pd.DataFrame, metadata: dict) -> mf.data.DataBundle:
    return mf.data.DataBundle(mf.data.as_panel(frame, date="date", metadata=metadata), metadata)


def test_combine_aligns_quarterly_source_to_monthly_grid():
    monthly = _bundle(
        pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=4, freq="MS"),
                "m": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        {"dataset": "monthly_source", "source_family": "test", "frequency": "monthly"},
    )
    quarterly = _bundle(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-04-01"]),
                "q": [10.0, 20.0],
            }
        ),
        {"dataset": "quarterly_source", "source_family": "test", "frequency": "quarterly"},
    )

    with pytest.warns(UserWarning, match="quarterly variables were aligned to monthly"):
        combined = mf.data.combine(monthly, quarterly, dataset="custom_combo", frequency="monthly")

    assert combined.metadata["dataset"] == "custom_combo"
    assert combined.metadata["frequency"] == "monthly"
    assert combined.panel["q"].tolist() == [10.0, 10.0, 10.0, 20.0]
    assert combined.metadata["source_by_column"] == {"m": "monthly_source", "q": "quarterly_source"}
    warning = combined.metadata["frequency_conversion_warnings"][0]
    assert warning["from_frequency"] == "quarterly"
    assert warning["to_frequency"] == "monthly"
    assert warning["variables"] == ["q"]


def test_load_fred_md_sd_combines_monthly_national_and_state_sources(tmp_path: Path):
    bundle = mf.data.load_fred_md_sd(
        cache_root=tmp_path,
        local_fred_md_source=FIXTURES / "fred_md_sample.csv",
        local_fred_sd_source=FIXTURES / "fred_sd_sample.csv",
        states=["CA"],
        variables=["UR"],
    )

    assert bundle.metadata["dataset"] == "fred_md+fred_sd"
    assert bundle.metadata["frequency"] == "monthly"
    assert "INDPRO" in bundle.panel.columns
    assert "UR_CA" in bundle.panel.columns
    assert bundle.metadata["transform_codes"]["INDPRO"] == 5
    assert bundle.metadata["source_by_column"]["UR_CA"] == "fred_sd"
    assert "fred_sd_series_metadata" in bundle.panel.attrs["macrocast_reports"]


def test_md_sd_monthly_warns_when_state_series_are_quarterly(tmp_path: Path):
    national = mf.data.load_fred_md(local_source=FIXTURES / "fred_md_sample.csv", cache_root=tmp_path)
    metadata = {"dataset": "fred_sd", "source_family": "fred-sd", "frequency": "state_monthly"}
    panel = mf.data.as_panel(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2000-01-01", "2000-04-01"]),
                "NQGSP_CA": [100.0, 110.0],
            }
        ),
        date="date",
        metadata=metadata,
    )
    panel.attrs["macrocast_reports"] = {
        "fred_sd_series_metadata": {
            "series": [{"column": "NQGSP_CA", "sd_variable": "NQGSP", "state": "CA", "native_frequency": "quarterly"}]
        }
    }
    regional = mf.data.DataBundle(panel, metadata)

    with pytest.warns(UserWarning, match="fred_sd quarterly variables were aligned to monthly"):
        bundle = mf.data.combine(national, regional, dataset="fred_md+fred_sd", frequency="monthly")

    warning = bundle.metadata["frequency_conversion_warnings"][0]
    assert warning["variables"] == ["NQGSP"]
    assert warning["rule"] == "repeat_within_quarter"


def test_load_fred_qd_sd_combines_quarterly_national_and_state_sources(tmp_path: Path):
    with pytest.warns(UserWarning, match="fred_sd monthly variables were aligned to quarterly"):
        bundle = mf.data.load_fred_qd_sd(
            cache_root=tmp_path,
            local_fred_qd_source=FIXTURES / "fred_qd_sample.csv",
            local_fred_sd_source=FIXTURES / "fred_sd_sample.csv",
            states=["CA"],
            variables=["UR"],
            monthly_to_quarterly="quarterly_endpoint",
        )

    assert bundle.metadata["dataset"] == "fred_qd+fred_sd"
    assert bundle.metadata["frequency"] == "quarterly"
    assert "GDPC1" in bundle.panel.columns
    assert "UR_CA" in bundle.panel.columns
    assert bundle.panel.loc[pd.Timestamp("2000-01-01"), "UR_CA"] == 5.2
    assert bundle.metadata["transform_codes"]["GDPC1"] == 5
    warning = bundle.metadata["frequency_conversion_warnings"][0]
    assert warning["variables"] == ["UR"]
    assert warning["rule"] == "quarterly_endpoint"


def test_cross_frequency_combination_records_not_recommended_note(tmp_path: Path):
    with pytest.warns(UserWarning, match="monthly variables were aligned to quarterly"):
        bundle = mf.data.load_fred_md_sd(
            cache_root=tmp_path,
            local_fred_md_source=FIXTURES / "fred_md_sample.csv",
            local_fred_sd_source=FIXTURES / "fred_sd_sample.csv",
            states=["CA"],
            variables=["UR"],
            frequency="quarterly",
        )

    assert any("not recommended" in note for note in bundle.metadata["parse_notes"])
