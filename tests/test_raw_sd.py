from __future__ import annotations

from pathlib import Path

from macroforecast.data import DataBundle, load_fred_sd, metadata

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_fred_sd_from_fixture_workbook(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_sd_sample.xlsx"
    bundle = load_fred_sd(local_source=fixture, cache_root=tmp_path)

    assert isinstance(bundle, DataBundle)
    assert metadata(bundle)["dataset"] == "fred_sd"
    assert "source_family" not in metadata(bundle)
    assert metadata(bundle)["frequency"] == "monthly"
    assert metadata(bundle)["native_frequency_counts"] == {"monthly": 6}
    assert metadata(bundle)["date_anchor_counts"] == {"month_start": 6}
    assert metadata(bundle)["support_tier"] == "stable"
    assert bundle.panel.shape[0] > 0
    assert any(col.startswith("UR_") for col in bundle.panel.columns)


def test_load_fred_sd_can_filter_variables_and_states(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_sd_sample.xlsx"
    bundle = load_fred_sd(
        local_source=fixture,
        cache_root=tmp_path,
        variables=["UR"],
        states=["CA", "TX"],
    )

    assert list(bundle.panel.columns) == ["UR_CA", "UR_TX"]
    report = bundle.panel.attrs["macrocast_reports"]["fred_sd_series_metadata"]
    assert report["contract_version"] == "fred_sd_series_metadata_v1"
    assert report["selector"] == {"states": ["CA", "TX"], "variables": ["UR"]}
    assert report["series_count"] == 2
    assert report["state_count"] == 2
    assert report["sd_variable_count"] == 1
    assert report["native_frequency_counts"] == {"monthly": 2}
    assert report["date_anchor_counts"] == {"month_start": 2}
    assert report["series"][0]["column"] == "UR_CA"
    assert report["series"][0]["sd_variable"] == "UR"
    assert report["series"][0]["state"] == "CA"
    assert report["series"][0]["source_sheet"] == "UR"
    assert report["series"][0]["date_anchor"] == "month_start"


def test_load_fred_sd_vintage_metadata(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_sd_sample.xlsx"
    bundle = load_fred_sd(
        vintage="2020-01",
        local_source=fixture,
        cache_root=tmp_path,
    )

    assert metadata(bundle)["vintage"] == "2020-01"
    assert metadata(bundle)["support_tier"] == "stable"


def test_load_fred_sd_detects_weekly_and_monthly_columns(tmp_path: Path) -> None:
    source = tmp_path / "fred_sd_weekly.csv"
    source.write_text(
        "date,W_CA,M_CA\n"
        "2020-01-01,1,10\n"
        "2020-01-08,2,\n"
        "2020-01-15,3,\n"
        "2020-01-22,4,\n"
        "2020-02-01,,20\n"
        "2020-02-05,10,\n"
        "2020-02-12,20,\n"
        "2020-02-19,30,\n",
        encoding="utf-8",
    )

    bundle = load_fred_sd(local_source=source, cache_root=tmp_path)
    report = bundle.panel.attrs["macrocast_reports"]["fred_sd_series_metadata"]

    assert metadata(bundle)["frequency"] == "mixed"
    assert metadata(bundle)["native_frequency_counts"] == {"monthly": 1, "weekly": 1}
    assert metadata(bundle)["date_anchor_counts"] == {"month_start": 1, "weekly": 1}
    assert metadata(bundle)["native_frequency_by_column"] == {"M_CA": "monthly", "W_CA": "weekly"}
    assert report["series"][0]["native_frequency"] == "weekly"
    assert report["series"][0]["date_anchor"] == "weekly"
