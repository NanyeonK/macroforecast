from __future__ import annotations

from pathlib import Path

from macrocast.raw import RawLoadResult, load_fred_sd

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_fred_sd_from_fixture_workbook(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_sd_sample.xlsx"
    result = load_fred_sd(local_source=fixture, cache_root=tmp_path)

    assert isinstance(result, RawLoadResult)
    assert result.dataset_metadata.dataset == "fred_sd"
    assert result.dataset_metadata.frequency == "state_monthly"
    assert result.dataset_metadata.support_tier == "stable"
    assert result.data.shape[0] > 0
    assert any(col.startswith("UR_") for col in result.data.columns)


def test_load_fred_sd_can_filter_variables_and_states(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_sd_sample.xlsx"
    result = load_fred_sd(
        local_source=fixture,
        cache_root=tmp_path,
        variables=["UR"],
        states=["CA", "TX"],
    )

    assert list(result.data.columns) == ["UR_CA", "UR_TX"]
    report = result.data.attrs["macrocast_reports"]["fred_sd_series_metadata"]
    assert report["contract_version"] == "fred_sd_series_metadata_v1"
    assert report["selector"] == {"states": ["CA", "TX"], "variables": ["UR"]}
    assert report["series_count"] == 2
    assert report["state_count"] == 2
    assert report["sd_variable_count"] == 1
    assert report["native_frequency_counts"] == {"monthly": 2}
    assert report["series"][0]["column"] == "UR_CA"
    assert report["series"][0]["sd_variable"] == "UR"
    assert report["series"][0]["state"] == "CA"
    assert report["series"][0]["source_sheet"] == "UR"


def test_load_fred_sd_vintage_is_marked_stable(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_sd_sample.xlsx"
    result = load_fred_sd(
        vintage="2020-01",
        local_source=fixture,
        cache_root=tmp_path,
    )

    assert result.dataset_metadata.vintage == "2020-01"
    assert result.dataset_metadata.support_tier == "stable"
