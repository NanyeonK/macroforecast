from __future__ import annotations

import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from macroforecast.data import load_fred_md, load_fred_qd, load_fred_sd, metadata
from macroforecast.data.errors import RawDownloadError, RawVersionFormatError
from macroforecast.data.loaders import (
    _atomic_copy,
    _extract_vintage_xlsx_from_zip,
    _fred_sd_series_xlsx_url,
    _fred_sd_series_zip_url,
    _historical_zip_links_from_html,
    _latest_series_url_from_html,
    _parse_fred_csv,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_fred_csv_accepts_official_header_then_transform_layout(tmp_path: Path) -> None:
    path = tmp_path / "official.csv"
    path.write_text(
        "sasdate,INDPRO,UNRATE\n"
        "Transform:,5,2\n"
        "1/1/2000,100.0,4.0\n"
        "2/1/2000,101.0,4.1\n"
    )

    df, tcodes = _parse_fred_csv(path)

    assert list(df.columns) == ["INDPRO", "UNRATE"]
    assert tcodes == {"INDPRO": 5, "UNRATE": 2}
    assert df.index[0].strftime("%Y-%m") == "2000-01"


def test_load_fred_md_uses_local_historical_zip_fallback(tmp_path: Path) -> None:
    zip_path = tmp_path / "fred_md_historical.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "2018-02.csv",
            ",5,5,2,6\n"
            "sasdate,INDPRO,RPI,UNRATE,CPIAUCSL\n"
            "1/1/2000,100.0,1000.0,4.0,170.0\n"
            "2/1/2000,101.0,1010.0,4.1,170.5\n",
        )

    bundle = load_fred_md(
        vintage="2018-02",
        cache_root=tmp_path,
        local_zip_source=zip_path,
    )

    assert metadata(bundle)["dataset"] == "fred_md"
    assert metadata(bundle)["vintage"] == "2018-02"
    assert bundle.panel.shape == (2, 4)


def test_fred_md_qd_historical_zip_links_are_parsed_from_official_page() -> None:
    html = """
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip?hash=abc">MD archive</a>
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-qd-2005-01-to-2024-10.zip?hash=def">QD archive</a>
    """

    md_links = _historical_zip_links_from_html(html, dataset="fred_md")
    qd_links = _historical_zip_links_from_html(html, dataset="fred_qd")

    assert md_links == [
        (
            "2015-01",
            "2024-12",
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2024-12.zip?hash=abc",
        )
    ]
    assert qd_links == [
        (
            "2005-01",
            "2024-10",
            "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-qd-2005-01-to-2024-10.zip?hash=def",
        )
    ]


def test_load_fred_md_auto_extracts_official_historical_zip(monkeypatch, tmp_path: Path) -> None:
    zip_path = tmp_path / "fred_md_historical.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "nested/2018-01.csv",
            ",5,5,2\n"
            "sasdate,INDPRO,RPI,UNRATE\n"
            "1/1/2000,100.0,1000.0,4.0\n"
            "2/1/2000,101.0,1010.0,4.1\n",
        )

    def fake_obtain(dataset, vintage, *, cache_root, force):
        assert dataset == "fred_md"
        assert vintage == "2018-01"
        return zip_path, "https://example.test/fred-md-history.zip"

    monkeypatch.setattr("macroforecast.data.loaders._obtain_fred_historical_zip", fake_obtain)
    bundle = load_fred_md(vintage="2018-01", cache_root=tmp_path)

    assert metadata(bundle)["dataset"] == "fred_md"
    assert metadata(bundle)["artifact"]["source_url"] == "https://example.test/fred-md-history.zip#nested/2018-01.csv"
    assert list(bundle.panel.columns) == ["INDPRO", "RPI", "UNRATE"]


def test_load_fred_qd_auto_extracts_official_historical_zip(monkeypatch, tmp_path: Path) -> None:
    zip_path = tmp_path / "fred_qd_historical.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "FRED-QD_2018m05.csv",
            ",5,2\n"
            "sasqdate,GDPC1,UNRATE\n"
            "1/1/2000,100.0,4.0\n"
            "4/1/2000,101.0,4.1\n",
        )

    def fake_obtain(dataset, vintage, *, cache_root, force):
        assert dataset == "fred_qd"
        assert vintage == "2018-05"
        return zip_path, "https://example.test/fred-qd-history.zip"

    monkeypatch.setattr("macroforecast.data.loaders._obtain_fred_historical_zip", fake_obtain)
    bundle = load_fred_qd(vintage="2018-05", cache_root=tmp_path)

    assert metadata(bundle)["dataset"] == "fred_qd"
    assert metadata(bundle)["artifact"]["source_url"] == "https://example.test/fred-qd-history.zip#FRED-QD_2018m05.csv"
    assert list(bundle.panel.columns) == ["GDPC1", "UNRATE"]


def test_load_fred_md_rejects_bad_vintage_format(tmp_path: Path) -> None:
    with pytest.raises(RawVersionFormatError):
        load_fred_md(vintage="201802", cache_root=tmp_path, local_source=FIXTURES / "fred_md_sample.csv")


def test_load_fred_sd_accepts_local_csv_fixture_without_excel_extra(tmp_path: Path) -> None:
    bundle = load_fred_sd(
        cache_root=tmp_path,
        local_source=FIXTURES / "fred_sd_sample.csv",
    )

    assert metadata(bundle)["dataset"] == "fred_sd"
    assert "source_family" not in metadata(bundle)
    assert metadata(bundle)["frequency"] == "monthly"
    assert metadata(bundle)["native_frequency_counts"] == {"monthly": 4}
    assert metadata(bundle)["artifact"]["file_format"] == "csv"
    assert metadata(bundle)["artifact"]["source_url"].endswith("fred_sd_sample.csv")
    assert list(bundle.panel.columns) == ["BPPRIVSA_CA", "UR_CA", "BPPRIVSA_TX", "UR_TX"]
    assert bundle.panel.index[0].strftime("%Y-%m") == "2000-01"


def test_load_fred_sd_local_csv_supports_state_variable_filters(tmp_path: Path) -> None:
    bundle = load_fred_sd(
        cache_root=tmp_path,
        local_source=FIXTURES / "fred_sd_sample.csv",
        states=["CA"],
        variables=["UR"],
    )

    assert list(bundle.panel.columns) == ["UR_CA"]
    assert metadata(bundle)["artifact"]["file_format"] == "csv"


def test_fred_sd_current_source_resolver_prefers_latest_by_series_workbook() -> None:
    html = """
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-02.xlsx">Series 2026 02</a>
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-sd/state/state-2026-04.xlsx">State 2026 04</a>
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-03.xlsx">Series 2026 03</a>
    """

    url = _latest_series_url_from_html(html)

    assert url == "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-03.xlsx"


def test_fred_sd_vintage_source_urls_use_official_by_series_layout() -> None:
    assert _fred_sd_series_xlsx_url("2026-03").endswith("/fred-sd/series/series-2026-03.xlsx")
    assert _fred_sd_series_zip_url("2024-12").endswith("/fred-sd/series/fredsd_byseries_2024.zip")
    assert _fred_sd_series_zip_url("2020-05").endswith("/fred-sd/series/fredsd_byseries_2019_2020.zip")
    assert _fred_sd_series_zip_url("2005-06").endswith("/fred-sd/series/fredsd_byseries_2005_2006.zip")


def test_fred_sd_zip_fallback_extracts_requested_by_series_vintage() -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("Series 2005-01.xlsx", b"PK workbook")
        zf.writestr("Series 2005-02.xlsx", b"wrong")

    entry, workbook = _extract_vintage_xlsx_from_zip(payload.getvalue(), "2005-01")

    assert entry == "Series 2005-01.xlsx"
    assert workbook == b"PK workbook"


def test_load_fred_qd_wraps_download_failure(monkeypatch, tmp_path: Path) -> None:
    def fail_urlopen(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr("macroforecast.data.loaders.urlopen", fail_urlopen)
    with pytest.raises(RawDownloadError):
        load_fred_qd(vintage="2020-01", cache_root=tmp_path)


def test_atomic_cache_copy_supports_concurrent_writers(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "cache" / "raw.csv"
    source.write_text("sasdate,INDPRO\nTransform:,5\n1/1/2000,100\n")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_atomic_copy, source, target) for _ in range(16)]
        for future in futures:
            future.result()

    assert target.read_text() == source.read_text()
    assert not list(target.parent.glob("*.tmp"))
