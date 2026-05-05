from __future__ import annotations

import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from macroforecast.raw.cache import atomic_copy_to_cache
from macroforecast.raw import load_fred_md, load_fred_qd, load_fred_sd, parse_fred_csv
from macroforecast.raw.datasets.fred_sd import (
    _extract_vintage_xlsx_from_zip,
    _latest_series_url_from_html,
    _series_xlsx_url,
    _series_zip_url,
)
from macroforecast.raw.errors import RawDownloadError, RawVersionFormatError

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_fred_csv_accepts_official_header_then_transform_layout(tmp_path: Path) -> None:
    path = tmp_path / "official.csv"
    path.write_text(
        "sasdate,INDPRO,UNRATE\n"
        "Transform:,5,2\n"
        "1/1/2000,100.0,4.0\n"
        "2/1/2000,101.0,4.1\n"
    )

    df, tcodes = parse_fred_csv(path)

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

    result = load_fred_md(
        vintage="2018-02",
        cache_root=tmp_path,
        local_zip_source=zip_path,
    )

    assert result.dataset_metadata.dataset == "fred_md"
    assert result.dataset_metadata.vintage == "2018-02"
    assert result.data.shape == (2, 4)


def test_load_fred_md_rejects_bad_vintage_format(tmp_path: Path) -> None:
    with pytest.raises(RawVersionFormatError):
        load_fred_md(vintage="201802", cache_root=tmp_path, local_source=FIXTURES / "fred_md_sample.csv")


def test_load_fred_sd_accepts_local_csv_fixture_without_excel_extra(tmp_path: Path) -> None:
    result = load_fred_sd(
        cache_root=tmp_path,
        local_source=FIXTURES / "fred_sd_sample.csv",
    )

    assert result.dataset_metadata.dataset == "fred_sd"
    assert result.dataset_metadata.frequency == "state_monthly"
    assert result.artifact.file_format == "csv"
    assert result.artifact.source_url.endswith("fred_sd_sample.csv")
    assert list(result.data.columns) == ["BPPRIVSA_CA", "UR_CA", "BPPRIVSA_TX", "UR_TX"]
    assert result.data.index[0].strftime("%Y-%m") == "2000-01"


def test_load_fred_sd_local_csv_supports_state_variable_filters(tmp_path: Path) -> None:
    result = load_fred_sd(
        cache_root=tmp_path,
        local_source=FIXTURES / "fred_sd_sample.csv",
        states=["CA"],
        variables=["UR"],
    )

    assert list(result.data.columns) == ["UR_CA"]
    assert result.artifact.file_format == "csv"


def test_fred_sd_current_source_resolver_prefers_latest_by_series_workbook() -> None:
    html = """
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-02.xlsx">Series 2026 02</a>
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-sd/state/state-2026-04.xlsx">State 2026 04</a>
    <a href="/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-03.xlsx">Series 2026 03</a>
    """

    url = _latest_series_url_from_html(html)

    assert url == "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-sd/series/series-2026-03.xlsx"


def test_fred_sd_vintage_source_urls_use_official_by_series_layout() -> None:
    assert _series_xlsx_url("2026-03").endswith("/fred-sd/series/series-2026-03.xlsx")
    assert _series_zip_url("2024-12").endswith("/fred-sd/series/fredsd_byseries_2024.zip")
    assert _series_zip_url("2020-05").endswith("/fred-sd/series/fredsd_byseries_2019_2020.zip")
    assert _series_zip_url("2005-06").endswith("/fred-sd/series/fredsd_byseries_2005_2006.zip")


def test_fred_sd_zip_fallback_extracts_requested_by_series_vintage() -> None:
    payload = BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("Series 2005-01.xlsx", b"PK workbook")
        zf.writestr("Series 2005-02.xlsx", b"wrong")

    entry, workbook = _extract_vintage_xlsx_from_zip(payload.getvalue(), "2005-01")

    assert entry == "Series 2005-01.xlsx"
    assert workbook == b"PK workbook"


def test_load_fred_qd_wraps_download_failure(tmp_path: Path) -> None:
    with pytest.raises(RawDownloadError):
        load_fred_qd(vintage="2020-01", cache_root=tmp_path)


def test_atomic_cache_copy_supports_concurrent_writers(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    target = tmp_path / "cache" / "raw.csv"
    source.write_text("sasdate,INDPRO\nTransform:,5\n1/1/2000,100\n")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(atomic_copy_to_cache, source, target) for _ in range(16)]
        for future in futures:
            future.result()

    assert target.read_text() == source.read_text()
    assert not list(target.parent.glob("*.tmp"))
