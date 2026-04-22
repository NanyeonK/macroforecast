from __future__ import annotations

from pathlib import Path

import pandas as pd

from macrocast.raw import (
    RawLoadResult,
    load_fred_md,
    load_fred_qd,
    parse_fred_csv,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_fred_md_sample_csv() -> None:
    df, tcodes = parse_fred_csv(FIXTURES / "fred_md_sample.csv")

    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["INDPRO", "RPI", "UNRATE", "CPIAUCSL"]
    assert tcodes["INDPRO"] == 5
    assert tcodes["UNRATE"] == 2


def test_parse_fred_qd_sample_csv() -> None:
    df, tcodes = parse_fred_csv(FIXTURES / "fred_qd_sample.csv")

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["GDPC1", "CPIAUCSL", "FEDFUNDS"]
    assert tcodes["GDPC1"] == 5
    assert tcodes["FEDFUNDS"] == 2


def test_parse_fred_qd_ignores_factors_row(tmp_path: Path) -> None:
    path = tmp_path / "fred_qd_with_factors.csv"
    path.write_text(
        "\n".join(
            [
                "sasdate,GDPC1,PCECC96,UNRATE",
                "factors,0,1,0",
                "transform,5,5,2",
                "1/1/2000,100.0,80.0,4.0",
                "4/1/2000,101.0,81.0,4.1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    df, tcodes = parse_fred_csv(path)

    assert list(df.columns) == ["GDPC1", "PCECC96", "UNRATE"]
    assert df.index[0].strftime("%Y-%m") == "2000-01"
    assert tcodes == {"GDPC1": 5, "PCECC96": 5, "UNRATE": 2}


def test_load_fred_md_from_fixture_copy(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_md_sample.csv"
    result = load_fred_md(local_source=fixture, cache_root=tmp_path)

    assert isinstance(result, RawLoadResult)
    assert result.dataset_metadata.dataset == "fred_md"
    assert result.dataset_metadata.frequency == "monthly"
    assert result.artifact.file_format == "csv"
    assert result.data.index[0].strftime("%Y-%m") == "2000-01"


def test_load_fred_qd_from_fixture_copy(tmp_path: Path) -> None:
    fixture = FIXTURES / "fred_qd_sample.csv"
    result = load_fred_qd(local_source=fixture, cache_root=tmp_path)

    assert isinstance(result, RawLoadResult)
    assert result.dataset_metadata.dataset == "fred_qd"
    assert result.dataset_metadata.frequency == "quarterly"
    assert result.artifact.file_format == "csv"
    assert result.data.index[-1].strftime("%Y-%m") == "2001-04"
