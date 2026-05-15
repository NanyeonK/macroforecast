"""F-P1-5 -- custom CSV with FRED-official Transform: header raises RuntimeError.

The fix (F-P1-5) detects the FRED-MD/QD CSV format (first row starts with
"Transform:") and raises RuntimeError with a helpful hint, instead of silently
corrupting the first data row.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from macroforecast.core.runtime import _read_custom_panel_path


def test_transform_row_raises_runtime_error(tmp_path: Path):
    """CSV with Transform: header should raise RuntimeError."""
    csv_file = tmp_path / "fred_md.csv"
    csv_file.write_text(
        "Transform:,2,2,5,5,1\n"
        "sasdate,INDPRO,PAYEMS,FEDFUNDS,TB3MS,GS10\n"
        "1959-01-01,6.9,49897,2.35,2.43,4.02\n"
        "1959-02-01,7.0,50000,2.35,2.40,4.00\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Transform:"):
        _read_custom_panel_path(csv_file)


def test_transform_row_error_suggests_fred_md(tmp_path: Path):
    """RuntimeError message should mention fred_md."""
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text(
        "Transform:,2,2\n"
        "sasdate,x,y\n"
        "2020-01-01,1.0,2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="fred_md|dataset"):
        _read_custom_panel_path(csv_file)


def test_normal_csv_accepted(tmp_path: Path):
    """A normal CSV without Transform: header should be read normally."""
    csv_file = tmp_path / "panel.csv"
    csv_file.write_text(
        "date,y,x1\n"
        "2020-01-01,1.0,10.0\n"
        "2020-02-01,2.0,20.0\n",
        encoding="utf-8",
    )
    df = _read_custom_panel_path(csv_file)
    assert list(df.columns) == ["date", "y", "x1"]
    assert len(df) == 2


def test_transform_code_variant_also_raises(tmp_path: Path):
    """CSV with transform_code: header (alternative format) also raises."""
    csv_file = tmp_path / "alt.csv"
    csv_file.write_text(
        "transform_code:,2,5\n"
        "sasdate,INDPRO,PAYEMS\n"
        "2020-01-01,1.0,2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        _read_custom_panel_path(csv_file)
