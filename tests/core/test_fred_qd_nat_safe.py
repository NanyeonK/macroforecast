"""Regression test for FRED-QD loader NaT-safe data_through."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from macroforecast.data import load_fred_qd, metadata


def _patch_loader(monkeypatch, tmp_path: Path, fake_df, fake_tcodes):
    fake_target = tmp_path / "raw.csv"
    fake_target.write_text("cached")
    monkeypatch.setattr("macroforecast.data.loaders._raw_file_path", lambda *a, **kw: fake_target)
    monkeypatch.setattr("macroforecast.data.loaders._parse_fred_csv", lambda *a, **kw: (fake_df, fake_tcodes))


def test_nat_safe_data_through_no_nat(monkeypatch, tmp_path):
    fake_df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01"]),
    )
    fake_df.index.name = "date"
    _patch_loader(monkeypatch, tmp_path, fake_df, {"A": 1})

    bundle = load_fred_qd(cache_root=tmp_path)
    assert metadata(bundle)["data_through"] == "2023-07"


def test_nat_safe_data_through_trailing_nat(monkeypatch, tmp_path):
    fake_df = pd.DataFrame(
        {"A": [1.0, 2.0, float("nan")]},
        index=pd.DatetimeIndex([pd.Timestamp("2023-01-01"), pd.Timestamp("2023-04-01"), pd.NaT]),
    )
    fake_df.index.name = "date"
    _patch_loader(monkeypatch, tmp_path, fake_df, {"A": 1})

    bundle = load_fred_qd(cache_root=tmp_path)
    assert metadata(bundle)["data_through"] == "2023-04"


def test_nat_safe_data_through_all_nat(monkeypatch, tmp_path):
    fake_df = pd.DataFrame({"A": [float("nan")]}, index=pd.DatetimeIndex([pd.NaT]))
    fake_df.index.name = "date"
    _patch_loader(monkeypatch, tmp_path, fake_df, {"A": 1})

    bundle = load_fred_qd(cache_root=tmp_path)
    assert metadata(bundle)["data_through"] is None


def test_nat_safe_data_through_empty_df(monkeypatch, tmp_path):
    fake_df = pd.DataFrame({"A": []}, index=pd.DatetimeIndex([]))
    fake_df.index.name = "date"
    _patch_loader(monkeypatch, tmp_path, fake_df, {})

    bundle = load_fred_qd(cache_root=tmp_path)
    assert metadata(bundle)["data_through"] is None
