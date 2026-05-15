"""Regression test for Cycle 14 J-1: FRED-QD loader NaT-safe data_through.

Verifies that load_fred_qd() does not crash (ValueError/AttributeError) when
the DataFrame index contains NaT values in the last position — e.g., when the
raw FRED CSV has trailing empty rows that produce a NaT after date parsing.

Closes: Cycle 14 F11 (P1-1)
"""
from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_fake_request():
    req = MagicMock()
    req.mode = "current"
    req.vintage = "current"
    return req


def _make_fake_artifact():
    return MagicMock()


def _patch_infrastructure(monkeypatch, fake_df, fake_tcodes):
    """Patch all I/O and infrastructure so load_fred_qd() can run without network."""
    import macroforecast.raw.datasets.fred_qd as fqd

    fake_req = _make_fake_request()
    fake_target = MagicMock(spec=Path)
    fake_target.exists.return_value = True  # treat as cache hit

    monkeypatch.setattr(
        "macroforecast.raw.datasets.fred_qd.normalize_version_request",
        lambda *a, **kw: fake_req,
    )
    monkeypatch.setattr(
        "macroforecast.raw.datasets.fred_qd.get_raw_file_path",
        lambda *a, **kw: fake_target,
    )
    monkeypatch.setattr(
        "macroforecast.raw.datasets.fred_qd.parse_fred_csv",
        lambda *a, **kw: (fake_df, fake_tcodes),
    )
    monkeypatch.setattr(
        "macroforecast.raw.datasets.fred_qd.build_raw_artifact_record",
        lambda *a, **kw: _make_fake_artifact(),
    )
    monkeypatch.setattr(
        "macroforecast.raw.datasets.fred_qd.append_raw_manifest_entry",
        lambda *a, **kw: None,
    )


def test_nat_safe_data_through_no_nat(monkeypatch):
    """When index has no NaT, data_through should be the last date formatted."""
    fake_df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01"]),
    )
    _patch_infrastructure(monkeypatch, fake_df, {"A": 1})

    from macroforecast.raw.datasets.fred_qd import load_fred_qd
    result = load_fred_qd()
    assert result.dataset_metadata.data_through == "2023-07", (
        f"Expected 2023-07, got {result.dataset_metadata.data_through!r}"
    )


def test_nat_safe_data_through_trailing_nat(monkeypatch):
    """When last index entry is NaT, data_through should fall back to last valid date."""
    fake_df = pd.DataFrame(
        {"A": [1.0, 2.0, float("nan")]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-04-01"), pd.NaT]
        ),
    )
    _patch_infrastructure(monkeypatch, fake_df, {"A": 1})

    from macroforecast.raw.datasets.fred_qd import load_fred_qd
    result = load_fred_qd()
    assert result.dataset_metadata.data_through == "2023-04", (
        f"Expected 2023-04, got {result.dataset_metadata.data_through!r}"
    )


def test_nat_safe_data_through_all_nat(monkeypatch):
    """When all index entries are NaT, data_through should be None (not crash)."""
    fake_df = pd.DataFrame(
        {"A": [float("nan")]},
        index=pd.DatetimeIndex([pd.NaT]),
    )
    _patch_infrastructure(monkeypatch, fake_df, {"A": 1})

    from macroforecast.raw.datasets.fred_qd import load_fred_qd
    result = load_fred_qd()
    assert result.dataset_metadata.data_through is None, (
        f"Expected None, got {result.dataset_metadata.data_through!r}"
    )


def test_nat_safe_data_through_empty_df(monkeypatch):
    """When DataFrame is empty, data_through should be None (not crash)."""
    fake_df = pd.DataFrame({"A": []}, index=pd.DatetimeIndex([]))
    _patch_infrastructure(monkeypatch, fake_df, {})

    from macroforecast.raw.datasets.fred_qd import load_fred_qd
    result = load_fred_qd()
    assert result.dataset_metadata.data_through is None, (
        f"Expected None, got {result.dataset_metadata.data_through!r}"
    )
