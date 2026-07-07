from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from macroforecast.data import DataBundle, load_fred_series, metadata


def test_load_fred_series_caches_local_source_and_reuses_offline(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "usrec_source.csv"
    source.write_text(
        "observation_date,USREC\n"
        "2020-01-01,0\n"
        "2020-02-01,1\n"
        "2020-03-01,1\n",
        encoding="utf-8",
    )

    first = load_fred_series("USREC", cache_root=tmp_path, local_source=source)

    assert isinstance(first, DataBundle)
    assert first.panel.index.name == "date"
    assert list(first.panel.columns) == ["USREC"]
    assert first.panel.index[0] == pd.Timestamp("2020-01-01")
    assert metadata(first)["dataset"] == "fred_series"
    assert metadata(first)["series_id"] == "USREC"
    assert metadata(first)["frequency"] == "monthly"
    assert metadata(first)["date_anchor_by_column"] == {"USREC": "month_start"}
    assert (tmp_path / "fred_series" / "USREC.csv").exists()

    manifest_path = tmp_path / "manifest" / "raw_artifacts.jsonl"
    manifest_rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert manifest_rows[-1]["dataset"] == "fred_series"
    assert manifest_rows[-1]["local_path"].endswith("fred_series/USREC.csv")
    assert manifest_rows[-1]["cache_hit"] is False

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("cache hit must not touch the network")

    monkeypatch.setattr("macroforecast.data.loaders.urlopen", fail_urlopen)
    second = load_fred_series("USREC", cache_root=tmp_path)

    assert second.panel.equals(first.panel)
    assert metadata(second)["artifact"]["cache_hit"] is True
