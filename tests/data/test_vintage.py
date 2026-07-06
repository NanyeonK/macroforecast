from __future__ import annotations

import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.data import vintage as vintage_mod


def _bundle(label: str) -> mf.data.DataBundle:
    idx = pd.DatetimeIndex([pd.Timestamp("2000-01-31")], name="date")
    panel = pd.DataFrame({"x": [1.0]}, index=idx)
    return mf.data.DataBundle(panel, {"dataset": "fred_md", "vintage": label})


def test_fred_md_vintages_resolve_by_latest_label_and_memoize(monkeypatch) -> None:
    loads: list[str] = []

    def fake_list_vintages(dataset, start=None, end=None):
        assert dataset == "fred_md"
        assert start == "2000-01"
        assert end == "2000-03"
        return ["2000-01", "2000-02", "2000-03"]

    def fake_load_fred_md(*, vintage, force=False, cache_root=None, local_zip_source=None):
        loads.append(vintage)
        return _bundle(vintage)

    monkeypatch.setattr(vintage_mod, "list_vintages", fake_list_vintages)
    monkeypatch.setattr(vintage_mod, "load_fred_md", fake_load_fred_md)

    source = mf.data.fred_md_vintages(start="2000-01", end="2000-03")

    first = source.resolve(pd.Timestamp("2000-02-15"))
    second = source.resolve(pd.Timestamp("2000-02-28"))
    third = source.resolve(pd.Timestamp("2000-03-01"))

    assert first is second
    assert first.metadata["vintage"] == "2000-02"
    assert third.metadata["vintage"] == "2000-03"
    assert loads == ["2000-02", "2000-03"]
    assert tuple(source.available_vintages()) == ("2000-01", "2000-02", "2000-03")


def test_fred_qd_vintages_resolve_and_raise_before_first(monkeypatch) -> None:
    monkeypatch.setattr(
        vintage_mod,
        "list_vintages",
        lambda dataset, start=None, end=None: ["2000-02", "2000-03"],
    )
    monkeypatch.setattr(
        vintage_mod,
        "load_fred_qd",
        lambda *, vintage, force=False, cache_root=None, local_zip_source=None: _bundle(vintage),
    )

    source = mf.data.fred_qd_vintages(end="2000-03")

    assert source.resolve(pd.Timestamp("2000-03-31")).metadata["vintage"] == "2000-03"
    with pytest.raises(mf.data.VintageUnavailableError):
        source.resolve(pd.Timestamp("2000-01-31"))


def test_vintage_panel_spec_validates_reference_calendar() -> None:
    class Source:
        def resolve(self, origin_date):
            return _bundle("v1")

        def available_vintages(self):
            return ["v1"]

    source = Source()
    with pytest.raises(ValueError, match="must not be empty"):
        mf.data.VintagePanelSpec(source, pd.DatetimeIndex([], name="date"))
    with pytest.raises(ValueError, match="monotonic"):
        mf.data.VintagePanelSpec(
            source,
            pd.DatetimeIndex(
                [pd.Timestamp("2000-02-29"), pd.Timestamp("2000-01-31")],
                name="date",
            ),
        )
