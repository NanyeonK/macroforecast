from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

import macroforecast as mf
from macroforecast.data import vintage as vintage_mod


def _bundle(label: str) -> mf.data.DataBundle:
    idx = pd.DatetimeIndex([pd.Timestamp("2000-01-31")], name="date")
    panel = pd.DataFrame({"x": [1.0]}, index=idx)
    return mf.data.DataBundle(panel, {"dataset": "fred_md", "vintage": label})


def _custom_frames() -> dict[pd.Timestamp, pd.DataFrame]:
    dates = pd.date_range("2000-01-31", periods=3, freq="ME", name="date")
    return {
        pd.Timestamp("2000-02-15"): pd.DataFrame(
            {"x": [1.0, 2.0]},
            index=dates[:2],
        ),
        pd.Timestamp("2000-03-15"): pd.DataFrame(
            {"x": [1.0, 20.0, 3.0]},
            index=dates,
        ),
    }


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


def test_custom_vintages_three_shapes_resolve_identical_bundles() -> None:
    frames = _custom_frames()

    def callable_source(origin_date: pd.Timestamp) -> pd.DataFrame:
        return frames[pd.Timestamp(origin_date)]

    callable_vintages = mf.data.custom_vintages(callable_source, frequency="monthly")
    mapping_vintages = mf.data.custom_vintages(frames, frequency="monthly")
    long = pd.concat(
        [
            frame.reset_index().assign(vintage=key)
            for key, frame in frames.items()
        ],
        ignore_index=True,
    )
    long_vintages = mf.data.custom_vintages(
        long,
        vintage_column="vintage",
        date_column="date",
        frequency="monthly",
    )

    for origin in frames:
        resolved = [
            source.resolve(origin)
            for source in (callable_vintages, mapping_vintages, long_vintages)
        ]
        for bundle in resolved:
            assert bundle.metadata["dataset"] == "custom_vintages"
            assert bundle.metadata["frequency"] == "monthly"
            assert bundle.metadata["vintage"] == str(origin)
        pdt.assert_frame_equal(resolved[0].panel, resolved[1].panel, check_freq=False)
        pdt.assert_frame_equal(resolved[0].panel, resolved[2].panel, check_freq=False)


def test_custom_vintages_mapping_resolves_latest_available_and_memoizes() -> None:
    frames = _custom_frames()
    source = mf.data.custom_vintages(frames, vintage_id=lambda key: pd.Timestamp(key).strftime("%Y%m"))

    first = source.resolve(pd.Timestamp("2000-03-20"))
    second = source.resolve(pd.Timestamp("2000-03-31"))

    assert first is second
    assert first.metadata["vintage"] == "200003"
    with pytest.raises(mf.data.VintageUnavailableError):
        source.resolve(pd.Timestamp("2000-01-01"))


def test_custom_vintages_callable_memoizes_by_vintage_id() -> None:
    frames = _custom_frames()
    calls: list[pd.Timestamp] = []

    def callable_source(origin_date: pd.Timestamp) -> pd.DataFrame:
        calls.append(pd.Timestamp(origin_date))
        return frames[pd.Timestamp("2000-03-15")]

    source = mf.data.custom_vintages(callable_source, vintage_id=lambda origin: "live")

    assert source.resolve(pd.Timestamp("2000-03-20")) is source.resolve(pd.Timestamp("2000-04-20"))
    assert calls == [pd.Timestamp("2000-03-20")]


def test_custom_vintages_long_frame_requires_columns() -> None:
    with pytest.raises(ValueError, match="vintage_column and date_column"):
        mf.data.custom_vintages(pd.DataFrame({"date": ["2000-01-31"], "x": [1.0]}))


@pytest.mark.parametrize(
    ("join", "expected_index"),
    [
        ("outer", ["2000-01-31", "2000-02-29", "2000-03-31"]),
        ("inner", ["2000-02-29"]),
        ("left", ["2000-01-31", "2000-02-29"]),
    ],
)
def test_with_static_extras_join_semantics(join: str, expected_index: list[str]) -> None:
    frames = {
        pd.Timestamp("2000-03-15"): pd.DataFrame(
            {"x": [1.0, 2.0]},
            index=pd.DatetimeIndex(["2000-01-31", "2000-02-29"], name="date"),
        )
    }
    extra = pd.DataFrame(
        {"z": [9.0, 10.0]},
        index=pd.DatetimeIndex(["2000-02-29", "2000-03-31"], name="date"),
    )
    source = mf.data.with_static_extras(mf.data.custom_vintages(frames), extra, join=join)

    panel = source.resolve(pd.Timestamp("2000-03-20")).panel

    assert list(panel.index) == [pd.Timestamp(value) for value in expected_index]
    assert "z" in panel.columns
    if pd.Timestamp("2000-02-29") in panel.index:
        assert panel.loc[pd.Timestamp("2000-02-29"), "z"] == 9.0


def test_with_static_extras_fingerprint_changes_vintage_id() -> None:
    frames = _custom_frames()
    base = mf.data.custom_vintages(frames)
    extra_a = pd.DataFrame({"z": [1.0]}, index=pd.DatetimeIndex(["2000-01-31"], name="date"))
    extra_b = pd.DataFrame({"z": [2.0]}, index=pd.DatetimeIndex(["2000-01-31"], name="date"))

    with_a = mf.data.with_static_extras(base, extra_a)
    with_b = mf.data.with_static_extras(base, extra_b)

    a_bundle = with_a.resolve(pd.Timestamp("2000-02-15"))
    b_bundle = with_b.resolve(pd.Timestamp("2000-02-15"))

    assert a_bundle.metadata["base_vintage"] == str(pd.Timestamp("2000-02-15"))
    assert a_bundle.metadata["vintage"] != b_bundle.metadata["vintage"]
    assert "static_extra_sha256=" in a_bundle.metadata["vintage"]
