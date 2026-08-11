from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import macroforecast as mf
from macroforecast.data import vintage as vintage_mod
from macroforecast.data.vintage import VintagePanelSpec


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


def test_custom_vintages_mapping_rejects_unparseable_vintage_key() -> None:
    with pytest.raises(ValueError, match="not-a-date.*timestamp"):
        mf.data.custom_vintages({"not-a-date": pd.DataFrame({"x": [1.0]})})


def test_vintage_panel_spec_rejects_first_release_callable_without_vintages() -> None:
    def callable_source(origin_date: pd.Timestamp) -> pd.DataFrame:
        return pd.DataFrame(
            {"x": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp(origin_date)], name="date"),
        )

    source = mf.data.custom_vintages(callable_source)

    with pytest.raises(ValueError, match="available_vintages.*first-release actuals"):
        mf.data.VintagePanelSpec(
            source,
            pd.date_range("2000-01-31", periods=2, freq="ME", name="date"),
            actuals_vintage="first_release",
        )


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
        ("outer", ["2000-01-31", "2000-02-29"]),
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


def test_with_static_extras_truncate_rows_at_each_origin() -> None:
    frames = _custom_frames()
    extra = pd.DataFrame(
        {"z": [100.0, 200.0, 300.0]},
        index=pd.date_range("2000-01-31", periods=3, freq="ME", name="date"),
    )
    source = mf.data.with_static_extras(mf.data.custom_vintages(frames), extra)

    first = source.resolve(pd.Timestamp("2000-02-15"))
    second = source.resolve(pd.Timestamp("2000-03-15"))

    assert "z" in first.panel.columns
    assert "z" in second.panel.columns
    assert first.panel.loc[pd.Timestamp("2000-01-31"), "z"] == 100.0
    assert pd.isna(first.panel.loc[pd.Timestamp("2000-02-29"), "z"])
    assert second.panel.loc[pd.Timestamp("2000-02-29"), "z"] == 200.0
    assert pd.isna(second.panel.loc[pd.Timestamp("2000-03-31"), "z"])


# --------------------------------------------------------------------------- #
# F-012: static-extra vintage IDs inherit the full-content panel identity
# --------------------------------------------------------------------------- #

def _extras_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two large extras differing only at the cell the old stride skipped."""
    index = pd.date_range("1990-01-31", periods=100, freq="ME", name="date")
    rng = np.random.default_rng(7)
    base = pd.DataFrame(
        rng.normal(size=(100, 10)), index=index, columns=[f"e{i}" for i in range(10)]
    )
    mutated = base.copy()
    mutated.iloc[0, 1] = mutated.iloc[0, 1] + 1.0
    return base, mutated


def test_static_extra_fingerprints_differ_at_a_formerly_unsampled_cell(monkeypatch) -> None:
    """The identity a static-extras source is cached on must see the whole panel.

    Under the old sampled digest this pair produced one fingerprint, so a resolved
    bundle built from the first extra could be served for the second (F-012). The chunk
    size is patched small so the streaming path -- not the small-panel path -- is what
    is being checked.
    """
    from macroforecast.data import identity as identity_mod

    monkeypatch.setattr(identity_mod, "_FINGERPRINT_CHUNK_CELLS", 7)
    base, mutated = _extras_pair()

    assert (
        identity_mod.panel_fingerprint(base)["value"]
        != identity_mod.panel_fingerprint(mutated)["value"]
    )


def test_static_extra_vintage_ids_differ_at_a_formerly_unsampled_cell(monkeypatch) -> None:
    """And that difference has to reach the vintage/cache label, not stop at the dict."""
    from macroforecast.data import identity as identity_mod
    from macroforecast.data.vintage import _extra_vintage_label

    monkeypatch.setattr(identity_mod, "_FINGERPRINT_CHUNK_CELLS", 7)
    base, mutated = _extras_pair()
    origin = pd.Timestamp("2020-01-31")

    label_base = _extra_vintage_label(
        "base-2020-01-01", identity_mod.panel_fingerprint(base), origin
    )
    label_mutated = _extra_vintage_label(
        "base-2020-01-01", identity_mod.panel_fingerprint(mutated), origin
    )

    assert label_base != label_mutated


def test_static_extra_label_form_is_unchanged_and_carries_the_full_digest() -> None:
    """F-012 is fixed inside the digest, so the label form must NOT move.

    A ``VintageSource``'s stable ID is supposed to change if and only if the content
    changes. Reshaping the label would have given every unchanged extras panel a new ID
    and missed its cached resolved bundle for nothing, so the fix stays where the defect
    was: the digest now reads the whole panel, and the label keeps saying ``sha256=``.
    """
    from macroforecast.data import identity as identity_mod
    from macroforecast.data.vintage import _extra_vintage_label

    base, _ = _extras_pair()
    fingerprint = identity_mod.panel_fingerprint(base)
    label = _extra_vintage_label("base-2020-01-01", fingerprint, pd.Timestamp("2020-01-31"))

    assert "|static_extra_sha256=" in label
    assert fingerprint["value"] in label
    assert label.endswith("|origin=2020-01-31")


def test_static_extra_label_refuses_a_partial_fingerprint() -> None:
    """Enforced rather than documented: a cache key cannot rest on a partial digest.

    Unreachable today -- ``panel_fingerprint`` has one method -- and kept so that
    reintroducing a sampled one fails loudly instead of quietly weakening every
    static-extra vintage ID again.
    """
    from macroforecast.data.vintage import _extra_vintage_label

    with pytest.raises(ValueError, match="full-content"):
        _extra_vintage_label(
            "base-2020-01-01",
            {"algorithm": "sha256", "method": "strided_subsample", "value": "deadbeef"},
            pd.Timestamp("2020-01-31"),
        )


# --------------------------------------------------------------------------- #
# F-010: a known vintage calendar must have unique labels
# --------------------------------------------------------------------------- #

def _snapshot(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {"y": [value, value]},
        index=pd.DatetimeIndex(["2020-01-31", "2020-02-29"], name="date"),
    )


def _mapping_pair() -> dict[str, pd.DataFrame]:
    return {"2020-01-01": _snapshot(1.0), "2020-02-01": _snapshot(2.0)}


def _long_pair() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "v": ["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"],
            "date": ["2020-01-31", "2020-02-29"] * 2,
            "y": [1.0, 1.0, 2.0, 2.0],
        }
    )


def test_mapping_vintage_id_collision_rejects_at_construction() -> None:
    """Two snapshots, one identifier: the cache cannot tell them apart.

    Resolved snapshots are memoized by the identifier, so with
    ``vintage_id=lambda _: "same"`` resolving the January origin populated the cache and
    the February origin was handed January's panel — a wrong point-in-time snapshot with
    nothing in the result to show for it. Construction is the only place this is still
    distinguishable.
    """
    with pytest.raises(ValueError, match="distinct string"):
        mf.data.custom_vintages(_mapping_pair(), vintage_id=lambda _: "same")


def test_grouped_wide_vintage_id_collision_rejects_at_construction() -> None:
    """Same invariant on the long-frame form."""
    with pytest.raises(ValueError, match="distinct string"):
        mf.data.custom_vintages(
            _long_pair(), vintage_column="v", date_column="date", vintage_id=lambda _: "same"
        )


def test_distinct_vintage_ids_still_resolve() -> None:
    source = mf.data.custom_vintages(
        _mapping_pair(), vintage_id=lambda key: pd.Timestamp(key).strftime("%Y%m")
    )

    assert source.resolve(pd.Timestamp("2020-02-15")).metadata["vintage"] == "202002"


def test_callable_only_constant_vintage_id_is_still_supported() -> None:
    """The boundary the fix deliberately does not cross.

    A callable source has no enumerable calendar, so uniqueness cannot be proven. There
    the identifier is the caller's declaration of cache identity: a constant one means
    "one snapshot, reused", and that is a supported thing to say.
    """
    calls = {"n": 0}

    def live(origin):
        calls["n"] += 1
        return _snapshot(float(calls["n"]))

    source = mf.data.custom_vintages(live, vintage_id=lambda origin: "live")
    source.resolve(pd.Timestamp("2020-01-15"))
    source.resolve(pd.Timestamp("2020-02-15"))

    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# F-011: calendars and keys canonical enough for bisect
# --------------------------------------------------------------------------- #

def _good_calendar() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(["2020-01-31", "2020-02-29"], name="date")


def test_reference_calendar_rejects_duplicate_origins() -> None:
    """``is_monotonic_increasing`` permits repeats; a repeated origin is one forecast
    row scored twice against one actual."""
    with pytest.raises(ValueError, match="duplicate origins"):
        VintagePanelSpec(
            source=mf.data.custom_vintages(_mapping_pair()),
            reference_calendar=pd.DatetimeIndex(["2020-01-31", "2020-01-31"]),
        )


def test_reference_calendar_rejects_nat() -> None:
    with pytest.raises(ValueError, match="NaT"):
        VintagePanelSpec(
            source=mf.data.custom_vintages(_mapping_pair()),
            reference_calendar=pd.DatetimeIndex(["2020-01-31", pd.NaT]),
        )


@pytest.mark.parametrize("limit", [1.0, 1.9, True, "2", 0, -1])
def test_first_release_max_vintages_rejects_non_integral_or_non_positive(limit) -> None:
    """It used to survive construction and be truncated later, so ``1.9`` became a probe
    budget of 1 without anyone saying so."""
    with pytest.raises((TypeError, ValueError)):
        VintagePanelSpec(
            source=mf.data.custom_vintages(_mapping_pair()),
            reference_calendar=_good_calendar(),
            first_release_max_vintages=limit,
        )


@pytest.mark.parametrize("limit", [2, np.int64(2)])
def test_first_release_max_vintages_accepts_integer_scalars(limit) -> None:
    spec = VintagePanelSpec(
        source=mf.data.custom_vintages(_mapping_pair()),
        reference_calendar=_good_calendar(),
        first_release_max_vintages=limit,
    )

    assert spec.first_release_max_vintages == 2


class _ExternalVintageSource:
    """A source whose ``available_vintages()`` order is the caller's, not ours."""

    def __init__(self, keys) -> None:
        self._keys = tuple(keys)

    def available_vintages(self):
        return self._keys

    def resolve(self, origin_date):
        return mf.data.DataBundle(mf.data.as_panel(_snapshot(1.0)), {"vintage": "x"})


def test_first_release_rejects_an_unsorted_available_calendar() -> None:
    """First release walks forward from a bisect, so the reported order IS the search
    order. Unsorted, it returns a later release and calls it the first one."""
    with pytest.raises(ValueError, match="increasing order"):
        VintagePanelSpec(
            source=_ExternalVintageSource(["2020-02-01", "2020-01-01"]),
            reference_calendar=_good_calendar(),
            actuals_vintage="first_release",
        )


def test_first_release_rejects_canonical_duplicate_keys() -> None:
    """Different raw keys, same instant: bisect cannot choose between them."""
    with pytest.raises(ValueError, match="distinct vintage"):
        VintagePanelSpec(
            source=_ExternalVintageSource(["2020-01-01", pd.Timestamp("2020-01-01", tz="UTC")]),
            reference_calendar=_good_calendar(),
            actuals_vintage="first_release",
        )


def test_first_release_accepts_a_sorted_unique_calendar() -> None:
    spec = VintagePanelSpec(
        source=_ExternalVintageSource(["2020-01-01", "2020-02-01"]),
        reference_calendar=_good_calendar(),
        actuals_vintage="first_release",
    )

    assert spec.actuals_vintage == "first_release"


def test_mixed_naive_and_aware_keys_order_and_resolve() -> None:
    """Comparing a naive key to an aware one is a raw pandas ``TypeError`` about
    operands, and it used to come out of ``bisect_right`` — from a search the caller
    never wrote. Canonical UTC-naive instants put both on one line."""
    source = mf.data.custom_vintages(
        {"2020-01-01": _snapshot(1.0), pd.Timestamp("2020-02-01", tz="UTC"): _snapshot(2.0)}
    )

    resolved = source.resolve(pd.Timestamp("2020-02-15"))

    assert resolved.panel["y"].iloc[0] == 2.0


def test_mixed_key_source_reports_the_raw_keys() -> None:
    """Canonicalisation governs ordering, not public labels."""
    aware = pd.Timestamp("2020-02-01", tz="UTC")
    source = mf.data.custom_vintages({"2020-01-01": _snapshot(1.0), aware: _snapshot(2.0)})

    keys = list(source.available_vintages())

    assert keys[0] == "2020-01-01"
    assert keys[1] == aware and keys[1].tz is not None


def test_timezone_equivalent_duplicate_instants_reject() -> None:
    with pytest.raises(ValueError, match="same instant"):
        mf.data.custom_vintages(
            {"2020-01-01": _snapshot(1.0), pd.Timestamp("2020-01-01", tz="UTC"): _snapshot(2.0)}
        )


@pytest.mark.parametrize("strict", [True, False])
def test_long_frame_rejects_missing_vintage_keys(strict: bool) -> None:
    """``groupby`` drops NaN keys silently, so the snapshot and its rows vanished.

    Refused in both strict modes: a missing vintage key is not a value to coerce, it is
    an unanswerable question about which snapshot the row belongs to.
    """
    frame = pd.DataFrame(
        {
            "v": ["2020-01-01", None, "2020-02-01"],
            "date": ["2020-01-31", "2020-02-29", "2020-02-29"],
            "y": [1.0, 9.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        mf.data.custom_vintages(frame, vintage_column="v", date_column="date", strict=strict)


# --------------------------------------------------------------------------- #
# Controller round 2: the canonical path must not reach user code or metadata
# --------------------------------------------------------------------------- #

def test_callable_source_receives_the_origin_it_was_given() -> None:
    """Canonicalising is for comparing against a calendar, and a callable has none.

    Making every origin UTC-naive on the way in changed the argument the caller's own
    function received, and changed what ``vintage_id`` derived the cache label from — a
    regression introduced by the F-011 canonicalisation, not by the defect it fixed.
    """
    seen: list[pd.Timestamp] = []

    def live(origin):
        seen.append(pd.Timestamp(origin))
        return _snapshot(1.0)

    aware_origin = pd.Timestamp("2020-02-15", tz="UTC")
    source = mf.data.custom_vintages(live, vintage_id=lambda _: "aware-live")
    source.resolve(aware_origin)

    assert seen == [aware_origin]
    assert seen[0].tz is not None


def test_callable_source_vintage_id_receives_the_raw_origin() -> None:
    """The label is derived from what the caller passed, so it must not be rewritten."""
    labels: list[object] = []

    def live(origin):
        return _snapshot(1.0)

    def label(origin):
        labels.append(origin)
        return "live"

    aware_origin = pd.Timestamp("2020-02-15", tz="UTC")
    mf.data.custom_vintages(live, vintage_id=label).resolve(aware_origin)

    assert labels == [aware_origin]


def test_probe_limit_is_stored_as_a_plain_int() -> None:
    """Validating a numpy scalar and keeping it is not the same as accepting it.

    The runner deliberately no longer coerces this value, so whatever the field holds
    reaches run metadata and provenance — where a ``np.int64`` is not JSON-serialisable.
    """
    spec = VintagePanelSpec(
        source=mf.data.custom_vintages(_mapping_pair()),
        reference_calendar=_good_calendar(),
        first_release_max_vintages=np.int64(2),
    )

    assert type(spec.first_release_max_vintages) is int
    assert json.loads(json.dumps({"limit": spec.first_release_max_vintages})) == {"limit": 2}
