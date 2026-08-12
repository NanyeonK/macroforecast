"""F-063: ``ForecastResult.to_dict()`` is a real JSON boundary, and it fails loudly.

``_json_ready`` in ``forecasting/types.py`` was a best-effort coercion, not a
boundary. Three separate defect classes came out of that, and they are pinned
separately below because they have different severities and different fixes.

*It produced values ``json`` cannot encode.* A ``datetime.datetime``,
``datetime.date``, ``datetime.time``, a ``pd.Timedelta``, a Python
``timedelta``, and a ``np.datetime64`` at day resolution all reached
``json.dumps`` unconverted and raised ``TypeError`` there -- from inside
``to_json``, i.e. after the caller had already been handed a "JSON-ready" dict.
``Series.name`` was copied out RAW, so a non-string name broke the same way even
though the index beside it was converted.

*It produced values that encoded but were WRONG.* ``np.datetime64`` at
nanosecond resolution became its epoch integer (``949276800000000000``), and the
same instant at picosecond resolution became a DIFFERENT integer, so the JSON
silently depended on the array's dtype unit. ``np.timedelta64`` did the same.
Worse, two mapping keys that normalize to one string (``{1: "int", "1": "str"}``)
silently DROPPED one value, and a DataFrame with colliding column labels silently
dropped a whole column of data -- both encoded fine and were simply not the
object the caller passed.

*It died in ways a caller could not act on.* ``np.longdouble`` -- finite ones
included -- hit ``RecursionError``, because ``.item()`` on a longdouble returns
another longdouble and the helper followed it (the same fixed point F-062
documented for the checkpoint writer). A container cycle, and an object whose
``to_dict()`` returns itself, did the same. An unsupported leaf nested deep in
metadata raised a bare ``TypeError`` naming only the type, with nothing to say
WHERE it was.

What is pinned here is therefore a contract, not a patch: the result of
``to_dict()`` is accepted by ``json.dumps(..., allow_nan=False)``, or the call
raises with a path that says which value to fix. Already-healthy output keeps its
structure and spelling -- the golden below is the literal c30b48eb output and is
asserted whole, so a coercion change that "improves" a working spelling fails
here.

Three later sections record where that summary needs qualifying, and they are
separated out because each is a different kind of claim.

*Section 14 is the one exception to "spelling preserved".* A tuple-valued
``Series.name`` used to be copied out RAW and is recursively sanitized now, so
``to_dict()`` returns a ``list`` where it returned a ``tuple``. ``json`` writes
both as the same array, so no ``to_json`` text and no artifact on disk moves --
but a caller who type-checks or mutates that field in the ``to_dict()`` payload
does get a different Python object.

*Section 15 is compatibility running the other way.* ``to_dict()`` was never the
only pass a payload made: ``output.write_artifacts`` runs it through
``output.core._json_ready`` on the way to the file, and THAT pass turned a
``set`` into a sorted list. Making this boundary strict took the set out before
the second pass could normalize it, breaking a caller who had nothing wrong with
them, so the sorted list is produced here now.

*Section 13 is a defect the first pass at this file missed.* A FINITE
``np.longdouble`` above ``float64``'s range narrows to ``inf`` and was then
written out as ``null``, indistinguishable from a genuine missing value. That is
a data corruption wearing a conversion's clothes, and it is refused.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from macroforecast.forecasting.types import ForecastResult


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _result(**metadata: Any) -> ForecastResult:
    """A one-row forecast table plus whatever metadata the test is probing."""
    forecasts = pd.DataFrame(
        {"date": [pd.Timestamp("2000-01-31")], "prediction": [1.0]}
    )
    return ForecastResult(forecasts, metadata=dict(metadata))


def _strict(payload: Any) -> str:
    """Encode as RFC 8259 JSON: no NaN/Infinity tokens, no ``default`` hook.

    Every assertion about "this is JSON" goes through here rather than through a
    bare ``json.dumps``, so a test cannot pass merely because Python's encoder is
    willing to emit tokens a third-party parser rejects.
    """
    return json.dumps(payload, allow_nan=False)


def _ready(value: Any) -> Any:
    """The normalized ``metadata`` payload for one probe value."""
    return _result(probe=value).to_dict()["metadata"]["probe"]


def _raises(value: Any, exc: type[BaseException]) -> str:
    """Attach ``value`` as metadata, expect ``exc`` from ``to_dict``, return it."""
    with pytest.raises(exc) as info:
        _result(probe=value).to_dict()
    return str(info.value)


# --------------------------------------------------------------------------- #
# 1. The healthy golden: structure and spelling are preserved EXACTLY
# --------------------------------------------------------------------------- #
#: The literal ``to_dict()`` output at c30b48eb for :func:`_healthy_result`.
#: Asserted whole rather than key by key: F-063 widens what the boundary
#: ACCEPTS, and this is what stops it from also changing what it EMITS.
_HEALTHY_GOLDEN: dict[str, Any] = {
    "forecasts": [{"date": "2000-01-31T00:00:00", "prediction": 1.0}],
    "metadata": {
        "metadata_schema": {"kind": "forecast_result", "version": 1},
        "run": {"input_path": "panel", "n": 3, "score": 0.5, "miss": None},
        "path": "/tmp/x.csv",
        "arr": [1.0, 2.0],
        "tup": [1, "a"],
        "ts": "2000-01-31T00:00:00",
        "series": {"name": "s", "index": ["a", "b"], "data": [1.0, 2.0]},
        "frame": {"columns": ["c"], "index": [0, 1], "data": {"c": [1, 2]}},
        "flag": True,
        "none": None,
    },
    "sidecars": {},
}


def _healthy_result() -> ForecastResult:
    """A result whose every metadata value ALREADY encoded before F-063."""
    return _result(
        metadata_schema={"kind": "forecast_result", "version": 1},
        run={
            "input_path": "panel",
            "n": np.int64(3),
            "score": np.float64(0.5),
            "miss": float("nan"),
        },
        path=Path("/tmp/x.csv"),
        arr=np.array([1.0, 2.0]),
        tup=(1, "a"),
        ts=pd.Timestamp("2000-01-31"),
        series=pd.Series([1.0, 2.0], index=["a", "b"], name="s"),
        frame=pd.DataFrame({"c": [1, 2]}, index=[0, 1]),
        flag=True,
        none=None,
    )


def test_healthy_to_dict_structure_and_spelling_are_unchanged() -> None:
    assert _healthy_result().to_dict() == _HEALTHY_GOLDEN


def test_healthy_to_dict_is_accepted_by_strict_json() -> None:
    assert json.loads(_strict(_healthy_result().to_dict())) == _HEALTHY_GOLDEN


def test_healthy_to_json_round_trips_through_a_strict_parser(tmp_path: Path) -> None:
    result = _healthy_result()
    target = tmp_path / "forecast_result.json"

    text = result.to_json(target)

    assert json.loads(text) == _HEALTHY_GOLDEN
    assert json.loads(target.read_text(encoding="utf-8")) == _HEALTHY_GOLDEN


# --------------------------------------------------------------------------- #
# 2. datetime-like values become ISO strings; missing ones become None
# --------------------------------------------------------------------------- #
def test_np_datetime64_is_iso_at_every_resolution_not_an_epoch_integer() -> None:
    # Nanosecond resolution encoded as ``949276800000000000`` before F-063: it is
    # valid JSON and completely wrong, and the same instant at another unit
    # produced a different integer.
    assert _ready(np.datetime64("2000-01-31T00:00:00", "ns")) == (
        "2000-01-31T00:00:00.000000000"
    )
    assert _ready(np.datetime64("2000-01-31", "D")) == "2000-01-31"
    # Picoseconds cannot reach the year 2000 (numpy's own int64 range), so an
    # in-range instant is used; the point is that the UNIT survives.
    assert _ready(np.datetime64(1234567, "ps")) == "1970-01-01T00:00:00.000001234567"


def test_np_datetime64_resolutions_agree_on_the_same_instant() -> None:
    instant = np.datetime64("2000-01-31T12:30:00", "s")
    seconds = _ready(instant)
    nanos = _ready(instant.astype("datetime64[ns]"))

    assert seconds == "2000-01-31T12:30:00"
    # Same instant, more precision -- not a different number entirely.
    assert nanos.startswith(seconds)


def test_missing_temporal_values_become_none() -> None:
    assert _ready(np.datetime64("NaT", "ns")) is None
    assert _ready(np.datetime64("NaT", "D")) is None
    assert _ready(pd.NaT) is None
    assert _ready(np.timedelta64("NaT", "ns")) is None


def test_python_and_pandas_datetimes_become_iso_strings() -> None:
    assert _ready(pd.Timestamp("2000-01-31")) == "2000-01-31T00:00:00"
    assert _ready(dt.datetime(2000, 1, 31, 12, 30)) == "2000-01-31T12:30:00"
    assert _ready(dt.date(2000, 1, 31)) == "2000-01-31"
    assert _ready(dt.time(12, 30)) == "12:30:00"


def test_datetime_like_metadata_encodes_strictly() -> None:
    result = _result(
        stamp=dt.datetime(2000, 1, 31, 12, 30),
        day=dt.date(2000, 1, 31),
        clock=dt.time(12, 30),
        numpy_day=np.datetime64("2000-01-31", "D"),
        missing=np.datetime64("NaT", "ns"),
    )

    payload = json.loads(_strict(result.to_dict()))["metadata"]

    assert payload == {
        "stamp": "2000-01-31T12:30:00",
        "day": "2000-01-31",
        "clock": "12:30:00",
        "numpy_day": "2000-01-31",
        "missing": None,
    }


# --------------------------------------------------------------------------- #
# 3. timedelta-like values become ISO 8601 durations
# --------------------------------------------------------------------------- #
def test_np_timedelta64_is_an_iso_duration_at_more_than_one_unit() -> None:
    # ``ns`` encoded as the bare integer ``1500`` before F-063; ``s`` and ``D``
    # unboxed to a ``datetime.timedelta`` and then raised in ``json.dumps``.
    assert _ready(np.timedelta64(3, "D")) == "P3DT0H0M0S"
    assert _ready(np.timedelta64(90, "s")) == "P0DT0H1M30S"
    assert _ready(np.timedelta64(1500, "ns")) == "P0DT0H0M0.0000015S"


def test_np_timedelta64_units_agree_on_the_same_duration() -> None:
    assert _ready(np.timedelta64(1, "D")) == _ready(np.timedelta64(24, "h"))
    assert _ready(np.timedelta64(1, "D")) == _ready(np.timedelta64(86400, "s"))


def test_pandas_and_python_timedeltas_are_iso_durations() -> None:
    assert _ready(pd.Timedelta(days=1, seconds=90)) == "P1DT0H1M30S"
    assert _ready(dt.timedelta(days=1, seconds=90)) == "P1DT0H1M30S"
    assert _ready(dt.timedelta(0)) == "P0DT0H0M0S"


def test_timedelta_spelling_is_the_same_across_all_three_types() -> None:
    """One duration, three source types, one string -- or the boundary lies."""
    spellings = {
        _ready(np.timedelta64(90, "s")),
        _ready(pd.Timedelta(seconds=90)),
        _ready(dt.timedelta(seconds=90)),
    }

    assert spellings == {"P0DT0H1M30S"}


def test_negative_durations_use_the_pandas_floor_decomposition() -> None:
    assert _ready(dt.timedelta(days=-1, seconds=-90)) == "P-2DT23H58M30S"
    assert _ready(pd.Timedelta(days=-1, seconds=-90)) == "P-2DT23H58M30S"
    assert _ready(np.timedelta64(-86490, "s")) == "P-2DT23H58M30S"


def test_timedelta_metadata_encodes_strictly() -> None:
    result = _result(gap=np.timedelta64(3, "D"), lag=pd.Timedelta(seconds=90))

    assert json.loads(_strict(result.to_dict()))["metadata"] == {
        "gap": "P3DT0H0M0S",
        "lag": "P0DT0H1M30S",
    }


# --------------------------------------------------------------------------- #
# 4. numeric leaves: finite reals become numbers, non-finite become None
# --------------------------------------------------------------------------- #
def test_longdouble_is_a_number_instead_of_a_recursionerror() -> None:
    assert np.finfo(np.longdouble).bits >= np.finfo(np.float64).bits
    value = _ready(np.longdouble("1.5"))

    assert isinstance(value, float)
    assert value == 1.5


def test_non_finite_reals_become_none_at_every_width() -> None:
    for value in (
        np.longdouble("nan"),
        np.longdouble("inf"),
        np.longdouble("-inf"),
        np.float64("nan"),
        np.float32("inf"),
        float("nan"),
        float("-inf"),
    ):
        assert _ready(value) is None, value


def test_finite_numeric_metadata_encodes_strictly() -> None:
    result = _result(wide=np.longdouble("1.5"), narrow=np.float32("0.5"), n=np.int64(7))

    payload = json.loads(_strict(result.to_dict()))["metadata"]

    assert payload["wide"] == 1.5
    assert payload["n"] == 7
    assert math.isclose(payload["narrow"], 0.5)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(complex(1, 2), id="python-complex"),
        pytest.param(np.complex128(1 + 2j), id="numpy-complex128"),
        pytest.param(np.clongdouble(1 + 2j), id="numpy-clongdouble"),
    ],
)
def test_complex_values_are_refused_with_a_path(value: Any) -> None:
    message = _raises(value, TypeError)

    assert "$.metadata.probe" in message


def test_arbitrary_objects_are_refused_with_a_path() -> None:
    message = _raises(object(), TypeError)

    assert "$.metadata.probe" in message


# --------------------------------------------------------------------------- #
# 5. Series and DataFrame are traversed on every side
# --------------------------------------------------------------------------- #
def test_series_name_index_and_data_are_all_converted() -> None:
    # ``name`` was copied out RAW before F-063, so this Series encoded its index
    # and then died on its own name.
    series = pd.Series(
        [1.0, float("inf")],
        index=[pd.Timestamp("2000-01-31"), np.datetime64("2000-02-29", "D")],
        name=pd.Timestamp("2001-01-01"),
    )

    assert _ready(series) == {
        "name": "2001-01-01T00:00:00",
        "index": ["2000-01-31T00:00:00", "2000-02-29T00:00:00"],
        "data": [1.0, None],
    }


def test_dataframe_labels_index_and_data_are_all_converted() -> None:
    frame = pd.DataFrame(
        {"a": [1.0, float("nan")]},
        index=[pd.Timestamp("2000-01-31"), pd.Timestamp("2000-02-29")],
    )

    assert _ready(frame) == {
        "columns": ["a"],
        "index": ["2000-01-31T00:00:00", "2000-02-29T00:00:00"],
        "data": {"a": [1.0, None]},
    }


def test_series_and_dataframe_metadata_encode_strictly() -> None:
    result = _result(
        series=pd.Series([1.0], index=[pd.Timestamp("2000-01-31")], name="s"),
        frame=pd.DataFrame({"a": [1.0]}, index=[pd.Timestamp("2000-01-31")]),
    )

    assert json.loads(_strict(result.to_dict()))["metadata"] == {
        "series": {
            "name": "s",
            "index": ["2000-01-31T00:00:00"],
            "data": [1.0],
        },
        "frame": {
            "columns": ["a"],
            "index": ["2000-01-31T00:00:00"],
            "data": {"a": [1.0]},
        },
    }


def test_unsupported_series_and_frame_leaves_carry_their_position() -> None:
    series_message = _raises(pd.Series([complex(1, 2)], name="s"), TypeError)
    frame_message = _raises(pd.DataFrame({"a": [complex(1, 2)]}), TypeError)

    assert "$.metadata.probe.data[0]" in series_message
    assert "$.metadata.probe.data" in frame_message
    assert "a" in frame_message


# --------------------------------------------------------------------------- #
# 6. A normalized collision is refused, never silently resolved
# --------------------------------------------------------------------------- #
def test_mapping_key_collision_is_refused_instead_of_dropping_a_value() -> None:
    # ``{1: "int", "1": "str"}`` normalized to ``{"1": "str"}`` before F-063:
    # valid JSON, one value gone, no warning.
    message = _raises({1: "int", "1": "str"}, ValueError)

    assert "$.metadata.probe" in message
    assert "'1'" in message


def test_dataframe_column_collision_is_refused_instead_of_dropping_a_column() -> None:
    frame = pd.DataFrame([[1, 2]], columns=[1, "1"])

    message = _raises(frame, ValueError)

    assert "$.metadata.probe" in message
    assert "'1'" in message


def test_a_collision_leaves_no_partial_output_behind() -> None:
    """The whole call fails; a caller never receives a half-normalized dict."""
    result = _result(good="kept", bad={1: "int", "1": "str"})

    with pytest.raises(ValueError):
        result.to_dict()
    with pytest.raises(ValueError):
        result.to_json()


def test_distinct_keys_that_do_not_collide_still_pass() -> None:
    assert _ready({1: "int", 2: "str"}) == {"1": "int", "2": "str"}


# --------------------------------------------------------------------------- #
# 7. Cycles are refused instead of exhausting the stack
# --------------------------------------------------------------------------- #
def test_direct_container_cycles_are_refused() -> None:
    looping_list: list[Any] = [1]
    looping_list.append(looping_list)
    looping_map: dict[str, Any] = {}
    looping_map["self"] = looping_map

    assert "$.metadata.probe" in _raises(looping_list, ValueError)
    assert "$.metadata.probe" in _raises(looping_map, ValueError)


def test_an_object_whose_to_dict_returns_itself_is_refused() -> None:
    class SelfDict:
        def to_dict(self) -> "SelfDict":
            return self

    message = _raises(SelfDict(), ValueError)

    assert "$.metadata.probe" in message


def test_the_same_object_twice_as_a_sibling_is_not_a_cycle() -> None:
    """``_active`` tracks the current PATH, so shared structure still encodes."""
    shared = {"a": 1}

    assert _ready([shared, shared]) == [{"a": 1}, {"a": 1}]
    assert _ready({"left": shared, "right": shared}) == {
        "left": {"a": 1},
        "right": {"a": 1},
    }


# --------------------------------------------------------------------------- #
# 8. A deep failure says WHERE it is
# --------------------------------------------------------------------------- #
def test_a_nested_unsupported_value_reports_its_full_path() -> None:
    message = _raises({"outer": [{"inner": complex(1, 2)}]}, TypeError)

    assert "$.metadata.probe.outer[0].inner" in message


def test_a_path_survives_a_non_identifier_key() -> None:
    message = _raises({"a key": complex(1, 2)}, TypeError)

    assert "'a key'" in message


def test_a_failure_inside_a_sidecar_names_the_sidecar() -> None:
    result = ForecastResult(
        pd.DataFrame({"a": [1.0]}), sidecars={"anatomy": {"weight": complex(1, 2)}}
    )

    with pytest.raises(TypeError) as info:
        result.to_dict()

    assert "$.sidecars.anatomy.weight" in str(info.value)


def test_a_failure_inside_the_forecast_table_names_the_row() -> None:
    forecasts = pd.DataFrame({"date": [pd.Timestamp("2000-01-31")]})
    forecasts["odd"] = [complex(1, 2)]

    with pytest.raises(TypeError) as info:
        ForecastResult(forecasts).to_dict()

    assert "$.forecasts[0].odd" in str(info.value)


# --------------------------------------------------------------------------- #
# 9. Sidecar schemas are validated when they are ATTACHED
# --------------------------------------------------------------------------- #
class _Sidecar:
    """The sidecar shape ``_sidecar_metadata`` reads: schema plus metadata.

    ``to_dict`` is present because the real sidecars have it -- both
    ``interpretation.dual`` and ``interpretation.anatomy`` define one -- and it
    is how the sidecar itself reaches the payload under ``$.sidecars``.
    """

    def __init__(self, schema: Any, metadata: Mapping[str, Any] | None = None) -> None:
        self.metadata_schema = schema
        self.metadata = dict(metadata or {"b": 2, "a": 1})

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": dict(self.metadata)}


class _OpaqueSidecar:
    """A sidecar with a valid schema and no JSON form of its own."""

    metadata_schema = {"kind": "opaque", "version": 1}
    metadata = {"a": 1}


def test_a_valid_sidecar_schema_is_converted_at_attachment() -> None:
    sidecar = _Sidecar({"kind": "demo", "version": np.int64(1)})

    attached = _result().with_sidecar("demo", sidecar)
    registered = attached.metadata["sidecars"]["demo"]

    assert registered["metadata_schema"] == {"kind": "demo", "version": 1}
    assert registered["metadata_keys"] == ["a", "b"]
    assert registered["object_type"].endswith("._Sidecar")
    _strict(attached.to_dict())


def test_a_callable_sidecar_schema_is_still_supported() -> None:
    class CallableSchema:
        metadata: dict[str, Any] = {"a": 1}

        def metadata_schema(self) -> dict[str, Any]:
            return {"kind": "callable", "version": 1}

    attached = _result().with_sidecar("demo", CallableSchema())

    assert attached.metadata["sidecars"]["demo"]["metadata_schema"] == {
        "kind": "callable",
        "version": 1,
    }


def test_an_invalid_sidecar_schema_fails_in_with_sidecar_not_later() -> None:
    # Before F-063 the bad schema was stored verbatim and only surfaced when
    # someone later called ``to_json``, by which point the result carrying it had
    # been passed around and the attachment site was long gone.
    sidecar = _Sidecar({"kind": "demo", "bad": complex(1, 2)})

    with pytest.raises(TypeError) as info:
        _result().with_sidecar("demo", sidecar)

    message = str(info.value)
    assert "metadata_schema" in message
    assert "demo" in message


def test_a_sidecar_object_with_no_json_form_is_refused_with_a_path() -> None:
    """Attaching succeeds; EXPORTING is what needs a JSON form, and says so.

    This is a deliberate F-063 change. The schema is fine, so ``with_sidecar``
    has nothing to object to, but the sidecar object itself has no ``to_dict``.
    ``to_dict()`` used to hand that object back INSIDE the payload it called
    JSON-ready, and the ``TypeError`` landed later in ``json.dumps`` naming only
    the type. It is refused at the boundary now, with the sidecar's name.
    """
    attached = _result().with_sidecar("opaque", _OpaqueSidecar())

    assert attached.sidecar_names() == ("opaque",)
    with pytest.raises(TypeError) as info:
        attached.to_dict()

    assert "$.sidecars.opaque" in str(info.value)


def test_a_rejected_sidecar_does_not_mutate_the_original_result() -> None:
    result = _result()
    sidecar = _Sidecar({"bad": complex(1, 2)})

    with pytest.raises(TypeError):
        result.with_sidecar("demo", sidecar)

    assert result.sidecar_names() == ()
    assert "sidecars" not in result.metadata


# --------------------------------------------------------------------------- #
# 10. ``to_json`` is strict, and it never damages an existing file
# --------------------------------------------------------------------------- #
def test_to_json_refuses_the_non_standard_nan_token() -> None:
    """``allow_nan=False`` is the switch; ``_json_ready`` is why it can be on."""
    text = _result(miss=float("nan"), gone=np.float64("inf")).to_json()

    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text)["metadata"] == {"miss": None, "gone": None}


def test_to_json_serializes_before_it_writes(tmp_path: Path) -> None:
    target = tmp_path / "forecast_result.json"
    original = "PREEXISTING CONTENT, MUST SURVIVE\n"
    target.write_text(original, encoding="utf-8")
    before = target.read_bytes()

    with pytest.raises(TypeError):
        _result(bad=complex(1, 2)).to_json(target)

    assert target.read_bytes() == before
    assert target.read_text(encoding="utf-8") == original


def test_to_json_does_not_create_a_file_when_serialization_fails(
    tmp_path: Path,
) -> None:
    target = tmp_path / "never_written.json"

    with pytest.raises(ValueError):
        _result(bad={1: "int", "1": "str"}).to_json(target)

    assert not target.exists()


def test_to_json_still_writes_a_healthy_result(tmp_path: Path) -> None:
    target = tmp_path / "forecast_result.json"

    text = _healthy_result().to_json(target)

    assert target.read_text(encoding="utf-8") == text + "\n"
    assert json.loads(text) == _HEALTHY_GOLDEN


# --------------------------------------------------------------------------- #
# 11. A temporal ARRAY means what its scalars mean
# --------------------------------------------------------------------------- #
# ``ndarray`` is normalized by handing ``.tolist()`` back to the traversal, and
# for ``datetime64``/``timedelta64`` that is a lossy detour: ``.tolist()`` unboxes
# per ELEMENT with the same unit-dependent rule that broke the scalar branch. At
# ``us`` it silently drops the sub-second digits, and at ``ns`` it abandons dates
# altogether and yields the raw epoch integer -- so ``[949276800000000000]`` and
# ``[1500]`` reached JSON while the identical scalars encoded as ISO. The array
# branch has to defer to the scalar branch, not race it.
_ARRAY_UNITS = ("D", "s", "ms", "us", "ns")


def test_datetime64_arrays_are_iso_at_every_dimensionality() -> None:
    one_d = np.array(["2000-01-31", "2000-02-29"], dtype="datetime64[ns]")
    nested = np.array([["2000-01-31"], ["2000-02-29"]], dtype="datetime64[ns]")
    zero_d = np.array("2000-01-31", dtype="datetime64[ns]")

    assert _ready(one_d) == [
        "2000-01-31T00:00:00.000000000",
        "2000-02-29T00:00:00.000000000",
    ]
    assert _ready(nested) == [
        ["2000-01-31T00:00:00.000000000"],
        ["2000-02-29T00:00:00.000000000"],
    ]
    assert _ready(zero_d) == "2000-01-31T00:00:00.000000000"


def test_timedelta64_arrays_are_iso_at_every_dimensionality() -> None:
    one_d = np.array([1500, 3000], dtype="timedelta64[ns]")
    nested = np.array([[3], [4]], dtype="timedelta64[D]")
    zero_d = np.array(90, dtype="timedelta64[s]")

    assert _ready(one_d) == ["P0DT0H0M0.0000015S", "P0DT0H0M0.000003S"]
    assert _ready(nested) == [["P3DT0H0M0S"], ["P4DT0H0M0S"]]
    assert _ready(zero_d) == "P0DT0H1M30S"


@pytest.mark.parametrize("unit", _ARRAY_UNITS)
def test_a_temporal_array_element_spells_what_that_scalar_spells(unit: str) -> None:
    """The whole contract in one line: array[i] must equal scalar.

    ``D`` and ``s`` already agreed before this fix. ``ms`` and ``us`` lost their
    sub-second digits, and ``ns`` lost the date entirely -- and all three are the
    same defect, so all three are pinned the same way.
    """
    stamps = np.array(["2000-01-31T12:30:00"], dtype=f"datetime64[{unit}]")
    gaps = np.array([90], dtype=f"timedelta64[{unit}]")

    assert _ready(stamps) == [_ready(stamps[0])]
    assert _ready(gaps) == [_ready(gaps[0])]


def test_missing_values_inside_a_temporal_array_become_none() -> None:
    assert _ready(np.array(["2000-01-31", "NaT"], dtype="datetime64[ns]")) == [
        "2000-01-31T00:00:00.000000000",
        None,
    ]
    assert _ready(np.array([1500, "NaT"], dtype="timedelta64[ns]")) == [
        "P0DT0H0M0.0000015S",
        None,
    ]
    assert _ready(np.array([], dtype="datetime64[ns]")) == []


def test_temporal_array_metadata_encodes_strictly() -> None:
    result = _result(
        stamps=np.array(["2000-01-31"], dtype="datetime64[ns]"),
        gaps=np.array([1500], dtype="timedelta64[ns]"),
    )

    assert json.loads(_strict(result.to_dict()))["metadata"] == {
        "stamps": ["2000-01-31T00:00:00.000000000"],
        "gaps": ["P0DT0H0M0.0000015S"],
    }


def test_non_temporal_array_spellings_are_left_exactly_as_they_were() -> None:
    """The fix is scoped to ``datetime64``/``timedelta64``, by dtype kind."""
    assert _ready(np.array([1.0, float("nan")])) == [1.0, None]
    assert _ready(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]
    assert _ready(np.array(1.5)) == 1.5
    assert _ready(np.array(["a", "b"])) == ["a", "b"]
    assert _ready(np.array([True, False])) == [True, False]
    assert _ready(np.array([{"a": 1}], dtype=object)) == [{"a": 1}]
    assert _ready(np.array([1.5], dtype=np.longdouble)) == [1.5]
    assert _ready(np.array([(1, 2.0)], dtype=[("a", "i8"), ("b", "f8")])) == [[1, 2.0]]


def test_masked_arrays_keep_reporting_their_mask_as_null() -> None:
    """A masked entry is missing, and stays ``null`` -- temporal ones included."""
    numbers = np.ma.MaskedArray([1.0, 2.0], mask=[True, False])
    stamps = np.ma.MaskedArray(
        np.array(["2000-01-31", "2000-02-29"], dtype="datetime64[ns]"),
        mask=[True, False],
    )
    gaps = np.ma.MaskedArray(
        np.array([1500, 3000], dtype="timedelta64[ns]"), mask=[True, False]
    )

    assert _ready(numbers) == [None, 2.0]
    # The unmasked half was ``951782400000000000`` before this fix: a masked
    # temporal array had the epoch-integer defect too, on its VISIBLE entries.
    assert _ready(stamps) == [None, "2000-02-29T00:00:00.000000000"]
    assert _ready(gaps) == [None, "P0DT0H0M0.000003S"]


def test_a_temporal_array_failure_still_carries_its_position() -> None:
    message = _raises({"gaps": [np.array(["NaT"], dtype="datetime64[ns]"), object()]},
                      TypeError)

    assert "$.metadata.probe.gaps[1]" in message


# --------------------------------------------------------------------------- #
# 12. The forecast table's own columns are checked BEFORE pandas resolves them
# --------------------------------------------------------------------------- #
# ``to_dict`` normalized ``self.forecasts`` with ``to_dict(orient="records")``,
# and pandas resolves duplicate labels itself: it keeps the LAST and drops the
# rest, announcing it with a ``UserWarning`` that a caller redirecting warnings
# never sees. ``["x", "x"]`` therefore exported ``[{"x": 2}]`` -- valid JSON, one
# column of forecasts gone. The nested-DataFrame path already refused this via
# ``_json_columns``; the top-level table skipped that guard entirely.
def _forecast_columns(*labels: Any) -> ForecastResult:
    """A one-row forecast table with the given (possibly colliding) labels."""
    frame = pd.DataFrame([list(range(len(labels)))], columns=list(labels))
    return ForecastResult(frame)


def test_exactly_duplicated_forecast_columns_are_refused() -> None:
    with pytest.raises(ValueError) as info:
        _forecast_columns("x", "x").to_dict()

    message = str(info.value)
    assert "$.forecasts.columns" in message
    assert "'x'" in message


def test_normalized_collision_in_forecast_columns_is_refused() -> None:
    with pytest.raises(ValueError) as info:
        _forecast_columns(1, "1").to_dict()

    message = str(info.value)
    assert "$.forecasts.columns" in message
    assert "'1'" in message


def test_the_column_guard_runs_before_pandas_can_drop_a_column() -> None:
    """Structural, not incidental: pandas must never get the chance to warn.

    With warnings promoted to errors, reaching ``to_dict(orient="records")``
    first would surface pandas' ``UserWarning`` instead of our ``ValueError``.
    """
    result = _forecast_columns("x", "x")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError) as info:
            result.to_dict()

    assert "$.forecasts.columns" in str(info.value)


def test_duplicated_forecast_columns_never_reach_a_file(tmp_path: Path) -> None:
    target = tmp_path / "forecast_result.json"

    with pytest.raises(ValueError):
        _forecast_columns("x", "x").to_json(target)

    assert not target.exists()


def test_healthy_forecast_records_keep_their_exact_spelling() -> None:
    """The guard adds a refusal, never a change to what a good table exports."""
    assert _healthy_result().to_dict()["forecasts"] == _HEALTHY_GOLDEN["forecasts"]
    assert _forecast_columns("a", "b").to_dict()["forecasts"] == [{"a": 0, "b": 1}]


# --------------------------------------------------------------------------- #
# 13. A finite real wider than float64 is refused, never written out as null
# --------------------------------------------------------------------------- #
# ``np.longdouble`` is the case, and it is a DATA CORRUPTION rather than a crash.
# This package represents reals as native Python floats -- IEEE 754 binary64 --
# because that is what ``json`` emits and what consumers parse back; RFC 8259's own
# number grammar is wider and fixes no precision, so the limit under test is the
# package's representation policy, not the format's syntax. The boundary therefore
# narrows through ``float()``. Above ``float64``'s range that returns ``inf``, and
# the traversal's non-finite rule then writes ``null`` -- exactly the spelling a
# genuine NaN gets. A reader could not tell a measurement of ``1e400`` from a
# missing one. Section 16 is the same corruption at the other end of the range.
#
# ``longdouble`` is only wider than ``float64`` on some platforms (80-bit extended
# on x86-64 Linux and Intel macOS; an alias for ``float64`` on Windows and some
# ARM builds). Where it is an alias no finite value outside ``float64``'s range
# exists to construct, so these are skipped rather than faked.
_LONGDOUBLE_EXCEEDS_FLOAT64 = bool(
    np.finfo(np.longdouble).max > np.finfo(np.float64).max
)
_NEEDS_WIDE_LONGDOUBLE = pytest.mark.skipif(
    not _LONGDOUBLE_EXCEEDS_FLOAT64,
    reason="longdouble is float64 here, so a finite value out of its range cannot exist",
)


@_NEEDS_WIDE_LONGDOUBLE
def test_a_finite_longdouble_above_float64_range_is_refused_not_nulled() -> None:
    wide = np.longdouble("1e400")
    assert bool(np.isfinite(wide)), "precondition: the source value is finite"

    message = _raises(wide, TypeError)

    assert "$.metadata.probe" in message
    assert "float64" in message


@_NEEDS_WIDE_LONGDOUBLE
def test_a_finite_longdouble_below_negative_float64_range_is_refused() -> None:
    wide = np.longdouble("-1e400")
    assert bool(np.isfinite(wide))

    assert "$.metadata.probe" in _raises(wide, TypeError)


@_NEEDS_WIDE_LONGDOUBLE
def test_an_out_of_range_longdouble_inside_an_array_carries_its_position() -> None:
    """The array path unboxes to longdouble too, so it needs the same refusal."""
    message = _raises(np.array([1.5, np.longdouble("1e400")], dtype=np.longdouble),
                      TypeError)

    assert "$.metadata.probe[1]" in message


@_NEEDS_WIDE_LONGDOUBLE
def test_an_out_of_range_longdouble_never_reaches_a_file(tmp_path: Path) -> None:
    target = tmp_path / "forecast_result.json"

    with pytest.raises(TypeError):
        _result(probe=np.longdouble("1e400")).to_json(target)

    assert not target.exists()


# --------------------------------------------------------------------------- #
# 14. ``Series.name`` is traversed, so a TUPLE name changes Python type
# --------------------------------------------------------------------------- #
# This is the one place the F-063 traversal changes an already-working payload,
# and it is recorded here rather than glossed. ``name`` used to be copied out RAW,
# so a tuple name left ``to_dict()`` as a ``tuple``; it is recursively sanitized
# now and leaves as a ``list``. ``json`` encodes both as the same array, so no
# ``to_json`` text and no artifact on disk changes -- only the Python object a
# caller gets back from ``to_dict()`` and then indexes or type-checks.
def test_a_tuple_valued_series_name_is_sanitized_to_a_list() -> None:
    series = pd.Series([1.0], index=["a"], name=("model", "h1"))

    name = _ready(series)["name"]

    assert name == ["model", "h1"]
    assert isinstance(name, list)


def test_a_tuple_valued_series_name_keeps_its_json_text() -> None:
    """The type changed; the SPELLING on disk did not, and that is the claim."""
    series = pd.Series([1.0], index=["a"], name=("model", "h1"))

    text = _result(probe=series).to_json()

    assert json.loads(text)["metadata"]["probe"]["name"] == ["model", "h1"]
    # What the pre-F-063 raw tuple would have encoded as, character for character.
    assert json.dumps(("model", "h1")) == json.dumps(["model", "h1"])


def test_a_series_name_that_needs_converting_is_converted() -> None:
    """The reason ``name`` is traversed at all: a raw one could not encode."""
    named = pd.Series([1.0], index=["a"], name=np.int64(3))

    assert _ready(named)["name"] == 3


# --------------------------------------------------------------------------- #
# 15. A set keeps the sorted list ``write_artifacts`` already produced for it
# --------------------------------------------------------------------------- #
# ``to_dict()`` was never the only normalization a set went through. ``output``'s
# ``write_artifacts`` runs the payload through ``output.core._json_ready`` on the
# way to the file, and THAT pass turned a ``set`` into ``sorted(...)`` -- so a set
# in ``metadata`` wrote a JSON array and the two-phase arrangement worked. Making
# ``to_dict()`` strict took the set out at the first phase, before the second
# could normalize it, and broke a public behavior that had nothing wrong with it.
#
# The sorted list is reproduced here, in this module's own traversal: ``output``
# imports ``forecasting``, so the second pass cannot be imported back to get it.
# ``frozenset`` joins ``set`` because ``output``'s branch tested
# ``isinstance(value, set)``, which a ``frozenset`` is not -- it reached
# ``json.dumps`` unconverted and raised, so it never worked at all.
@pytest.mark.parametrize(
    ("members", "expected"),
    [
        pytest.param({"beta", "alpha", "gamma"}, ["alpha", "beta", "gamma"], id="str"),
        pytest.param({3, 1, 2}, [1, 2, 3], id="int"),
        pytest.param(set(), [], id="empty"),
    ],
)
def test_a_set_becomes_a_deterministic_sorted_list(
    members: Any, expected: list[Any]
) -> None:
    assert _ready(members) == expected


def test_a_frozenset_is_normalized_exactly_like_a_set() -> None:
    assert _ready(frozenset({"beta", "alpha"})) == ["alpha", "beta"]
    assert _ready(frozenset()) == []


def test_set_members_are_converted_before_they_are_sorted() -> None:
    """Order follows the JSON spelling, not the Python objects."""
    stamps = {pd.Timestamp("2001-01-01"), pd.Timestamp("2000-01-01")}

    assert _ready(stamps) == ["2000-01-01T00:00:00", "2001-01-01T00:00:00"]


def test_a_set_in_metadata_encodes_strictly() -> None:
    result = _result(tags={"beta", "alpha"})

    assert json.loads(_strict(result.to_dict()))["metadata"] == {
        "tags": ["alpha", "beta"]
    }


def test_a_set_in_metadata_still_survives_write_artifacts(tmp_path: Path) -> None:
    """End to end, against the caller that made this a public behavior.

    ``output`` is imported inside the test on purpose: the boundary under test
    lives in ``forecasting``, and a module-level import here would make this file
    depend on the layer above it just to collect.
    """
    from macroforecast.output import write_artifacts

    write_artifacts(
        {"run": _result(tags={"beta", "alpha", "gamma"})},
        tmp_path,
        formats=("json",),
    )

    payload = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["tags"] == ["alpha", "beta", "gamma"]


def test_a_set_whose_members_have_no_order_is_refused_with_a_path() -> None:
    """No deterministic list exists, so none is invented.

    Sorting ``{1, "a"}`` raises in ``output``'s pass too -- but as a bare
    ``TypeError`` about ``'<'``, from inside a write, naming nothing. Picking an
    order anyway would make the artifact depend on the interpreter's hash seed.
    """
    message = _raises({1, "a"}, TypeError)

    assert "$.metadata.probe" in message
    assert "order" in message


def test_a_set_of_unencodable_members_names_the_member_not_the_set() -> None:
    """``[*]`` is JSONPath's wildcard: a set member has no stable position."""
    message = _raises({complex(1, 2)}, TypeError)

    assert "$.metadata.probe[*]" in message
    assert "complex" in message


# --------------------------------------------------------------------------- #
# 16. A NESTED ``ForecastResult`` keeps the outer traversal's state
# --------------------------------------------------------------------------- #
# A nested result reached the generic ``to_dict`` protocol branch, which calls
# ``value.to_dict()`` -- and that starts a FRESH traversal: path back to ``$``,
# cycle set empty. Two things followed. A result reachable from its own
# ``metadata`` recursed until the stack gave out, so the documented ``ValueError``
# for a cycle never arrived. And a failure inside a nested result reported a path
# rooted at that result, not at the caller's, which for an unencodable leaf was
# worse still: the inner ``TypeError`` was swallowed by the branch's own
# ``except TypeError`` and re-raised as "a ForecastResult cannot be expressed",
# blaming the container instead of the value.
def _nested(**metadata: Any) -> ForecastResult:
    """A second result, to be attached inside another one's metadata."""
    return ForecastResult(pd.DataFrame({"a": [1.0]}), metadata=dict(metadata))


def test_a_forecast_result_reachable_from_its_own_metadata_is_a_cycle() -> None:
    result = _result()
    result.metadata["self"] = result

    with pytest.raises(ValueError) as info:
        result.to_dict()

    assert "$.metadata.self" in str(info.value)


def test_a_cycle_through_a_second_forecast_result_is_refused() -> None:
    outer = _result()
    inner = _nested()
    outer.metadata["inner"] = inner
    inner.metadata["back"] = outer

    with pytest.raises(ValueError) as info:
        outer.to_dict()

    assert "$.metadata.inner.metadata.back" in str(info.value)


def test_an_unencodable_leaf_in_a_nested_result_reports_the_full_outer_path() -> None:
    outer = _result(inner=_nested(bad=complex(1, 2)))

    with pytest.raises(TypeError) as info:
        outer.to_dict()

    assert "$.metadata.inner.metadata.bad" in str(info.value)


def test_a_column_collision_in_a_nested_result_names_the_nested_table() -> None:
    nested = ForecastResult(pd.DataFrame([[1, 2]], columns=["x", "x"]))

    with pytest.raises(ValueError) as info:
        _result(inner=nested).to_dict()

    assert "$.metadata.inner.forecasts.columns" in str(info.value)


def test_the_same_nested_result_twice_as_a_sibling_is_not_a_cycle() -> None:
    """Shared structure is not a back-reference, and still encodes twice."""
    inner = _nested(k=1)

    payload = _result(left=inner, right=inner).to_dict()["metadata"]

    assert payload["left"] == payload["right"]
    assert payload["left"]["metadata"] == {"k": 1}
    assert payload["left"]["forecasts"] == [{"a": 1.0}]


def test_a_healthy_nested_result_encodes_strictly() -> None:
    result = _result(inner=_nested(when=pd.Timestamp("2000-01-31")))

    assert json.loads(_strict(result.to_dict()))["metadata"]["inner"] == {
        "forecasts": [{"a": 1.0}],
        "metadata": {"when": "2000-01-31T00:00:00"},
        "sidecars": {},
    }


# --------------------------------------------------------------------------- #
# 16. A finite real too SMALL for float64 is refused, never written out as zero
# --------------------------------------------------------------------------- #
# The mirror image of section 13, and the same data corruption seen from the other
# end. This package represents reals as native Python floats -- IEEE 754 binary64
# -- because that is what ``json`` emits and what every consumer of these
# artifacts parses back; RFC 8259's own number grammar is wider than that, so the
# constraint is the package's representation and interoperability policy, not the
# format's syntax. Narrowing a wider numpy float therefore goes through
# ``float()``. ABOVE binary64's range that returns an infinity (section 13); BELOW
# its smallest subnormal it returns a ZERO, and zero is not a spelling the
# traversal rejects, so a finite NONZERO measurement was written out as ``0.0``.
#
# That is worse than the overflow case in one respect. ``null`` at least reads as
# missingness, so a careful consumer treats it with suspicion; ``0.0`` reads as a
# measurement that was really taken and really was zero.
#
# The sign survives the underflow -- ``float(np.longdouble("-1e-4000"))`` is
# ``-0.0`` -- so both signs are asserted rather than assumed symmetric.
#
# Whether a finite value below ``float64``'s range EXISTS is a platform question,
# exactly as in section 13, and is asked by construction rather than by inspecting
# ``np.finfo``: build the value, and check it is finite, nonzero, and narrows away.
_LONGDOUBLE_UNDERFLOWS_FLOAT64 = bool(
    np.isfinite(np.longdouble("1e-4000"))
    and np.longdouble("1e-4000") != np.longdouble(0)
    and float(np.longdouble("1e-4000")) == 0.0
)
_NEEDS_NARROW_LONGDOUBLE = pytest.mark.skipif(
    not _LONGDOUBLE_UNDERFLOWS_FLOAT64,
    reason="longdouble is float64 here, so a finite value under its range cannot exist",
)


@_NEEDS_NARROW_LONGDOUBLE
def test_a_finite_longdouble_below_float64_range_is_refused_not_zeroed() -> None:
    tiny = np.longdouble("1e-4000")
    assert bool(np.isfinite(tiny)), "precondition: the source value is finite"
    assert tiny != np.longdouble(0), "precondition: the source value is nonzero"

    message = _raises(tiny, TypeError)

    assert "$.metadata.probe" in message
    assert "float64" in message


@_NEEDS_NARROW_LONGDOUBLE
def test_a_negative_finite_longdouble_underflow_is_refused_symmetrically() -> None:
    """``-1e-4000`` narrows to ``-0.0``, which is just as much a lie as ``0.0``."""
    tiny = np.longdouble("-1e-4000")
    assert bool(np.isfinite(tiny))
    assert tiny != np.longdouble(0)

    assert "$.metadata.probe" in _raises(tiny, TypeError)


@_NEEDS_NARROW_LONGDOUBLE
def test_the_out_of_range_error_admits_zero_and_not_only_null() -> None:
    """The message has to describe the failure the CALLER actually hit.

    An explanation that only discusses overflow and ``null`` is wrong for half the
    values it now refuses, and a caller reading it would go looking for an
    infinity that never existed.
    """
    message = _raises(np.longdouble("1e-4000"), TypeError)

    assert "zero" in message.lower()


@_NEEDS_NARROW_LONGDOUBLE
def test_an_underflowing_longdouble_inside_an_array_carries_its_position() -> None:
    message = _raises(
        np.array([1.5, np.longdouble("1e-4000")], dtype=np.longdouble), TypeError
    )

    assert "$.metadata.probe[1]" in message


@_NEEDS_NARROW_LONGDOUBLE
def test_an_underflowing_longdouble_never_reaches_a_file(tmp_path: Path) -> None:
    target = tmp_path / "forecast_result.json"

    with pytest.raises(TypeError):
        _result(probe=np.longdouble("1e-4000")).to_json(target)

    assert not target.exists()


def test_a_true_zero_is_still_encoded_as_zero() -> None:
    """The refusal is about a value that DISAPPEARS, not about the number zero."""
    assert _ready(np.longdouble(0)) == 0.0
    assert _ready(np.longdouble("-0.0")) == 0.0
    assert _ready(np.float64(0)) == 0.0
    assert _ready(0.0) == 0.0


def test_ordinary_finite_longdouble_values_still_narrow_to_a_number() -> None:
    """Values with a finite nonzero float64 image are untouched by the new guard."""
    assert _ready(np.longdouble("1.5")) == 1.5
    assert _ready(np.longdouble("-1.5")) == -1.5
    # Representable at float64 and far smaller than 1.0: still a number, not a refusal.
    assert _ready(np.longdouble("1e-300")) == 1e-300
    assert _ready(np.longdouble("1e300")) == 1e300


def test_a_non_finite_longdouble_source_is_still_null_not_refused() -> None:
    """Only a FINITE source is refused; a genuine NaN or infinity still nulls."""
    for value in (
        np.longdouble("nan"),
        np.longdouble("inf"),
        np.longdouble("-inf"),
    ):
        assert _ready(value) is None


# --------------------------------------------------------------------------- #
# 17. pandas narrows a wide column BEFORE the traversal can refuse it
# --------------------------------------------------------------------------- #
# Sections 13 and 16 guard the traversal, and the traversal is not where a frame's
# cells arrive from. ``DataFrame.to_dict`` builds its own Python objects first, and
# for a ``float128`` column that conversion goes through ``float()`` itself --
# ``to_dict(orient="list")`` hands back a Python ``inf`` where the column held a
# finite ``np.longdouble("1e400")``, and a Python ``0.0`` where it held
# ``np.longdouble("1e-4000")``. By the time ``_json_ready`` sees the cell the
# evidence that it was ever finite is gone, the non-finite rule spells it ``null``,
# and the guard that exists to prevent exactly that never runs.
#
# ``Series.to_list()`` does NOT do this -- it yields the ``np.longdouble`` back --
# which is why a Series in ``metadata`` was already caught and a FRAME was not. The
# repair is to stop routing frames through ``to_dict`` at all and extract each
# column as a Series instead, so every cell reaches the traversal at its own dtype.
#
# Which ``orient`` corrupts is a pandas-version detail (``orient="records"``
# preserves ``float128`` on some versions and narrows it on others), so both the
# nested-frame path and the top-level ``forecasts`` path are pinned here rather
# than only the one that happens to fail on the machine running this file.
def _wide_column(value: Any) -> pd.DataFrame:
    """A one-cell frame whose column really is ``longdouble``, not ``object``."""
    frame = pd.DataFrame({"v": np.array([value], dtype=np.longdouble)})
    assert frame["v"].dtype == np.dtype(np.longdouble), "precondition: dtype survived"
    return frame


@_NEEDS_WIDE_LONGDOUBLE
def test_a_longdouble_series_refuses_an_overflowing_value() -> None:
    """The control: ``Series.to_list`` preserves the dtype, so this always worked."""
    series = pd.Series([np.longdouble("1e400")], dtype=np.longdouble)
    assert series.dtype == np.dtype(np.longdouble)

    assert "$.metadata.probe" in _raises(series, TypeError)


@_NEEDS_NARROW_LONGDOUBLE
def test_a_longdouble_series_refuses_an_underflowing_value() -> None:
    series = pd.Series([np.longdouble("1e-4000")], dtype=np.longdouble)
    assert series.dtype == np.dtype(np.longdouble)

    assert "$.metadata.probe" in _raises(series, TypeError)


@_NEEDS_WIDE_LONGDOUBLE
def test_a_nested_frame_refuses_an_overflowing_cell_instead_of_nulling_it() -> None:
    message = _raises(_wide_column(np.longdouble("1e400")), TypeError)

    assert "$.metadata.probe.data.v[0]" in message


@_NEEDS_NARROW_LONGDOUBLE
def test_a_nested_frame_refuses_an_underflowing_cell_instead_of_zeroing_it() -> None:
    message = _raises(_wide_column(np.longdouble("1e-4000")), TypeError)

    assert "$.metadata.probe.data.v[0]" in message


@_NEEDS_WIDE_LONGDOUBLE
def test_the_forecast_table_refuses_an_overflowing_longdouble() -> None:
    with pytest.raises(TypeError) as info:
        ForecastResult(_wide_column(np.longdouble("1e400"))).to_dict()

    assert "$.forecasts[0].v" in str(info.value)


@_NEEDS_NARROW_LONGDOUBLE
def test_the_forecast_table_refuses_an_underflowing_longdouble() -> None:
    with pytest.raises(TypeError) as info:
        ForecastResult(_wide_column(np.longdouble("1e-4000"))).to_dict()

    assert "$.forecasts[0].v" in str(info.value)


@_NEEDS_WIDE_LONGDOUBLE
def test_a_wide_forecast_cell_never_reaches_a_file(tmp_path: Path) -> None:
    target = tmp_path / "forecast_result.json"

    with pytest.raises(TypeError):
        ForecastResult(_wide_column(np.longdouble("1e400"))).to_json(target)

    assert not target.exists()


# --- the compatibility half: the replacement must not move ordinary columns --- #
#: Every dtype family a forecast table realistically carries, including the two
#: that pandas represents with its OWN missing sentinel rather than ``nan``.
def _compatibility_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": np.array([1, 2], dtype=np.int64),
            "f": np.array([1.5, np.nan]),
            "b": np.array([True, False]),
            "s": ["a", "b"],
            "dt": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "td": pd.to_timedelta(["1 days", "2 days"]),
            "nullable": pd.array([1, None], dtype="Int64"),
            "cat": pd.Categorical(["x", "y"]),
        }
    )


#: The normalized payload for :func:`_compatibility_frame` BEFORE the extraction
#: changed. Hard-coded rather than recomputed, so this asserts "unchanged" against
#: a recorded fact instead of against whatever the code currently does.
_COMPATIBILITY_COLUMNS: dict[str, Any] = {
    "i": [1, 2],
    "f": [1.5, None],
    "b": [True, False],
    "s": ["a", "b"],
    "dt": ["2020-01-01T00:00:00", "2020-01-02T00:00:00"],
    "td": ["P1DT0H0M0S", "P2DT0H0M0S"],
    "nullable": [1, None],
    "cat": ["x", "y"],
}


def test_ordinary_columns_normalize_identically_in_a_nested_frame() -> None:
    """``pd.NA`` reaches the traversal now where ``None`` used to; both spell null."""
    payload = _ready(_compatibility_frame())

    assert payload["columns"] == list(_COMPATIBILITY_COLUMNS)
    assert payload["index"] == [0, 1]
    assert payload["data"] == _COMPATIBILITY_COLUMNS


def test_ordinary_columns_normalize_identically_in_the_forecast_table() -> None:
    records = ForecastResult(_compatibility_frame()).to_dict()["forecasts"]

    assert records == [
        {name: values[row] for name, values in _COMPATIBILITY_COLUMNS.items()}
        for row in (0, 1)
    ]


def test_the_compatibility_payloads_still_encode_strictly() -> None:
    result = ForecastResult(
        _compatibility_frame(), metadata={"probe": _compatibility_frame()}
    )

    assert json.loads(_strict(result.to_dict()))["metadata"]["probe"]["data"] == (
        _COMPATIBILITY_COLUMNS
    )
