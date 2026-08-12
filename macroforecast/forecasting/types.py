"""``ForecastResult``, and the JSON boundary its export methods stand on.

``to_dict`` is the package's public "give me this run as plain data" call, and
``to_json`` is the same payload written down. F-063 made that a BOUNDARY rather
than a best-effort coercion: what comes back is accepted by
``json.dumps(..., allow_nan=False)``, or the call raises and names the value.

What follows is the set of choices that boundary implies. Each is deliberate
rather than a side effect, and each is stated so the next reader can tell an
intended refusal from a bug.

*It is strict on the way out.* Non-finite numeric leaves become ``null`` and
``to_json`` encodes with ``allow_nan=False``, because the bare ``NaN`` /
``Infinity`` tokens ``json.dumps`` emits by default are ECMAScript literals that
every RFC 8259 parser rejects. This mirrors the checkpoint sidecar writer
(F-062); unlike that writer there is no lenient reader to keep in step, because
nothing in this package reads a ``to_json`` file back.

*It FAILS rather than guesses.* A value with no JSON form raises ``TypeError``
-- a finite real outside ``float64``'s range, and a set whose converted members
have no single order, count as having none -- while a reference cycle and a
normalized key collision raise ``ValueError``. The
collision case is the reason this is not merely defensive: ``{1: "int",
"1": "str"}`` and a DataFrame with column labels ``[1, "1"]`` both used to encode
CLEANLY, having silently dropped one of the two values on the way. Emitting a
payload that is not the object the caller passed is the failure this boundary
exists to end, so it is refused instead. Every message carries a ``$``-rooted
path, because the value that needs fixing is usually nested several levels into
``metadata``.

*It does not disturb what already worked.* Only values that previously CRASHED
or were silently wrong have new spellings. In particular the JSON object key
rule is unchanged -- a mapping key and a DataFrame column label are still
``str(label)``, so a ``pd.Timestamp`` column keeps its ``str`` spelling and does
not switch to ISO. Consistency is enforced on TRAVERSAL (every side of a Series
or DataFrame is now visited, ``Series.name`` included, where it used to be
copied out raw) rather than by re-spelling working output.

Traversal changes an already-encodable spelling in exactly TWO places. A
``datetime64`` or ``timedelta64`` ARRAY had to change: ``.tolist()`` unboxed
those per element with the same unit-dependent rule that broke the scalars, so
an array and its own elements disagreed, and they are walked element-wise now.
The other is a TUPLE-valued ``Series.name``. It used to be copied out raw and is
recursively sanitized like every other leaf now, so ``to_dict`` hands back a
``list`` where it once handed back a ``tuple``. ``json`` writes both as the same
array, so no ``to_json`` text and no artifact on disk moves; a caller who indexes
or type-checks that field in the ``to_dict`` payload does see a different Python
type.

*A real with no float64 image is REFUSED, not rounded away.* This package
represents reals as native Python floats -- IEEE 754 binary64 -- because that is
what ``json`` emits and what consumers of these artifacts parse back. RFC 8259's
own number grammar is wider than that and fixes no precision, so the constraint
here is this package's representation and interoperability policy rather than the
format's syntax. A wider numpy float narrows through ``float()``, and a FINITE
``np.longdouble`` outside binary64's range loses its VALUE and not merely its
precision: above the range it narrows to ``inf``, which the non-finite rule would
then write as ``null``, and below the smallest subnormal it narrows to ``0.0``.
Both spellings already mean something else -- a missing value and a true zero --
so a real measurement would reach the reader indistinguishable from one of them.
Both cases raise. Genuinely non-finite reals still become ``null``, a true zero
is still ``0.0``, and a finite ``longdouble`` with a finite nonzero image still
narrows to a number, losing only the precision binary64 has no room for.

*pandas is not allowed to narrow a column first.* ``DataFrame.to_dict`` builds
its own Python objects BEFORE this traversal ever runs, and for a ``float128``
column that conversion applies ``float()`` itself -- so a finite wide cell
arrived here already spelled ``inf`` or ``0.0``, and the refusal above could not
fire because the evidence that it had been finite was gone. Frames are extracted
column-wise through ``Series.to_list`` instead, which preserves the dtype all the
way to the leaf.

*A set keeps the list it was already writing.* ``output.write_artifacts`` runs
this payload through ``output.core._json_ready`` on the way to the file, and that
second pass turned a ``set`` into ``sorted(...)`` -- so a set in ``metadata``
wrote a JSON array, and refusing it at this boundary would break a caller who
never had a problem. ``set`` and ``frozenset`` are normalized to that same sorted
list here, members converted BEFORE sorting so the order follows the JSON
spelling. ``frozenset`` is newly supported rather than restored: that second pass
tested ``isinstance(value, set)``, which a ``frozenset`` is not, so it used to
reach ``json.dumps`` unconverted and raise. When the converted members have no
single order there is no deterministic list to write, and that is refused rather
than made to depend on the interpreter's hash seed.

*A NESTED ForecastResult keeps this traversal's state.* One used to reach the
generic ``to_dict`` protocol, which calls ``value.to_dict()`` and so began a
FRESH walk with the path back at ``$`` and the cycle set empty. A result
reachable from its own ``metadata`` then recursed until the stack gave out
instead of raising the documented ``ValueError``, and a failure inside a nested
result reported a path rooted at that result rather than at the caller's.
:func:`_forecast_result_payload` is the shared body both entry points run, with
the path and the cycle set threaded through it.

A collision is also checked on the forecast table ITSELF, before its rows are
built, because a duplicate column label collapses however they are assembled --
pandas kept the last and dropped the rest behind a warning, and a row dict keyed
by label would drop one just as quietly -- and a payload missing a column of
forecasts is exactly what this boundary refuses.

This helper is deliberately LOCAL and narrow rather than a reuse of
``output.core._json_ready``: ``output`` imports ``forecasting``, so importing it
back here would invert that dependency.
"""
from __future__ import annotations

from collections.abc import Mapping
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    """Forecast runner output."""

    forecasts: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    sidecars: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """Return a copy of the forecast table."""

        return self.forecasts.copy()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready forecast result.

        The result is accepted by ``json.dumps(..., allow_nan=False)`` for every
        supported value, or this raises: ``TypeError`` for a value with no JSON
        form -- a finite real outside ``float64``'s range, and a set whose members
        have no single order, included -- and ``ValueError`` for a reference cycle
        or a key collision. Each message carries the ``$``-rooted path of the
        offending value, so a leaf buried in ``metadata`` can be found without
        bisecting the payload.

        The body is :func:`_forecast_result_payload`, which is also what a
        ForecastResult NESTED inside another one goes through. Entering there
        rather than re-entering this method is what keeps the outer walk's path
        and cycle set intact.
        """

        return _forecast_result_payload(
            self, _JSON_ROOT, _cycle_free(self, _JSON_ROOT, frozenset())
        )

    def evaluate(self, **kwargs: Any) -> pd.DataFrame:
        """Evaluate this forecast result with ``macroforecast.metrics``."""

        from macroforecast.metrics import evaluate_forecasts

        return evaluate_forecasts(self, **kwargs)

    def anatomy_explain(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Explain a precomputed ``anatomy.Anatomy`` object for this run."""

        from macroforecast.interpretation import anatomy_explain

        table = anatomy_explain(anatomy, **kwargs)
        table.attrs["macroforecast_forecast_result"] = self.metadata
        return table

    def anatomy_pbsv(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute backend PBSV rows for a precomputed ``anatomy.Anatomy`` object."""

        return self.pbsv(anatomy, **kwargs)

    def pbsv(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute PBSV rows for a precomputed forecast-Shapley backend object."""

        from macroforecast.interpretation import pbsv

        table = pbsv(anatomy, **kwargs)
        table.attrs["macroforecast_forecast_result"] = self.metadata
        return table

    def anatomy_oshapley_vi(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute backend oShapley-VI rows for a precomputed anatomy object."""

        return self.oshapley_vi(anatomy, **kwargs)

    def oshapley_vi(self, anatomy: Any, **kwargs: Any) -> pd.DataFrame:
        """Compute oShapley-VI rows for a precomputed forecast-Shapley object."""

        from macroforecast.interpretation import oshapley_vi

        table = oshapley_vi(anatomy, **kwargs)
        table.attrs["macroforecast_forecast_result"] = self.metadata
        return table

    def with_sidecar(self, name: str, value: Any) -> "ForecastResult":
        """Return a copy with a named runtime sidecar attached.

        The sidecar's ``metadata_schema`` is converted HERE rather than at export
        time, so a schema with no JSON form raises from this call, where the
        offending sidecar is still in the caller's hand. It used to be stored
        verbatim and to surface only from a later ``to_json``, by which point the
        attachment site was several calls away. Nothing is mutated on the way
        out: this raises before a new ``ForecastResult`` is built, so ``self`` is
        left exactly as it was.
        """

        key = str(name)
        if not key:
            raise ValueError("sidecar name must not be empty")
        entry = _sidecar_metadata(value, key)
        sidecars = dict(self.sidecars)
        sidecars[key] = value
        metadata = dict(self.metadata)
        registry = dict(metadata.get("sidecars", {}))
        registry[key] = entry
        metadata["sidecars"] = registry
        return ForecastResult(self.forecasts.copy(), metadata=metadata, sidecars=sidecars)

    def get_sidecar(self, name: str, default: Any = None) -> Any:
        """Return a named sidecar, or ``default`` when it is absent."""

        return self.sidecars.get(str(name), default)

    def sidecar_names(self) -> tuple[str, ...]:
        """Return attached sidecar names."""

        return tuple(self.sidecars)

    def with_anatomy(
        self,
        X: Any,
        y: Any,
        models: Any,
        *,
        window: Any,
        sidecar_name: str = "anatomy",
        **kwargs: Any,
    ) -> "ForecastResult":
        """Build and attach a forecast-accuracy anatomy sidecar.

        ``window`` is required because a completed forecast table does not
        contain the feature matrix, target vector, and origin-wise refit design
        needed by the anatomy backend.
        """

        from macroforecast.interpretation import anatomy_from_forecast_result

        return anatomy_from_forecast_result(
            self,
            X,
            y,
            models,
            window=window,
            attach=True,
            sidecar_name=sidecar_name,
            **kwargs,
        )

    def with_oshapley(
        self,
        X: Any,
        y: Any,
        models: Any,
        *,
        window: Any,
        sidecar_name: str = "oshapley",
        **kwargs: Any,
    ) -> "ForecastResult":
        """Build and attach an oShapley/PBSV forecast-accuracy sidecar."""

        from macroforecast.interpretation import oshapley_from_forecast_result

        return oshapley_from_forecast_result(
            self,
            X,
            y,
            models,
            window=window,
            attach=True,
            sidecar_name=sidecar_name,
            **kwargs,
        )

    def with_dual(
        self,
        model: Any | None,
        X_train: Any,
        y_train: Any,
        X_test: Any | None = None,
        *,
        sidecar_name: str = "dual",
        **kwargs: Any,
    ) -> "ForecastResult":
        """Build and attach a dual interpretation sidecar."""

        from macroforecast.interpretation import dual_from_forecast_result

        return dual_from_forecast_result(
            self,
            model,
            X_train,
            y_train,
            X_test,
            attach=True,
            sidecar_name=sidecar_name,
            **kwargs,
        )

    def to_json(self, path: str | Path | None = None, *, indent: int | None = 2) -> str:
        """Return JSON text, and optionally write it to ``path``.

        ``allow_nan=False`` makes this RFC 8259 rather than ECMAScript;
        :func:`_json_ready` has already turned every non-finite numeric leaf into
        ``null``, which is what lets the switch be on at all.

        Serialization completes BEFORE ``path`` is opened, and that order is
        load-bearing: a payload the encoder refuses raises here, so an existing
        file at ``path`` is left byte-for-byte as it was and a missing one is
        never created. A caller re-exporting over last run's artifact therefore
        cannot lose it to a bad ``metadata`` value.
        """

        text = json.dumps(
            self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False
        )
        if path is not None:
            Path(path).write_text(text + "\n", encoding="utf-8")
        return text


#: Root of every path reported by this module's errors, JSONPath-style.
_JSON_ROOT = "$"

#: Seconds per ``np.timedelta64`` unit, as an exact ``(numerator, denominator)``
#: pair so sub-second units stay lossless. ``Y`` and ``M`` are absent on purpose:
#: they are not fixed durations and get their own ISO designators.
_DURATION_UNIT_SECONDS: dict[str, tuple[int, int]] = {
    "W": (604800, 1),
    "D": (86400, 1),
    "h": (3600, 1),
    "m": (60, 1),
    "s": (1, 1),
    "ms": (1, 10**3),
    "us": (1, 10**6),
    "ns": (1, 10**9),
    "ps": (1, 10**12),
    "fs": (1, 10**15),
    "as": (1, 10**18),
}

#: ``dtype.kind`` for ``datetime64`` and ``timedelta64``. Membership decides
#: which arrays are walked element-wise instead of via ``.tolist()``; every other
#: kind -- object, structured, masked-numeric -- keeps the spelling it had.
_TEMPORAL_DTYPE_KINDS = frozenset("Mm")

#: Returned by :func:`_iso_duration_from_timedelta64` for a unitless
#: ``timedelta64``, which is a bare integer count and keeps rendering as one.
_NOT_A_DURATION = object()


def _child_path(path: str, key: str) -> str:
    """Extend a path by one mapping key, quoting it when it is not an identifier."""
    return f"{path}.{key}" if key.isidentifier() else f"{path}[{key!r}]"


def _item_path(path: str, index: int) -> str:
    """Extend a path by one sequence position."""
    return f"{path}[{index}]"


def _member_path(path: str) -> str:
    """Extend a path by one SET member, which has no stable position.

    JSONPath's ``[*]`` wildcard rather than an index, because ``[0]`` would be a
    lie: iterate the same set again and a different member comes first.
    """
    return f"{path}[*]"


def _unsupported(path: str, value: Any) -> TypeError:
    """The error for a leaf JSON cannot express, naming WHERE it sits."""
    return TypeError(
        f"{path} holds a {type(value).__name__}, which RFC 8259 JSON cannot "
        "express, so no forecast-result payload was produced. Convert it to a "
        "string, a number, or a mapping before attaching it."
    )


def _out_of_float64_range(path: str, value: Any) -> TypeError:
    """The error for a FINITE real with no finite nonzero ``float64`` image.

    This package represents reals as native Python floats -- IEEE 754 binary64 --
    because that is what ``json`` emits and what consumers parse back. RFC 8259's
    number grammar is itself wider and fixes no precision, so the limit is this
    package's representation policy, not the format's syntax. A wider numpy float
    has to pass through ``float()``, and OUTSIDE binary64's range that call costs
    the VALUE rather than the precision. Above the range it returns an infinity,
    which the traversal's non-finite rule would write as ``null`` -- what a
    genuine NaN writes. Below the smallest subnormal it returns a zero, written as
    ``0.0`` -- what a true zero writes. Either way the reader cannot tell the
    measurement from the thing it collided with, so refusing is the only spelling
    that does not lie about which of them it was.
    """
    return TypeError(
        f"{path} holds a finite {type(value).__name__} ({value!r}) with no finite "
        "nonzero float64 image: narrowing it reaches infinity or zero. This "
        "package encodes reals as native Python floats (IEEE 754 binary64), so "
        "encoding this value would write it out as null or as 0.0 -- the "
        "spellings a missing value and a true zero already have. No "
        "forecast-result payload was produced. Narrow or stringify it yourself "
        "before attaching it."
    )


def _cycle_free(container: Any, path: str, active: frozenset[int]) -> frozenset[int]:
    """Extend the current path, refusing a container that already sits on it.

    ``active`` holds the ids on the CURRENT PATH only, so the same object
    appearing twice as a SIBLING is encoded twice, which is not a cycle.
    """
    if id(container) in active:
        raise ValueError(
            f"{path} closes a reference cycle through a "
            f"{type(container).__name__}; RFC 8259 JSON has no way to express a "
            "back-reference, so no forecast-result payload was produced."
        )
    return active | {id(container)}


def _iso_duration(seconds: int, per_second: int) -> str:
    """An exact duration as ISO 8601, in the spelling ``pd.Timedelta`` uses.

    ``seconds / per_second`` is the exact duration; ``per_second`` is a power of
    ten, so the fractional part is rendered without floating-point rounding.

    The decomposition FLOORS, which is what makes a negative duration read
    ``P-2DT23H58M30S`` rather than a mix of signs, matching
    ``pd.Timedelta.isoformat`` exactly. That agreement is the point: the same
    duration arriving as ``np.timedelta64``, ``pd.Timedelta`` or
    ``datetime.timedelta`` has to produce one string, or the boundary is lying
    about which of those it was handed.
    """
    whole, fraction = divmod(seconds, per_second)
    days, rest = divmod(whole, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, second = divmod(rest, 60)
    text = str(second)
    if fraction:
        digits = len(str(per_second)) - 1
        text = f"{second}.{fraction:0{digits}d}".rstrip("0")
    return f"P{days}DT{hours}H{minutes}M{text}S"


def _iso_duration_from_timedelta64(value: np.timedelta64) -> Any:
    """One ``np.timedelta64`` as an ISO duration, whatever its unit.

    Before F-063 this went through ``.item()``, which returns a bare ``int`` for
    the sub-microsecond units and a ``datetime.timedelta`` for the rest -- so a
    nanosecond duration encoded as a UNIT-DEPENDENT integer and a day duration
    raised in ``json.dumps``. Both are replaced here.

    ``Y`` and ``M`` are not fixed durations, so they keep their own ISO
    designators instead of being expanded into days. A unitless ``timedelta64``
    is not a duration at all -- it is an integer with no scale -- and is reported
    as :data:`_NOT_A_DURATION` so it keeps rendering as the integer it always did.
    """
    if np.isnat(value):
        return None
    unit, step = np.datetime_data(value.dtype)
    count = int(value.astype("int64")) * step
    if unit in ("Y", "M"):
        return f"P{count}{unit}"
    scale = _DURATION_UNIT_SECONDS.get(unit)
    if scale is None:
        return _NOT_A_DURATION
    numerator, denominator = scale
    return _iso_duration(count * numerator, denominator)


def _numpy_scalar(value: np.generic, path: str) -> Any:
    """Narrow one numpy scalar to a native Python value, in ONE step.

    ``.item()`` is not trustworthy on its own: ``np.longdouble.item()`` returns
    another ``np.longdouble`` wherever ``longdouble`` is wider than ``float64``,
    so it is a FIXED POINT, and following it is what made the pre-F-063 helper
    die with ``RecursionError`` on a longdouble leaf -- a FINITE one included.
    One unboxing is therefore followed by an explicit narrowing.

    Complex scalars are refused up front rather than unboxed. ``np.clongdouble``
    is the same fixed point as ``np.longdouble``, and JSON has no complex number
    to narrow either of them to, so guessing at a spelling here would invent a
    representation nothing agrees on.

    The narrowing itself can lose the VALUE and not just precision, at EITHER end
    of the range, and both of those cases are refused. See
    :func:`_out_of_float64_range`.
    """
    if isinstance(value, np.complexfloating):
        raise _unsupported(path, value)
    unboxed = value.item()
    if not isinstance(unboxed, np.generic):
        return unboxed
    if isinstance(unboxed, np.floating):
        # The ``longdouble`` case. ``float()`` is lossy wherever ``longdouble`` is
        # wider, and binary64 is the representation this package hands back, so the
        # precision goes either way. The VALUE is a different matter, and it goes
        # at BOTH ends of the range: above it ``float()`` returns an infinity, and
        # the non-finite rule further down would spell a finite measurement
        # ``null``; below the smallest subnormal it returns a zero, which nothing
        # further down rejects at all and which reads as a measured zero.
        #
        # ``narrowed == 0.0`` is true of ``-0.0`` as well, so the underflow is
        # caught with its sign rather than only in the positive direction.
        narrowed = float(unboxed)
        if bool(np.isfinite(unboxed)) and (
            not np.isfinite(narrowed) or (narrowed == 0.0 and bool(unboxed != 0))
        ):
            raise _out_of_float64_range(path, value)
        return narrowed
    if isinstance(unboxed, np.integer):
        return int(unboxed)
    raise _unsupported(path, value)


def _json_object(
    items: Any,
    path: str,
    active: frozenset[int],
) -> dict[str, Any]:
    """Build one JSON object, REFUSING any collision the key rewrite creates.

    A plain dict comprehension silently drops a value here: ``{1: "int",
    "1": "str"}`` collapses to ``{"1": "str"}``, and the caller gets a payload
    that encodes perfectly and is not the object they passed. The key rule itself
    is unchanged from before F-063 -- still ``str(label)``, so working spellings
    are untouched -- and only the collapse is refused.
    """
    out: dict[str, Any] = {}
    for key, item in items:
        name = str(key)
        if name in out:
            raise ValueError(
                f"{path} has two keys that both normalize to {name!r}, so "
                "encoding it would silently drop one of their values. No "
                "forecast-result payload was produced."
            )
        out[name] = _json_ready(item, _path=_child_path(path, name), _active=active)
    return out


def _json_columns(frame: pd.DataFrame, path: str) -> list[str]:
    """The frame's column labels as JSON keys, refusing a normalized collision.

    Checked BEFORE any per-column extraction runs, because a duplicate label
    collapses whichever way the cells are gathered: pandas' own ``to_dict``
    dropped all but one column, with a ``UserWarning`` that says so and a payload
    that does not, and a dict keyed by label drops one just as quietly.
    """
    names: list[str] = []
    seen: set[str] = set()
    for column in frame.columns:
        name = str(column)
        if name in seen:
            raise ValueError(
                f"{path}.columns has two labels that both normalize to "
                f"{name!r}, so encoding this frame would silently drop a column "
                "of data. No forecast-result payload was produced."
            )
        seen.add(name)
        names.append(name)
    return names


def _frame_column_values(frame: pd.DataFrame) -> list[list[Any]]:
    """Every column's cells, gathered WITHOUT letting pandas narrow a dtype.

    ``DataFrame.to_dict`` is what this replaces, and it is not a neutral copy. It
    materializes Python objects itself, and for a ``float128`` column that means
    ``float()`` -- so a finite ``np.longdouble`` reached the traversal already
    spelled ``inf`` or ``0.0``, and :func:`_out_of_float64_range` never got to see
    that it had been finite. That is why a ``Series`` in ``metadata`` was already
    refused and the same value inside a FRAME was silently written out: a series
    goes through ``Series.to_list``, which preserves the dtype, and a frame did
    not. Taking each column as a series restores that.

    Columns are taken POSITIONALLY. Label-based access returns a frame rather than
    a series for a duplicated label, and the labels that could collide are already
    refused by :func:`_json_columns`.
    """
    return [frame.iloc[:, position].to_list() for position in range(frame.shape[1])]


def _frame_records(frame: pd.DataFrame) -> list[dict[Any, Any]]:
    """One dict per row, assembled from dtype-preserving columns.

    Keyed by the ORIGINAL column labels rather than by their JSON spellings, so
    :func:`_json_object` still sees both members of a normalized collision and can
    refuse it with a path instead of quietly keeping the last one.
    """
    columns = _frame_column_values(frame)
    return [
        {label: column[row] for label, column in zip(frame.columns, columns)}
        for row in range(frame.shape[0])
    ]


def _json_set(
    value: set[Any] | frozenset[Any],
    path: str,
    active: frozenset[int],
) -> list[Any]:
    """One ``set``/``frozenset`` as a SORTED list, refusing an unorderable one.

    JSON has no set and a set has no order, so an encoder has to choose one.
    ``output.core._json_ready`` -- the second normalization pass every
    ``write_artifacts`` payload already ran through -- chose ``sorted`` on the
    CONVERTED members, and that choice is reproduced here so a ``set`` in
    ``metadata`` keeps writing the artifact it was writing before this boundary
    turned strict. It cannot be imported to get it: ``output`` imports
    ``forecasting``.

    ``frozenset`` is newly supported rather than restored. That second pass tested
    ``isinstance(value, set)``, which a ``frozenset`` is not, so one reached
    ``json.dumps`` unconverted and raised there.

    Members are converted BEFORE they are sorted, and the order follows from
    that: a set of ``pd.Timestamp`` sorts as ISO strings. When the converted
    members have no single order -- ``{1, "a"}`` is the usual way -- there is no
    deterministic list to emit, and choosing one anyway would make the artifact
    depend on this interpreter's hash seed. That is refused, with the path.
    """
    members = [
        _json_ready(item, _path=_member_path(path), _active=active) for item in value
    ]
    try:
        return sorted(members)
    except TypeError as error:
        raise TypeError(
            f"{path} holds a {type(value).__name__} whose members have no single "
            "order once converted, so the JSON array it has to become has no "
            "deterministic spelling and would move with this interpreter's hash "
            "seed. No forecast-result payload was produced. Convert it to a list "
            "yourself, in the order you want."
        ) from error


def _json_temporal_array(
    value: np.ndarray,
    path: str,
    active: frozenset[int],
) -> Any:
    """One ``datetime64``/``timedelta64`` array, deferred to the scalar branch.

    Every other array is normalized by handing ``.tolist()`` back to the
    traversal, and for these two dtypes that is the WRONG detour: ``.tolist()``
    unboxes per element with the same unit-dependent rule that broke the scalar
    case before F-063. At ``us`` it drops the sub-second digits; at ``ns`` it
    gives up on dates entirely and yields the raw epoch integer. So
    ``np.array(["2000-01-31"], dtype="datetime64[ns]")`` encoded as
    ``[949276800000000000]`` while the identical SCALAR encoded as ISO.

    Walking the array instead hands each element to :func:`_json_ready` as the
    numpy scalar it is, which is the only way the two spellings can be one
    spelling. A masked entry arrives as ``np.ma.masked``, whose ``.tolist()`` is
    ``None``, so a mask still reads as JSON ``null`` -- masked temporal arrays had
    the epoch-integer defect on their VISIBLE entries too, and get the fix.

    A zero-dimensional array holds a scalar, not a sequence, and must not become
    a one-element list: ``value[()]`` unwraps it in place.
    """
    if value.ndim == 0:
        return _json_ready(value[()], _path=path, _active=active)
    return [
        _json_ready(item, _path=_item_path(path, position), _active=active)
        for position, item in enumerate(value)
    ]


def _json_ready(
    value: Any,
    *,
    _path: str = _JSON_ROOT,
    _active: frozenset[int] = frozenset(),
) -> Any:
    """Normalize one value for strict (``allow_nan=False``) JSON.

    Branch order is load-bearing and is kept as close to the pre-F-063 order as
    the new cases allow, so that an object matching several branches still takes
    the one it always took -- ``to_dict`` before ``Mapping`` most of all, since a
    mapping subclass that defines ``to_dict`` used to be exported through it.
    """
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    # Before the datetime branch below, deliberately: ``pd.NaT`` IS an instance
    # of ``datetime.datetime``, and ``NaT.isoformat()`` is the string "NaT".
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, np.datetime64):
        # ``str`` on a ``datetime64`` is already ISO 8601 and keeps the dtype's
        # unit, where ``.item()`` returned an epoch integer at nanoseconds and a
        # ``datetime.date`` at days -- one wrong, one unencodable.
        return None if np.isnat(value) else str(value)
    if isinstance(value, np.timedelta64):
        duration = _iso_duration_from_timedelta64(value)
        if duration is not _NOT_A_DURATION:
            return duration
        return _json_ready(int(value.astype("int64")), _path=_path, _active=_active)
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        microseconds = (
            (value.days * 86400 + value.seconds) * 10**6 + value.microseconds
        )
        return _iso_duration(microseconds, 10**6)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        active = _cycle_free(value, _path, _active)
        if value.dtype.kind in _TEMPORAL_DTYPE_KINDS:
            return _json_temporal_array(value, _path, active)
        return _json_ready(value.tolist(), _path=_path, _active=active)
    if isinstance(value, pd.Series):
        active = _cycle_free(value, _path, _active)
        return {
            "name": _json_ready(
                value.name, _path=_child_path(_path, "name"), _active=active
            ),
            "index": [
                _json_ready(
                    item,
                    _path=_item_path(_child_path(_path, "index"), position),
                    _active=active,
                )
                for position, item in enumerate(value.index)
            ],
            "data": [
                _json_ready(
                    item,
                    _path=_item_path(_child_path(_path, "data"), position),
                    _active=active,
                )
                for position, item in enumerate(value.to_list())
            ],
        }
    if isinstance(value, pd.DataFrame):
        active = _cycle_free(value, _path, _active)
        return {
            "columns": _json_columns(value, _path),
            "index": [
                _json_ready(
                    item,
                    _path=_item_path(_child_path(_path, "index"), position),
                    _active=active,
                )
                for position, item in enumerate(value.index)
            ],
            "data": _json_object(
                zip(value.columns, _frame_column_values(value)),
                _child_path(_path, "data"),
                active,
            ),
        }
    if isinstance(value, ForecastResult):
        # BEFORE the generic ``to_dict`` protocol below, deliberately. That branch
        # calls ``value.to_dict()``, which starts a FRESH walk -- path back to
        # ``$``, cycle set empty -- so a self-referencing result recursed past the
        # cycle check entirely and a nested failure lost the outer path.
        return _forecast_result_payload(
            value, _path, _cycle_free(value, _path, _active)
        )
    if hasattr(value, "to_dict") and not isinstance(value, type):
        try:
            converted = value.to_dict()
        except TypeError:
            # A ``to_dict`` that needs arguments is not the protocol we mean.
            # The ``try`` covers this call ONLY: it used to wrap the recursion
            # too, so a genuine failure deep inside a sidecar was swallowed here
            # and re-raised later without its path.
            pass
        else:
            return _json_ready(
                converted, _path=_path, _active=_cycle_free(value, _path, _active)
            )
    if isinstance(value, np.generic):
        return _json_ready(
            _numpy_scalar(value, _path), _path=_path, _active=_active
        )
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, complex):
        raise _unsupported(_path, value)
    if isinstance(value, Mapping):
        return _json_object(
            value.items(), _path, _cycle_free(value, _path, _active)
        )
    if isinstance(value, (set, frozenset)):
        return _json_set(value, _path, _cycle_free(value, _path, _active))
    if isinstance(value, (tuple, list)):
        active = _cycle_free(value, _path, _active)
        return [
            _json_ready(item, _path=_item_path(_path, position), _active=active)
            for position, item in enumerate(value)
        ]
    if isinstance(value, (bool, int, float, str)):
        # ``bool`` before ``int`` is why this is one check: both encode as
        # themselves, and ``json`` spells each correctly on its own.
        return value
    raise _unsupported(_path, value)


def _forecast_result_payload(
    result: ForecastResult,
    path: str,
    active: frozenset[int],
) -> dict[str, Any]:
    """One ``ForecastResult`` as JSON-ready data, at ``path`` and under ``active``.

    Shared by :meth:`ForecastResult.to_dict`, which enters at ``$``, and by the
    ForecastResult branch of :func:`_json_ready`, which enters at wherever the
    nested result sits. That sharing is the point. The alternative -- letting a
    nested result go through the generic ``to_dict`` protocol -- re-entered the
    public method, which begins a fresh walk: the path reset to ``$`` and the
    cycle set emptied. A result reachable from its own ``metadata`` therefore
    never met the cycle check and recursed until the stack gave out, and a bad
    leaf inside a nested result reported a path rooted at that result. For an
    unencodable leaf it was worse than a wrong path: the inner ``TypeError`` was
    swallowed by that branch's own ``except TypeError`` and re-raised as "a
    ForecastResult cannot be expressed", blaming the container for the value.

    ``active`` must already contain this result. Both callers put it there with
    :func:`_cycle_free`, which is also what refuses the cycle when it is already
    on the path.
    """
    forecasts_path = _child_path(path, "forecasts")
    # For its refusal, not its return value, and BEFORE the rows are built. A
    # duplicate label collapses however the rows are assembled -- pandas kept the
    # last and dropped the rest behind a ``UserWarning``, and a row dict keyed by
    # label does the same silently -- so a table with columns ``["x", "x"]``
    # exported ``[{"x": 2}]`` and lost a column of forecasts.
    _json_columns(result.forecasts, forecasts_path)
    return {
        "forecasts": _json_ready(
            _frame_records(result.forecasts),
            _path=forecasts_path,
            _active=active,
        ),
        "metadata": _json_ready(
            result.metadata, _path=_child_path(path, "metadata"), _active=active
        ),
        "sidecars": _json_ready(
            result.sidecars, _path=_child_path(path, "sidecars"), _active=active
        ),
    }


def _sidecar_metadata(value: Any, name: str) -> dict[str, Any]:
    """The registry entry for one attached sidecar.

    ``metadata_schema`` is normalized HERE, at attachment time, so a schema with
    no JSON form fails in :meth:`ForecastResult.with_sidecar` rather than in some
    later ``to_json``. ``name`` exists only to put that sidecar in the error's
    path; nothing else in the entry depends on it.
    """
    schema = getattr(value, "metadata_schema", None)
    if callable(schema):
        schema = schema()
    metadata = getattr(value, "metadata", None)
    schema_path = _child_path(
        _child_path(
            _child_path(_child_path(_JSON_ROOT, "metadata"), "sidecars"), name
        ),
        "metadata_schema",
    )
    return {
        "object_type": f"{type(value).__module__}.{type(value).__name__}",
        "metadata_schema": _json_ready(schema, _path=schema_path),
        "metadata_keys": sorted(str(key) for key in metadata)
        if isinstance(metadata, Mapping)
        else [],
    }


__all__ = ["ForecastResult"]
