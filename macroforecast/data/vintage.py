from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from .errors import VintageUnavailableError
from .loaders import list_vintages, load_fred_md, load_fred_qd
from .panel import DataBundle, as_panel, custom_dataset, validate_panel


@runtime_checkable
class VintageSource(Protocol):
    """A lazily-resolved source of point-in-time data, one bundle per real-time origin."""

    def resolve(self, origin_date: pd.Timestamp) -> DataBundle:
        """Return the DataBundle observable as of ``origin_date``.

        The returned bundle must contain only data that would have been
        publicly available at ``origin_date``; publication-lag handling belongs
        to the source. A source must raise ``VintageUnavailableError`` if no
        vintage exists at or before ``origin_date`` rather than returning an
        empty bundle. ``bundle.metadata["vintage"]`` must be a stable,
        JSON-serialisable identifier for the resolved content and must change
        if and only if that content changes.
        """

    def available_vintages(self) -> Sequence[Any]:
        """Return sorted canonical vintage identifiers this source can resolve."""


@dataclass(frozen=True)
class VintagePanelSpec:
    """Run-level wrapper for a point-in-time vintage source."""

    source: VintageSource
    reference_calendar: pd.DatetimeIndex
    actuals_vintage: Literal["latest", "first_release"] = "latest"
    first_release_max_vintages: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.source, VintageSource):
            raise TypeError("source must satisfy the VintageSource protocol")
        if not isinstance(self.reference_calendar, pd.DatetimeIndex):
            raise TypeError("reference_calendar must be a pandas DatetimeIndex")
        if self.reference_calendar.empty:
            raise ValueError("reference_calendar must not be empty")
        if self.reference_calendar.hasnans:
            raise ValueError("reference_calendar must not contain NaT")
        if not self.reference_calendar.is_monotonic_increasing:
            raise ValueError("reference_calendar must be monotonic increasing")
        # Monotonic increasing permits repeats. A repeated origin is a repeated forecast
        # row scored twice against one actual, so the calendar has to be STRICTLY
        # increasing (F-011).
        if not self.reference_calendar.is_unique:
            duplicated = self.reference_calendar[self.reference_calendar.duplicated()].unique()
            sample = ", ".join(str(value) for value in duplicated[:3])
            raise ValueError(f"reference_calendar must not contain duplicate origins: {sample}")
        if self.actuals_vintage not in {"latest", "first_release"}:
            raise ValueError("actuals_vintage must be 'latest' or 'first_release'")
        # Store what the validator returns, not what the caller passed. Validating
        # ``np.int64(2)`` and leaving it on the field means a NumPy scalar survives into
        # run metadata and provenance, where it is not JSON-serialisable -- and the
        # runner no longer coerces it, precisely so a bad value cannot be laundered late.
        object.__setattr__(
            self,
            "first_release_max_vintages",
            _validate_probe_limit(self.first_release_max_vintages),
        )
        if self.actuals_vintage == "first_release":
            keys = tuple(self.source.available_vintages())
            if not keys:
                raise ValueError(
                    "actuals_vintage='first_release' requires "
                    "source.available_vintages() to return timestamp-parsable "
                    "vintage keys; callable custom_vintages sources without "
                    "explicit vintages cannot supply first-release actuals"
                )
            # First release walks forward from a bisect over these keys, so the order the
            # source reports IS the search order. An unsorted or duplicated calendar
            # silently returns the wrong release rather than failing, so it is rejected
            # here, before the runner does any work with it.
            timestamps = [_canonical_vintage_timestamp(key) for key in keys]
            for previous, current, previous_key, current_key in zip(
                timestamps, timestamps[1:], keys, keys[1:], strict=False
            ):
                if current == previous:
                    raise ValueError(
                        "actuals_vintage='first_release' requires distinct vintage "
                        f"instants; keys {previous_key!r} and {current_key!r} are "
                        f"duplicate instants, both denoting {current}"
                    )
                if current < previous:
                    raise ValueError(
                        "actuals_vintage='first_release' requires "
                        "source.available_vintages() in increasing order; "
                        f"{current_key!r} ({current}) follows {previous_key!r} "
                        f"({previous})"
                    )


@dataclass
class _FredVintageSource:
    dataset: Literal["fred_md", "fred_qd"]
    start: str | None = None
    end: str | None = None
    cache_root: str | Path | None = None
    local_zip_source: str | Path | None = None
    force: bool = False
    _labels: list[str] = field(init=False, repr=False)
    _label_dates: list[pd.Timestamp] = field(init=False, repr=False)
    _cache: dict[str, DataBundle] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._labels = list_vintages(self.dataset, start=self.start, end=self.end)
        self._label_dates = [_vintage_label_timestamp(label) for label in self._labels]

    @property
    def kind(self) -> str:
        return f"{self.dataset}_vintages"

    def available_vintages(self) -> Sequence[Any]:
        return tuple(self._labels)

    def resolve(self, origin_date: pd.Timestamp) -> DataBundle:
        origin = pd.Timestamp(origin_date)
        pos = bisect_right(self._label_dates, origin) - 1
        if pos < 0:
            raise VintageUnavailableError(
                f"no {self.dataset} vintage is available at or before {origin.date()}"
            )
        label = self._labels[pos]
        cached = self._cache.get(label)
        if cached is not None:
            return cached
        loader = load_fred_md if self.dataset == "fred_md" else load_fred_qd
        bundle = loader(
            vintage=label,
            force=self.force,
            cache_root=self.cache_root,
            local_zip_source=self.local_zip_source,
        )
        self._cache[label] = bundle
        return bundle


def fred_md_vintages(
    *,
    start: str | None = None,
    end: str | None = None,
    cache_root: str | Path | None = None,
    local_zip_source: str | Path | None = None,
    force: bool = False,
) -> VintageSource:
    """Return a FRED-MD point-in-time source resolved by forecast origin.

    Parameters bound the available monthly vintage labels and cache/download
    behavior. ``start`` and ``end`` use ``YYYY-MM`` labels. ``cache_root``
    controls where raw vintage CSVs are stored. ``local_zip_source`` points to
    an official historical-vintage ZIP for offline or deterministic runs.
    ``force=True`` refreshes cached vintage files.

    Returns
    -------
    VintageSource
        Source object with ``resolve(origin_date)`` and
        ``available_vintages()``. Resolving an origin returns the latest
        FRED-MD vintage available at or before that origin and raises
        ``VintageUnavailableError`` when no eligible vintage exists.

    Example
    -------
    >>> import pandas as pd
    >>> import macroforecast as mf
    >>> source = mf.data.fred_md_vintages(start="2020-01", end="2020-03")
    >>> labels = source.available_vintages()
    >>> isinstance(labels, tuple)
    True
    """

    return _FredVintageSource(
        "fred_md",
        start=start,
        end=end,
        cache_root=cache_root,
        local_zip_source=local_zip_source,
        force=force,
    )


def fred_qd_vintages(
    *,
    start: str | None = None,
    end: str | None = None,
    cache_root: str | Path | None = None,
    local_zip_source: str | Path | None = None,
    force: bool = False,
) -> VintageSource:
    """Return a FRED-QD point-in-time source resolved by origin date."""

    return _FredVintageSource(
        "fred_qd",
        start=start,
        end=end,
        cache_root=cache_root,
        local_zip_source=local_zip_source,
        force=force,
    )


@dataclass
class _ResolvedVintage:
    key: Any
    timestamp: pd.Timestamp
    bundle: DataBundle | pd.DataFrame | None = None


@dataclass
class _CustomVintageSource:
    source: Callable[[pd.Timestamp], DataBundle | pd.DataFrame] | None = None
    vintages: tuple[_ResolvedVintage, ...] = ()
    vintage_id: Callable[[Any], Any] | None = None
    dataset: str = "custom_vintages"
    frequency: str = "unknown"
    strict: bool = True
    _cache: dict[str, DataBundle] = field(default_factory=dict, init=False, repr=False)

    @property
    def kind(self) -> str:
        return "custom_vintages"

    def available_vintages(self) -> Sequence[Any]:
        if self.source is not None:
            return tuple(v.key for v in self.vintages)
        return tuple(v.key for v in self.vintages)

    def resolve(self, origin_date: pd.Timestamp) -> DataBundle:
        # The RAW origin is what user code sees. Canonicalising is for comparing against
        # a known calendar, and a callable source has none: it receives the origin and
        # its ``vintage_id`` derives the cache label from it, so rewriting a tz-aware
        # origin to UTC-naive on the way in would silently change both the argument the
        # caller's function is given and the identity its snapshot is stored under.
        raw_origin = pd.Timestamp(origin_date)
        if self.source is not None:
            key = (
                raw_origin
                if not self.vintages
                # With an explicit calendar the KEY comes from a bisect, which needs
                # canonical instants; the callable itself still gets the raw origin.
                else self._resolved_key(_canonical_vintage_timestamp(raw_origin))
            )
            vintage_label = self._vintage_label(key)
            cached = self._cache.get(vintage_label)
            if cached is not None:
                return cached
            bundle = self._coerce(self.source(raw_origin), vintage_label=vintage_label)
            self._cache[vintage_label] = bundle
            return bundle

        resolved = self._resolved(_canonical_vintage_timestamp(raw_origin))
        vintage_label = self._vintage_label(resolved.key)
        cached = self._cache.get(vintage_label)
        if cached is not None:
            return cached
        if resolved.bundle is None:  # pragma: no cover - constructor invariant
            raise VintageUnavailableError(
                f"no custom vintage is available at or before {raw_origin.date()}"
            )
        bundle = self._coerce(resolved.bundle, vintage_label=vintage_label)
        self._cache[vintage_label] = bundle
        return bundle

    def _resolved(self, origin: pd.Timestamp) -> _ResolvedVintage:
        if not self.vintages:
            raise VintageUnavailableError("custom vintage source reports no available vintages")
        timestamps = [v.timestamp for v in self.vintages]
        pos = bisect_right(timestamps, origin) - 1
        if pos < 0:
            raise VintageUnavailableError(
                f"no custom vintage is available at or before {origin.date()}"
            )
        return self.vintages[pos]

    def _resolved_key(self, origin: pd.Timestamp) -> Any:
        if not self.vintages:
            return origin
        return self._resolved(origin).key

    def _vintage_label(self, key: Any) -> str:
        callback = self.vintage_id or (lambda value: str(value))
        return str(callback(key))

    def _coerce(
        self,
        value: DataBundle | pd.DataFrame,
        *,
        vintage_label: str,
    ) -> DataBundle:
        if isinstance(value, DataBundle):
            metadata = {
                **dict(value.metadata),
                "dataset": value.metadata.get("dataset", self.dataset),
                "frequency": value.metadata.get("frequency", self.frequency),
                "vintage": vintage_label,
            }
            panel = as_panel(value.panel, metadata=metadata, strict=self.strict)
            return DataBundle(panel, metadata)
        if isinstance(value, pd.DataFrame):
            return custom_dataset(
                value,
                dataset=self.dataset,
                source_family="custom_vintages",
                frequency=self.frequency,
                metadata={"vintage": vintage_label},
                strict=self.strict,
            )
        raise TypeError(
            "custom_vintages sources must return DataBundle or pandas DataFrame"
        )


def custom_vintages(
    source: (
        Callable[[pd.Timestamp], DataBundle | pd.DataFrame]
        | Mapping[Any, DataBundle | pd.DataFrame]
        | pd.DataFrame
    ),
    *,
    vintage_column: str | None = None,
    date_column: str | None = None,
    vintage_id: Callable[[Any], Any] | None = None,
    dataset: str = "custom_vintages",
    frequency: str = "unknown",
    strict: bool = True,
) -> VintageSource:
    """Return a custom point-in-time source.

    ``source`` may be a callable ``origin_date -> DataBundle | DataFrame``, a
    mapping from timestamp-parsable vintage keys to snapshots, or a grouped-wide
    DataFrame. The grouped-wide form has one ``vintage_column``, one
    ``date_column``, and one numeric column per series; each vintage group is a
    complete wide snapshot. Every snapshot is normalized through
    :func:`as_panel` / :func:`custom_dataset` and then validated. Resolved
    snapshots are memoized by the stable identifier produced by ``vintage_id``
    (default: ``str(resolved_key)``). If a callable reads from a
    non-deterministic source whose content can change for the same identifier,
    run the forecast with runner/pipeline preprocessing caching disabled.

    For the mapping and grouped-wide forms the vintage calendar is known at
    construction, so both invariants it needs are checked there: every key must denote a
    distinct instant, and every key must produce a distinct ``vintage_id``. A collision
    in either is refused rather than resolved, because the memoized snapshot would
    otherwise be served for the wrong vintage.

    A callable source has no enumerable calendar, so neither can be proven. There the
    identifier is the caller's declaration of cache identity and the caller owns it: a
    constant ``vintage_id`` such as ``lambda origin: "live"`` remains supported and
    means "one snapshot, reused". Vintage keys are compared as UTC-naive instants, so a
    mapping may mix naive and timezone-aware keys; the raw key is what
    ``available_vintages()`` reports and what ``vintage_id`` receives.
    """

    if callable(source):
        return _CustomVintageSource(
            source=source,
            vintage_id=vintage_id,
            dataset=dataset,
            frequency=frequency,
            strict=strict,
        )

    if isinstance(source, Mapping):
        vintages: list[_ResolvedVintage] = []
        for key, value in source.items():
            timestamp = _canonical_vintage_timestamp(key)
            vintages.append(_ResolvedVintage(key=key, timestamp=timestamp, bundle=value))
        vintages.sort(key=lambda v: v.timestamp)
        _validate_known_vintage_calendar(vintages, vintage_id=vintage_id, form="mapping")
        return _CustomVintageSource(
            vintages=tuple(vintages),
            vintage_id=vintage_id,
            dataset=dataset,
            frequency=frequency,
            strict=strict,
        )

    if isinstance(source, pd.DataFrame):
        if vintage_column is None or date_column is None:
            raise ValueError(
                "vintage_column and date_column are required for long DataFrame custom_vintages"
            )
        if vintage_column not in source.columns:
            raise ValueError(f"vintage column {vintage_column!r} is not in the DataFrame")
        if date_column not in source.columns:
            raise ValueError(f"date column {date_column!r} is not in the DataFrame")
        # ``groupby`` drops rows whose key is NaN/NaT without a word, so a snapshot
        # would vanish from the calendar and the rows it held would silently not exist
        # (F-011). Checked before the groupby, and in both strict modes: a missing
        # vintage key is not a value to coerce, it is an unanswerable question about
        # which snapshot the row belongs to.
        missing_vintage_rows = int(source[vintage_column].isna().sum())
        if missing_vintage_rows:
            raise ValueError(
                f"vintage column {vintage_column!r} has {missing_vintage_rows} missing "
                "values; every row must name the vintage it belongs to"
            )
        vintages_list: list[_ResolvedVintage] = []
        for key, group in source.groupby(vintage_column, sort=False):
            frame = group.drop(columns=[vintage_column]).copy()
            bundle = custom_dataset(
                frame,
                date=date_column,
                dataset=dataset,
                source_family="custom_vintages",
                frequency=frequency,
                metadata={"vintage": str(vintage_id(key) if vintage_id else key)},
                strict=strict,
            )
            vintages_list.append(
                _ResolvedVintage(
                    key=key,
                    timestamp=_canonical_vintage_timestamp(key),
                    bundle=bundle,
                )
            )
        vintages_list.sort(key=lambda v: v.timestamp)
        _validate_known_vintage_calendar(
            vintages_list, vintage_id=vintage_id, form="grouped-wide"
        )
        return _CustomVintageSource(
            vintages=tuple(vintages_list),
            vintage_id=vintage_id,
            dataset=dataset,
            frequency=frequency,
            strict=strict,
        )

    raise TypeError("source must be callable, mapping, or pandas DataFrame")


@dataclass
class _StaticExtrasVintageSource:
    source: VintageSource
    extra: DataBundle
    join: Literal["outer", "inner", "left"]
    _extra_fingerprint: dict[str, Any] = field(init=False, repr=False)
    _cache: dict[str, DataBundle] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        validate_panel(self.extra.panel)
        self._extra_fingerprint = _panel_fingerprint_for_vintage(self.extra.panel)

    @property
    def kind(self) -> str:
        return f"{_source_kind(self.source)}_with_static_extras"

    def available_vintages(self) -> Sequence[Any]:
        return self.source.available_vintages()

    def resolve(self, origin_date: pd.Timestamp) -> DataBundle:
        origin = pd.Timestamp(origin_date)
        bundle = self.source.resolve(origin)
        base_id = _bundle_vintage_label(bundle)
        vintage_label = _extra_vintage_label(base_id, self._extra_fingerprint, origin)
        cached = self._cache.get(vintage_label)
        if cached is not None:
            return cached
        validate_panel(bundle.panel)
        observable_extra = self.extra.panel.loc[self.extra.panel.index < origin]
        joined = bundle.panel.join(observable_extra, how=self.join)
        metadata = {
            **dict(bundle.metadata),
            "vintage": vintage_label,
            "base_vintage": base_id,
            "static_extras": {
                "join": self.join,
                "fingerprint": self._extra_fingerprint,
                "columns": [str(column) for column in self.extra.panel.columns],
                "truncated_before": origin.isoformat(),
            },
        }
        panel = as_panel(joined, metadata=metadata)
        out = DataBundle(panel, metadata)
        self._cache[vintage_label] = out
        return out


def with_static_extras(
    source: VintageSource,
    extra: DataBundle | pd.DataFrame,
    *,
    join: Literal["outer", "inner", "left"] = "outer",
) -> VintageSource:
    """Join non-revised extra columns observable before each origin."""

    if join not in {"outer", "inner", "left"}:
        raise ValueError("join must be one of 'outer', 'inner', or 'left'")
    if not isinstance(source, VintageSource):
        raise TypeError("source must satisfy the VintageSource protocol")
    extra_bundle = _coerce_static_extra(extra)
    return _StaticExtrasVintageSource(source=source, extra=extra_bundle, join=join)


def _validate_probe_limit(value: Any) -> int:
    """``first_release_max_vintages`` is a positive integral, non-boolean count.

    It used to survive construction as anything ``int()`` could truncate and be coerced
    later, so ``1.9`` became a two-vintage probe budget of 1 without anyone saying so
    (F-011). numpy integers are accepted because ``np.arange``/``len`` arithmetic
    produces them; floats are not, even when integral-valued, and ``bool`` is named
    because it is an ``int`` subclass.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"first_release_max_vintages must be a positive integer, not a bool: {value!r}"
        )
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            "first_release_max_vintages must be a positive integer; got "
            f"{type(value).__name__} {value!r}"
        )
    limit = int(value)
    if limit < 1:
        raise ValueError(f"first_release_max_vintages must be at least 1; got {limit}")
    return limit


def _vintage_label_timestamp(label: str) -> pd.Timestamp:
    return pd.Period(label, freq="M").start_time


def _coerce_vintage_timestamp(value: Any) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, pd.errors.OutOfBoundsDatetime) as exc:
        raise ValueError(
            f"custom_vintages vintage key {value!r} cannot be parsed as a timestamp"
        ) from exc
    if pd.isna(timestamp):
        raise ValueError(
            f"custom_vintages vintage key {value!r} cannot be parsed as a timestamp"
        )
    return timestamp


def _canonical_vintage_timestamp(value: Any) -> pd.Timestamp:
    """The instant a vintage key denotes, as a UTC-naive timestamp.

    Ordering and comparison go through this and nothing else. A mapping may mix naive
    and timezone-aware keys -- ``"2020-01-01"`` beside
    ``pd.Timestamp("2020-01-01", tz="UTC")`` -- and comparing those two directly is a
    raw pandas ``TypeError`` about tz-aware and tz-naive operands, which used to reach
    the caller straight out of ``bisect_right`` (F-011). Converting aware keys to UTC
    and dropping the offset puts every key on one line; naive keys are already there and
    are left alone.

    This governs ORDER, not identity. The raw key is what
    ``available_vintages()`` reports and what ``vintage_id`` receives, so canonicalising
    here never rewrites a public label.
    """
    timestamp = _coerce_vintage_timestamp(value)
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _validate_known_vintage_calendar(
    vintages: "Sequence[_ResolvedVintage]",
    *,
    vintage_id: Callable[[Any], Any] | None,
    form: str,
) -> None:
    """Every known vintage must denote a distinct instant and a distinct label.

    Two separate invariants, both checked before any snapshot can be resolved or cached.

    *Distinct instants*, because resolution is a bisect over these timestamps: two keys
    that canonicalise to the same instant leave no way to say which snapshot an origin
    selects. Raw keys that differ are not enough -- ``"2020-01-01"`` and
    ``pd.Timestamp("2020-01-01", tz="UTC")`` are the same instant.

    *Distinct labels*, because the resolved-snapshot cache is keyed by the label. With
    ``vintage_id=lambda _: "same"`` over two different snapshots, resolving the earlier
    origin populated the cache and the later origin was served that earlier content
    (F-010). Rejecting at construction is the only place this can be caught: by the time
    a caller sees a wrong panel there is nothing left to distinguish it from a right one.
    """
    seen_instants: dict[pd.Timestamp, Any] = {}
    seen_labels: dict[str, Any] = {}
    for vintage in vintages:
        previous = seen_instants.get(vintage.timestamp)
        if previous is not None:
            raise ValueError(
                f"custom_vintages {form} keys {previous!r} and {vintage.key!r} denote "
                f"the same instant {vintage.timestamp}; a vintage source cannot choose "
                "between two snapshots for one point in time"
            )
        seen_instants[vintage.timestamp] = vintage.key

        label = str(vintage_id(vintage.key)) if vintage_id else str(vintage.key)
        collided = seen_labels.get(label)
        if collided is not None:
            raise ValueError(
                f"custom_vintages {form} keys {collided!r} and {vintage.key!r} both "
                f"produce the vintage identifier {label!r}. Resolved snapshots are "
                "memoized by that identifier, so the second key would be served the "
                "first key's panel. Make vintage_id return a distinct string per key."
            )
        seen_labels[label] = vintage.key


def _coerce_static_extra(value: DataBundle | pd.DataFrame) -> DataBundle:
    if isinstance(value, DataBundle):
        metadata = dict(value.metadata)
        panel = as_panel(value.panel, metadata=metadata)
        return DataBundle(panel, metadata)
    if isinstance(value, pd.DataFrame):
        return custom_dataset(
            value,
            dataset="static_extras",
            source_family="static_extras",
            frequency="unknown",
        )
    raise TypeError("extra must be a DataBundle or pandas DataFrame")


def _panel_fingerprint_for_vintage(panel: pd.DataFrame) -> dict[str, Any]:
    from macroforecast.data.identity import panel_fingerprint as _panel_fingerprint

    return _panel_fingerprint(panel)


def _bundle_vintage_label(bundle: DataBundle) -> str:
    if "vintage" not in bundle.metadata:
        raise ValueError('wrapped VintageSource bundles must set metadata["vintage"]')
    return str(bundle.metadata["vintage"])


def _extra_vintage_label(
    base_id: str,
    fingerprint: Mapping[str, Any],
    origin: pd.Timestamp,
) -> str:
    """The vintage/cache ID for one origin of a static-extras source.

    The label form is unchanged. It does not need to change: the VintageSource
    contract is that a stable ID moves if and only if the content moves, and F-012 is
    fixed at the source -- the digest inside the label now reads the whole extras panel,
    so a cell the old sampling skipped changes the ID. Reshaping the label instead would
    have migrated every unchanged extras panel to a new ID for nothing.

    The guard below is the part that is new, and it changes no valid label.
    """
    method = str(fingerprint.get("method", ""))
    if method != "full_content":
        # A cache key must not be built on a digest that admits it did not read
        # everything. Unreachable today (``panel_fingerprint`` has one method), and
        # kept so that reintroducing a partial one is a failure rather than a silent
        # weakening of every static-extra vintage ID.
        raise ValueError(
            "static-extra vintage labels require a full-content panel fingerprint; "
            f"got method={method!r}"
        )
    origin_label = pd.Timestamp(origin).strftime("%Y-%m-%d")
    return f"{base_id}|static_extra_sha256={fingerprint.get('value')}|origin={origin_label}"


def _source_kind(source: Any) -> str:
    kind = getattr(source, "kind", None)
    if kind is not None:
        return str(kind)
    return type(source).__name__


__all__ = [
    "VintagePanelSpec",
    "VintageSource",
    "VintageUnavailableError",
    "custom_vintages",
    "fred_md_vintages",
    "fred_qd_vintages",
    "with_static_extras",
]
