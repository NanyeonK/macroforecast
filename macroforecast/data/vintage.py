from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

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
        if not self.reference_calendar.is_monotonic_increasing:
            raise ValueError("reference_calendar must be monotonic increasing")
        if self.actuals_vintage not in {"latest", "first_release"}:
            raise ValueError("actuals_vintage must be 'latest' or 'first_release'")
        if int(self.first_release_max_vintages) < 1:
            raise ValueError("first_release_max_vintages must be at least 1")
        if self.actuals_vintage == "first_release":
            keys = tuple(self.source.available_vintages())
            if not keys:
                raise ValueError(
                    "actuals_vintage='first_release' requires "
                    "source.available_vintages() to return timestamp-parsable "
                    "vintage keys; callable custom_vintages sources without "
                    "explicit vintages cannot supply first-release actuals"
                )
            for key in keys:
                _coerce_vintage_timestamp(key)


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
        origin = pd.Timestamp(origin_date)
        if self.source is not None:
            key = origin if not self.vintages else self._resolved_key(origin)
            vintage_label = self._vintage_label(key)
            cached = self._cache.get(vintage_label)
            if cached is not None:
                return cached
            bundle = self._coerce(self.source(origin), vintage_label=vintage_label)
            self._cache[vintage_label] = bundle
            return bundle

        resolved = self._resolved(origin)
        vintage_label = self._vintage_label(resolved.key)
        cached = self._cache.get(vintage_label)
        if cached is not None:
            return cached
        if resolved.bundle is None:  # pragma: no cover - constructor invariant
            raise VintageUnavailableError(f"no custom vintage is available at or before {origin.date()}")
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
            timestamp = _coerce_vintage_timestamp(key)
            vintages.append(_ResolvedVintage(key=key, timestamp=timestamp, bundle=value))
        return _CustomVintageSource(
            vintages=tuple(sorted(vintages, key=lambda v: v.timestamp)),
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
                    timestamp=_coerce_vintage_timestamp(key),
                    bundle=bundle,
                )
            )
        return _CustomVintageSource(
            vintages=tuple(sorted(vintages_list, key=lambda v: v.timestamp)),
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
    from macroforecast.pipeline.run import _panel_fingerprint

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
