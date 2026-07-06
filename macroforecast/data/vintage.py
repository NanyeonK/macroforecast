from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import pandas as pd

from .errors import VintageUnavailableError
from .loaders import list_vintages, load_fred_md, load_fred_qd
from .panel import DataBundle, as_panel, custom_dataset


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
    """Return a FRED-MD point-in-time source resolved by origin date."""

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
    mapping from date-like vintage keys to snapshots, or a long ALFRED-style
    DataFrame with one row per ``(date, vintage, series...)``. Every snapshot is
    normalized through :func:`as_panel` / :func:`custom_dataset` and then
    validated. Resolved snapshots are memoized by the stable identifier produced
    by ``vintage_id`` (default: ``str(resolved_key)``). If a callable reads from
    a non-deterministic source whose content can change for the same identifier,
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
        vintages = tuple(
            _ResolvedVintage(key=key, timestamp=_coerce_vintage_timestamp(key), bundle=value)
            for key, value in source.items()
        )
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


def _vintage_label_timestamp(label: str) -> pd.Timestamp:
    return pd.Period(label, freq="M").start_time


def _coerce_vintage_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError(f"vintage key {value!r} cannot be parsed as a timestamp")
    return timestamp


__all__ = [
    "VintagePanelSpec",
    "VintageSource",
    "VintageUnavailableError",
    "custom_vintages",
    "fred_md_vintages",
    "fred_qd_vintages",
]
