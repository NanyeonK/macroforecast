from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

import pandas as pd

from .errors import VintageUnavailableError
from .loaders import list_vintages, load_fred_md, load_fred_qd
from .panel import DataBundle


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


def _vintage_label_timestamp(label: str) -> pd.Timestamp:
    return pd.Period(label, freq="M").start_time


__all__ = [
    "VintagePanelSpec",
    "VintageSource",
    "VintageUnavailableError",
    "fred_md_vintages",
    "fred_qd_vintages",
]
