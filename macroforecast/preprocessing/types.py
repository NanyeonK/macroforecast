from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import pandas as pd

from macroforecast.data import DataBundle, DataSpec, attach_metadata


@dataclass(frozen=True)
class PreprocessedData:
    """Cleaned macroforecast panel plus metadata and data-spec choices."""

    panel: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    target: str | None = None
    targets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    start: str | None = None
    end: str | None = None
    predictors: Any = "all"
    steps: tuple[dict[str, Any], ...] = ()

    def __iter__(self):
        yield self.panel
        yield self.metadata

    def attach(self, stage: str, values: Mapping[str, Any]) -> PreprocessedData:
        return replace(self, metadata=attach_metadata(self.metadata, stage, values))


PreprocessInput: TypeAlias = (
    PreprocessedData
    | DataSpec
    | DataBundle
    | tuple[pd.DataFrame, Mapping[str, Any]]
    | pd.DataFrame
)


@dataclass(frozen=True)
class _InputBundle:
    panel: pd.DataFrame
    metadata: dict[str, Any]
    target: str | None = None
    targets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    start: str | None = None
    end: str | None = None
    predictors: Any = "all"


__all__ = ["PreprocessedData", "PreprocessInput"]
