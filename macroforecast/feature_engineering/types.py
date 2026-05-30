from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, TypeAlias

import pandas as pd

from macroforecast.data import DataBundle, DataSpec, attach_metadata
from macroforecast.preprocessing import PreprocessedData


@dataclass(frozen=True)
class FeatureSet:
    """Predictor matrix, target matrix, and feature-engineering metadata."""

    X: pd.DataFrame
    y: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    target: str | None = None
    targets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    predictors: tuple[str, ...] = ()

    def __iter__(self):
        yield self.X
        yield self.y
        yield self.metadata

    def attach(self, stage: str, values: Mapping[str, Any]) -> FeatureSet:
        return replace(self, metadata=attach_metadata(self.metadata, stage, values))


FeatureInput: TypeAlias = (
    PreprocessedData | DataSpec | DataBundle | tuple[pd.DataFrame, Mapping[str, Any]] | pd.DataFrame
)


@dataclass(frozen=True)
class _InputBundle:
    panel: pd.DataFrame
    metadata: dict[str, Any]
    target: str | None = None
    targets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    predictors: Any = "all"


__all__ = ["FeatureInput", "FeatureSet"]
