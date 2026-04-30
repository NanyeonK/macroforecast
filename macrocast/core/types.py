from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

@dataclass(frozen=True)
class DataType:
    """Base class for typed artifacts passed between DAG nodes."""


@dataclass(frozen=True)
class MetadataArtifact(DataType):
    values: dict[str, Any] = field(default_factory=dict)


class PanelMetadata(MetadataArtifact):
    pass


class LaggedPanelMetadata(MetadataArtifact):
    pass


class FactorMetadata(MetadataArtifact):
    pass


class SeriesMetadata(MetadataArtifact):
    pass


class FeatureMetadata(MetadataArtifact):
    pass


class ForecastMetadata(MetadataArtifact):
    pass


class TrainingMetadata(MetadataArtifact):
    pass


class ArtifactManifest(MetadataArtifact):
    pass


class DataFrameArtifact(MetadataArtifact):
    pass


class MappingArtifact(MetadataArtifact):
    pass


@dataclass(frozen=True)
class Panel(DataType):
    shape: tuple[Any, Any] | None = None
    column_names: tuple[str, ...] = ()
    index: pd.DatetimeIndex | None = None
    metadata: PanelMetadata = field(default_factory=PanelMetadata)


@dataclass(frozen=True)
class LaggedPanel(DataType):
    shape: tuple[Any, Any] | None = None
    column_names: tuple[str, ...] = ()
    n_lag: int = 0
    metadata: LaggedPanelMetadata = field(default_factory=LaggedPanelMetadata)


@dataclass(frozen=True)
class Factor(DataType):
    shape: tuple[Any, Any] | None = None
    column_names: tuple[str, ...] = ()
    extraction_method: str = ""
    metadata: FactorMetadata = field(default_factory=FactorMetadata)


@dataclass(frozen=True)
class Series(DataType):
    shape: tuple[Any] | None = None
    name: str = ""
    metadata: SeriesMetadata = field(default_factory=SeriesMetadata)


@dataclass(frozen=True)
class MaskedPanel(DataType):
    panel: Panel
    mask: Any
    mask_meaning: Literal["missing", "outlier", "both"]


@dataclass(frozen=True)
class FeatureBundle(DataType):
    X_final: Panel | LaggedPanel | Factor
    y_final: Series
    metadata: FeatureMetadata = field(default_factory=FeatureMetadata)


@dataclass(frozen=True)
class ModelArtifact(DataType):
    model_id: str
    family: str
    fitted_object: Any
    framework: Literal["sklearn", "xgboost", "lightgbm", "statsmodels", "torch", "tf", "custom_r"]
    fit_metadata: dict[str, Any] = field(default_factory=dict)
    feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelArtifactSet(DataType):
    artifacts: dict[str, ModelArtifact] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastArtifact(DataType):
    forecasts: dict[tuple[str, int, Any], float]
    forecast_intervals: dict[tuple[str, int, Any, float], float] = field(default_factory=dict)
    metadata: ForecastMetadata = field(default_factory=ForecastMetadata)


@dataclass(frozen=True)
class MetricTable(DataType):
    df: pd.DataFrame
    metric_names: tuple[str, ...] = ()
    benchmark_id: str | None = None


@dataclass(frozen=True)
class TestResult(DataType):
    test_name: str
    statistic: float
    p_value: float
    decision_at_alpha: dict[float, bool] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TestResultSet(DataType):
    results: dict[str, dict[str, TestResult]] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportanceResult(DataType):
    method: str
    scope: Literal["global", "local_per_origin", "local_per_observation"]
    values: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportanceResultSet(DataType):
    results: dict[str, ImportanceResult] = field(default_factory=dict)


@dataclass(frozen=True)
class DiagnosticArtifact(DataType):
    layer_hooked: str
    artifact_type: Literal["table", "figure", "json"] | str
    file_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
