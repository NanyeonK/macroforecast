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
class L0MetaArtifact(DataType):
    failure_policy: Literal["fail_fast", "continue_on_failure"]
    reproducibility_mode: Literal["seeded_reproducible", "exploratory"]
    compute_mode: Literal["serial", "parallel"]
    random_seed: int | None
    parallel_unit: Literal["models", "horizons", "targets", "oos_dates"] | None
    n_workers: int | Literal["auto"] | None
    gpu_deterministic: bool
    derived_study_scope: str
    derived_execution_route: str = "comparison_sweep"


@dataclass(frozen=True)
class L1DataDefinitionArtifact(DataType):
    custom_source_policy: Literal["official_only", "custom_panel_only", "official_plus_custom"]
    dataset: Literal["fred_md", "fred_qd", "fred_sd", "fred_md+fred_sd", "fred_qd+fred_sd"] | None
    frequency: Literal["monthly", "quarterly"]
    vintage_policy: Literal["current_vintage", "real_time_alfred"] | None
    target_structure: Literal["single_target", "multi_series_target"]
    target: str | None = None
    targets: tuple[str, ...] = ()
    variable_universe: (
        Literal[
            "all_variables",
            "core_variables",
            "category_variables",
            "target_specific_variables",
            "explicit_variable_list",
        ]
        | None
    ) = None
    target_geography_scope: Literal["single_state", "all_states", "selected_states"] | None = None
    predictor_geography_scope: Literal["match_target", "all_states", "selected_states", "national_only"] | None = None
    sample_start_rule: Literal["earliest_available", "fixed_date", "max_balanced"] = "max_balanced"
    sample_end_rule: Literal["latest_available", "fixed_date"] = "latest_available"
    horizon_set: Literal["standard_md", "standard_qd", "single", "custom_list", "range_up_to_h"] = "standard_md"
    target_horizons: tuple[int, ...] = ()
    regime_definition: str = "none"
    leaf_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class L1RegimeMetadataArtifact(DataType):
    definition: Literal[
        "none",
        "external_nber",
        "external_user_provided",
        "estimated_markov_switching",
        "estimated_threshold",
        "estimated_structural_break",
    ]
    n_regimes: int
    regime_label_series: "Series | None" = None
    regime_probabilities: "Series | None" = None
    transition_matrix: Any | None = None
    estimation_temporal_rule: str | None = None
    estimation_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Panel(DataType):
    shape: tuple[Any, Any] | None = None
    column_names: tuple[str, ...] = ()
    index: pd.DatetimeIndex | None = None
    metadata: PanelMetadata = field(default_factory=PanelMetadata)


@dataclass(frozen=True)
class L2CleanPanelArtifact(Panel):
    panel: Panel = field(default_factory=Panel)
    column_metadata: dict[str, Any] = field(default_factory=dict)
    cleaning_log: dict[str, Any] = field(default_factory=dict)
    n_imputed_cells: int = 0
    n_outliers_flagged: int = 0
    n_truncated_obs: int = 0
    transform_map_applied: dict[str, int] = field(default_factory=dict)
    upstream_hashes: dict[str, str] = field(default_factory=dict)
    cleaning_temporal_rules: dict[str, str] = field(default_factory=dict)


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
class StepRef(DataType):
    step_node_id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ColumnLineage(DataType):
    column_name: str
    source_variable_ids: tuple[str, ...] = ()
    step_chain: tuple[StepRef, ...] = ()
    pipeline_id: str | None = None
    cascade_depth: int = 0
    output_type: str = "Panel"


@dataclass(frozen=True)
class PipelineDefinition(DataType):
    pipeline_id: str
    endpoint_node_id: str
    source_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class L3FeaturesArtifact(DataType):
    X_final: Panel | LaggedPanel | Factor
    y_final: Series
    sample_index: pd.DatetimeIndex | None = None
    horizon_set: tuple[int, ...] = ()
    upstream_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class L3MetadataArtifact(DataType):
    column_lineage: dict[str, ColumnLineage] = field(default_factory=dict)
    pipeline_definitions: dict[str, PipelineDefinition] = field(default_factory=dict)
    cascade_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)
    transform_chain: dict[str, tuple[StepRef, ...]] = field(default_factory=dict)
    source_variables: dict[str, tuple[str, ...]] = field(default_factory=dict)


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
class L4ForecastsArtifact(DataType):
    forecasts: dict[tuple[str, str, int, Any], float] = field(default_factory=dict)
    forecast_intervals: dict[tuple[str, str, int, Any, float], float] = field(default_factory=dict)
    forecast_object: Literal["point", "quantile", "density"] = "point"
    sample_index: pd.DatetimeIndex | None = None
    targets: tuple[str, ...] = ()
    horizons: tuple[int, ...] = ()
    model_ids: tuple[str, ...] = ()
    upstream_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class L4ModelArtifactsArtifact(DataType):
    artifacts: dict[str, ModelArtifact] = field(default_factory=dict)
    is_benchmark: dict[str, bool] = field(default_factory=dict)
    upstream_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class L4TrainingMetadataArtifact(DataType):
    forecast_origins: tuple[Any, ...] = ()
    refit_origins: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    training_window_per_origin: dict[tuple[str, Any], tuple[Any, Any]] = field(default_factory=dict)
    runtime_per_origin: dict[tuple[str, Any], float] = field(default_factory=dict)
    cache_hits_per_origin: dict[tuple[str, Any], bool] = field(default_factory=dict)
    tuning_log: dict[tuple[str, Any], dict[str, Any]] = field(default_factory=dict)
    upstream_hashes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricTable(DataType):
    df: pd.DataFrame
    metric_names: tuple[str, ...] = ()
    benchmark_id: str | None = None


@dataclass(frozen=True)
class L5EvaluationArtifact(DataType):
    metrics_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    ranking_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_relative_metrics: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    per_regime_metrics: dict[tuple[Any, ...], Any] | None = None
    decomposition_results: dict[str, Any] | None = None
    per_state_metrics: dict[tuple[Any, ...], Any] | None = None
    report_artifacts: dict[str, Any] = field(default_factory=dict)
    upstream_hashes: dict[str, str] = field(default_factory=dict)
    l5_axis_resolved: dict[str, Any] = field(default_factory=dict)


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


def __getattr__(name: str) -> Any:
    if name == "LAYER_SINKS":
        from .layers.registry import LAYER_SINKS

        return LAYER_SINKS
    raise AttributeError(name)
