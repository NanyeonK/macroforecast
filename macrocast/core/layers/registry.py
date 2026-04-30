from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from ..dag import LayerCategory, LayerId
from ..types import (
    ArtifactManifest,
    DataFrameArtifact,
    DiagnosticArtifact,
    FeatureBundle,
    FeatureMetadata,
    ForecastArtifact,
    L0MetaArtifact,
    L1DataDefinitionArtifact,
    L1RegimeMetadataArtifact,
    L2CleanPanelArtifact,
    MetricTable,
    MappingArtifact,
    ModelArtifactSet,
    TestResultSet,
    TrainingMetadata,
    ImportanceResultSet,
)
from .l0 import L0StudySetup
from .l1 import L1Data
from .l2 import L2Preprocessing
from ..ops.registry import TypeSpec


@dataclass(frozen=True)
class LayerSpec:
    id: LayerId
    name: str
    category: LayerCategory
    expected_inputs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    ui_mode: Literal["adaptive", "list", "graph"] = "adaptive"
    cls: type | None = None


_LAYERS: dict[LayerId, LayerSpec] = {}


def clear_layer_registry() -> None:
    _LAYERS.clear()


def register_layer(
    *,
    id: LayerId,
    name: str,
    category: LayerCategory,
    expected_inputs: tuple[str, ...] = (),
    produces: tuple[str, ...] = (),
    ui_mode: Literal["adaptive", "list", "graph"] = "adaptive",
) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        if id in _LAYERS:
            raise ValueError(f"duplicate layer registration for {id!r}")
        _LAYERS[id] = LayerSpec(
            id=id,
            name=name,
            category=category,
            expected_inputs=expected_inputs,
            produces=produces,
            ui_mode=ui_mode,
            cls=cls,
        )
        return cls

    return decorator


def get_layer(layer_id: LayerId) -> LayerSpec:
    try:
        return _LAYERS[layer_id]
    except KeyError as exc:
        raise KeyError(f"unknown layer {layer_id!r}") from exc


def list_layers() -> dict[LayerId, LayerSpec]:
    return dict(_LAYERS)


LAYER_SINKS: dict[LayerId, dict[str, TypeSpec]] = {
    "l0": {
        "l0_meta_v1": L0MetaArtifact,
    },
    "l1": {
        "l1_data_definition_v1": L1DataDefinitionArtifact,
        "l1_regime_metadata_v1": L1RegimeMetadataArtifact,
    },
    "l2": {
        "l2_clean_panel_v1": L2CleanPanelArtifact,
    },
    "l3": {
        "features_v1": FeatureBundle,
        "feature_metadata_v1": FeatureMetadata,
    },
    "l4": {
        "forecasts_v1": ForecastArtifact,
        "model_artifacts_v1": ModelArtifactSet,
        "training_metadata_v1": TrainingMetadata,
    },
    "l5": {
        "evaluation_v1": MetricTable,
        "ranking_v1": DataFrameArtifact,
        "decomposition_v1": MappingArtifact,
    },
    "l6": {
        "tests_v1": TestResultSet,
    },
    "l7": {
        "importance_v1": ImportanceResultSet,
        "transformation_attribution_v1": MappingArtifact,
    },
    "l8": {
        "artifacts_v1": ArtifactManifest,
    },
    "l1_5": {"diagnostic_v1": DiagnosticArtifact},
    "l2_5": {"diagnostic_v1": DiagnosticArtifact},
    "l3_5": {"diagnostic_v1": DiagnosticArtifact},
    "l4_5": {"diagnostic_v1": DiagnosticArtifact},
}


LAYER_GLOBALS: dict[LayerId, tuple[str, ...]] = {
    "l0": ("failure_policy", "reproducibility_mode", "compute_mode"),
    "l1": (),
    "l1_5": ("enabled",),
    "l2": (),
    "l2_5": ("enabled",),
    "l3": (),
    "l3_5": ("enabled",),
    "l4": (),
    "l4_5": ("enabled",),
    "l5": (),
    "l6": ("test_scope", "dependence_correction", "overlap_handling"),
    "l7": (),
    "l8": (),
}


register_layer(
    id="l0",
    name="Study Setup",
    category="setup",
    produces=("l0_meta_v1",),
    ui_mode="list",
)(L0StudySetup)


register_layer(
    id="l1",
    name="Data",
    category="construction",
    produces=("l1_data_definition_v1", "l1_regime_metadata_v1"),
    ui_mode="list",
)(L1Data)


register_layer(
    id="l2",
    name="Preprocessing",
    category="construction",
    expected_inputs=("l1_data_definition_v1",),
    produces=("l2_clean_panel_v1",),
    ui_mode="list",
)(L2Preprocessing)


@register_layer(
    id="l3",
    name="Feature engineering",
    category="construction",
    expected_inputs=("l2.clean_panel_v1", "l1.raw_panel_v1"),
    produces=("l3.features_v1",),
    ui_mode="adaptive",
)
class L3FeatureEngineering:
    pass


@register_layer(
    id="l4",
    name="Forecasting model",
    category="construction",
    expected_inputs=("l3.features_v1",),
    produces=("l4.forecasts_v1", "l4.model_artifacts_v1"),
    ui_mode="adaptive",
)
class L4ForecastingModel:
    pass


@register_layer(id="l1_5", name="Data summary", category="diagnostic", ui_mode="adaptive")
class L15DataSummary:
    pass


@register_layer(id="l2_5", name="Pre vs post", category="diagnostic", ui_mode="adaptive")
class L25PrePost:
    pass


@register_layer(id="l3_5", name="Feature diagnostics", category="diagnostic", ui_mode="adaptive")
class L35FeatureDiagnostics:
    pass


@register_layer(id="l4_5", name="Generator diagnostics", category="diagnostic", ui_mode="adaptive")
class L45GeneratorDiagnostics:
    pass


@register_layer(
    id="l5",
    name="Evaluation",
    category="consumption",
    expected_inputs=("l4.forecasts_v1",),
    produces=("l5.evaluation_v1",),
    ui_mode="adaptive",
)
class L5Evaluation:
    pass


@register_layer(
    id="l6",
    name="Statistical tests",
    category="consumption",
    expected_inputs=("l5.evaluation_v1",),
    produces=("l6.tests_v1",),
    ui_mode="adaptive",
)
class L6StatisticalTests:
    pass


@register_layer(
    id="l7",
    name="Interpretation",
    category="consumption",
    expected_inputs=("l4.model_artifacts_v1", "l3.features_v1"),
    produces=("l7.importance_v1",),
    ui_mode="adaptive",
)
class L7Interpretation:
    pass


@register_layer(
    id="l8",
    name="Output / provenance",
    category="consumption",
    expected_inputs=("l5.evaluation_v1",),
    produces=("l8.artifacts_v1",),
    ui_mode="adaptive",
)
class L8OutputProvenance:
    pass
