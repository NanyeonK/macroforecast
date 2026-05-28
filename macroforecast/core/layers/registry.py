from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from ..pipeline import LayerCategory, LayerId
from ..types import (
    ArtifactManifest,
    DataFrameArtifact,
    DiagnosticArtifact,
    FeatureBundle,
    FeatureMetadata,
    ForecastArtifact,
    L1DataDefinitionArtifact,
    L1RegimeMetadataArtifact,
    L2CleanPanelArtifact,
    L3FeaturesArtifact,
    L3MetadataArtifact,
    L4ForecastsArtifact,
    L4ModelArtifactsArtifact,
    L4TrainingMetadataArtifact,
    L5EvaluationArtifact,
    L6TestsArtifact,
    L7ImportanceArtifact,
    L7TransformationAttributionArtifact,
    L8ArtifactsArtifact,
    MetricTable,
    MappingArtifact,
    ModelArtifactSet,
    TestResultSet,
    TrainingMetadata,
    ImportanceResultSet,
)
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


class _L6SinkMap(dict[str, TypeSpec]):
    """Compatibility lookup for pre-L6 foundation selector tests."""

    def __getitem__(self, key: str) -> TypeSpec:
        if key == "tests_v1":
            key = "l6_tests_v1"
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if key == "tests_v1":
            return True
        return super().__contains__(key)

    def __eq__(self, other: object) -> bool:
        return dict(self) == other


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
    "l1": {
        "l1_data_definition_v1": L1DataDefinitionArtifact,
        "l1_regime_metadata_v1": L1RegimeMetadataArtifact,
    },
    "l2": {
        "l2_clean_panel_v1": L2CleanPanelArtifact,
    },
    "l3": {
        "l3_features_v1": L3FeaturesArtifact,
        "l3_metadata_v1": L3MetadataArtifact,
    },
    "l4": {
        "l4_forecasts_v1": L4ForecastsArtifact,
        "l4_model_artifacts_v1": L4ModelArtifactsArtifact,
        "l4_training_metadata_v1": L4TrainingMetadataArtifact,
    },
    "l5": {
        "l5_evaluation_v1": L5EvaluationArtifact,
    },
    "l6": _L6SinkMap({
        "l6_tests_v1": L6TestsArtifact,
    }),
    "l7": {
        "l7_importance_v1": L7ImportanceArtifact,
        "l7_transformation_attribution_v1": L7TransformationAttributionArtifact,
    },
    "l8": {
        "l8_artifacts_v1": L8ArtifactsArtifact,
    },
    "l1_5": {"l1_5_diagnostic_v1": DiagnosticArtifact},
    "l2_5": {"l2_5_diagnostic_v1": DiagnosticArtifact},
    "l3_5": {"l3_5_diagnostic_v1": DiagnosticArtifact},
    "l4_5": {"l4_5_diagnostic_v1": DiagnosticArtifact},
}


LAYER_GLOBALS: dict[LayerId, tuple[str, ...]] = {
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

