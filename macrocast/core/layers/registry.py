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
from .l0 import L0StudySetup
from .l1 import L1Data
from .l2 import L2Preprocessing
from .l3 import L3FeatureEngineering
from .l4 import L4ForecastingModel
from .l5 import L5Evaluation
from .l6 import L6StatisticalTests
from .l7 import L7Interpretation
from .l8 import L8Output
from .l1_5 import L1_5DataSummary
from .l2_5 import L2_5PrePostPreprocessing
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


register_layer(
    id="l3",
    name="Feature engineering",
    category="construction",
    expected_inputs=("l2_clean_panel_v1", "l1_data_definition_v1", "l1_regime_metadata_v1"),
    produces=("l3_features_v1", "l3_metadata_v1"),
    ui_mode="graph",
)(L3FeatureEngineering)


register_layer(
    id="l4",
    name="Forecasting model",
    category="construction",
    expected_inputs=("l3_features_v1", "l3_metadata_v1", "l1_regime_metadata_v1"),
    produces=("l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1"),
    ui_mode="graph",
)(L4ForecastingModel)


register_layer(
    id="l1_5",
    name="Data summary",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1",),
    produces=("l1_5_diagnostic_v1",),
    ui_mode="list",
)(L1_5DataSummary)


register_layer(
    id="l2_5",
    name="Pre vs post preprocessing",
    category="diagnostic",
    expected_inputs=("l1_data_definition_v1", "l2_clean_panel_v1"),
    produces=("l2_5_diagnostic_v1",),
    ui_mode="list",
)(L2_5PrePostPreprocessing)


@register_layer(id="l3_5", name="Feature diagnostics", category="diagnostic", ui_mode="adaptive")
class L35FeatureDiagnostics:
    pass


@register_layer(id="l4_5", name="Generator diagnostics", category="diagnostic", ui_mode="adaptive")
class L45GeneratorDiagnostics:
    pass


register_layer(
    id="l5",
    name="Evaluation",
    category="consumption",
    expected_inputs=("l4_forecasts_v1", "l4_model_artifacts_v1", "l1_data_definition_v1", "l1_regime_metadata_v1", "l3_metadata_v1"),
    produces=("l5_evaluation_v1",),
    ui_mode="list",
)(L5Evaluation)


register_layer(
    id="l6",
    name="Statistical tests",
    category="consumption",
    expected_inputs=("l4_forecasts_v1", "l4_model_artifacts_v1", "l5_evaluation_v1", "l1_data_definition_v1", "l1_regime_metadata_v1"),
    produces=("l6_tests_v1",),
    ui_mode="list",
)(L6StatisticalTests)


register_layer(
    id="l7",
    name="Interpretation",
    category="consumption",
    expected_inputs=("l4_model_artifacts_v1", "l4_forecasts_v1", "l3_features_v1", "l3_metadata_v1", "l5_evaluation_v1", "l6_tests_v1", "l1_data_definition_v1", "l1_regime_metadata_v1"),
    produces=("l7_importance_v1", "l7_transformation_attribution_v1"),
    ui_mode="graph",
)(L7Interpretation)


register_layer(
    id="l8",
    name="Output / provenance",
    category="consumption",
    expected_inputs=(
        "l0_meta_v1", "l1_data_definition_v1", "l1_regime_metadata_v1", "l2_clean_panel_v1", "l3_features_v1", "l3_metadata_v1",
        "l4_forecasts_v1", "l4_model_artifacts_v1", "l4_training_metadata_v1", "l5_evaluation_v1", "l6_tests_v1",
        "l7_importance_v1", "l7_transformation_attribution_v1",
    ),
    produces=("l8_artifacts_v1",),
    ui_mode="list",
)(L8Output)
