from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="model_artifacts_format",
    layer="5_output_provenance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="pickle", description="pickle model artifact", status="operational", priority="A"),
        EnumRegistryEntry(id="joblib", description="joblib model artifact", status="operational", priority="A"),
        EnumRegistryEntry(id="onnx", description="ONNX model artifact", status="future", priority="B"),
        EnumRegistryEntry(id="pmml", description="PMML model artifact", status="future", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
