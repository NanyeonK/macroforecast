from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_shap",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="tree_shap", description="TreeSHAP", status="operational", priority="A"),
        EnumRegistryEntry(id="kernel_shap", description="KernelSHAP", status="operational", priority="A"),
        EnumRegistryEntry(id="linear_shap", description="LinearSHAP", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
