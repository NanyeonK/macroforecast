from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_model_agnostic",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no model-agnostic importance family selected", status="operational", priority="A"),
        EnumRegistryEntry(id="kernel_shap", description="KernelSHAP", status="operational", priority="A"),
        EnumRegistryEntry(id="permutation_importance", description="permutation importance", status="operational", priority="A"),
        EnumRegistryEntry(id="feature_ablation", description="feature ablation", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
