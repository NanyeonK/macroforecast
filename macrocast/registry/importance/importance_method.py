from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_method",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no importance artifact", status="operational", priority="A"),
        EnumRegistryEntry(id="minimal_importance", description="lightweight coefficient / impurity importance", status="operational", priority="A"),
        EnumRegistryEntry(id="tree_shap", description="TreeSHAP for supported tree models", status="operational", priority="A"),
        EnumRegistryEntry(id="kernel_shap", description="KernelSHAP for model-agnostic local explanations", status="operational", priority="A"),
        EnumRegistryEntry(id="linear_shap", description="LinearSHAP for linear estimators", status="operational", priority="A"),
        EnumRegistryEntry(id="permutation_importance", description="model-agnostic permutation feature importance", status="operational", priority="A"),
        EnumRegistryEntry(id="lime", description="LIME-style local surrogate explanation", status="operational", priority="A"),
        EnumRegistryEntry(id="feature_ablation", description="one-feature-at-a-time prediction ablation", status="operational", priority="A"),
        EnumRegistryEntry(id="pdp", description="partial dependence profiles", status="operational", priority="A"),
        EnumRegistryEntry(id="ice", description="individual conditional expectation profiles", status="operational", priority="A"),
        EnumRegistryEntry(id="ale", description="accumulated local effects profiles", status="operational", priority="A"),
        EnumRegistryEntry(id="grouped_permutation", description="grouped permutation importance", status="operational", priority="A"),
        EnumRegistryEntry(id="importance_stability", description="bootstrap / seed stability for feature importance", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
    component="importance",
)
