from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_model_native",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="tree_shap", description="native tree-based SHAP", status="operational", priority="A"),
        EnumRegistryEntry(id="linear_shap", description="native linear SHAP", status="operational", priority="A"),
        EnumRegistryEntry(id="feature_gain", description="native gain / impurity importance", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
