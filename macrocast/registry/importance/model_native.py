from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_model_native",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no model-native importance family selected", status="operational", priority="A"),
        EnumRegistryEntry(id="minimal_importance", description="coefficient / impurity importance summary", status="operational", priority="A"),
        EnumRegistryEntry(id="tree_shap", description="native tree-based SHAP", status="operational", priority="A"),
        EnumRegistryEntry(id="linear_shap", description="native linear SHAP", status="operational", priority="A"),
        EnumRegistryEntry(id="feature_gain", description="native gain / impurity importance", status="registry_only", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
