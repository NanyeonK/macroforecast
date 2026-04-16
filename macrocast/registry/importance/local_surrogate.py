from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_local_surrogate",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="lime", description="LIME-style local surrogate", status="operational", priority="A"),
        EnumRegistryEntry(id="feature_ablation", description="feature-ablation local explanation", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
