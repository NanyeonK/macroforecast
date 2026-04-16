from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_grouped",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="grouped_permutation", description="grouped permutation importance", status="operational", priority="A"),
        EnumRegistryEntry(id="variable_root_groups", description="group by feature root / lag prefix", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
