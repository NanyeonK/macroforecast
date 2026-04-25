from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_grouped",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no grouped importance selected", status="operational", priority="A"),
        EnumRegistryEntry(id="grouped_permutation", description="grouped permutation importance", status="operational", priority="A"),
        EnumRegistryEntry(id="variable_root_groups", description="group by feature root / lag prefix", status="registry_only", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
