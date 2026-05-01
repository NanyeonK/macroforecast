from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="dependence_correction",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="newey_west", description="Newey-West HAC correction", status="operational", priority="A"),
        EnumRegistryEntry(id="andrews", description="Andrews automatic bandwidth correction", status="operational", priority="A"),
        EnumRegistryEntry(id="parzen_kernel", description="Parzen-kernel HAC correction", status="operational", priority="A"),
        EnumRegistryEntry(id="none", description="IID variance assumption", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
