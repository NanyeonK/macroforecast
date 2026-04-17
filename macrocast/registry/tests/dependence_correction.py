from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="dependence_correction",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="IID variance assumption", status="operational", priority="A"),
        EnumRegistryEntry(id="nw_hac", description="Newey-West HAC with horizon-based bandwidth", status="operational", priority="A"),
        EnumRegistryEntry(id="nw_hac_auto", description="Newey-West HAC with automatic bandwidth", status="operational", priority="A"),
        EnumRegistryEntry(id="block_bootstrap", description="Moving block bootstrap dependence correction", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
