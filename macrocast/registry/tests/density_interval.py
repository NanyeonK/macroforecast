from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="density_interval",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="No density/interval calibration test", status="operational", priority="A"),
        EnumRegistryEntry(id="pit_uniformity",             description="Probability integral transform uniformity test", status="operational", priority="B"),
        EnumRegistryEntry(id="berkowitz",                  description="Berkowitz density-forecast test", status="operational", priority="B"),
        EnumRegistryEntry(id="kupiec",                     description="Kupiec unconditional coverage test", status="operational", priority="B"),
        EnumRegistryEntry(id="christoffersen_unconditional", description="Christoffersen unconditional coverage test", status="operational", priority="B"),
        EnumRegistryEntry(id="christoffersen_independence",  description="Christoffersen independence test", status="operational", priority="B"),
        EnumRegistryEntry(id="christoffersen_conditional",   description="Christoffersen conditional coverage test", status="operational", priority="B"),
        EnumRegistryEntry(id="interval_coverage",            description="Empirical coverage of prediction intervals", status="operational", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
