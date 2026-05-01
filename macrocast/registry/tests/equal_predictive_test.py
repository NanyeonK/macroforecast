from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="equal_predictive_test",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="dm_diebold_mariano", description="Diebold-Mariano equal predictive accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="gw_giacomini_white", description="Giacomini-White conditional predictive ability test", status="operational", priority="A"),
        EnumRegistryEntry(id="multi", description="run all supported equal-predictive tests", status="operational", priority="A"),
        EnumRegistryEntry(id="none", description="disable equal-predictive tests", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
