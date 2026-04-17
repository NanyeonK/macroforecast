from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="cpa_instability",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none",       description="No CPA/instability test", status="operational", priority="A"),
        EnumRegistryEntry(id="cpa",        description="Giacomini-White conditional predictive ability test", status="operational", priority="A"),
        EnumRegistryEntry(id="rossi",      description="Rossi-Sekhposyan forecast-stability statistic", status="operational", priority="A"),
        EnumRegistryEntry(id="rolling_dm", description="Rolling-window Diebold-Mariano summary", status="operational", priority="A"),
        EnumRegistryEntry(id="fluctuation_test",   description="Giacomini-Rossi fluctuation test", status="planned", priority="B"),
        EnumRegistryEntry(id="chow_break_forecast", description="Chow-style structural break test on forecast errors", status="planned", priority="B"),
        EnumRegistryEntry(id="cusum_on_loss",       description="CUSUM-of-loss-differential stability test", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
