from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="equal_predictive",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none",        description="No equal-predictive test", status="operational", priority="A"),
        EnumRegistryEntry(id="dm",          description="Diebold-Mariano equal predictive accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="dm_hln",      description="Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction", status="operational", priority="A"),
        EnumRegistryEntry(id="dm_modified", description="Modified Diebold-Mariano for long-horizon forecasts", status="operational", priority="A"),
        EnumRegistryEntry(id="paired_t_on_loss_diff", description="Paired t-test on loss differential", status="planned", priority="B"),
        EnumRegistryEntry(id="wilcoxon_signed_rank",  description="Wilcoxon signed-rank on loss differential", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
