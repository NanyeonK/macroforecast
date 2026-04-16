from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name="importance_output_style",
    layer="7_importance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="ranked_table", description="ranked tabular output", status="operational", priority="A"),
        EnumRegistryEntry(id="curve_bundle", description="curve / profile bundle", status="operational", priority="A"),
        EnumRegistryEntry(id="nested_report", description="nested JSON report", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
