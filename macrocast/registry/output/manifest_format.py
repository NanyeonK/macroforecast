from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="manifest_format",
    layer="5_output_provenance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="json", description="JSON manifest", status="operational", priority="A"),
        EnumRegistryEntry(id="yaml", description="YAML manifest", status="operational", priority="A"),
        EnumRegistryEntry(id="json_lines", description="JSON Lines manifest", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
