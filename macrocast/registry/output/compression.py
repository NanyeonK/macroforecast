from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="compression",
    layer="5_output_provenance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no compression", status="operational", priority="A"),
        EnumRegistryEntry(id="gzip", description="gzip compression", status="operational", priority="A"),
        EnumRegistryEntry(id="zip", description="zip archive compression", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
