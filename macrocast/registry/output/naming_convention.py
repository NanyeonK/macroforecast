from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="naming_convention",
    layer="5_output_provenance",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="cell_id", description="name artifacts by cell id", status="operational", priority="A"),
        EnumRegistryEntry(id="descriptive", description="descriptive artifact names", status="operational", priority="A"),
        EnumRegistryEntry(id="recipe_hash", description="name artifacts by recipe hash", status="operational", priority="A"),
        EnumRegistryEntry(id="custom", description="custom naming convention", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
