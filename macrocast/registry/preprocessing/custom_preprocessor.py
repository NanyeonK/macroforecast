from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="custom_preprocessor",
    layer="2_preprocessing",
    axis_type="plugin",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="no custom preprocessor", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
