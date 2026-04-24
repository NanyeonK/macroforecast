from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="feature_selection_semantics",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="select_before_factor",
            description="select raw predictor X before fitting factor blocks",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="select_after_factor",
            description="select among final Z columns after factor extraction",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="select_after_custom_blocks",
            description="select among final Z columns after custom blocks or a custom combiner",
            status="operational",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
