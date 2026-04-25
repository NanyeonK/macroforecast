from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="factor_rotation_order",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="rotation_then_factor",
            description="apply an executable rotation block to the X panel, then fit factor features",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="factor_then_rotation",
            description="fit factor-score histories first, then apply an executable rotation block to factor scores",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
