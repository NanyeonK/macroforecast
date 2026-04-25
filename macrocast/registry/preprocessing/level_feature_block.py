from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="level_feature_block",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="none",
            description="no level add-back block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="target_level_addback",
            description="target level added back as a feature",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="x_level_addback",
            description="level version of the predictor panel added as features",
            status="operational",
            priority="B",
        ),
        EnumRegistryEntry(
            id="selected_level_addbacks",
            description="selected level variables added as features",
            status="operational",
            priority="B",
        ),
        EnumRegistryEntry(
            id="level_growth_pairs",
            description="paired level and growth representations for selected variables",
            status="operational",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
