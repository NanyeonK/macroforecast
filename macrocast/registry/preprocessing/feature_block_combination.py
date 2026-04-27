from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="feature_block_combination",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="replace_with_selected_blocks",
            description="replace the base predictor panel with the selected blocks",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="append_to_base_predictors",
            description="append selected blocks to the base predictor panel",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="append_to_target_lags",
            description="append selected blocks to the target lag block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="concatenate_named_blocks",
            description="concatenate all selected named blocks into Z",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="custom_feature_combiner",
            description="researcher supplied feature-block combiner",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
