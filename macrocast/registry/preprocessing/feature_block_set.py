from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="feature_block_set",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="legacy_feature_builder_bridge",
            description="bridge to the current feature_builder based recipes",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="target_lags_only",
            description="target lag block only",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="transformed_x",
            description="current transformed predictor panel",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="transformed_x_lags",
            description="transformed predictor panel plus X lags",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="factors_plus_target_lags",
            description="factor block plus target lag block",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="high_dimensional_x",
            description="high dimensional transformed predictor panel",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="selected_sparse_x",
            description="selected sparse predictor block",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="level_augmented_x",
            description="transformed predictor panel augmented with level features",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="rotation_augmented_x",
            description="predictor panel augmented with rotated features",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="mixed_blocks",
            description="multiple named feature blocks composed into Z",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_blocks",
            description="researcher supplied feature-block set",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
