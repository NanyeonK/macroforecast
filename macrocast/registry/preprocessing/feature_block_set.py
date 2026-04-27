from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="feature_block_set",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="feature_builder_compatibility_bridge",
            description="bridge to the current feature_builder based recipes",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="target_lags_only",
            description="target lag block only",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="transformed_predictors",
            description="current transformed predictor panel",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="transformed_predictor_lags",
            description="transformed predictor panel plus X lags",
            status="operational_narrow",
            priority="A",
        ),
        EnumRegistryEntry(
            id="factors_plus_target_lags",
            description="factor block plus target lag block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="factor_blocks_only",
            description="factor block without target-lag or raw predictor block append",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="high_dimensional_predictors",
            description="high dimensional transformed predictor panel",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="selected_sparse_predictors",
            description="selected sparse predictor block",
            status="operational_narrow",
            priority="A",
        ),
        EnumRegistryEntry(
            id="level_augmented_predictors",
            description="transformed predictor panel augmented with level features",
            status="operational_narrow",
            priority="B",
        ),
        EnumRegistryEntry(
            id="rotation_augmented_predictors",
            description="predictor panel augmented with rotated features",
            status="operational_narrow",
            priority="B",
        ),
        EnumRegistryEntry(
            id="mixed_feature_blocks",
            description="multiple named feature blocks composed into Z",
            status="operational_narrow",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_feature_blocks",
            description="researcher supplied feature-block set",
            status="operational_narrow",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
