from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="temporal_feature_block",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="none",
            description="no temporal feature block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="moving_average_features",
            description="moving average features",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="rolling_moments",
            description="rolling mean variance and related moment features",
            status="operational",
            priority="B",
        ),
        EnumRegistryEntry(
            id="local_temporal_factors",
            description="local temporal factor features",
            status="operational",
            priority="B",
        ),
        EnumRegistryEntry(
            id="volatility_features",
            description="rolling volatility and instability features",
            status="operational",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_temporal_features",
            description="researcher supplied temporal feature block pending a block-local callable contract",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
