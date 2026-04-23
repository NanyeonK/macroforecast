from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="rotation_feature_block",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="none",
            description="no rotated feature block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="marx_rotation",
            description="moving average rotation of X",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="maf_rotation",
            description="moving average factor rotation",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="moving_average_rotation",
            description="generic moving average rotation block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_rotation",
            description="researcher supplied rotation block",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
