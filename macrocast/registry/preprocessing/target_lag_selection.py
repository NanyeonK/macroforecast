from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="target_lag_selection",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="none",
            description="no target-lag feature selection",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="fixed",
            description="fixed target-lag feature count",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="ic_select",
            description="information-criterion selected target-lag feature count",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="cv_select",
            description="cross-validation selected target-lag feature count",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="horizon_specific",
            description="horizon-specific target-lag feature selection",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom",
            description="researcher supplied target-lag feature selection",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
