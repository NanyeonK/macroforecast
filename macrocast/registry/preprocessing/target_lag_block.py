from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="target_lag_block",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="none",
            description="no target lag block",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="fixed_target_lags",
            description="fixed target lag block",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="ic_selected_target_lags",
            description="information-criterion selected target lag block",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="horizon_specific_target_lags",
            description="horizon-specific target lag block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_target_lags",
            description="researcher supplied target lag block",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
