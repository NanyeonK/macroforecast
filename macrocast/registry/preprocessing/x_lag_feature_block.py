from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="x_lag_feature_block",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="none",
            description="no X lag feature block",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="fixed_x_lags",
            description="fixed X lag feature block",
            status="registry_only",
            priority="A",
        ),
        EnumRegistryEntry(
            id="variable_specific_x_lags",
            description="variable-specific X lag feature block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="category_specific_x_lags",
            description="category-specific X lag feature block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="cv_selected_x_lags",
            description="cross-validation selected X lag feature block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_x_lags",
            description="researcher supplied X lag feature block",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="preprocessing",
)
