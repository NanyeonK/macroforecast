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
            description="no predictor-lag feature block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="fixed_predictor_lags",
            description="fixed predictor-lag feature block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="variable_specific_predictor_lags",
            description="variable-specific predictor-lag feature block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="category_specific_predictor_lags",
            description="category-specific predictor-lag feature block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="cv_selected_predictor_lags",
            description="cross-validation selected predictor-lag feature block",
            status="registry_only",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_predictor_lags",
            description="researcher supplied predictor-lag feature block",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
