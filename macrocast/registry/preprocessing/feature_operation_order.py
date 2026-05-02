from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="feature_operation_order",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="lag_then_pca",
            description="create predictor lags first, then summarize the expanded lag panel with PCA",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="pca_then_factor_lag",
            description="extract PCA factors first, then create lags of those factors",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="pca_then_lagged_factors",
            description="extract PCA factors first and emphasize lagged factor outputs",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="parallel_lag_and_pca",
            description="build predictor lags and PCA factors as parallel feature blocks",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="custom_feature_sequence",
            description="researcher supplied feature operation sequence",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
