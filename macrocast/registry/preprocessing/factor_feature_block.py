from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="factor_feature_block",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="none",
            description="no factor feature block",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="pca_static_factors",
            description="static PCA factors extracted from the predictor panel",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="pca_factor_lags",
            description="static PCA factors plus factor lags",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="supervised_factors",
            description="supervised factor block fit on the training window",
            status="operational",
            priority="B",
        ),
        EnumRegistryEntry(
            id="custom_factors",
            description="researcher supplied factor feature block",
            status="registry_only",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
