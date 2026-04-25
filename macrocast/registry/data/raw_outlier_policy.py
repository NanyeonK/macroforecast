from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="raw_outlier_policy",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="preserve_raw_outliers",
            description="leave raw-source outliers unchanged before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="winsorize_raw",
            description="winsorize raw numeric columns before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="iqr_clip_raw",
            description="clip raw numeric columns by IQR fences before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="mad_clip_raw",
            description="clip raw numeric columns by MAD fences before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="zscore_clip_raw",
            description="clip raw numeric columns by z-score bounds before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="raw_outlier_to_missing",
            description="convert raw numeric outliers to missing before official transforms",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
