from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="raw_missing_policy",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="preserve_raw_missing",
            description="leave raw-source missing values unchanged before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="zero_fill_leading_x_before_tcode",
            description="zero-fill predictor leading missing values in the raw source panel before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="x_impute_raw",
            description="impute raw predictor missing values before official transforms",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="drop_rows_with_raw_missing",
            description="drop rows with any raw-source missing values before official transforms",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
