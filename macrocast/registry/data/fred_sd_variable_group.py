from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="fred_sd_variable_group",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="all_sd_variables",
            description="load all FRED-SD workbook variable sheets unless sd_variable_selection explicitly provides sd_variables",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="labor_market_core",
            description="load core labor-market FRED-SD variables: claims, labor force, payrolls, participation, unemployment",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="employment_sector",
            description="load sector employment and hours FRED-SD variables",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="gsp_output",
            description="load aggregate and sector gross-state-product or output FRED-SD variables",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="housing",
            description="load permits, rents, and house-price FRED-SD variables",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="trade",
            description="load exports and imports FRED-SD variables",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="income",
            description="load income-related FRED-SD variables",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="direct_analog_high_confidence",
            description="load variables with direct high-confidence national analogs in macrocast's FRED-SD t-code review",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="provisional_analog_medium",
            description="load variables with medium-confidence direct or near-direct analogs in macrocast's FRED-SD t-code review",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="semantic_review_outputs",
            description="load output/GSP variables whose analog status requires semantic review",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="no_reliable_analog",
            description="load variables marked as lacking reliable national analogs in macrocast's FRED-SD t-code review",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="custom_sd_variable_group",
            description="load a user-provided workbook-variable group from leaf_config.sd_variable_group_members or sd_variable_groups",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
