from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="fred_sd_state_group",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="all_states",
            description="load all FRED-SD states unless state_selection explicitly provides sd_states",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_region_northeast",
            description="load FRED-SD states in the U.S. Census Northeast region",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_region_midwest",
            description="load FRED-SD states in the U.S. Census Midwest region",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_region_south",
            description="load FRED-SD states in the U.S. Census South region",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_region_west",
            description="load FRED-SD states in the U.S. Census West region",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_new_england",
            description="load FRED-SD states in the New England division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_middle_atlantic",
            description="load FRED-SD states in the Middle Atlantic division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_east_north_central",
            description="load FRED-SD states in the East North Central division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_west_north_central",
            description="load FRED-SD states in the West North Central division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_south_atlantic",
            description="load FRED-SD states in the South Atlantic division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_east_south_central",
            description="load FRED-SD states in the East South Central division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_west_south_central",
            description="load FRED-SD states in the West South Central division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_mountain",
            description="load FRED-SD states in the Mountain division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="census_division_pacific",
            description="load FRED-SD states in the Pacific division",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="contiguous_48_plus_dc",
            description="load the contiguous 48 states plus DC, excluding Alaska and Hawaii",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="custom_state_group",
            description="load a user-provided state group from leaf_config.sd_state_group_members or sd_state_groups",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
