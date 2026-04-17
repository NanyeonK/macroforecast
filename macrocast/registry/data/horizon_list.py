from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='horizon_list',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='arbitrary_grid',
            description='arbitrary horizon grid from leaf_config.horizons',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='default_1_3_6_12',
            description='default monthly horizons {1,3,6,12}',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='short_only_1_3',
            description='short horizons only {1,3}',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='long_only_12_24',
            description='long horizons only {12,24}',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='paper_specific',
            description='paper-specific horizon list',
            status='registry_only',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
