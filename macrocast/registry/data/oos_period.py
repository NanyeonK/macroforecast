from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='oos_period',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='all_oos_data',
            description='all oos data (no filter, default)',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='recession_only_oos',
            description='recession only oos',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='expansion_only_oos',
            description='expansion only oos',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
