from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='information_set_type',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='revised',
            description='revised',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='pseudo_oos_revised',
            description='pseudo oos revised',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='pseudo_oos_vintage_aware',
            description='pseudo oos vintage aware',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
