from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='dataset',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fred_md',
            description='fred md',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='fred_qd',
            description='fred qd',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='fred_sd',
            description='fred sd',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
