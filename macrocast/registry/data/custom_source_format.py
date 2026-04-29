from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='custom_source_format',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='no custom source file is used',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='csv',
            description='custom source file is a csv panel',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='parquet',
            description='custom source file is a parquet panel',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
