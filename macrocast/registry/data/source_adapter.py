from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='source_adapter',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fred_md',
            description='fred md source',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fred_qd',
            description='fred qd source',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fred_sd',
            description='fred sd source',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='custom_csv',
            description='custom csv',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='custom_parquet',
            description='custom parquet',
            status="operational",
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
