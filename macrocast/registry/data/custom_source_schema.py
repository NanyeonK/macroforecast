from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='custom_source_schema',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='no custom source schema because no custom source file is used',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='fred_md',
            description='custom file follows a FRED-MD-like monthly macro panel schema',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='fred_qd',
            description='custom file follows a FRED-QD-like quarterly macro panel schema',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='fred_sd',
            description='custom file follows a FRED-SD-like state-level panel schema',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
