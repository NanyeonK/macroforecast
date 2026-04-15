from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='primary_metric',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='msfe',
            description='msfe',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='relative_msfe',
            description='relative msfe',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='oos_r2',
            description='oos r2',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='csfe',
            description='csfe',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
