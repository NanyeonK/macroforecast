from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='training_start_rule',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='earliest_possible',
            description='earliest possible (default)',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_start',
            description='fixed start (leaf_config.training_start_date)',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
