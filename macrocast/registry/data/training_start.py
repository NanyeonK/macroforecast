from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='training_start_rule',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='earliest_possible',
            description='earliest possible',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_start',
            description='fixed start',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='post_warmup_start',
            description='post warmup start',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='post_break_start',
            description='post break start',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='rolling_train_start',
            description='rolling train start',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
