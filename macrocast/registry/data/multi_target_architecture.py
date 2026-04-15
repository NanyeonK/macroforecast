from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='multi_target_architecture',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='separate_univariate_runs',
            description='separate univariate runs',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='same_design_different_targets',
            description='same design different targets',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='joint_multivariate_model',
            description='joint multivariate model',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='multitask_shared_representation',
            description='multitask shared representation',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
