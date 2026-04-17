from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='min_train_size',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fixed_n_obs',
            description='fixed n obs',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_years',
            description='fixed years',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='model_specific_min_train',
            description='model specific min train',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='target_specific_min_train',
            description='target specific min train',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='horizon_specific_min_train',
            description='horizon specific min train',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
