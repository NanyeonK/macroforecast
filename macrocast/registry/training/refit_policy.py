from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='refit_policy',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='refit_every_step',
            description='refit every step',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='refit_every_k_steps',
            description='refit every k steps',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fit_once_predict_many',
            description='fit once predict many',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='warm_start_refit',
            description='warm start refit',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
