from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='model_family',
    layer='3_training',
    axis_type='enum',
    default_policy='sweep',
    entries=(
        EnumRegistryEntry(
            id='ar',
            description='ar',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='ridge',
            description='ridge',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='lasso',
            description='lasso',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='elasticnet',
            description='elasticnet',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='randomforest',
            description='randomforest',
            status='operational',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
