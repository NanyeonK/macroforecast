from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='search_algorithm',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='grid_search',
            description='grid search',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='random_search',
            description='random search',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='bayesian_optimization',
            description='bayesian optimization',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='genetic_algorithm',
            description='genetic algorithm',
            status='operational',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
