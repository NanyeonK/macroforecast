from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='importance_method',
    layer='7_importance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='minimal_importance',
            description='minimal importance',
            status='operational',
            priority="A",
        ),
        EnumRegistryEntry(
            id='shap',
            description='shap',
            status='planned',
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
