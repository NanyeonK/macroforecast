from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='recipe_mode',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fixed_recipe',
            description='fixed recipe',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='recipe_grid',
            description='recipe grid',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='recipe_ablation',
            description='recipe ablation',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='paper_exact_recipe',
            description='paper exact recipe',
            status='registry_only',
            priority='A',
        ),
        EnumRegistryEntry(
            id='model_specific_recipe',
            description='model specific recipe',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
