from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='horizon_modelization',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='separate_model_per_h',
            description='separate model per h',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='shared_model_multi_h',
            description='shared model multi h',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='shared_backbone_multi_head',
            description='shared backbone multi head',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='recursive_one_step_model',
            description='recursive one step model',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='hybrid_h_specific',
            description='hybrid h specific',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
