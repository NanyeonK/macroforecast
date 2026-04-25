from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='decomposition_target',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='nonlinearity_effect', description='nonlinearity effect', status='registry_only', priority='B'),
        EnumRegistryEntry(id='regularization_effect', description='regularization effect', status='registry_only', priority='B'),
        EnumRegistryEntry(id='cv_scheme_effect', description='cv scheme effect', status='registry_only', priority='B'),
        EnumRegistryEntry(id='loss_function_effect', description='loss function effect', status='registry_only', priority='B'),
        EnumRegistryEntry(id='preprocessing_effect', description='preprocessing effect', status="operational", priority='A'),
        EnumRegistryEntry(id='feature_representation_effect', description='feature representation effect', status="operational", priority='A'),
        EnumRegistryEntry(id='feature_builder_effect', description='legacy alias for feature representation effect', status="operational", priority='A'),
        EnumRegistryEntry(id='benchmark_effect', description='benchmark effect', status="operational", priority='A'),
        EnumRegistryEntry(id='importance_method_effect', description='importance method effect', status='registry_only', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
