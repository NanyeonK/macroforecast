from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='variable_universe',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='all_variables',
            description='use all variables in the dataset',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='preselected_core',
            description='preselected core macro indicators',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='category_subset',
            description='subset by FRED category',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='paper_replication_subset',
            description='paper-specific replication variable subset',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_specific_subset',
            description='subset chosen per target',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='expert_curated_subset',
            description='expert-curated subset',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='stability_filtered_subset',
            description='subset filtered by series stability test',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='correlation_screened_subset',
            description='subset chosen by correlation screening',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='feature_selection_dynamic_subset',
            description='dynamic subset selected per fold',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
