from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='saved_objects',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='forecasts', description='forecast artifacts', status='operational', priority='A'),
        EnumRegistryEntry(id='forecast_intervals', description='forecast interval artifacts', status='operational', priority='A'),
        EnumRegistryEntry(id='metrics', description='evaluation metrics', status='operational', priority='A'),
        EnumRegistryEntry(id='ranking', description='evaluation rankings', status='operational', priority='A'),
        EnumRegistryEntry(id='decomposition', description='metric decomposition outputs', status='operational', priority='A'),
        EnumRegistryEntry(id='regime_metrics', description='regime-specific metrics', status='operational', priority='A'),
        EnumRegistryEntry(id='state_metrics', description='state-specific metrics', status='operational', priority='A'),
        EnumRegistryEntry(id='model_artifacts', description='model artifacts', status='operational', priority='B'),
        EnumRegistryEntry(id='combination_weights', description='forecast-combination weights', status='operational', priority='A'),
        EnumRegistryEntry(id='feature_metadata', description='feature metadata', status='operational', priority='B'),
        EnumRegistryEntry(id='clean_panel', description='cleaned panel', status='operational', priority='B'),
        EnumRegistryEntry(id='raw_panel', description='raw panel', status='operational', priority='B'),
        EnumRegistryEntry(id='diagnostics_l1_5', description='L1.5 diagnostics', status='operational', priority='A'),
        EnumRegistryEntry(id='diagnostics_l2_5', description='L2.5 diagnostics', status='operational', priority='A'),
        EnumRegistryEntry(id='diagnostics_l3_5', description='L3.5 diagnostics', status='operational', priority='A'),
        EnumRegistryEntry(id='diagnostics_l4_5', description='L4.5 diagnostics', status='operational', priority='A'),
        EnumRegistryEntry(id='diagnostics_all', description='all active diagnostics', status='operational', priority='A'),
        EnumRegistryEntry(id='tests', description='statistical test outputs', status='operational', priority='A'),
        EnumRegistryEntry(id='importance', description='importance outputs', status='operational', priority='A'),
        EnumRegistryEntry(id='transformation_attribution', description='transformation attribution outputs', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
