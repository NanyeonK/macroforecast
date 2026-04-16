from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='report_style',
    layer='4_evaluation',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='tidy_dataframe', description='tidy dataframe', status='operational', priority='A'),
        EnumRegistryEntry(id='latex_table', description='latex table', status='planned', priority='A'),
        EnumRegistryEntry(id='markdown_table', description='markdown table', status='planned', priority='A'),
        EnumRegistryEntry(id='plot_dashboard', description='plot dashboard', status='registry_only', priority='B'),
        EnumRegistryEntry(id='paper_ready_bundle', description='paper ready bundle', status='registry_only', priority='B'),
    ),
    compatible_with={},
    incompatible_with={},
)
