from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='export_format',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='json', description='JSON format for structured artifacts', status='operational', priority='A'),
        EnumRegistryEntry(id='csv', description='CSV format for tabular artifacts', status='operational', priority='A'),
        EnumRegistryEntry(id='parquet', description='Parquet format for large tabular artifacts', status='operational', priority='A'),
        EnumRegistryEntry(id='json_csv', description='JSON for metadata, CSV for tabular', status='operational', priority='A'),
        EnumRegistryEntry(id='json_parquet', description='JSON for metadata, parquet for tabular', status='operational', priority='A'),
        EnumRegistryEntry(id='latex_tables', description='paper-ready LaTeX tables', status='operational', priority='A'),
        EnumRegistryEntry(id='markdown_report', description='Markdown summary report', status='operational', priority='A'),
        EnumRegistryEntry(id='html_report', description='HTML summary report', status='operational', priority='A'),
        EnumRegistryEntry(id='all', description='all supported artifact formats', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
