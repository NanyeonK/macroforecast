from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry

AXIS_DEFINITION = AxisDefinition(
    axis_name='provenance_fields',
    layer='5_output_provenance',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(id='recipe_yaml_full', description='full recipe YAML', status='operational', priority='A'),
        EnumRegistryEntry(id='recipe_hash', description='canonical recipe hash', status='operational', priority='A'),
        EnumRegistryEntry(id='package_version', description='macrocast package version', status='operational', priority='A'),
        EnumRegistryEntry(id='python_version', description='Python version', status='operational', priority='A'),
        EnumRegistryEntry(id='r_version', description='R version', status='operational', priority='A'),
        EnumRegistryEntry(id='julia_version', description='Julia version', status='operational', priority='B'),
        EnumRegistryEntry(id='dependency_lockfile', description='dependency lockfile content', status='operational', priority='A'),
        EnumRegistryEntry(id='git_commit_sha', description='git commit SHA', status='operational', priority='A'),
        EnumRegistryEntry(id='git_branch_name', description='git branch name', status='operational', priority='A'),
        EnumRegistryEntry(id='data_revision_tag', description='data revision tag', status='operational', priority='A'),
        EnumRegistryEntry(id='random_seed_used', description='random seed used', status='operational', priority='A'),
        EnumRegistryEntry(id='runtime_environment', description='runtime environment metadata', status='operational', priority='A'),
        EnumRegistryEntry(id='runtime_duration', description='runtime duration', status='operational', priority='A'),
        EnumRegistryEntry(id='cell_resolved_axes', description='resolved axes by cell', status='operational', priority='A'),
    ),
    compatible_with={},
    incompatible_with={},
)
