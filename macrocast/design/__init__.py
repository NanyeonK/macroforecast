"""Backward-compatible experiment spec aliases.

Package-facing imports should prefer macrocast.specs.
"""

from macrocast.design.resolver import ResolvedExperimentSpec, resolve_experiment_spec, resolve_experiment_spec_from_dict

__all__ = [
    'ResolvedExperimentSpec',
    'resolve_experiment_spec',
    'resolve_experiment_spec_from_dict',
]
