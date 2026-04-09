from __future__ import annotations

from pathlib import Path
from typing import Any

from macrocast.design.resolver import ResolvedExperimentSpec
from macrocast.design.resolver import resolve_experiment_spec as _resolve_experiment_spec
from macrocast.design.resolver import resolve_experiment_spec_from_dict as _resolve_experiment_spec_from_dict

CompiledExperimentSpec = ResolvedExperimentSpec


def validate_compiled_experiment_spec(spec: CompiledExperimentSpec) -> CompiledExperimentSpec:
    return spec.validate_compiled_spec()


def compile_experiment_spec(path: str | Path, *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    return validate_compiled_experiment_spec(
        _resolve_experiment_spec(path, preset_id=preset_id, experiment_overrides=experiment_overrides)
    )


def compile_experiment_spec_from_dict(raw: dict[str, Any], *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    return validate_compiled_experiment_spec(
        _resolve_experiment_spec_from_dict(raw, preset_id=preset_id, experiment_overrides=experiment_overrides)
    )


__all__ = [
    'CompiledExperimentSpec',
    'validate_compiled_experiment_spec',
    'compile_experiment_spec',
    'compile_experiment_spec_from_dict',
]
