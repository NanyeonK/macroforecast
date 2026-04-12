from __future__ import annotations

from pathlib import Path
from typing import Any

from macrocast.design.resolver import ResolvedExperimentSpec
from macrocast.design.resolver import resolve_experiment_spec as _resolve_experiment_spec
from macrocast.design.resolver import resolve_experiment_spec_from_dict as _resolve_experiment_spec_from_dict
from macrocast.recipes import load_recipe, load_recipe_schema, recipe_to_runtime_config, validate_recipe, validate_recipe_schema

CompiledExperimentSpec = ResolvedExperimentSpec


def validate_compiled_experiment_spec(spec: CompiledExperimentSpec) -> CompiledExperimentSpec:
    return spec.validate_compiled_spec()


def compile_experiment_spec(path: str | Path, *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    return validate_compiled_experiment_spec(_resolve_experiment_spec(path, preset_id=preset_id, experiment_overrides=experiment_overrides))


def compile_experiment_spec_from_dict(raw: dict[str, Any], *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    return validate_compiled_experiment_spec(_resolve_experiment_spec_from_dict(raw, preset_id=preset_id, experiment_overrides=experiment_overrides))


def compile_experiment_spec_from_recipe(recipe_path: str, *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    schema = validate_recipe_schema(load_recipe_schema())
    recipe = validate_recipe(load_recipe(recipe_path), schema)
    raw = recipe_to_runtime_config(recipe)
    if 'benchmark' in recipe['taxonomy_path']:
        raw['benchmark_family'] = recipe['taxonomy_path']['benchmark']
        raw['benchmark_options'] = recipe.get('benchmark_options', {})
    compiled = compile_experiment_spec_from_dict(raw, preset_id=preset_id, experiment_overrides=experiment_overrides)
    compiled.meta_config['recipe_id'] = recipe['recipe_id']
    compiled.meta_config['taxonomy_path'] = recipe['taxonomy_path']
    return compiled

__all__ = ['CompiledExperimentSpec','validate_compiled_experiment_spec','compile_experiment_spec','compile_experiment_spec_from_dict','compile_experiment_spec_from_recipe']
