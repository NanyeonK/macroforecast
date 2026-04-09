from __future__ import annotations

from typing import Any

from macrocast.meta.exceptions import (
    AxisClassificationError,
    BenchmarkRegistryError,
    IllegalOverrideError,
    NamingPolicyError,
    PresetResolutionError,
)

_REQUIRED_BENCHMARK_FIELDS = {
    'id', 'family', 'description', 'denominator_rule', 'target_construction_rule',
    'estimation_window_rule', 'applicable_datasets', 'applicable_targets',
    'applicable_horizons', 'notes'
}

_REQUIRED_PROVENANCE_FIELDS = {
    'failure_stage',
    'exception_class',
    'exception_message',
    'retry_count',
    'cell_id',
    'model_id',
    'horizon',
}

_REQUIRED_SEVERITY_LEVELS = {'warning', 'degraded_run', 'hard_error'}


def _require_keys(data: dict[str, Any], keys: list[str], ctx: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ValueError(f'{ctx} missing required keys: {missing}')


def _axis_classes(reg: dict[str, Any]) -> dict[str, list[str]]:
    if 'axis_classes' in reg:
        axis_classes = reg['axis_classes']
        if not isinstance(axis_classes, dict):
            raise AxisClassificationError('axis_classes must be a dict')
        _require_keys(axis_classes, ['invariant', 'experiment_fixed', 'research_sweep', 'conditional'], 'axis_classes')
        out = axis_classes
    else:
        _require_keys(reg, ['invariant_axes', 'experiment_fixed_axes', 'research_sweep_axes', 'conditional_axes'], 'legacy axis classes')
        out = {
            'invariant': reg['invariant_axes'],
            'experiment_fixed': reg['experiment_fixed_axes'],
            'research_sweep': reg['research_sweep_axes'],
            'conditional': reg['conditional_axes'],
        }
    for name, axes in out.items():
        if not isinstance(axes, list):
            raise AxisClassificationError(f'{name} axes must be a list')
    return out


def validate_axes_registry(reg: dict[str, Any]) -> dict[str, Any]:
    _require_keys(reg, ['unit_of_run', 'axis_classes', 'comparability_rules'], 'axes registry')
    if not isinstance(reg['unit_of_run'], dict):
        raise AxisClassificationError('unit_of_run must be a dict')
    _require_keys(reg['unit_of_run'], ['experiment', 'run', 'cell'], 'unit_of_run')

    classes = _axis_classes(reg)
    seen: dict[str, str] = {}
    for cls_name in ['invariant', 'experiment_fixed', 'research_sweep', 'conditional']:
        for axis in classes[cls_name]:
            if axis in seen:
                raise AxisClassificationError(f'axis {axis!r} appears in both {seen[axis]} and {cls_name}')
            seen[axis] = cls_name

    legacy_map = {
        'experiment_unit': reg.get('experiment_unit'),
        'invariant_axes': reg.get('invariant_axes'),
        'experiment_fixed_axes': reg.get('experiment_fixed_axes'),
        'research_sweep_axes': reg.get('research_sweep_axes'),
        'conditional_axes': reg.get('conditional_axes'),
    }
    expected_legacy = {
        'experiment_unit': reg['unit_of_run'],
        'invariant_axes': classes['invariant'],
        'experiment_fixed_axes': classes['experiment_fixed'],
        'research_sweep_axes': classes['research_sweep'],
        'conditional_axes': classes['conditional'],
    }
    for key, legacy in legacy_map.items():
        if legacy is not None and legacy != expected_legacy[key]:
            raise AxisClassificationError(f'{key} does not mirror canonical axis definitions')
    return reg


def validate_benchmark_registry(reg: dict[str, Any]) -> dict[str, Any]:
    benchmarks = reg.get('benchmarks')
    if not isinstance(benchmarks, list) or not benchmarks:
        raise BenchmarkRegistryError('benchmarks must be a non-empty list')
    seen: set[str] = set()
    for bench in benchmarks:
        if not isinstance(bench, dict):
            raise BenchmarkRegistryError('each benchmark entry must be a dict')
        missing = _REQUIRED_BENCHMARK_FIELDS - set(bench)
        if missing:
            raise BenchmarkRegistryError(f'benchmark entry missing fields: {sorted(missing)}')
        bench_id = bench['id']
        if bench_id in seen:
            raise BenchmarkRegistryError(f'duplicate benchmark id: {bench_id}')
        seen.add(bench_id)
    return reg


def validate_preset_registry(reg: dict[str, Any], invariant_axes: set[str] | None = None, known_axes: set[str] | None = None) -> dict[str, Any]:
    presets = reg.get('presets')
    if not isinstance(presets, list) or not presets:
        raise PresetResolutionError('presets must be a non-empty list')
    invariant_axes = invariant_axes or set()
    seen: set[str] = set()
    for preset in presets:
        _require_keys(preset, ['id', 'mode', 'description', 'base_defaults', 'allowed_overrides', 'notes'], 'preset entry')
        pid = preset['id']
        if pid in seen:
            raise PresetResolutionError(f'duplicate preset id: {pid}')
        seen.add(pid)
        illegal = invariant_axes.intersection(set(preset.get('allowed_overrides', [])))
        if illegal:
            raise PresetResolutionError(f'preset {pid} allows invariant overrides: {sorted(illegal)}')
        if known_axes is not None:
            unknown = set(preset.get('allowed_overrides', [])) - known_axes
            if unknown:
                raise PresetResolutionError(f'preset {pid} allows unknown axes: {sorted(unknown)}')
    return reg


def validate_execution_policy(reg: dict[str, Any]) -> dict[str, Any]:
    policies = reg.get('policies')
    if not isinstance(policies, list) or not policies:
        raise ValueError('policies must be a non-empty list')
    seen: set[str] = set()
    for policy in policies:
        _require_keys(
            policy,
            [
                'id', 'max_retries_per_cell', 'max_failed_cells', 'timeout_per_cell_seconds',
                'timeout_per_run_seconds', 'benchmark_failure_policy', 'config_failure_policy',
                'partial_save_policy', 'canary_mode', 'cell_failure', 'model_failure',
                'partial_horizon_failure', 'severity_levels', 'provenance_fields',
            ],
            'execution policy',
        )
        pid = policy['id']
        if pid in seen:
            raise ValueError(f'duplicate execution policy id: {pid}')
        seen.add(pid)
        if policy['benchmark_failure_policy'] != 'hard_error':
            raise ValueError('benchmark failure policy must be hard_error')
        if policy['config_failure_policy'] != 'hard_error':
            raise ValueError('config failure policy must be hard_error')
        if policy['partial_save_policy'] != 'save_partial_with_flag':
            raise ValueError('partial_save_policy must be save_partial_with_flag')
        for key in ['max_retries_per_cell', 'max_failed_cells', 'timeout_per_cell_seconds', 'timeout_per_run_seconds']:
            if policy[key] < 0:
                raise ValueError(f'{key} must be non-negative')

        cell_failure = policy['cell_failure']
        model_failure = policy['model_failure']
        partial_horizon_failure = policy['partial_horizon_failure']
        if cell_failure.get('default') != 'fail_and_record':
            raise ValueError('cell_failure.default must be fail_and_record')
        allowed_actions = set(cell_failure.get('allowed_actions', []))
        if not allowed_actions.issubset({'retry_once', 'skip_with_failure_record'}):
            raise ValueError('cell_failure.allowed_actions contains unsupported action')
        if model_failure.get('default') != 'continue_other_models_and_mark_run_degraded':
            raise ValueError('model_failure.default must mark degraded run')
        if 'benchmark_model_fails' not in set(model_failure.get('escalate_to_stop_when', [])):
            raise ValueError('model_failure must stop when benchmark model fails')
        if partial_horizon_failure.get('default') != 'save_partial_with_flag':
            raise ValueError('partial_horizon_failure.default must be save_partial_with_flag')
        required_metadata = set(partial_horizon_failure.get('required_metadata', []))
        if not {'failed_horizons', 'skipped_cells', 'warnings'}.issubset(required_metadata):
            raise ValueError('partial_horizon_failure.required_metadata incomplete')
        if not _REQUIRED_SEVERITY_LEVELS.issubset(set(policy.get('severity_levels', []))):
            raise ValueError('severity_levels missing required research-grade values')
        if not _REQUIRED_PROVENANCE_FIELDS.issubset(set(policy.get('provenance_fields', []))):
            raise ValueError('provenance_fields missing required failure tracking fields')
    return reg


def validate_extension_registry(reg: dict[str, Any]) -> dict[str, Any]:
    families = reg.get('extension_families')
    required = reg.get('required_fields')
    provenance = reg.get('provenance_requirements', {}).get('mandatory')
    if not isinstance(families, list) or not families:
        raise ValueError('extension_families must be a non-empty list')
    if not isinstance(required, dict):
        raise ValueError('required_fields must be a dict')
    for fam in families:
        if fam not in required:
            raise ValueError(f'missing required_fields entry for extension family: {fam}')
    if not provenance:
        raise ValueError('provenance_requirements.mandatory must be non-empty')
    return reg


def validate_naming_policy(reg: dict[str, Any]) -> dict[str, Any]:
    _require_keys(reg, ['naming', 'hashing'], 'naming policy')
    naming = reg['naming']
    hashing = reg['hashing']
    _require_keys(naming, ['slug_style', 'experiment_id_format', 'run_id_format', 'cell_id_format', 'display_name_policy'], 'naming')
    _require_keys(hashing, ['config_hash_fields', 'environment_fingerprint_fields'], 'hashing')
    if naming['slug_style'] not in {'snake_case', 'kebab_case'}:
        raise NamingPolicyError('slug_style must be snake_case or kebab_case')
    return reg


def validate_meta_bundle(bundle: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    axes = validate_axes_registry(bundle['axes'])
    classes = _axis_classes(axes)
    validate_benchmark_registry(bundle['benchmarks'])
    validate_execution_policy(bundle['execution'])
    validate_extension_registry(bundle['extensions'])
    validate_naming_policy(bundle['naming'])
    known_axes = set().union(*classes.values())
    validate_preset_registry(bundle['presets'], invariant_axes=set(classes['invariant']), known_axes=known_axes)
    return bundle


def check_override_legality(axis_name: str, axes_registry: dict[str, Any], *, stage: str) -> None:
    classes = _axis_classes(axes_registry)
    invariant = set(classes['invariant'])
    experiment_fixed = set(classes['experiment_fixed'])
    if axis_name in invariant:
        raise IllegalOverrideError(f'cannot override invariant axis {axis_name!r} at stage {stage}')
    if axis_name in experiment_fixed and stage in {'run', 'cell'}:
        raise IllegalOverrideError(f'cannot override experiment-fixed axis {axis_name!r} at stage {stage}')
