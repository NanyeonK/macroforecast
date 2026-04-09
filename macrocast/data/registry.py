from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path, *, label: str) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'{label} must decode to a dict')
    return data


def load_dataset_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'datasets.yaml', label='dataset registry')


def load_target_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'targets.yaml', label='target registry')


def load_data_task_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'data_tasks.yaml', label='data task registry')


def _index_by_id(items: list[dict[str, Any]], item_id: str, *, label: str) -> dict[str, Any]:
    for item in items:
        if item['id'] == item_id:
            return item
    raise KeyError(f'unknown {label}: {item_id}')


def get_dataset_defaults(registry: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    datasets = registry.get('datasets', [])
    item = _index_by_id(datasets, dataset_id, label='dataset id')
    return dict(item.get('defaults', {}))


def get_target_defaults(registry: dict[str, Any], target_id: str, *, dataset_id: str | None = None) -> dict[str, Any]:
    targets = registry.get('targets', [])
    fallback: dict[str, Any] = {}
    for item in targets:
        applicable = item.get('applicable_datasets', [])
        dataset_ok = dataset_id is None or not applicable or dataset_id in applicable
        if item['id'] == '__default__' and dataset_ok:
            fallback = dict(item.get('defaults', {}))
        if item['id'] == target_id and dataset_ok:
            out = dict(fallback)
            out.update(dict(item.get('defaults', {})))
            return out
    return fallback


def get_data_task_defaults(registry: dict[str, Any], task_id: str | None = None, *, dataset_id: str | None = None) -> dict[str, Any]:
    tasks = registry.get('data_tasks', [])
    fallback: dict[str, Any] = {}
    for item in tasks:
        applicable = item.get('applicable_datasets', [])
        dataset_ok = dataset_id is None or not applicable or dataset_id in applicable
        if item['id'] == '__default__' and dataset_ok:
            fallback = dict(item.get('defaults', {}))
        if task_id is not None and item['id'] == task_id and dataset_ok:
            out = dict(fallback)
            out.update(dict(item.get('defaults', {})))
            return out
    return fallback


def validate_dataset_registry(registry: dict[str, Any]) -> dict[str, Any]:
    datasets = registry.get('datasets')
    if not isinstance(datasets, list) or not datasets:
        raise ValueError('datasets must be a non-empty list')
    seen: set[str] = set()
    required = {
        'id', 'family', 'frequency', 'source_type', 'loader', 'supports_vintages',
        'default_vintage_mode', 'cache_subdir', 'target_style', 'notes'
    }
    for item in datasets:
        if not isinstance(item, dict):
            raise ValueError('each dataset registry entry must be a dict')
        missing = required - set(item)
        if missing:
            raise ValueError(f'dataset registry entry missing fields: {sorted(missing)}')
        dataset_id = item['id']
        if dataset_id in seen:
            raise ValueError(f'duplicate dataset id: {dataset_id}')
        seen.add(dataset_id)
        defaults = item.get('defaults', {})
        if defaults and not isinstance(defaults, dict):
            raise ValueError('dataset defaults must be a dict when provided')
        if defaults and 'task_id' not in defaults:
            raise ValueError('dataset defaults must include task_id for package-facing task resolution')
    return registry


def validate_target_registry(registry: dict[str, Any]) -> dict[str, Any]:
    targets = registry.get('targets')
    if not isinstance(targets, list) or not targets:
        raise ValueError('targets must be a non-empty list')
    seen: set[str] = set()
    for item in targets:
        if not isinstance(item, dict):
            raise ValueError('each target registry entry must be a dict')
        for key in ['id', 'applicable_datasets', 'defaults', 'notes']:
            if key not in item:
                raise ValueError(f'target registry entry missing field: {key}')
        item_id = item['id']
        if item_id != '__default__' and item_id in seen:
            raise ValueError(f'duplicate target id: {item_id}')
        seen.add(item_id)
        if not isinstance(item['applicable_datasets'], list):
            raise ValueError('target applicable_datasets must be a list')
        if not isinstance(item['defaults'], dict):
            raise ValueError('target defaults must be a dict')
        if 'evaluation_scale' not in item['defaults']:
            raise ValueError('target defaults must include evaluation_scale')
    return registry


def validate_data_task_registry(registry: dict[str, Any]) -> dict[str, Any]:
    tasks = registry.get('data_tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ValueError('data_tasks must be a non-empty list')
    seen: set[str] = set()
    for item in tasks:
        if not isinstance(item, dict):
            raise ValueError('each data task registry entry must be a dict')
        for key in ['id', 'applicable_datasets', 'defaults', 'notes']:
            if key not in item:
                raise ValueError(f'data task registry entry missing field: {key}')
        item_id = item['id']
        if item_id != '__default__' and item_id in seen:
            raise ValueError(f'duplicate data task id: {item_id}')
        seen.add(item_id)
        if not isinstance(item['applicable_datasets'], list):
            raise ValueError('data task applicable_datasets must be a list')
        defaults = item['defaults']
        if not isinstance(defaults, dict):
            raise ValueError('data task defaults must be a dict')
        for required in ['sample_period', 'estimation_start', 'oos_period', 'minimum_train_size', 'validation_design', 'outer_window', 'refit_policy', 'horizon_grid_default']:
            if required not in defaults:
                raise ValueError(f'data task defaults missing field: {required}')
        if not isinstance(defaults['sample_period'], dict) or not isinstance(defaults['oos_period'], dict):
            raise ValueError('data task sample_period and oos_period must be dicts')
        if not isinstance(defaults['horizon_grid_default'], list) or not defaults['horizon_grid_default']:
            raise ValueError('data task horizon_grid_default must be a non-empty list')
    return registry
