from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_LEARNED_OPERATIONS = {
    'em_fill',
    'mean_impute',
    'median_impute',
    'kalman_fill',
    'dfm_fill',
    'standardize',
    'robust_standardize',
    'minmax_scale',
    'pca',
    'dynamic_factors',
    'supervised_factors',
    'autoencoder',
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_preprocessing_registry() -> dict[str, Any]:
    path = _repo_root() / 'config' / 'preprocessing.yaml'
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError('preprocessing registry must decode to a dict')
    return data


def _root(registry: dict[str, Any]) -> dict[str, Any]:
    root = registry.get('preprocessing')
    if not isinstance(root, dict):
        raise ValueError('preprocessing root must be a dict')
    return root


def get_target_recipe(registry: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    for item in _root(registry).get('target_recipes', []):
        if item['id'] == recipe_id:
            return dict(item)
    raise KeyError(f'unknown target recipe id: {recipe_id}')


def get_x_recipe(registry: dict[str, Any], recipe_id: str) -> dict[str, Any]:
    for item in _root(registry).get('x_recipes', []):
        if item['id'] == recipe_id:
            return dict(item)
    raise KeyError(f'unknown x recipe id: {recipe_id}')


def validate_preprocessing_registry(registry: dict[str, Any]) -> dict[str, Any]:
    root = _root(registry)
    required_root = {'target_recipes', 'x_recipes', 'policies'}
    missing_root = required_root - set(root)
    if missing_root:
        raise ValueError(f'preprocessing root missing keys: {sorted(missing_root)}')

    def _validate_recipe_list(items: Any, family: str) -> None:
        if not isinstance(items, list) or not items:
            raise ValueError(f'{family}_recipes must be a non-empty list')
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f'{family} recipe entries must be dicts')
            required = {'id', 'family', 'description', 'operation_order', 'fit_scope', 'learned_parameters', 'allowed_datasets', 'notes'}
            missing = required - set(item)
            if missing:
                raise ValueError(f'{family} recipe missing fields: {sorted(missing)}')
            if item['family'] != family:
                raise ValueError(f"{family} recipe family mismatch: {item['id']}")
            if item['id'] in seen:
                raise ValueError(f'duplicate {family} recipe id: {item["id"]}')
            seen.add(item['id'])
            if not isinstance(item['operation_order'], list) or not item['operation_order']:
                raise ValueError(f'{family} recipe {item["id"]} must define non-empty operation_order')
            if item['fit_scope'] not in {'none', 'train_window_only'}:
                raise ValueError(f'{family} recipe {item["id"]} invalid fit_scope')
            if not isinstance(item['learned_parameters'], bool):
                raise ValueError(f'{family} recipe {item["id"]} learned_parameters must be bool')
            if not isinstance(item['allowed_datasets'], list) or not item['allowed_datasets']:
                raise ValueError(f'{family} recipe {item["id"]} must define allowed_datasets')
            if item['learned_parameters'] and item['fit_scope'] != 'train_window_only':
                raise ValueError(f'{family} recipe {item["id"]} with learned parameters must use train_window_only fit_scope')
            if any(op in _LEARNED_OPERATIONS for op in item['operation_order']) and item['fit_scope'] != 'train_window_only':
                raise ValueError(f'{family} recipe {item["id"]} has learned operation but non-train-only fit_scope')

    _validate_recipe_list(root['target_recipes'], 'target')
    _validate_recipe_list(root['x_recipes'], 'x')

    policies = root['policies']
    if not isinstance(policies, dict):
        raise ValueError('preprocessing.policies must be a dict')
    for key in [
        'target_x_must_be_separate',
        'operation_order_is_explicit',
        'train_only_fit_required_when_fit_scope_present',
        'fit_scope_must_be_train_only_for_learned_steps',
        'target_recipe_key',
        'x_recipe_key',
    ]:
        if key not in policies:
            raise ValueError(f'missing preprocessing policy: {key}')
    if policies['target_recipe_key'] == policies['x_recipe_key']:
        raise ValueError('target and x recipe keys must be distinct')
    return registry
