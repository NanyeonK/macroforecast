from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(name: str) -> dict[str, Any]:
    path = _repo_root() / 'config' / name
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'{name} must decode to a dict')
    return data


def load_feature_registry() -> dict[str, Any]:
    return _load_yaml('features.yaml')


def load_model_registry() -> dict[str, Any]:
    return _load_yaml('models.yaml')


def get_feature_defaults(registry: dict[str, Any], feature_id: str) -> dict[str, Any]:
    for item in registry.get('features', []):
        if item['id'] == feature_id:
            return dict(item.get('defaults', {}))
    raise KeyError(f'unknown feature id: {feature_id}')


def get_model_defaults(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    for item in registry.get('models', []):
        if item['id'] == model_id:
            return dict(item.get('defaults', {}))
    raise KeyError(f'unknown model id: {model_id}')


def get_feature_entry(registry: dict[str, Any], feature_id: str) -> dict[str, Any]:
    for item in registry.get('features', []):
        if item['id'] == feature_id:
            return dict(item)
    raise KeyError(f'unknown feature id: {feature_id}')


def get_model_entry(registry: dict[str, Any], model_id: str) -> dict[str, Any]:
    for item in registry.get('models', []):
        if item['id'] == model_id:
            return dict(item)
    raise KeyError(f'unknown model id: {model_id}')


def validate_feature_model_compatibility(feature_registry: dict[str, Any], model_registry: dict[str, Any]) -> list[tuple[str, str]]:
    feature_items = feature_registry.get('features', [])
    model_items = model_registry.get('models', [])
    compatible: list[tuple[str, str]] = []
    for feature in feature_items:
        allowed = set(feature['compatibility']['allowed_model_families'])
        for model in model_items:
            if model['family'] in allowed and feature['family'] in set(model['compatibility']['allowed_feature_families']):
                compatible.append((feature['id'], model['id']))
    if not compatible:
        raise ValueError('feature/model registries define no compatible pairs')
    return compatible


def validate_feature_registry(registry: dict[str, Any]) -> dict[str, Any]:
    items = registry.get('features')
    if not isinstance(items, list) or not items:
        raise ValueError('features must be a non-empty list')
    seen: set[str] = set()
    for item in items:
        required = {'id', 'family', 'description', 'factor_type', 'default_n_factors', 'default_n_lags', 'supports_levels', 'supports_marx', 'defaults', 'compatibility', 'extension_slot', 'notes'}
        missing = required - set(item)
        if missing:
            raise ValueError(f'feature entry missing fields: {sorted(missing)}')
        if item['id'] in seen:
            raise ValueError(f'duplicate feature id: {item["id"]}')
        seen.add(item['id'])
        if not isinstance(item['defaults'], dict):
            raise ValueError('feature defaults must be a dict')
        if not isinstance(item['compatibility'], dict):
            raise ValueError('feature compatibility must be a dict')
        for key in ['allowed_model_families', 'requires_target_preprocess_families', 'requires_x_preprocess_families']:
            if key not in item['compatibility'] or not isinstance(item['compatibility'][key], list) or not item['compatibility'][key]:
                raise ValueError(f'feature compatibility missing non-empty list: {key}')
        if item['extension_slot'] != 'feature_recipe':
            raise ValueError('feature extension_slot must be feature_recipe')
    return registry


def validate_model_registry(registry: dict[str, Any]) -> dict[str, Any]:
    items = registry.get('models')
    if not isinstance(items, list) or not items:
        raise ValueError('models must be a non-empty list')
    seen: set[str] = set()
    for item in items:
        required = {'id', 'family', 'backend', 'supports_multihorizon', 'supports_featureless', 'supports_interpretation', 'defaults', 'required_inputs', 'compatibility', 'extension_slot', 'notes'}
        missing = required - set(item)
        if missing:
            raise ValueError(f'model entry missing fields: {sorted(missing)}')
        if item['id'] in seen:
            raise ValueError(f'duplicate model id: {item["id"]}')
        seen.add(item['id'])
        if not isinstance(item['defaults'], dict):
            raise ValueError('model defaults must be a dict when provided')
        if not isinstance(item['required_inputs'], list) or not item['required_inputs']:
            raise ValueError('model required_inputs must be a non-empty list')
        if not isinstance(item['compatibility'], dict):
            raise ValueError('model compatibility must be a dict')
        for key in ['allowed_feature_families']:
            if key not in item['compatibility'] or not isinstance(item['compatibility'][key], list) or not item['compatibility'][key]:
                raise ValueError(f'model compatibility missing non-empty list: {key}')
        if item['extension_slot'] != 'model_wrapper':
            raise ValueError('model extension_slot must be model_wrapper')
    return registry
