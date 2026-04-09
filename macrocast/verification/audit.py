from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def audit_benchmark_registry(benchmark_registry: dict[str, Any], benchmark_id: str) -> bool:
    return any(item.get('id') == benchmark_id for item in benchmark_registry.get('benchmarks', []))


def audit_preprocessing_registry(preprocessing_registry: dict[str, Any], recipe_id: str, family: str = 'x') -> bool:
    root = preprocessing_registry.get('preprocessing', {})
    key = 'x_recipes' if family == 'x' else 'target_recipes'
    return any(item.get('id') == recipe_id for item in root.get(key, []))


def audit_target_mapping(mapping: dict[str, str], target_id: str) -> bool:
    return target_id in mapping and isinstance(mapping[target_id], str) and bool(mapping[target_id])


def load_verification_registry() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[2] / 'config' / 'verification.yaml'
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError('verification registry must decode to a dict')
    return data


def validate_verification_registry(registry: dict[str, Any]) -> dict[str, Any]:
    items = registry.get('verification_audits')
    if not isinstance(items, list) or not items:
        raise ValueError('verification_audits must be a non-empty list')
    seen: set[str] = set()
    for item in items:
        required = {'id', 'family', 'entrypoint', 'required_inputs', 'produced_outputs', 'tolerance_policy', 'notes'}
        missing = required - set(item)
        if missing:
            raise ValueError(f'verification audit missing fields: {sorted(missing)}')
        if item['id'] in seen:
            raise ValueError(f'duplicate verification audit id: {item["id"]}')
        seen.add(item['id'])
    return registry
