from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_interpretation_registry() -> dict[str, Any]:
    path = _repo_root() / 'config' / 'interpretation.yaml'
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError('interpretation registry must decode to a dict')
    return data


def validate_interpretation_registry(registry: dict[str, Any]) -> dict[str, Any]:
    items = registry.get('interpretation_methods')
    if not isinstance(items, list) or not items:
        raise ValueError('interpretation_methods must be a non-empty list')
    seen: set[str] = set()
    for item in items:
        required = {'id', 'family', 'entrypoint', 'required_inputs', 'produced_outputs', 'compatibility_rules', 'provenance_fields', 'extension_slot', 'notes'}
        missing = required - set(item)
        if missing:
            raise ValueError(f'interpretation entry missing fields: {sorted(missing)}')
        if item['id'] in seen:
            raise ValueError(f'duplicate interpretation method id: {item["id"]}')
        seen.add(item['id'])
        compat = item['compatibility_rules']
        if 'allowed_model_families' not in compat or not isinstance(compat['allowed_model_families'], list) or not compat['allowed_model_families']:
            raise ValueError('interpretation compatibility must include non-empty allowed_model_families')
        if item['extension_slot'] != 'interpretation_method':
            raise ValueError('interpretation extension_slot must be interpretation_method')
    return registry
