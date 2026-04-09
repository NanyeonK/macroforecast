from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_output_registry() -> dict[str, Any]:
    path = _repo_root() / 'config' / 'output.yaml'
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError('output registry must decode to a dict')
    return data


def validate_output_registry(registry: dict[str, Any]) -> dict[str, Any]:
    root = registry.get('output')
    if not isinstance(root, dict):
        raise ValueError('output root must be a dict')
    for key in ['canonical_formats', 'required_manifest_fields', 'required_artifacts', 'required_failure_fields']:
        if key not in root:
            raise ValueError(f'missing output registry key: {key}')
    return registry
