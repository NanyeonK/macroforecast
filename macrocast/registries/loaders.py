from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REGISTRY_LAYERS = ['meta', 'data', 'training', 'evaluation', 'output']


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registries_root() -> Path:
    return _repo_root() / 'registries'


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'registry YAML at {path} must decode to dict')
    return data


def load_registry_file(relative_path: str) -> dict[str, Any]:
    return _load_yaml(_registries_root() / relative_path)


def load_registry_layer(layer: str) -> dict[str, dict[str, Any]]:
    layer_dir = _registries_root() / layer
    if not layer_dir.exists():
        raise FileNotFoundError(f'unknown registry layer: {layer}')
    bundle: dict[str, dict[str, Any]] = {}
    for path in sorted(layer_dir.glob('*.yaml')):
        bundle[path.stem] = _load_yaml(path)
    return bundle


def load_registry_bundle() -> dict[str, dict[str, dict[str, Any]]]:
    return {layer: load_registry_layer(layer) for layer in REGISTRY_LAYERS}
