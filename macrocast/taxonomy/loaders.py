from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


TAXONOMY_LAYERS = [
    '0_meta',
    '1_data',
    '2_target_x',
    '3_preprocess',
    '4_training',
    '5_evaluation',
    '6_stat_tests',
    '7_importance',
    '8_output_provenance',
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _taxonomy_root() -> Path:
    return _repo_root() / 'macrocast' / 'taxonomy'


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'taxonomy YAML at {path} must decode to dict')
    return data


def load_taxonomy_file(relative_path: str) -> dict[str, Any]:
    return _load_yaml(_taxonomy_root() / relative_path)


def load_taxonomy_layer(layer: str) -> dict[str, dict[str, Any]]:
    layer_dir = _taxonomy_root() / layer
    if not layer_dir.exists():
        raise FileNotFoundError(f'unknown taxonomy layer: {layer}')
    bundle: dict[str, dict[str, Any]] = {}
    for path in sorted(layer_dir.glob('*.yaml')):
        bundle[path.stem] = _load_yaml(path)
    return bundle


def load_taxonomy_bundle() -> dict[str, dict[str, dict[str, Any]]]:
    return {layer: load_taxonomy_layer(layer) for layer in TAXONOMY_LAYERS}
