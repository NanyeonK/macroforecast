from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _recipes_root() -> Path:
    return _repo_root() / 'recipes'


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'recipe YAML at {path} must decode to dict')
    return data


def load_recipe_schema() -> dict[str, Any]:
    return _load_yaml(_recipes_root() / 'schema' / 'recipe_schema.yaml')


def load_recipe(relative_path: str) -> dict[str, Any]:
    return _load_yaml(_recipes_root() / relative_path)


def list_recipe_files() -> list[str]:
    root = _recipes_root()
    files = []
    for sub in ['papers', 'baselines', 'benchmarks', 'ablations']:
        for path in sorted((root / sub).glob('*.yaml')):
            files.append(str(path.relative_to(root)))
    return files
