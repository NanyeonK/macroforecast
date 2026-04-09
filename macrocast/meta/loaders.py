from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'YAML at {path} must decode to a dict')
    return data


def load_axes_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'meta' / 'axes.yaml')


def load_benchmark_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'meta' / 'benchmarks.yaml')


def load_preset_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'meta' / 'presets.yaml')


def load_execution_policy_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'meta' / 'execution.yaml')


def load_global_defaults_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'meta' / 'global_defaults.yaml')


def load_extension_registry() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'meta' / 'extensions.yaml')


def load_naming_policy() -> dict[str, Any]:
    return _load_yaml(_repo_root() / 'config' / 'meta' / 'naming.yaml')
