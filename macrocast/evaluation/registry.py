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
        raise TypeError(f'{name} registry must decode to a dict')
    return data


def load_evaluation_registry() -> dict[str, Any]:
    return _load_yaml('evaluation.yaml')


def load_test_registry() -> dict[str, Any]:
    return _load_yaml('tests.yaml')


def validate_evaluation_registry(registry: dict[str, Any]) -> dict[str, Any]:
    items = registry.get('metric_suites')
    if not isinstance(items, list) or not items:
        raise ValueError('metric_suites must be a non-empty list')
    seen: set[str] = set()
    for item in items:
        required = {'id', 'family', 'metrics', 'aggregation', 'benchmark_requirement', 'provenance_fields', 'notes'}
        missing = required - set(item)
        if missing:
            raise ValueError(f'evaluation suite missing fields: {sorted(missing)}')
        if item['id'] in seen:
            raise ValueError(f'duplicate metric suite id: {item["id"]}')
        seen.add(item['id'])
        if not isinstance(item['metrics'], list) or not item['metrics']:
            raise ValueError('metric suite metrics must be non-empty list')
        if not isinstance(item['aggregation'], dict):
            raise ValueError('metric suite aggregation must be dict')
    return registry


def validate_test_registry(registry: dict[str, Any]) -> dict[str, Any]:
    suites = registry.get('test_suites')
    if not isinstance(suites, list) or not suites:
        raise ValueError('test_suites must be a non-empty list')
    suite_seen: set[str] = set()
    for suite in suites:
        required_suite = {'id', 'family', 'members', 'multiple_comparison_correction', 'hac_or_bootstrap', 'provenance_fields', 'notes'}
        missing_suite = required_suite - set(suite)
        if missing_suite:
            raise ValueError(f'test suite missing fields: {sorted(missing_suite)}')
        if suite['id'] in suite_seen:
            raise ValueError(f'duplicate test suite id: {suite["id"]}')
        suite_seen.add(suite['id'])
        if not isinstance(suite['members'], list) or not suite['members']:
            raise ValueError('test suite members must be non-empty list')

    items = registry.get('statistical_tests')
    if not isinstance(items, list) or not items:
        raise ValueError('statistical_tests must be a non-empty list')
    seen: set[str] = set()
    for item in items:
        required = {'id', 'family', 'entrypoint', 'required_inputs', 'produced_outputs', 'compatibility_rules', 'provenance_fields', 'notes'}
        missing = required - set(item)
        if missing:
            raise ValueError(f'test entry missing fields: {sorted(missing)}')
        if item['id'] in seen:
            raise ValueError(f'duplicate statistical test id: {item["id"]}')
        seen.add(item['id'])
    return registry
