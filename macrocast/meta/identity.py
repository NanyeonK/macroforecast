from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r'[^a-z0-9]+', '_', value)
    return value.strip('_')


def make_experiment_id(*, target_id: str, dataset_id: str, sample_id: str, preprocess_id: str, split_id: str, benchmark_id: str) -> str:
    return '__'.join(map(_slugify, [target_id, dataset_id, sample_id, preprocess_id, split_id, benchmark_id]))


def make_run_id(*, experiment_id: str, feature_set_id: str, model_set_id: str, tuning_policy_id: str, code_version: str) -> str:
    return '__'.join([
        experiment_id,
        _slugify(feature_set_id),
        _slugify(model_set_id),
        _slugify(tuning_policy_id),
        _slugify(code_version),
    ])


def make_cell_id(*, run_id: str, horizon: int, model_id: str, feature_recipe_id: str) -> str:
    return '__'.join([
        run_id,
        f'h{int(horizon):02d}',
        _slugify(model_id),
        _slugify(feature_recipe_id),
    ])


def make_config_hash(payload: dict[str, Any]) -> str:
    serial = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serial.encode('utf-8')).hexdigest()[:16]
