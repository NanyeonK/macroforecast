from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def build_run_manifest(*, run_id: str, experiment_id: str, config_hash: str, code_version: str, dataset_ids: list[str], benchmark_ids: list[str], artifact_paths: dict[str, str], success: bool = True, failure_summary: list[str] | None = None, environment_fingerprint: dict[str, Any] | None = None, degraded: bool = False, provenance_fields: list[str] | None = None, recipe_id: str | None = None, taxonomy_path: dict[str, Any] | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        'run_id': run_id,
        'experiment_id': experiment_id,
        'recipe_id': recipe_id,
        'taxonomy_path': taxonomy_path or {},
        'config_hash': config_hash,
        'code_version': code_version,
        'dataset_ids': dataset_ids,
        'benchmark_ids': benchmark_ids,
        'artifact_paths': artifact_paths,
        'success': success,
        'degraded': degraded,
        'failure_summary': failure_summary or [],
        'environment_fingerprint': environment_fingerprint or {},
        'provenance_fields': provenance_fields or [],
        'generated_at': now,
    }


def write_run_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return out
