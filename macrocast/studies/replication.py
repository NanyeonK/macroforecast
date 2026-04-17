"""Replication runner — frozen recipe + overrides + diff report.

The source recipe is passed in directly (not loaded from artifact metadata)
because ``manifest.json`` stores compiled-spec fields, not the raw YAML dict.
Callers own the source recipe; they typically load it from a YAML or from
the prior study_manifest's ``parent_recipe_dict`` on a sweep.

If ``source_artifact_dir`` is supplied, the runner compares the source run's
``predictions.csv`` and ``metrics.json`` against the replay's outputs and
writes the differences into ``<output_root>/replication_diff.json``. Without
``source_artifact_dir`` the runner still executes and produces the diff
file, but the comparison section is empty.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..compiler.override_diff import apply_overrides

REPLICATION_DIFF_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ReplicationResult:
    source_recipe_id: str
    replayed_recipe_id: str
    overrides_applied: dict[str, Any]
    diff_report_path: str
    byte_identical_predictions: bool
    execution_result: Any  # ExecutionResult (lazy — avoids circular import at module load)


def execute_replication(
    *,
    source_recipe_dict: dict[str, Any],
    overrides: dict[str, Any],
    output_root: str | Path,
    source_artifact_dir: str | Path | None = None,
    local_raw_source: str | Path | None = None,
    provenance_payload: dict[str, Any] | None = None,
) -> ReplicationResult:
    from ..compiler.build import compile_recipe_dict
    from ..execution.build import execute_recipe

    new_dict, diff_entries = apply_overrides(source_recipe_dict, overrides)
    compile_result = compile_recipe_dict(new_dict)
    compiled = compile_result.compiled
    if compiled.execution_status != "executable":
        raise ValueError(
            f"replicated recipe is not executable: "
            f"status={compiled.execution_status!r} "
            f"warnings={list(compiled.warnings)} "
            f"blocked={list(compiled.blocked_reasons)}"
        )

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    execution = execute_recipe(
        recipe=compiled.recipe_spec,
        preprocess=compiled.preprocess_contract,
        output_root=output_root_path,
        local_raw_source=local_raw_source,
        provenance_payload=provenance_payload,
    )

    byte_identical, metrics_delta = _compare_against_source(
        source_artifact_dir=source_artifact_dir,
        replayed_artifact_dir=execution.artifact_dir,
        overrides_empty=(len(overrides) == 0),
    )

    diff_report_path = output_root_path / "replication_diff.json"
    report = {
        "schema_version": REPLICATION_DIFF_SCHEMA_VERSION,
        "source_recipe_id": source_recipe_dict.get("recipe_id"),
        "replayed_recipe_id": new_dict.get("recipe_id"),
        "overrides_applied": dict(overrides),
        "override_diff_entries": diff_entries,
        "source_artifact_dir": str(source_artifact_dir)
        if source_artifact_dir is not None
        else None,
        "replayed_artifact_dir": execution.artifact_dir,
        "source_package_version": _read_package_version(source_artifact_dir),
        "replayed_package_version": _package_version(),
        "metrics_delta": metrics_delta,
        "byte_identical_predictions": byte_identical,
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    diff_report_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False)
    )

    return ReplicationResult(
        source_recipe_id=str(source_recipe_dict.get("recipe_id", "")),
        replayed_recipe_id=str(new_dict.get("recipe_id", "")),
        overrides_applied=dict(overrides),
        diff_report_path=str(diff_report_path),
        byte_identical_predictions=byte_identical,
        execution_result=execution,
    )


def _compare_against_source(
    *,
    source_artifact_dir: str | Path | None,
    replayed_artifact_dir: str,
    overrides_empty: bool,
) -> tuple[bool, dict[str, Any]]:
    if source_artifact_dir is None:
        return False, {}

    src = Path(source_artifact_dir)
    rep = Path(replayed_artifact_dir)

    src_pred = src / "predictions.csv"
    rep_pred = rep / "predictions.csv"
    if overrides_empty and src_pred.is_file() and rep_pred.is_file():
        byte_identical = _sha256(src_pred) == _sha256(rep_pred)
    else:
        byte_identical = False

    metrics_delta: dict[str, Any] = {}
    src_metrics_file = src / "metrics.json"
    rep_metrics_file = rep / "metrics.json"
    if src_metrics_file.is_file() and rep_metrics_file.is_file():
        src_m = json.loads(src_metrics_file.read_text())
        rep_m = json.loads(rep_metrics_file.read_text())
        metrics_delta = _compute_metrics_delta(src_m, rep_m)

    return byte_identical, metrics_delta


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_metrics_delta(
    source: dict[str, Any], replay: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, src_val in source.items():
        if key not in replay:
            continue
        if isinstance(src_val, (int, float)) and isinstance(replay[key], (int, float)):
            delta_abs = float(replay[key]) - float(src_val)
            denom = abs(float(src_val))
            delta_pct = 100.0 * delta_abs / float(src_val) if denom > 1e-12 else None
            out[key] = {
                "source": float(src_val),
                "replayed": float(replay[key]),
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
            }
        elif isinstance(src_val, dict) and isinstance(replay[key], dict):
            sub = _compute_metrics_delta(src_val, replay[key])
            if sub:
                out[key] = sub
    return out


def _read_package_version(source_artifact_dir: str | Path | None) -> str | None:
    if source_artifact_dir is None:
        return None
    mpath = Path(source_artifact_dir) / "manifest.json"
    if not mpath.is_file():
        return None
    try:
        data = json.loads(mpath.read_text())
        return data.get("package_version")
    except Exception:
        return None


def _package_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("macrocast")
    except Exception:  # pragma: no cover
        return None


__all__ = [
    "ReplicationResult",
    "REPLICATION_DIFF_SCHEMA_VERSION",
    "execute_replication",
]
