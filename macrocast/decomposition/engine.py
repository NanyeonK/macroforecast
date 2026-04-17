"""Phase 7 decomposition engine — ANOVA attribution over a completed sweep.

``run_decomposition`` loads a Phase-1 ``study_manifest.json``, builds the
(variant, horizon) long-format primary-metric table, runs one-way ANOVA
per axis within each requested component, and emits both
``decomposition_result.parquet`` and ``decomposition_report.json``.

The engine is intentionally decoupled from the sweep runner — any manifest
emitted by ``execute_sweep`` (including ``execute_ablation``'s output) is a
valid input.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .attribution import one_way_anova
from .components import COMPONENT_NAMES
from .schema import DECOMPOSITION_RESULT_SCHEMA_VERSION, expected_columns


@dataclass(frozen=True)
class DecompositionPlan:
    study_manifest_path: str
    components_to_decompose: tuple[str, ...] = COMPONENT_NAMES
    attribution_method: str = "anova"
    primary_metric: str = "msfe"


@dataclass(frozen=True)
class DecompositionResult:
    study_id: str
    plan: DecompositionPlan
    result_parquet_path: str
    report_json_path: str
    per_component_shares: dict[str, float]
    per_axis_rows: tuple[dict[str, Any], ...] = field(default_factory=tuple)


def run_decomposition(
    plan: DecompositionPlan,
    *,
    output_dir: str | Path | None = None,
) -> DecompositionResult:
    if plan.attribution_method != "anova":
        raise ValueError(
            f"attribution_method={plan.attribution_method!r} not supported in v0.9; "
            "ANOVA baseline only (Shapley deferred to v1.1)."
        )

    for c in plan.components_to_decompose:
        if c not in COMPONENT_NAMES:
            raise ValueError(f"unknown component {c!r}; valid: {COMPONENT_NAMES}")

    manifest_path = Path(plan.study_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    study_id = str(manifest.get("study_id", "unknown-study"))

    out_dir = Path(output_dir) if output_dir is not None else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    observations = _build_observations(manifest=manifest, primary_metric=plan.primary_metric)

    axis_to_component = _axis_component_index()

    per_axis_rows: list[dict[str, Any]] = []
    per_component_ss: dict[str, float] = {c: 0.0 for c in plan.components_to_decompose}
    total_ss = 0.0

    if observations:
        values_all = np.asarray([o["value"] for o in observations], dtype=np.float64)
        grand_mean = float(values_all.mean())
        total_ss = float(((values_all - grand_mean) ** 2).sum())

    axis_keys_in_observations = sorted({
        key
        for obs in observations
        for key in obs["axis_values"].keys()
    })

    for axis_key in axis_keys_in_observations:
        axis_name = axis_key.split(".")[-1]
        component = axis_to_component.get(axis_name)
        if component is None or component not in plan.components_to_decompose:
            continue

        rows_for_axis = [o for o in observations if axis_key in o["axis_values"]]
        if not rows_for_axis:
            continue
        rows_for_axis.sort(key=lambda o: (o["variant_id"], o["horizon"]))
        values = np.asarray([o["value"] for o in rows_for_axis], dtype=np.float64)
        groups = np.asarray([str(o["axis_values"][axis_key]) for o in rows_for_axis])

        res = one_way_anova(values, groups)
        share = res.ss_between / total_ss if total_ss > 0 else 0.0
        per_axis_rows.append(
            {
                "component": component,
                "axis_name": axis_name,
                "ss_between": float(res.ss_between),
                "ss_total": float(total_ss),
                "share": float(share),
                "n_variants": int(len({o["variant_id"] for o in rows_for_axis})),
                "n_groups": int(res.n_groups),
                "significance_p": float(res.p_value) if res.p_value is not None else float("nan"),
            }
        )
        per_component_ss[component] = per_component_ss.get(component, 0.0) + float(res.ss_between)

    per_component_shares = {
        c: (per_component_ss[c] / total_ss if total_ss > 0 else 0.0)
        for c in plan.components_to_decompose
    }

    parquet_path = out_dir / "decomposition_result.parquet"
    _write_parquet(rows=per_axis_rows, path=parquet_path)

    report_path = out_dir / "decomposition_report.json"
    _write_report(
        path=report_path,
        plan=plan,
        study_id=study_id,
        per_component_shares=per_component_shares,
        per_axis_rows=per_axis_rows,
        total_ss=total_ss,
        n_observations=len(observations),
    )

    return DecompositionResult(
        study_id=study_id,
        plan=plan,
        result_parquet_path=str(parquet_path),
        report_json_path=str(report_path),
        per_component_shares=per_component_shares,
        per_axis_rows=tuple(per_axis_rows),
    )


def _build_observations(*, manifest: dict, primary_metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in manifest.get("variants", []):
        if variant.get("status") != "success":
            continue
        axis_values = variant.get("axis_values") or {}
        metrics_summary = variant.get("metrics_summary") or {}
        metrics_by_horizon = metrics_summary.get("metrics_by_horizon")
        if isinstance(metrics_by_horizon, dict):
            for horizon, mh in sorted(metrics_by_horizon.items()):
                if primary_metric not in mh:
                    continue
                try:
                    value = float(mh[primary_metric])
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "variant_id": str(variant.get("variant_id", "")),
                        "horizon": str(horizon),
                        "value": value,
                        "axis_values": dict(axis_values),
                    }
                )
        elif primary_metric in metrics_summary:
            try:
                value = float(metrics_summary[primary_metric])
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "variant_id": str(variant.get("variant_id", "")),
                    "horizon": "aggregate",
                    "value": value,
                    "axis_values": dict(axis_values),
                }
            )
    rows.sort(key=lambda r: (r["variant_id"], r["horizon"]))
    return rows


def _axis_component_index() -> dict[str, str | None]:
    from ..registry.base import AxisDefinition
    from ..registry.build import _discover_axis_definitions
    

    definitions = list(_discover_axis_definitions().values())
    return {d.axis_name: d.component for d in definitions}


def _write_parquet(*, rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        from ..execution.errors import ExecutionError

        raise ExecutionError(
            "decomposition_result.parquet requires the [parquet] extra. "
            "Install with: pip install macrocast[parquet]"
        ) from exc

    cols = expected_columns()
    if rows:
        data = {c: [row[c] for row in rows] for c in cols}
    else:
        data = {c: [] for c in cols}
    table = pa.table(data)
    pq.write_table(table, path)


def _write_report(
    *,
    path: Path,
    plan: DecompositionPlan,
    study_id: str,
    per_component_shares: dict[str, float],
    per_axis_rows: list[dict[str, Any]],
    total_ss: float,
    n_observations: int,
) -> None:
    payload = {
        "schema_version": DECOMPOSITION_RESULT_SCHEMA_VERSION,
        "study_id": study_id,
        "plan": {
            "components_to_decompose": list(plan.components_to_decompose),
            "attribution_method": plan.attribution_method,
            "primary_metric": plan.primary_metric,
        },
        "per_component_shares": per_component_shares,
        "per_axis_rows": per_axis_rows,
        "total_ss": float(total_ss),
        "n_observations": int(n_observations),
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False))


__all__ = [
    "DecompositionPlan",
    "DecompositionResult",
    "run_decomposition",
]
