"""Ablation runner — baseline + N drop-one variants via Phase 1 sweep runner.

An ablation spec names one or more recipe paths to neutralize individually;
the runner builds a sweep plan (``baseline`` + one variant per component) and
hands it to ``execute_sweep``. After the sweep completes, per-component metric
deltas vs. baseline are emitted to ``<output_root>/ablation_report.json``.

The sweep plan is assembled directly (not via ``compile_sweep_plan``) because
drop-one semantics don't map onto Cartesian ``sweep_axes`` expansion.
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

ABLATION_REPORT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AblationSpec:
    """Specification for one ablation study.

    Attributes
    ----------
    baseline_recipe_dict : dict
        The YAML-form baseline recipe. The dict is not mutated.
    components_to_ablate : tuple of (path, neutral_value)
        Each component is identified by a literal dotted path into the
        recipe dict (same convention as ``apply_overrides``) and a
        *neutral* value — the "off" / no-op value for that axis. What
        "neutral" means is per-axis and left to the caller.
    ablation_study_id : str or None
        If ``None``, auto-derived as ``abl-<12-hex>`` hashing the
        baseline + component list.
    """

    baseline_recipe_dict: dict[str, Any]
    components_to_ablate: tuple[tuple[str, Any], ...]
    ablation_study_id: str | None = None


def _ensure_baseline_failure_policy(spec) -> None:
    """Default the baseline recipe's failure_policy to skip_failed_cell.

    Ablation studies are tolerant of individual cell failures by design; a
    broken ablation variant should not abort the full study. If the user
    pinned a specific policy in the baseline recipe, honour it.
    """
    recipe = spec.baseline_recipe_dict
    path = recipe.setdefault("path", {})
    meta = path.setdefault("0_meta", {})
    fixed = meta.setdefault("fixed_axes", {})
    fixed.setdefault("failure_policy", "skip_failed_cell")


def execute_ablation(
    *,
    spec: AblationSpec,
    output_root: str | Path,
    local_raw_source: str | Path | None = None,
):
    from ..compiler.sweep_plan import SweepPlan, SweepVariant, _variant_id
    from ..execution.sweep_runner import execute_sweep

    study_id = spec.ablation_study_id or _hash_ablation_id(spec)
    parent_recipe_id = spec.baseline_recipe_dict.get("recipe_id", "baseline")

    _ensure_baseline_failure_policy(spec)
    baseline_variant = SweepVariant(
        variant_id="v-baseline",
        axis_values={},
        parent_recipe_id=parent_recipe_id,
        variant_recipe_dict=copy.deepcopy(spec.baseline_recipe_dict),
    )
    variants: list[SweepVariant] = [baseline_variant]
    for path, neutral_value in spec.components_to_ablate:
        new_dict, _ = apply_overrides(
            spec.baseline_recipe_dict, {path: neutral_value}
        )
        axis_values = {path: _as_str_for_axis(neutral_value)}
        variants.append(
            SweepVariant(
                variant_id=_variant_id(axis_values),
                axis_values=axis_values,
                parent_recipe_id=parent_recipe_id,
                variant_recipe_dict=new_dict,
            )
        )

    plan = SweepPlan(
        study_id=study_id,
        parent_recipe_id=parent_recipe_id,
        parent_recipe_dict=copy.deepcopy(spec.baseline_recipe_dict),
        axes_swept=tuple(p for p, _ in spec.components_to_ablate),
        variants=tuple(variants),
    )

    sweep_result = execute_sweep(
        plan=plan,
        output_root=output_root,
        local_raw_source=local_raw_source,
    )

    _write_ablation_report(
        spec=spec, plan=plan, sweep_result=sweep_result, study_id=study_id
    )
    return sweep_result


def _hash_ablation_id(spec: AblationSpec) -> str:
    payload = {
        "baseline": spec.baseline_recipe_dict,
        "components": [list(c) for c in spec.components_to_ablate],
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()
    return f"abl-{digest[:12]}"


def _as_str_for_axis(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _resolve_original_value(recipe_dict: dict[str, Any], dotted_path: str) -> Any:
    cur: Any = recipe_dict
    for segment in dotted_path.split("."):
        if not isinstance(cur, dict) or segment not in cur:
            return None
        cur = cur[segment]
    return cur


def _delta(source_val: float, replay_val: float) -> dict[str, float | None]:
    delta_abs = replay_val - source_val
    if abs(source_val) > 1e-12:
        delta_pct = 100.0 * delta_abs / source_val
    else:
        delta_pct = None
    return {
        "source": source_val,
        "replayed": replay_val,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }


def _write_ablation_report(
    *,
    spec: AblationSpec,
    plan,
    sweep_result,
    study_id: str,
) -> None:
    baseline_vr = next(
        (v for v in sweep_result.per_variant_results if v.variant_id == "v-baseline"),
        None,
    )
    baseline_metrics: dict[str, Any] = (
        dict(baseline_vr.metrics_summary) if baseline_vr is not None else {}
    )

    component_reports: list[dict[str, Any]] = []
    for i, (path, neutral_value) in enumerate(spec.components_to_ablate):
        variant = plan.variants[i + 1]
        vr = next(
            (
                v
                for v in sweep_result.per_variant_results
                if v.variant_id == variant.variant_id
            ),
            None,
        )
        if vr is None:
            continue
        v_metrics = dict(vr.metrics_summary) if vr.metrics_summary else {}
        delta: dict[str, dict[str, float | None]] = {}
        for k, base_v in baseline_metrics.items():
            if k not in v_metrics:
                continue
            try:
                delta[k] = _delta(float(base_v), float(v_metrics[k]))
            except (TypeError, ValueError):
                continue

        component_reports.append(
            {
                "axis_name": path,
                "original_value": _resolve_original_value(
                    spec.baseline_recipe_dict, path
                ),
                "neutral_value": neutral_value,
                "variant_id": variant.variant_id,
                "status": vr.status,
                "metrics": v_metrics,
                "delta_vs_baseline": delta,
            }
        )

    report = {
        "schema_version": ABLATION_REPORT_SCHEMA_VERSION,
        "ablation_study_id": study_id,
        "baseline_variant_id": "v-baseline",
        "baseline_metrics": baseline_metrics,
        "components": component_reports,
        "package_version": _package_version(),
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }

    out_path = Path(sweep_result.output_root) / "ablation_report.json"
    out_path.write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False)
    )


def _package_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("macrocast")
    except Exception:  # pragma: no cover
        return None


__all__ = [
    "AblationSpec",
    "ABLATION_REPORT_SCHEMA_VERSION",
    "execute_ablation",
]
