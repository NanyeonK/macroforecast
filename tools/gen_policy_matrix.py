"""Generate the model x forecast-policy compatibility guide page.

This writes one generated page under a guide base directory:

    <base>/model_policy_matrix.md

The matrix is derived from the live model registry plus the direct-policy guard,
so it cannot drift from the package's actual supported policy surface.

Usage::

    python tools/gen_policy_matrix.py --out docs/guide
    python tools/gen_policy_matrix.py --check docs/guide
"""
from __future__ import annotations

import argparse
import json
import os as _os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPO_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), _os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import macroforecast as mf  # noqa: E402
from macroforecast.pipeline.spec import (  # noqa: E402
    DIRECT_AVERAGE_GUARD_MODELS,
    DIRECT_POLICY_GUARD_MODELS,
)

_POLICIES = ("direct", "direct_average", "path_average", "recursive")
_DIRECT_LIKE = frozenset({"direct", "direct_average"})
_SCAN_PATH = Path(".dev-notes/policy_matrix_results.json")

_HEADER = """# Model x Forecast Policy Matrix

[Back to Models and Features](model_overview.md)

This page states which forecast policies are statistically supported for each
registered model. It is generated from `mf.list_model_specs()`, each model's
`ModelSpec.default_params`, and the package's direct-policy guard.

Status meanings:

- `supported`: the runner policy applies to this model family.
- `supported-via-direct-projection`: the model has an explicit validated direct
  projection mode for this policy.
- `guarded-unsupported`: the combination is blocked by default because the model
  forecasts by iterating its own dynamics rather than by fitting an h-step direct
  projection, or because the model's direct projection is a point target rather
  than the requested horizon-average target.

The measured-scan column is only a point-in-time runtime smoke scan. `OK` means a
combination ran, not that its forecast object has the right statistical meaning.
"""


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _has_direct_projection(model_name: str) -> bool:
    spec = mf.get_model(model_name)
    return "direct" in getattr(spec, "default_params", {})


def _policy_status(model_name: str, input_kind: str, policy: str) -> str:
    del input_kind
    if policy == "direct_average" and model_name in DIRECT_AVERAGE_GUARD_MODELS:
        return "guarded-unsupported"
    if policy in _DIRECT_LIKE:
        if model_name in DIRECT_POLICY_GUARD_MODELS:
            return "guarded-unsupported"
        if _has_direct_projection(model_name):
            return "supported-via-direct-projection"
        return "supported"
    return "supported"


def _scan_status(value: Any) -> str | None:
    if isinstance(value, Mapping):
        status = value.get("status") or value.get("runtime_status") or value.get("result")
        return None if status is None else str(status)
    return str(value)


def _load_scan(path: Path = _SCAN_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[Mapping[str, Any]] = []
    if isinstance(data, list):
        rows = [item for item in data if isinstance(item, Mapping)]
    elif isinstance(data, Mapping):
        if isinstance(data.get("results"), list):
            rows = [item for item in data["results"] if isinstance(item, Mapping)]
        else:
            parsed: dict[str, dict[str, str]] = {}
            for model_name, per_policy in data.items():
                if not isinstance(per_policy, Mapping):
                    continue
                parsed[str(model_name)] = {}
                for policy, status in per_policy.items():
                    if str(policy) not in _POLICIES:
                        continue
                    parsed_status = _scan_status(status)
                    if parsed_status is not None:
                        parsed[str(model_name)][str(policy)] = parsed_status
            return parsed
    parsed: dict[str, dict[str, str]] = {}
    for row in rows:
        model_name = row.get("model") or row.get("model_spec") or row.get("name")
        policy = row.get("forecast_policy") or row.get("policy")
        status = row.get("status") or row.get("runtime_status") or row.get("result")
        if model_name is None or policy is None or status is None:
            continue
        if str(policy) not in _POLICIES:
            continue
        parsed.setdefault(str(model_name), {})[str(policy)] = str(status)
    return parsed


def _scan_cell(model_name: str, scan: Mapping[str, Mapping[str, str]]) -> str:
    per_policy = scan.get(model_name, {})
    if not per_policy:
        return "not available"
    parts = [
        f"{policy}: {per_policy[policy]}"
        for policy in _POLICIES
        if policy in per_policy
    ]
    return ", ".join(parts) if parts else "not available"


def build_page(df: Any, scan: Mapping[str, Mapping[str, str]]) -> str:
    parts = [_HEADER.rstrip(), ""]
    parts.append(
        "| Model | Family | Input | direct | direct_average | path_average | "
        "recursive | Measured scan |"
    )
    parts.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in df.sort_values(["family", "name"]).iterrows():
        model_name = str(row["name"])
        input_kind = str(row["input_kind"])
        statuses = [
            _policy_status(model_name, input_kind, policy) for policy in _POLICIES
        ]
        parts.append(
            f"| `{_cell(model_name)}` | {_cell(row['family'])} | "
            f"`{_cell(input_kind)}` | "
            f"{_cell(statuses[0])} | {_cell(statuses[1])} | "
            f"{_cell(statuses[2])} | {_cell(statuses[3])} | "
            f"{_cell(_scan_cell(model_name, scan))} |"
        )
    parts.append("")
    if scan:
        parts.append(
            "Measured scan source: `.dev-notes/policy_matrix_results.json`. "
            "Treat it as a runtime smoke scan, not as support metadata."
        )
    else:
        parts.append(
            "Measured scan source unavailable in this checkout "
            "(`.dev-notes/policy_matrix_results.json` was not present when this "
            "page was generated)."
        )
    parts.append("")
    return "\n".join(parts)


def _targets(base: Path, df: Any, scan: Mapping[str, Mapping[str, str]]) -> dict[Path, str]:
    return {base / "model_policy_matrix.md": build_page(df, scan)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--out", help="Base guide directory to write into.")
    group.add_argument(
        "--check", help="Base guide directory to verify; exit 1 if stale."
    )
    args = parser.parse_args(argv)

    df = mf.list_model_specs()
    scan = _load_scan()

    if args.check:
        base = Path(args.check)
        targets = _targets(base, df, scan)
        stale = []
        for path, content in targets.items():
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != content:
                stale.append(path)
        if stale:
            listing = "\n".join(f"  {p}" for p in stale)
            print(
                f"Policy matrix page out of sync with the package:\n{listing}\n"
                f"Run: python tools/gen_policy_matrix.py --out {base}",
                file=sys.stderr,
            )
            return 1
        print(f"{len(targets)} policy matrix page in sync with the package.")
        return 0

    base = Path(args.out)
    targets = _targets(base, df, scan)
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"wrote {len(targets)} page under {base}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
