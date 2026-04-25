from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import build_navigation_view_from_yaml
from .replications import get_replication_entry, list_replication_entries, write_replication_recipe


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _cmd_tree(args: argparse.Namespace) -> int:
    _print_json(build_navigation_view_from_yaml(args.recipe, include_downstream=not args.upstream_only))
    return 0


def _cmd_resolve(args: argparse.Namespace) -> int:
    from ..compiler import compile_recipe_yaml

    compiled = compile_recipe_yaml(args.recipe)
    _print_json(
        {
            "input_yaml_path": args.recipe,
            "execution_status": compiled.compiled.execution_status,
            "warnings": list(compiled.manifest.get("warnings", [])),
            "blocked_reasons": list(compiled.manifest.get("blocked_reasons", [])),
            "tree_context": dict(compiled.manifest.get("tree_context", {})),
            "layer3_capability_matrix": dict(compiled.manifest.get("layer3_capability_matrix", {})),
        }
    )
    return 0


def _cmd_replications(args: argparse.Namespace) -> int:
    if args.replication_id:
        payload = get_replication_entry(args.replication_id)
        if args.write_yaml:
            write_replication_recipe(args.replication_id, args.write_yaml)
            payload["written_yaml"] = str(Path(args.write_yaml))
        _print_json(payload)
        return 0
    _print_json(list_replication_entries())
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from ..compiler import compile_recipe_yaml, run_compiled_recipe

    compiled = compile_recipe_yaml(args.recipe)
    if compiled.compiled.execution_status != "executable":
        _print_json(
            {
                "execution_status": compiled.compiled.execution_status,
                "warnings": list(compiled.manifest.get("warnings", [])),
                "blocked_reasons": list(compiled.manifest.get("blocked_reasons", [])),
            }
        )
        return 2
    result = run_compiled_recipe(
        compiled.compiled,
        output_root=args.output_root,
        local_raw_source=Path(args.local_raw_source) if args.local_raw_source else None,
    )
    _print_json(
        {
            "execution_status": "executed",
            "artifact_dir": str(result.artifact_dir),
            "run_id": result.run.run_id,
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="macrocast-navigate",
        description="Constraint-aware decision tree, path resolver, and replication navigator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tree = sub.add_parser("tree", help="Show selectable and disabled branches for a YAML recipe.")
    tree.add_argument("recipe")
    tree.add_argument("--upstream-only", action="store_true", help="Show only Layers 0-3.")
    tree.set_defaults(func=_cmd_tree)

    resolve = sub.add_parser("resolve", help="Compile a YAML recipe and show route/capability status.")
    resolve.add_argument("recipe")
    resolve.set_defaults(func=_cmd_resolve)

    replications = sub.add_parser("replications", help="List replication entries or write one recipe YAML.")
    replications.add_argument("replication_id", nargs="?")
    replications.add_argument("--write-yaml")
    replications.set_defaults(func=_cmd_replications)

    run = sub.add_parser("run", help="Compile and execute one YAML recipe.")
    run.add_argument("recipe")
    run.add_argument("--output-root", default="results/macrocast")
    run.add_argument("--local-raw-source")
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
