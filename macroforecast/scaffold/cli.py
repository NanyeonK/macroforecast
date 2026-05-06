"""``python -m macroforecast`` entry point.

Subcommands:

* ``scaffold`` -- walk the gate-following recipe wizard.
* ``run`` -- execute a recipe YAML end-to-end (forwards to
  ``macroforecast.run``); writes manifest + per-cell artifacts.
* ``replicate`` -- re-run a stored manifest and verify per-cell sink
  hashes match bit-for-bit (forwards to ``macroforecast.replicate``).
* ``validate`` -- parse + schema-validate a recipe without executing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .wizard import run_wizard


def _cmd_scaffold(args: argparse.Namespace) -> int:
    run_wizard(
        output_path=Path(args.output),
        include_diagnostics=args.include_diagnostics,
    )
    return 0


def _cmd_encyclopedia(args: argparse.Namespace) -> int:
    """Emit the source-committed encyclopedia tree under ``out_dir``.

    Lives under ``scaffold`` because the encyclopedia is the schema
    catalogue (every layer / sub-layer / axis / option) -- the same
    introspection surface the wizard walks at recipe-authoring time."""

    from . import render_encyclopedia

    written = render_encyclopedia.write_all(Path(args.output))
    print(
        f"[macroforecast scaffold encyclopedia] wrote "
        f"{len(written)} pages to {args.output}",
        file=sys.stderr,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import macroforecast

    recipe_path = Path(args.recipe).resolve()
    if not recipe_path.exists():
        print(f"error: recipe not found: {recipe_path}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_directory).resolve()
    print(f"[macroforecast run] {recipe_path} -> {output_dir}", file=sys.stderr)
    result = macroforecast.run(recipe_path, output_directory=output_dir)
    failed = [c for c in result.cells if c.error is not None]
    print(
        f"[macroforecast run] {len(result.cells)} cells "
        f"({len(result.cells) - len(failed)} ok, {len(failed)} failed)",
        file=sys.stderr,
    )
    if failed:
        for cell in failed:
            print(f"  ! {cell.cell_id}: {cell.error}", file=sys.stderr)
        return 1
    return 0


def _cmd_replicate(args: argparse.Namespace) -> int:
    import macroforecast

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    print(f"[macroforecast replicate] {manifest_path}", file=sys.stderr)
    result = macroforecast.replicate(manifest_path)
    summary = {
        "recipe_match": bool(getattr(result, "recipe_match", False)),
        "sink_hashes_match": bool(getattr(result, "sink_hashes_match", False)),
        "per_cell_match": getattr(result, "per_cell_match", None),
    }
    print(json.dumps(summary, indent=2))
    return 0 if summary["sink_hashes_match"] else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Pre-flight schema check: parse + per-layer ``validate_layer``
    walk. Catches schema-level mistakes (unknown axis, missing required
    leaf_config field, gate violation) before the user pays for an
    end-to-end run."""

    import importlib

    import yaml

    recipe_path = Path(args.recipe).resolve()
    if not recipe_path.exists():
        print(f"error: recipe not found: {recipe_path}", file=sys.stderr)
        return 2
    with recipe_path.open(encoding="utf-8") as fh:
        recipe = yaml.safe_load(fh)
    if not isinstance(recipe, dict):
        print(f"error: {recipe_path} is not a YAML mapping", file=sys.stderr)
        return 2

    layer_validators = (
        ("0_meta", "macroforecast.core.layers.l0"),
        ("1_data", "macroforecast.core.layers.l1"),
        ("2_preprocessing", "macroforecast.core.layers.l2"),
        ("3_feature_engineering", "macroforecast.core.layers.l3"),
        ("4_forecasting_model", "macroforecast.core.layers.l4"),
        ("5_evaluation", "macroforecast.core.layers.l5"),
        ("6_statistical_tests", "macroforecast.core.layers.l6"),
        ("7_interpretation", "macroforecast.core.layers.l7"),
        ("8_output", "macroforecast.core.layers.l8"),
    )
    hard_errors: list[str] = []
    for key, module_name in layer_validators:
        block = recipe.get(key)
        if block is None:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        validator = getattr(module, "validate_layer", None)
        if validator is None:
            continue
        try:
            report = validator(block)
        except Exception as exc:  # noqa: BLE001
            hard_errors.append(f"{key}: {type(exc).__name__}: {exc}")
            continue
        for issue in getattr(report, "hard_errors", ()) or ():
            hard_errors.append(f"{key}.{issue.location}: {issue.message}")
    if hard_errors:
        print(f"[macroforecast validate] {recipe_path}: {len(hard_errors)} error(s)", file=sys.stderr)
        for err in hard_errors:
            print(f"  ! {err}", file=sys.stderr)
        return 1
    print(f"[macroforecast validate] {recipe_path}: OK", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="macroforecast",
        description="macroforecast CLI -- recipe scaffold wizard, runner, replicator, validator.",
    )
    sub = parser.add_subparsers(dest="command")

    scaffold = sub.add_parser("scaffold", help="Walk the gate-following recipe wizard.")
    scaffold.add_argument(
        "-o", "--output",
        default="recipe.yaml",
        help="Where to write the generated recipe (default: ./recipe.yaml).",
    )
    scaffold.add_argument(
        "--include-diagnostics",
        action="store_true",
        help="Also walk the L1.5/L2.5/L3.5/L4.5 diagnostic layers.",
    )
    scaffold.set_defaults(func=_cmd_scaffold)

    encyc_p = sub.add_parser(
        "encyclopedia",
        help=(
            "Emit the source-committed encyclopedia tree under <output>. "
            "Used to refresh ``docs/encyclopedia/`` after editing the "
            "OptionDoc registry; CI diffs the output to enforce sync."
        ),
    )
    encyc_p.add_argument(
        "output",
        help="Output directory (e.g. ``docs/encyclopedia``).",
    )
    encyc_p.set_defaults(func=_cmd_encyclopedia)

    run_p = sub.add_parser("run", help="Execute a recipe end-to-end.")
    run_p.add_argument("recipe", help="Path to the recipe YAML.")
    run_p.add_argument(
        "-o", "--output-directory",
        default="out/",
        help="Output directory (default: ./out/).",
    )
    run_p.set_defaults(func=_cmd_run)

    rep_p = sub.add_parser("replicate", help="Replicate a stored manifest.")
    rep_p.add_argument("manifest", help="Path to manifest.json from a prior run.")
    rep_p.set_defaults(func=_cmd_replicate)

    val_p = sub.add_parser("validate", help="Parse + schema-validate a recipe (no execution).")
    val_p.add_argument("recipe", help="Path to the recipe YAML.")
    val_p.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
