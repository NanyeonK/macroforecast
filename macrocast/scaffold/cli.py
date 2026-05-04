"""``python -m macrocast`` entry point.

Currently only the ``scaffold`` sub-command is wired; future v1.x
sub-commands (e.g., ``run``, ``replicate``, ``validate``) plug in here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .wizard import run_wizard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="macrocast",
        description="macrocast CLI -- recipe scaffold wizard, runner, replicator.",
    )
    sub = parser.add_subparsers(dest="command")

    scaffold = sub.add_parser("scaffold", help="Walk the gate-following recipe wizard.")
    scaffold.add_argument(
        "-o", "--output",
        default="recipe.yaml",
        help="Where to write the generated recipe (default: ./recipe.yaml).",
    )

    args = parser.parse_args(argv)
    if args.command == "scaffold":
        run_wizard(output_path=Path(args.output))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
