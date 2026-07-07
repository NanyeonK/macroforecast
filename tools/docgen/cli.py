"""CLI for regenerating and checking ``docs/reference``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .renderer import check_all, write_all


def main(argv: list[str] | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    if args_in and args_in[0] == "encyclopedia":
        args_in.pop(0)

    parser = argparse.ArgumentParser(
        prog="python -m tools.docgen",
        description="Regenerate or check the committed docs/reference tree.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="docs/reference",
        help="Reference output directory (default: docs/reference).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if output differs from generated reference pages.",
    )
    args = parser.parse_args(args_in)

    output = Path(args.output)
    if args.check:
        ok, messages = check_all(output)
        if ok:
            print(f"[tools.docgen] {output} is up to date")
            return 0
        print(f"[tools.docgen] {output} is out of date", file=sys.stderr)
        for message in messages:
            print(message, file=sys.stderr)
        return 1

    written = write_all(output)
    print(f"[tools.docgen] wrote {len(written)} pages to {output}", file=sys.stderr)
    return 0
