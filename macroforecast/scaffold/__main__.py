"""``python -m macroforecast.scaffold`` entry point.

Currently dispatches to the encyclopedia renderer:

    python -m macroforecast.scaffold encyclopedia <out_dir>

This is the form used by the CI sync gate (``ci-docs.yml``) to keep
``docs/encyclopedia/`` aligned with the live ``LayerImplementationSpec``
+ ``OPTION_DOCS`` registries.
"""
from __future__ import annotations

import argparse
import sys


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m macroforecast.scaffold",
        description="Scaffold-package CLI entry point (encyclopedia renderer).",
    )
    sub = parser.add_subparsers(dest="command")
    encyc_p = sub.add_parser(
        "encyclopedia",
        help="Emit the source-committed encyclopedia tree under <output>.",
    )
    encyc_p.add_argument(
        "output",
        help="Output directory (e.g. ``docs/encyclopedia``).",
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "encyclopedia":
        from . import render_encyclopedia

        written = render_encyclopedia.write_all(args.output)
        print(
            f"[macroforecast.scaffold encyclopedia] wrote "
            f"{len(written)} pages to {args.output}",
            file=sys.stderr,
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
