"""``python -m tools.docgen`` entry point.

Currently dispatches to the encyclopedia renderer:

    python -m tools.docgen encyclopedia <out_dir>

This is the form used by the CI sync gate (``ci-docs.yml``) to keep
``docs/reference/`` aligned with the live ``LayerImplementationSpec``
+ ``OPTION_DOCS`` registries.
"""
from __future__ import annotations

import argparse
import sys


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.docgen",
        description="Doc-generator CLI entry point (encyclopedia renderer).",
    )
    sub = parser.add_subparsers(dest="command")
    encyc_p = sub.add_parser(
        "encyclopedia",
        help="Emit the source-committed encyclopedia tree under <output>.",
    )
    encyc_p.add_argument(
        "output",
        help="Output directory (e.g. ``docs/reference``).",
    )
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "encyclopedia":
        from . import render_encyclopedia

        written = render_encyclopedia.write_all(args.output)
        print(
            f"[tools.docgen encyclopedia] wrote "
            f"{len(written)} pages to {args.output}",
            file=sys.stderr,
        )
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
