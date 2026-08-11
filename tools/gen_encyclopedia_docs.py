"""Retired: this generator targets a package layout that no longer exists.

It imports ``macroforecast.features`` through an ops bootstrap that predates the current
layout and fails before writing anything, including under ``--dry-run`` (D-003).
"""
from __future__ import annotations

import sys

try:  # package import: ``import tools.gen_encyclopedia_docs`` from the repo root
    from ._retired_doc_tool import retire
except ImportError:  # direct execution: ``python /abs/path/tools/<script>.py``
    from _retired_doc_tool import retire


def main() -> int:
    return retire(
        "tools/gen_encyclopedia_docs.py",
        reason=(
            "It imports macroforecast.features through an ops bootstrap that predates the\n"
            "current layout, and fails before writing anything, including under --dry-run.\n"
            "The canonical generated reference is owned by python -m tools.docgen."
        ),
        replacement=(
            "python -m tools.docgen docs/reference",
            "python -m tools.docgen --check docs/reference",
            "python -m pytest tests/test_docgen.py",
            "python -m sphinx -b html -W --keep-going docs <build-dir>",
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
