"""Retired: this generator targets a package layout that no longer exists.

It imports ``macroforecast.functions``, a namespace removed when the package moved to
semantic submodules, so it fails before writing anything. Kept as a path that explains
itself rather than a command that half-works (D-003).
"""
from __future__ import annotations

import sys

try:  # package import: ``import tools.gen_standalone_docs`` from the repo root
    from ._retired_doc_tool import retire
except ImportError:  # direct execution: ``python /abs/path/tools/<script>.py``
    from _retired_doc_tool import retire


def main() -> int:
    return retire(
        "tools/gen_standalone_docs.py",
        reason=(
            "It imports macroforecast.functions, a namespace removed when the package\n"
            "moved to semantic submodules such as macroforecast.metrics and\n"
            "macroforecast.models. The canonical generated reference is owned by\n"
            "python -m tools.docgen."
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
