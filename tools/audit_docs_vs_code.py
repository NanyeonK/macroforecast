"""Retired: this audit targets a package layout that no longer exists.

It fails in the same removed ops bootstrap as the encyclopedia generator, before any
audit artifact is written. The drift check it was meant to provide is now
``python -m tools.docgen --check`` (D-003).
"""
from __future__ import annotations

import sys

try:  # package import: ``import tools.audit_docs_vs_code`` from the repo root
    from ._retired_doc_tool import retire
except ImportError:  # direct execution: ``python /abs/path/tools/<script>.py``
    from _retired_doc_tool import retire


def main() -> int:
    return retire(
        "tools/audit_docs_vs_code.py",
        reason=(
            "It fails in the same removed ops bootstrap as gen_encyclopedia_docs.py,\n"
            "before any audit artifact is written.\n"
            "\n"
            "python -m tools.docgen --check compares the committed API reference against\n"
            "the docstrings it is generated from. It is not a replacement for every\n"
            "heuristic this script once attempted, and it does not audit the\n"
            "hand-maintained guides under docs/datasets/ or docs/guide/."
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
