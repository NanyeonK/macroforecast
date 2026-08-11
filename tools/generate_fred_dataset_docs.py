"""Retired: the FRED dataset guides are hand-maintained.

This generator produced the pre-2026 dataset pages from live or cached external data,
stamped with ``date.today()``. It wrote to ``docs/fred_dataset/`` -- a tree that no
longer exists -- with source artifacts under ``build/fred_dataset_sources/``.

It would therefore not overwrite the current guides: the hand-maintained pages live under
``docs/datasets/`` and the old output path is disjoint from them. What it would do is
fetch or reuse external data, write date-dependent artifacts, and recreate a stale
competing documentation tree beside the real one -- which is the more confusing failure,
because both would then exist and only one would be current (D-002).
"""
from __future__ import annotations

import sys

try:  # package import: ``import tools.generate_fred_dataset_docs`` from the repo root
    from ._retired_doc_tool import retire
except ImportError:  # direct execution: ``python /abs/path/tools/<script>.py``
    from _retired_doc_tool import retire


def main() -> int:
    return retire(
        "tools/generate_fred_dataset_docs.py",
        reason=(
            "The FRED dataset guides in docs/datasets/ are hand-maintained prose about\n"
            "loaders, vintages, and dataset contracts. Nothing generates them; edit the\n"
            "pages directly.\n"
            "\n"
            "This script does not write there. It would fetch or reuse external data,\n"
            "write date-dependent source artifacts under build/fred_dataset_sources/, and\n"
            "create a stale competing tree under docs/fred_dataset/ -- a second set of\n"
            "dataset pages beside the current ones, with nothing marking which is which.\n"
            "\n"
            "python -m tools.docgen owns the generated API REFERENCE under docs/reference/\n"
            "only. It does not generate, validate, or replace the dataset guides."
        ),
        replacement=(
            "# edit the dataset guides by hand:",
            "$EDITOR docs/datasets/<dataset>.md",
            "",
            "# regenerate and check the API reference (a different thing):",
            "python -m tools.docgen docs/reference",
            "python -m tools.docgen --check docs/reference",
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
