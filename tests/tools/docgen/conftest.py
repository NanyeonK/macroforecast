"""conftest for tests/tools/docgen/ — namespace guard.

pytest with importlib mode + __init__.py packages will register
``tests/tools/docgen/`` under the ``tools.docgen`` namespace (because
``tests/`` is added to sys.path when it discovers the sub-package, making
``tests/tools/`` shadow the top-level ``tools/``).

This conftest runs before any test module in this directory is imported.
It ensures ``tools.docgen`` and ``tools.docgen.option_docs`` are registered
in sys.modules pointing to the *real* top-level ``tools/docgen/`` package,
preventing the namespace shadow.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Ensure project root is at the front of sys.path.
_PROJECT_ROOT = str(Path(__file__).parents[3])
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

# Force-reload tools and tools.docgen from the real top-level location
# so that subsequent imports in test modules find the correct package.
for _mod in list(sys.modules):
    if _mod == "tools" or _mod.startswith("tools."):
        del sys.modules[_mod]

import tools  # noqa: E402 — must be after sys.path fix
import tools.docgen  # noqa: E402
import tools.docgen.option_docs  # noqa: E402
