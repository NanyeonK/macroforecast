"""Root conftest: ensure the project root is on sys.path.

Required so that ``import tools.docgen`` works in tests without a full
package install. The ``tools/`` tree is a dev-only namespace (not part of
the published macroforecast wheel).

Note: we forcibly insert at position 0 even if the path is already present
elsewhere in sys.path. pytest (importlib mode) adds ``tests/`` to sys.path
when it collects test packages, which would shadow top-level ``tools/`` with
``tests/tools/``. Ensuring project root is always first prevents that shadow.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).parent)
# Remove any existing entry for the project root so we can re-insert at 0.
while _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)
