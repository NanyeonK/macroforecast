"""``python -m macroforecast`` entry point.

Forwards to the full macroforecast CLI implemented in ``tools/docgen/cli.py``.

Subcommands:
  run          Execute a recipe YAML end-to-end.
  replicate    Re-run a stored manifest and verify per-cell sink hashes.
  validate     Parse and schema-validate a recipe without executing.
  scaffold     Generate a recipe from a template.
  encyclopedia Emit the source-committed encyclopedia tree.
"""
from __future__ import annotations

import sys

from tools.docgen.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
