"""Module entry: ``python -m macrocast`` forwards to the scaffold CLI."""
from __future__ import annotations

import sys

from .scaffold.cli import main


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
