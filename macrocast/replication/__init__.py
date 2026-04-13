"""Replication helpers retained only where they remain generic enough to support migration.

Paper studies should live as recipes/paths, not package-specific core modules.
Prefer `recipes/papers/*.yaml` plus recipe-aware compilation paths for new work.
"""

from macrocast.replication.clss2021 import CLSS2021, get_preset

__all__ = [
    "CLSS2021",
    "get_preset",
]
