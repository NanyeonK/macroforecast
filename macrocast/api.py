"""Top-level public entry points for macrocast v0.1.

These are thin re-exports over :mod:`macrocast.core.execution` so that the
common case ``import macrocast; macrocast.run("recipe.yaml")`` works without
reaching into ``macrocast.core``.
"""
from __future__ import annotations

from .core.execution import (
    ManifestExecutionResult,
    ReplicationResult,
    execute_recipe as run,
    replicate_recipe as replicate,
)

__all__ = [
    "ManifestExecutionResult",
    "ReplicationResult",
    "replicate",
    "run",
]
