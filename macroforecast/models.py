"""Compatibility alias for public L4 model classes.

The canonical implementation lives in ``macroforecast.layers.l4_models``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODELS = import_module("macroforecast.layers.l4_models")

__all__ = list(getattr(_MODELS, "__all__", ()))


def __getattr__(name: str) -> Any:
    if name in __all__:
        value = getattr(_MODELS, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
