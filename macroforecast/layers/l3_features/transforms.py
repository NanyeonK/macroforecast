"""Compatibility shim for macroforecast.features.transforms."""

from __future__ import annotations

from macroforecast.features.transforms import *  # noqa: F401,F403
try:
    from macroforecast.features.transforms import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
