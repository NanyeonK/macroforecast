"""Compatibility shim for macroforecast.features.selection."""

from __future__ import annotations

from macroforecast.features.selection import *  # noqa: F401,F403
try:
    from macroforecast.features.selection import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
