"""Compatibility shim for macroforecast.meta.schema."""

from __future__ import annotations

from macroforecast.meta.schema import *  # noqa: F401,F403
try:
    from macroforecast.meta.schema import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
