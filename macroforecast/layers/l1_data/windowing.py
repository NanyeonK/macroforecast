"""Compatibility shim for macroforecast.data.windowing."""

from __future__ import annotations

from macroforecast.data.windowing import *  # noqa: F401,F403
try:
    from macroforecast.data.windowing import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
