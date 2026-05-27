"""Compatibility shim for macroforecast.interpretation.methods."""

from __future__ import annotations

from macroforecast.interpretation.methods import *  # noqa: F401,F403
try:
    from macroforecast.interpretation.methods import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
