"""Compatibility shim for macroforecast.meta.option_docs."""

from __future__ import annotations

from macroforecast.meta.option_docs import *  # noqa: F401,F403
try:
    from macroforecast.meta.option_docs import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
