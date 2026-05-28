"""Compatibility shim for macroforecast.preprocessing.option_docs."""

from __future__ import annotations

from macroforecast.preprocessing.option_docs import *  # noqa: F401,F403
try:
    from macroforecast.preprocessing.option_docs import __all__ as __all__  # type: ignore[attr-defined]
except ImportError:
    __all__ = [name for name in globals() if not name.startswith("_")]
