"""Backward-compatibility shim for macroforecast.functions.

This package was moved to macroforecast.api.functions in v0.10 Phase 4
restructure. This shim re-exports everything so existing imports continue
to work, including submodule paths like macroforecast.functions.ridge.
"""
from macroforecast.api.functions import *  # noqa: F401, F403
from macroforecast.api.functions import __all__  # noqa: F401
