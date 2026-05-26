"""Backward-compatibility shim for macroforecast.defaults.

This module was moved to macroforecast.api.defaults in v0.10 Phase 4
restructure. This shim re-exports everything so existing imports continue
to work.
"""
from macroforecast.api.defaults import *  # noqa: F401, F403
from macroforecast.api.defaults import (  # noqa: F401
    DEFAULT_PROFILE_NAME,
    DEFAULT_PROFILE,
    build_default_recipe_dict,
)
