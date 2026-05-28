"""Backward-compatible facade for standalone functions.

The canonical implementation lives in macroforecast.api.functions. This
package keeps the top-level macroforecast.functions surface importable while
new code should use macroforecast.api.functions for module-level imports.
"""
from macroforecast.api.functions import *  # noqa: F401, F403
from macroforecast.api.functions import __all__  # noqa: F401
