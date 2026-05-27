"""Compatibility alias for preprocessing contract helpers.

The canonical implementation lives in ``macroforecast.layers.l2_preprocessing``.
"""

from macroforecast.layers.l2_preprocessing import *  # noqa: F401, F403
from macroforecast.layers.l2_preprocessing import __all__ as _PREPROCESSING_ALL

__all__ = list(_PREPROCESSING_ALL)
