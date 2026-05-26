"""L6 statistical test ops (collocated).

This module re-exports the op registrations from ``macroforecast.core.ops.l6_ops``
so callers can import from the collocated path. The side-effect registrations
happen exactly once via the original module.
"""
from __future__ import annotations

# Trigger op registration side-effects from the original core module.
import macroforecast.core.ops.l6_ops  # noqa: F401

from macroforecast.core.ops.l6_ops import (  # noqa: F401
    l6_collect_inputs,
)
