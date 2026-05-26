"""L8 output ops (collocated).

This module re-exports the op registrations from ``macroforecast.core.ops.l8_ops``
so callers can import from the collocated path. The side-effect registrations
happen exactly once via the original module.
"""
from __future__ import annotations

# Trigger op registration side-effects from the original core module.
import macroforecast.core.ops.l8_ops  # noqa: F401
