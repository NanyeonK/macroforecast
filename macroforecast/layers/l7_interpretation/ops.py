"""L7 interpretation ops -- collocated copy.

This module re-exports everything from ``macroforecast.core.ops.l7_ops`` so
that ``macroforecast.layers.l7_interpretation.ops`` is the canonical import
path going forward, while keeping the old path alive for backward-compat.

All op registration side-effects live in the original module; this file
simply ensures the collocated namespace resolves correctly.
"""
from __future__ import annotations

# Re-export every public symbol from the original ops module.
from macroforecast.core.ops.l7_ops import (  # noqa: F401
    DEFAULT_FIGURE_MAPPING,
    FIGURE_TYPES,
    FUTURE_OPS,
    HONESTY_DEMOTED_L7_OPS,
    OPERATIONAL_OPS,
    PRE_DEFINED_BLOCKS,
    _schema,
    _stub,
)
