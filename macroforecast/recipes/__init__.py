"""Pre-built recipe builders for the v0.9 Phase 2 paper-coverage pass.

Each helper in :mod:`.paper_methods` returns a recipe dict ready for
:func:`macroforecast.run`, encoding the canonical decomposition of the
paper's method over the existing atomic-primitive vocabulary. Helpers
are *thin wrappers* that emit the same recipe shape as the paired YAML
in ``examples/recipes/replications/``; both surfaces are kept in sync
so the algorithmic decomposition is visible from either entry point.
"""
from __future__ import annotations

from . import paper_methods

__all__ = ["paper_methods"]
