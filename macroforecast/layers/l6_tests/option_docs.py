"""L6 statistical tests -- per-option documentation (collocated).

This module is the canonical Phase-3f collocated option_docs entry for L6.
It delegates to the original scaffold modules so all ``register(...)`` calls
happen exactly once, avoiding duplicate-registration errors.

L6 statistical tests axes docs: ``tools.docgen.option_docs.l6``

Imported here for side-effect registration.
"""
from __future__ import annotations

# Trigger registration side-effects from the original scaffold module.
import tools.docgen.option_docs.l6  # noqa: F401
