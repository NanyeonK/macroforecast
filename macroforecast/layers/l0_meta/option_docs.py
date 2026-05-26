"""L0 study setup -- per-option documentation (collocated).

This module is the canonical Phase-3f collocated option_docs entry for L0.
It delegates to the original scaffold modules so all ``register(...)`` calls
happen exactly once, avoiding duplicate-registration errors.

L0.A policy axes docs: ``macroforecast.scaffold.option_docs.l0``

Imported here for side-effect registration.
"""
from __future__ import annotations

# Trigger registration side-effects from the original scaffold module.
import macroforecast.scaffold.option_docs.l0  # noqa: F401
