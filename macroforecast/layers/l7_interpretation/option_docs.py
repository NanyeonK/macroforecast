"""L7 interpretation -- per-option documentation (collocated).

This module is the canonical Phase-3e collocated option_docs entry for L7.
It delegates to the original scaffold modules so all ``register(...)`` calls
happen exactly once, avoiding duplicate-registration errors.

L7.B output/export axes docs: ``tools.docgen.option_docs.l7``
L7.A importance ops docs:     ``tools.docgen.option_docs.l7_a``

Both are imported here for side-effect registration.
"""
from __future__ import annotations

# Trigger registration side-effects from the original scaffold modules.
import tools.docgen.option_docs.l7   # noqa: F401
import tools.docgen.option_docs.l7_a  # noqa: F401
