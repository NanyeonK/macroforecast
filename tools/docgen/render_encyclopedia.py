"""Compatibility wrapper for the current reference renderer.

The old encyclopedia was tied to the removed layered-ops architecture. This
module keeps the historical import path available while rendering the live
``docs/reference`` public-API tree.
"""

from __future__ import annotations

from .renderer import check_all, collect_pages, write_all

__all__ = ["check_all", "collect_pages", "write_all"]
