"""Compiler migration shims (Phase 2+)."""

from .stat_test_split import migrate_legacy_stat_test

__all__ = ["migrate_legacy_stat_test"]
