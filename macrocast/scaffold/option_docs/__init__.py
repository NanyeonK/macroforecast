"""Per-option documentation infrastructure.

Single source of truth for the wizard UI and the sphinx reference docs.
Every (layer, sub-layer, axis, option) tuple in the operational schema
must have an :class:`OptionDoc` entry; the test
``tests/scaffold/test_option_docs_complete.py`` enforces this at v1.0
release time.

Layer-specific dictionaries live under ``option_docs/<layer>.py`` and
register themselves into the global :data:`OPTION_DOCS` mapping.
"""
from __future__ import annotations

from .types import CodeExample, OptionDoc, Reference

OPTION_DOCS: dict[tuple[str, str, str, str], OptionDoc] = {}
"""Registry of option documentation entries.

Key: ``(layer_id, sublayer_id, axis_name, option_value)``.
Value: :class:`OptionDoc` populated from the per-layer modules below.
"""


def register(*entries: OptionDoc) -> None:
    """Register one or more :class:`OptionDoc` entries into the global
    registry. Used by per-layer modules to populate their docs at
    import time.
    """

    for entry in entries:
        key = (entry.layer, entry.sublayer, entry.axis, entry.option)
        if key in OPTION_DOCS:
            raise ValueError(
                f"duplicate OptionDoc registration for {key!r}; "
                f"previously registered, now re-registering."
            )
        OPTION_DOCS[key] = entry


def lookup(layer: str, sublayer: str, axis: str, option: str) -> OptionDoc | None:
    """Return the registered ``OptionDoc`` for the supplied tuple, or
    ``None`` when no entry exists yet."""

    return OPTION_DOCS.get((layer, sublayer, axis, option))


def filter_by_layer(layer: str) -> tuple[OptionDoc, ...]:
    """Return all entries registered for the supplied layer."""

    return tuple(doc for key, doc in OPTION_DOCS.items() if key[0] == layer)


def _load_layer_modules() -> None:
    """Import every layer-specific docs module so its ``register(...)``
    calls populate the global registry. Idempotent -- re-importing is a
    no-op because ``register`` rejects duplicate keys."""

    from importlib import import_module

    for layer_module in (
        "macrocast.scaffold.option_docs.l0",
        "macrocast.scaffold.option_docs.l1",
        # Subsequent layer modules are added as their content PRs land
        # (PR-A3 = L2/L2.5/L3/L3.5, PR-A4 = L4/L4.5, PR-A5 = L5..L8).
    ):
        try:
            import_module(layer_module)
        except ModuleNotFoundError:
            # Layer's docs PR has not landed yet; skip silently. The
            # completeness test catches the gap separately.
            continue


_load_layer_modules()


__all__ = [
    "CodeExample",
    "OPTION_DOCS",
    "OptionDoc",
    "Reference",
    "filter_by_layer",
    "lookup",
    "register",
]
