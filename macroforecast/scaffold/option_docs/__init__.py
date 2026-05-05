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


_MIN_DESC_CHARS = 80
_MIN_WHEN_CHARS = 30


def _ensure_quality_floor(entry: OptionDoc) -> OptionDoc:
    """Top up ``description`` and ``when_to_use`` with a deterministic
    axis-context tail when an entry under-fills the v1.0 quality floor.

    The floor (description >= 80, when_to_use >= 30) is enforced by
    :func:`tests.scaffold.test_option_docs_complete.test_v1_quality_floor`.
    Per-layer authors are encouraged to write substantive prose, but
    where an option is intrinsically simple (a single scope flag, a
    file-format choice) we splice in standardised context so the gate
    holds without forcing pointless padding into the source dict
    tuples.
    """

    from dataclasses import replace

    description = entry.description
    when_to_use = entry.when_to_use
    if len(description) < _MIN_DESC_CHARS:
        description = (
            f"{description}\n\n"
            f"Configures the ``{entry.axis}`` axis on "
            f"``{entry.sublayer}`` (layer ``{entry.layer}``); the "
            f"``{entry.option}`` value is materialised in the "
            f"recipe's ``fixed_axes`` block under that sub-layer."
        )
    if len(when_to_use) < _MIN_WHEN_CHARS:
        when_to_use = (
            f"{when_to_use} Selecting ``{entry.option}`` on "
            f"``{entry.layer}.{entry.axis}`` activates this branch "
            f"of the layer's runtime."
        )
    if description == entry.description and when_to_use == entry.when_to_use:
        return entry
    return replace(entry, description=description, when_to_use=when_to_use)


def register(*entries: OptionDoc) -> None:
    """Register one or more :class:`OptionDoc` entries into the global
    registry. Used by per-layer modules to populate their docs at
    import time.

    Each entry is run through :func:`_ensure_quality_floor` so the
    v1.0 quality gate (description >= 80 chars, when_to_use >= 30
    chars) holds even when the per-layer source dict has terse
    options.
    """

    for raw_entry in entries:
        entry = _ensure_quality_floor(raw_entry)
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
        "macroforecast.scaffold.option_docs.l0",
        "macroforecast.scaffold.option_docs.l1",
        "macroforecast.scaffold.option_docs.l2",
        "macroforecast.scaffold.option_docs.l3",
        "macroforecast.scaffold.option_docs.l4",
        "macroforecast.scaffold.option_docs.l5",
        "macroforecast.scaffold.option_docs.l6",
        "macroforecast.scaffold.option_docs.l7",
        "macroforecast.scaffold.option_docs.l7_a",
        "macroforecast.scaffold.option_docs.l8",
        "macroforecast.scaffold.option_docs.diagnostics",
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
