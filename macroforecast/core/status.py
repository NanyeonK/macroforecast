"""Unified status vocabulary for macroforecast schema items.

Single source of truth: every typed status field across the package
(``NodeStatus`` on DAG nodes, ``OpStatus`` on op specs, ``status`` on
``AxisSpec`` / ``Option``, the ad-hoc ``MODEL_FAMILY_STATUS`` dict in
``ops/l4_ops.py``) maps onto this 2-value vocabulary:

* :data:`OPERATIONAL` -- runtime executes the full design-spec procedure.
  The output matches the published method named in the design document.
* :data:`FUTURE` -- schema-only. The validator rejects use at recipe time
  (or the runtime raises ``NotImplementedError``). Tracked for v0.2+
  implementation in the GitHub issue tracker.

Earlier macroforecast releases experimented with intermediate values
(``planned``, ``approximation``, ``simplified``, ``registry_only``,
``contract_defined_gated``). These are kept as deprecated aliases:
:func:`normalize_status` collapses them to either ``operational`` or
``future`` so existing recipes / docs continue to type-check, but new
code should write ``operational`` or ``future`` explicitly.

Rationale: the grey middle band ("runs but is an approximation of the
named procedure") is itself a form of false advertising. v0.1 prefers
to *reject* recipes that select procedures we haven't truly implemented
yet -- the user gets a clear error pointing at the tracking issue --
over silently returning numbers that look right but aren't.
"""
from __future__ import annotations

from typing import Final, Literal


ItemStatus = Literal["operational", "future"]
"""The two-value status vocabulary used across the package."""


OPERATIONAL: Final[ItemStatus] = "operational"
FUTURE: Final[ItemStatus] = "future"

KNOWN_STATUSES: Final[frozenset[str]] = frozenset({OPERATIONAL, FUTURE})


# Legacy values that previously carried distinct semantics. Mapped here so
# old recipes / serialized manifests / third-party schemas continue to work.
# All "intermediate" labels collapse to FUTURE because they all marked items
# whose runtime did not match the named published procedure.
_LEGACY_ALIASES: Final[dict[str, ItemStatus]] = {
    "planned": FUTURE,
    "approximation": FUTURE,
    "simplified": FUTURE,
    "registry_only": FUTURE,
    "contract_defined_gated": FUTURE,
    "stub": FUTURE,
}


def normalize_status(status: str | None) -> ItemStatus:
    """Convert any legacy or current status string into the canonical
    2-value :data:`ItemStatus`.

    >>> normalize_status("operational")
    'operational'
    >>> normalize_status("planned")
    'future'
    >>> normalize_status("registry_only")
    'future'
    >>> normalize_status(None)
    'operational'

    Unknown strings are treated as ``operational`` (callers that need
    strict validation should compare against :data:`KNOWN_STATUSES`).
    """

    if status is None or status == "":
        return OPERATIONAL
    if status in KNOWN_STATUSES:
        return status  # type: ignore[return-value]
    if status in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[status]
    return OPERATIONAL


def is_runnable(status: str | None) -> bool:
    """True iff the status indicates the runtime will execute the item.

    Operational items run; future items are rejected by the validator
    or raise ``NotImplementedError``. Legacy aliases collapse first.
    """

    return normalize_status(status) == OPERATIONAL


def is_future(status: str | None) -> bool:
    """Convenience inverse of :func:`is_runnable`."""

    return normalize_status(status) == FUTURE


__all__ = [
    "FUTURE",
    "ItemStatus",
    "KNOWN_STATUSES",
    "OPERATIONAL",
    "is_future",
    "is_runnable",
    "normalize_status",
]
