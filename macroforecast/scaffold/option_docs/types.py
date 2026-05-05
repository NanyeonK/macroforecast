"""Dataclasses backing the per-option documentation registry.

Three types:

* :class:`Reference` -- a single bibliographic / URL citation.
* :class:`CodeExample` -- a runnable code snippet illustrating an option.
* :class:`OptionDoc` -- the full per-option documentation entry,
  consumed by the wizard UI and the sphinx reference docs.

Two completeness tiers:

* **Tier 1** (v1.0 required): ``summary`` + ``description`` + ``when_to_use``
  + at least one ``Reference``.
* **Tier 2** (v1.1): adds ``when_not_to_use``, ``related_options``,
  ``examples``, ``gates_required`` / ``gates_blocked``.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Reference:
    """A single citation. ``url`` is optional for paper-only references."""

    citation: str
    url: str | None = None
    doi: str | None = None

    def to_rst(self) -> str:
        """Render as a sphinx bibliographic line."""

        text = self.citation
        if self.doi:
            text += f" (doi:{self.doi})"
        if self.url:
            text += f" <{self.url}>"
        return text


@dataclass(frozen=True)
class CodeExample:
    """A runnable code snippet. ``language`` is typically ``yaml`` for
    recipe fragments or ``python`` for builder-API snippets."""

    title: str
    code: str
    language: str = "yaml"


@dataclass(frozen=True)
class OptionDoc:
    """Documentation for a single (layer, sub-layer, axis, option) tuple.

    Used by both the CLI wizard (where it surfaces on ``?`` expansion at
    prompt time) and the sphinx reference docs (where it's auto-emitted
    as RST).
    """

    layer: str
    sublayer: str
    axis: str
    option: str

    # Tier 1 (v1.0 required)
    summary: str
    description: str
    when_to_use: str
    references: tuple[Reference, ...] = ()

    # Tier 2 (v1.1)
    when_not_to_use: str = ""
    related_options: tuple[str, ...] = ()
    examples: tuple[CodeExample, ...] = ()
    gates_required: tuple[str, ...] = ()
    gates_blocked: tuple[str, ...] = ()

    # Provenance
    last_reviewed: str = ""
    reviewer: str = ""

    def is_tier1_complete(self) -> bool:
        """``True`` when summary / description / when_to_use are non-empty,
        at least one reference is supplied, **and** ``last_reviewed`` is
        non-empty (machine-generated placeholders leave ``last_reviewed``
        blank so the v1.0 gate distinguishes reviewed from un-reviewed
        entries)."""

        return bool(
            self.summary.strip()
            and self.description.strip()
            and self.when_to_use.strip()
            and self.references
            and self.last_reviewed.strip()
        )

    def is_tier2_complete(self) -> bool:
        return self.is_tier1_complete() and bool(
            self.when_not_to_use.strip() and (self.related_options or self.examples)
        )

    def key(self) -> tuple[str, str, str, str]:
        return (self.layer, self.sublayer, self.axis, self.option)


__all__ = ["CodeExample", "OptionDoc", "Reference"]
