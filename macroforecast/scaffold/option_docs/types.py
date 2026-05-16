"""Dataclasses backing the per-option documentation registry.

Three types:

* :class:`Reference` -- a single bibliographic / URL citation.
* :class:`CodeExample` -- a runnable code snippet illustrating an option.
* :class:`OptionDoc` -- the full per-option documentation entry,
  consumed by the wizard UI and the sphinx reference docs.
* :class:`ParameterDoc` -- documents a single function-arg parameter
  accepted by an option (distinct from the axis categorical switch).

Two completeness tiers:

* **Tier 1** (v1.0 required): ``summary`` + ``description`` + ``when_to_use``
  + at least one ``Reference``.
* **Tier 2** (v1.1): adds ``when_not_to_use``, ``related_options``,
  ``examples``, ``gates_required`` / ``gates_blocked``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final


# ---------------------------------------------------------------------------
# Sentinel for required parameters (no default value)
# ---------------------------------------------------------------------------

REQUIRED: Final = object()
"""Sentinel indicating a :class:`ParameterDoc` field has no default value.

Use ``default=REQUIRED`` (or omit ``default=``) to mark a parameter as
required.  Distinguishes "no default" from "default is ``None``" (which
is a valid actual runtime default for optional kwargs like ``vol_model``
and ``random_state``).

The renderer checks ``p.default is REQUIRED`` (identity, not equality)
and renders such parameters as positional required (no ``= ...`` suffix)
or with a ``—`` in the default column of the parameters table.
"""


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
class ParameterDoc:
    """Per-option (function-arg) parameter documentation.

    Distinguishes the categorical axis switch from the function's
    own parameters that the chosen option accepts.

    Attributes:
        name:        Keyword-argument name in the option's runtime call or
                     the ``leaf_config`` key that configures the option.
        type:        Human-readable type hint (e.g. ``"int"``, ``"str"``,
                     ``"float"``, ``"tuple[str, ...]"``, or an enum
                     abbreviation like ``"str enum {cells, models}"``).
        default:     Default value, or ``None`` when the parameter has no
                     default (i.e. it is *required* when the option is active).
        constraint:  Free-text constraint description (e.g. ``">=0"``,
                     ``"must be set if compute_mode=parallel"``).
        description: One-line summary of what the parameter controls.
    """

    name: str
    type: str
    default: Any = REQUIRED
    constraint: str | None = None
    description: str = ""


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

    # Parameters (for options that accept function-level / leaf_config arguments)
    parameters: tuple[ParameterDoc, ...] = ()

    # Data arguments for per-op page signature (Cycle 26)
    # Positional data inputs (X/y, y_true/y_pred, etc.) that precede the *
    # separator in the rendered function signature.  Each entry is a
    # ParameterDoc with default=REQUIRED (always positional).  Stored
    # separately from ``parameters`` so the renderer can emit them before
    # the ``*,`` group.
    data_args: tuple["ParameterDoc", ...] = ()

    # Return type annotation for the per-op page signature (Cycle 26).
    # Non-empty string causes ``-> {return_type}`` to appear after the
    # closing ``)`` in the rendered ## Function signature block.
    return_type: str = ""

    # Return-value attribute table for the ## Returns section (Cycle 26).
    # Each entry is a (attr_name, type_str, description) triple rendered
    # as a markdown table row.  Empty tuple = only the return_type header
    # line is emitted (for scalar returns like ``float``).
    returns_attrs: tuple[tuple[str, str, str], ...] = ()

    # Per-op page (Cycle 22 POC)
    # When True, render_encyclopedia.py emits a dedicated page at
    # ``docs/encyclopedia/<layer>/<axis>/<option>.md`` with Function signature
    # + Parameters table + Behavior + In recipe context + References sections.
    # The axis page's option section is replaced by a 1-2 line stub + link.
    op_page: bool = False

    # Standalone callable name in ``mf.functions`` namespace (used in per-op
    # page header). E.g. ``"ridge_fit"`` for the ridge OptionDoc.
    op_func_name: str = ""

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


__all__ = ["CodeExample", "OptionDoc", "ParameterDoc", "Reference", "REQUIRED"]

