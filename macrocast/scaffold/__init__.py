"""Scaffold module -- builders, wizards, and per-option documentation.

The ``macrocast.scaffold`` namespace ships the v1.0 user-facing entry
points promised in the design's "gate-following YAML scaffold" pillar:

* :mod:`macrocast.scaffold.option_docs` -- per-(layer, sub-layer, axis,
  option) documentation registry. Single source of truth for the wizard
  UI and the sphinx reference docs.
* :mod:`macrocast.scaffold.introspect` -- walks the
  :class:`LayerImplementationSpec` registry and surfaces axes / options
  in a wizard-friendly form.

Subsequent infrastructure PRs add:

* ``builder.py`` -- programmatic ``RecipeBuilder`` (PR-INFRA-3).
* ``wizard.py`` -- CLI wizard with ``?`` expansion (PR-INFRA-4).
* ``templates.py`` -- 5 starter recipes (PR-INFRA-6).
"""
from __future__ import annotations

from . import introspect, option_docs
from .builder import RecipeBuilder
from .option_docs import OPTION_DOCS, CodeExample, OptionDoc, Reference
from .templates import from_template, list_templates
from .wizard import run_wizard

__all__ = [
    "CodeExample",
    "OPTION_DOCS",
    "OptionDoc",
    "RecipeBuilder",
    "Reference",
    "from_template",
    "introspect",
    "list_templates",
    "option_docs",
    "run_wizard",
]
