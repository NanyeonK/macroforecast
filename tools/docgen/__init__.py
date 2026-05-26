"""Scaffold module -- builders and per-option documentation.

The ``macroforecast.scaffold`` namespace ships the recipe-authoring
infrastructure:

* :mod:`macroforecast.scaffold.option_docs` -- per-(layer, sub-layer, axis,
  option) documentation registry. Single source of truth for the sphinx
  reference docs.
* :mod:`macroforecast.scaffold.introspect` -- walks the
  :class:`LayerImplementationSpec` registry and surfaces axes / options.
* ``builder.py`` -- programmatic ``RecipeBuilder``.
* ``templates.py`` -- starter recipe templates.
"""
from __future__ import annotations

from . import introspect, option_docs
from .builder import RecipeBuilder
from .option_docs import OPTION_DOCS, CodeExample, OptionDoc, Reference
from .templates import from_template, list_templates

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
]
