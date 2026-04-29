"""Registry layer for the Layer 0 meta axes.

Holds enum definitions for axis_type, compute_mode, study_scope,
failure_policy, and reproducibility_mode — i.e. the catalogs consumed by the Stage 0 framework
(``macrocast.stage0``) and referenced in every recipe.

To enumerate the catalogs at runtime, prefer
``macrocast.registry.get_axis_registry()`` / ``get_axis_registry_entry()``
over importing the individual axis modules directly.
"""
