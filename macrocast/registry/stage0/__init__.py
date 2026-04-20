"""Registry layer for the 7 Layer 0 meta axes.

Holds enum definitions for axis_type, compute_mode, experiment_unit,
failure_policy, registry_type, reproducibility_mode, and study_mode —
i.e. the catalogs consumed by the Stage 0 framework
(``macrocast.stage0``) and referenced in every recipe.

To enumerate the catalogs at runtime, prefer
``macrocast.registry.get_axis_registry()`` / ``get_axis_registry_entry()``
over importing the individual axis modules directly.
"""
