"""Recipe orchestration tier for macroforecast.

The ``recipes`` module provides YAML-driven orchestration for end-to-end
forecasting studies. It is the appropriate entry point for replication
research and systematic benchmarking where every decision must be recorded
in a YAML recipe for reproducibility. For exploratory analysis, custom model
development, or one-off forecasts, the standalone API
(``macroforecast.layers.l4_models``, ``macroforecast.functions``) is simpler
and does not require YAML.

**Canonical API (new in v0.9.5a)**

- ``macroforecast.recipes.run(recipe, ...)`` -- execute a recipe end-to-end.
- ``macroforecast.recipes.run_file(path, ...)`` -- execute by file path.
- ``macroforecast.recipes.replicate(manifest_path, ...)`` -- bit-exact replay.
- ``macroforecast.recipes.forecast(dataset, target, ...)`` -- one-shot helper.
- ``macroforecast.recipes.Experiment(...)`` -- builder-pattern study object.
- ``macroforecast.recipes.ForecastResult`` -- result facade.
- ``macroforecast.recipes.ManifestExecutionResult`` -- full execution result.
- ``macroforecast.recipes.ReplicationResult`` -- replication result.

**Top-level aliases (retained for backward compatibility, no deprecation)**

``macroforecast.run``, ``macroforecast.run_file``, ``macroforecast.replicate``,
``macroforecast.forecast``, ``macroforecast.Experiment``,
``macroforecast.ForecastResult``, ``macroforecast.ManifestExecutionResult``,
``macroforecast.ReplicationResult`` are silent aliases for the canonical names
above and will continue to work through v0.9.5a without deprecation warnings.

**Paper-method recipe builders**

``macroforecast.layers.l4_models.paper_methods`` contains per-paper recipe
constructor functions (v0.9 Phase 2 paper-coverage pass). Each helper returns
a recipe dict ready for ``run()``.
"""
from __future__ import annotations

# Recipe-running API -- imported from implementation modules.
# api/recipe.py is the thin re-export over core.execution (Phase 4 restructure).
from ..api.recipe import (
    ManifestExecutionResult,
    ReplicationResult,
    run,
    run_file,
    replicate,
)

# api/quick.py is the high-level facade (Experiment, ForecastResult, forecast).
from ..api.quick import (
    Experiment,
    ForecastResult,
    forecast,
)

__all__ = [
    "Experiment",
    "ForecastResult",
    "ManifestExecutionResult",
    "ReplicationResult",
    "forecast",
    "replicate",
    "run",
    "run_file",
]
