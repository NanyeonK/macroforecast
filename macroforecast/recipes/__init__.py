"""macroforecast.recipes -- recipe-orchestration namespace.

Canonical entry point for YAML-driven forecasting studies. Re-exports the
public API from macroforecast.api.recipe and macroforecast.api.quick.

Paper-method recipe builders live at macroforecast.layers.l4_models.paper_methods
(relocated from this module in Phase 3b).

Public API:
- run, run_file, replicate -- recipe execution
- Experiment, ForecastResult, forecast -- high-level facade
- ManifestExecutionResult, ReplicationResult -- result types
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
