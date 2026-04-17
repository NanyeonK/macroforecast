from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="test_scope",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="per_target",     description="Run each stat test once per target", status="operational", priority="A"),
        EnumRegistryEntry(id="per_horizon",    description="Run each stat test once per (target, horizon)", status="operational", priority="A"),
        EnumRegistryEntry(id="per_model_pair", description="Run each pairwise stat test across all model pairs", status="operational", priority="A"),
        EnumRegistryEntry(id="full_grid_pairwise",     description="Full Cartesian grid of pairwise tests", status="planned", priority="B"),
        EnumRegistryEntry(id="benchmark_vs_all",       description="Benchmark-vs-all-models comparison only", status="planned", priority="B"),
        EnumRegistryEntry(id="regime_specific_tests",  description="Tests run separately per detected regime", status="planned", priority="B"),
        EnumRegistryEntry(id="subsample_tests",        description="Tests run on rolling/expanding subsamples", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
