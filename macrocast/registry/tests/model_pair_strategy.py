from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="model_pair_strategy",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="vs_benchmark_only", description="compare each model only against the benchmark", status="operational", priority="A"),
        EnumRegistryEntry(id="all_pairs", description="compare all model pairs", status="operational", priority="A"),
        EnumRegistryEntry(id="user_list", description="compare user-provided model pairs", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
