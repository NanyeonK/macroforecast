from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="multiple_model",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none",          description="No multiple-model test", status="operational", priority="A"),
        EnumRegistryEntry(id="reality_check", description="White Reality Check bootstrap against benchmark", status="operational", priority="A"),
        EnumRegistryEntry(id="spa",           description="Hansen SPA bootstrap against benchmark", status="operational", priority="A"),
        EnumRegistryEntry(id="mcs",           description="Model Confidence Set", status="operational", priority="A"),
        EnumRegistryEntry(id="stepwise_mcs",         description="Stepwise MCS variant", status="planned", priority="B"),
        EnumRegistryEntry(id="bootstrap_best_model", description="Bootstrap best-model selection", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
