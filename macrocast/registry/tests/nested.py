from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="nested",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none",    description="No nested-model test", status="operational", priority="A"),
        EnumRegistryEntry(id="cw",      description="Clark-West nested-model equal predictive accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="enc_new", description="ENC-NEW forecast-encompassing test", status="operational", priority="A"),
        EnumRegistryEntry(id="mse_f",   description="MSE-F nested-model comparison statistic", status="operational", priority="A"),
        EnumRegistryEntry(id="mse_t",   description="MSE-t nested-model comparison statistic", status="operational", priority="A"),
        EnumRegistryEntry(id="forecast_encompassing_nested", description="Forecast encompassing regression for nested models", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
