from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="residual_diagnostics",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none",             description="No residual diagnostic", status="operational", priority="A"),
        EnumRegistryEntry(id="mincer_zarnowitz", description="Mincer-Zarnowitz regression diagnostic", status="operational", priority="A"),
        EnumRegistryEntry(id="ljung_box",        description="Ljung-Box serial-correlation diagnostic", status="operational", priority="A"),
        EnumRegistryEntry(id="arch_lm",          description="ARCH-LM heteroskedasticity diagnostic", status="operational", priority="A"),
        EnumRegistryEntry(id="bias_test",        description="Forecast-bias t-test", status="operational", priority="A"),
        EnumRegistryEntry(id="diagnostics_full", description="Residual diagnostic bundle (MZ, Ljung-Box, ARCH-LM, bias)", status="operational", priority="A"),
        EnumRegistryEntry(id="autocorrelation_of_errors",    description="Standalone autocorrelation diagnostic", status="operational", priority="B"),
        EnumRegistryEntry(id="serial_dependence_loss_diff",  description="Serial dependence diagnostic on loss differential", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
