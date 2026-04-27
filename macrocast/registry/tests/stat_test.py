from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="stat_test",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none", description="No statistical test", status="operational", priority="A"),
        EnumRegistryEntry(id="dm", description="Diebold-Mariano equal predictive accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="dm_hln", description="Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction", status="operational", priority="A"),
        EnumRegistryEntry(id="dm_modified", description="Modified Diebold-Mariano for long-horizon forecasts", status="operational", priority="A"),
        EnumRegistryEntry(id="cw", description="Clark-West nested-model equal predictive accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="mcs", description="Model Confidence Set for current model vs benchmark slice", status="operational", priority="A"),
        EnumRegistryEntry(id="enc_new", description="ENC-NEW forecast-encompassing test", status="operational", priority="A"),
        EnumRegistryEntry(id="mse_f", description="MSE-F nested-model comparison statistic", status="operational", priority="A"),
        EnumRegistryEntry(id="mse_t", description="MSE-t nested-model comparison statistic", status="operational", priority="A"),
        EnumRegistryEntry(id="cpa", description="Giacomini-White conditional predictive ability test (constant-only minimal slice)", status="operational", priority="A"),
        EnumRegistryEntry(id="rossi", description="Rossi-Sekhposyan forecast-stability statistic", status="operational", priority="A"),
        EnumRegistryEntry(id="rolling_dm", description="Rolling-window Diebold-Mariano summary", status="operational", priority="A"),
        EnumRegistryEntry(id="reality_check", description="White Reality Check bootstrap against benchmark", status="operational", priority="A"),
        EnumRegistryEntry(id="spa", description="Hansen SPA bootstrap against benchmark", status="operational", priority="A"),
        EnumRegistryEntry(id="mincer_zarnowitz", description="Mincer-Zarnowitz regression diagnostic", status="operational", priority="A"),
        EnumRegistryEntry(id="ljung_box", description="Ljung-Box serial-correlation diagnostic", status="operational", priority="A"),
        EnumRegistryEntry(id="arch_lm", description="ARCH-LM heteroskedasticity diagnostic", status="operational", priority="A"),
        EnumRegistryEntry(id="bias_test", description="Forecast-bias t-test", status="operational", priority="A"),
        EnumRegistryEntry(id="pesaran_timmermann", description="Pesaran-Timmermann directional-accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="binomial_hit", description="Binomial hit-rate test for directional accuracy", status="operational", priority="A"),
        EnumRegistryEntry(id="full_residual_diagnostics", description="Residual diagnostic bundle: Mincer-Zarnowitz, Ljung-Box, ARCH-LM, bias test", status="operational", priority="A"),
    ),
    compatible_with={},
    incompatible_with={},
)
