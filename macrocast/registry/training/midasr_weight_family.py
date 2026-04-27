from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="midasr_weight_family",
    layer="3_training",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="nealmon",
            description="R midasr-compatible normalized exponential Almon weights",
            status="operational_narrow",
            priority="B",
        ),
        EnumRegistryEntry(
            id="almonp",
            description="R midasr-compatible raw polynomial Almon weights",
            status="operational_narrow",
            priority="B",
        ),
        EnumRegistryEntry(
            id="nbeta",
            description="R midasr-compatible normalized beta weights",
            status="operational_narrow",
            priority="C",
        ),
        EnumRegistryEntry(
            id="genexp",
            description="R midasr-compatible generalized exponential weights",
            status="operational_narrow",
            priority="C",
        ),
        EnumRegistryEntry(
            id="harstep",
            description="R midasr-compatible HAR-style step weights; requires midas_max_lag=20",
            status="operational_narrow",
            priority="C",
        ),
    ),
    compatible_with={"model_family": ("midasr", "midasr_nealmon")},
    incompatible_with={},
    component="nonlinearity",
)
