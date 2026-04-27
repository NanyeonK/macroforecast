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
            description="R midasr beta-family weights",
            status="future",
            priority="C",
        ),
        EnumRegistryEntry(
            id="genexp",
            description="R midasr generalized exponential weights",
            status="future",
            priority="C",
        ),
        EnumRegistryEntry(
            id="harstep",
            description="R midasr HAR-style step weights",
            status="future",
            priority="C",
        ),
    ),
    compatible_with={"model_family": ("midasr", "midasr_nealmon")},
    incompatible_with={},
    component="nonlinearity",
)
