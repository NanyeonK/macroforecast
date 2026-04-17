from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="direction",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(id="none",               description="No directional-accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="pesaran_timmermann", description="Pesaran-Timmermann directional-accuracy test", status="operational", priority="A"),
        EnumRegistryEntry(id="binomial_hit",       description="Binomial hit-rate test for directional accuracy", status="operational", priority="A"),
        EnumRegistryEntry(id="mcnemar",        description="McNemar test for paired binary hits", status="planned", priority="B"),
        EnumRegistryEntry(id="roc_comparison", description="Paired ROC-curve comparison", status="planned", priority="B"),
    ),
    compatible_with={},
    incompatible_with={},
)
