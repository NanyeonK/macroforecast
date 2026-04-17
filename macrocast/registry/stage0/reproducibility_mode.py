from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


REPRODUCIBILITY_MODE_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="strict_reproducible", description="Strict deterministic reproducibility contract.", status="operational", priority="A"),
    EnumRegistryEntry(id="seeded_reproducible", description="Seeded best-effort reproducibility contract.", status="operational", priority="A"),
    EnumRegistryEntry(id="best_effort", description="Best-effort reproducibility without required seed.", status="operational", priority="A"),
    EnumRegistryEntry(id="exploratory", description="Exploratory execution without reproducibility guarantees.", status="registry_only", priority="B"),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="reproducibility_mode",
    layer="0_meta",
    axis_type="enum",
    entries=REPRODUCIBILITY_MODE_ENTRIES,
    compatible_with={},
    incompatible_with={},
    registry_type="enum_registry",
    default_policy="fixed",
)
