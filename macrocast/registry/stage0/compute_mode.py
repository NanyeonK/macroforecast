from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


COMPUTE_MODE_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="serial", description="Default local execution: run one work unit at a time.", status="operational", priority="A"),
    EnumRegistryEntry(id="parallel_by_model", description="Parallelize model-family sweep variants with local threading, capped at 4 workers.", status="operational", priority="A"),
    EnumRegistryEntry(id="parallel_by_horizon", description="Parallelize forecast horizons with local threading, capped at 4 workers.", status="operational", priority="A"),
    EnumRegistryEntry(id="parallel_by_target", description="Parallelize targets in multi-target recipes with local threading, capped at 4 workers.", status="operational", priority="A"),
    EnumRegistryEntry(id="parallel_by_oos_date", description="Parallelize OOS origin-date fits within each horizon loop with local threading, capped at 4 workers. Refit-policy state is computed serially in a pre-pass.", status="operational", priority="A"),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="compute_mode",
    layer="0_meta",
    axis_type="enum",
    entries=COMPUTE_MODE_ENTRIES,
    compatible_with={},
    incompatible_with={},
    default_policy="fixed",
)
