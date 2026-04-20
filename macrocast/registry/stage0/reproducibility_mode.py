from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


REPRODUCIBILITY_MODE_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(
        id="strict_reproducible",
        description=(
            "Pin Python/NumPy/torch seeds AND enable torch deterministic algorithms + cuDNN deterministic=True / benchmark=False. "
            "Sets CUBLAS_WORKSPACE_CONFIG=':4096:8' when unset. Warns if PYTHONHASHSEED is not set in the shell. "
            "Variant seeds are hash-derived from (recipe_id, variant_id, model_family) so different variants get distinct but reproducible seeds."
        ),
        status="operational",
        priority="A",
    ),
    EnumRegistryEntry(
        id="seeded_reproducible",
        description=(
            "Pin Python/NumPy/torch seeds to a single base_seed (default 42). Does not flip cuDNN / deterministic-algorithms flags; small numerical drift across library versions is accepted. Default mode."
        ),
        status="operational",
        priority="A",
    ),
    EnumRegistryEntry(
        id="best_effort",
        description=(
            "Identical install-time behaviour to seeded_reproducible — pins seeds but does not enforce strict deterministic flags. The label exists to mark runs the caller explicitly does not want counted as strict for CI regression checks."
        ),
        status="operational",
        priority="A",
    ),
    EnumRegistryEntry(
        id="exploratory",
        description=(
            "No-op at install time: does not reset Python/NumPy/torch global RNG state. Each variant still receives a fresh non-deterministic seed via np.random.randint from current_seed(). Suitable for ad-hoc exploration where reproducibility is explicitly waived."
        ),
        status="operational",
        priority="A",
    ),
)

AXIS_DEFINITION = AxisDefinition(
    axis_name="reproducibility_mode",
    layer="0_meta",
    axis_type="enum",
    entries=REPRODUCIBILITY_MODE_ENTRIES,
    compatible_with={},
    incompatible_with={},
    default_policy="fixed",
)
