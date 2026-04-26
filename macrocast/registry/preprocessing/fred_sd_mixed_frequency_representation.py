from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="fred_sd_mixed_frequency_representation",
    layer="2_preprocessing",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="calendar_aligned_frame",
            description="keep selected FRED-SD series on the recipe target calendar after Layer 1 frequency conversion",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="drop_unknown_native_frequency",
            description="drop selected FRED-SD series whose inferred native frequency is unknown before representation building",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="drop_non_target_native_frequency",
            description="keep only selected FRED-SD series whose inferred native frequency matches the recipe target frequency",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="native_frequency_block_payload",
            description="emit separate monthly/quarterly/unknown FRED-SD blocks for a future mixed-frequency feature builder",
            status="planned",
            priority="B",
        ),
        EnumRegistryEntry(
            id="mixed_frequency_model_adapter",
            description="delegate native-frequency blocks to a future MIDAS/state-space style model adapter",
            status="planned",
            priority="B",
        ),
    ),
    compatible_with={},
    incompatible_with={},
    component="feature_representation",
)
