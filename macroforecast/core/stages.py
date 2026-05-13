"""Stage label constants and helpers for the macroforecast layer system.

Stage labels derive from existing sink contract naming (``l0_meta_v1``,
``l1_data_definition_v1``, etc.).  Used by the Kedro adapter (future P1)
for layer tag assignment and by the Wizard (future P2) for navigator colour
coding.  Pure constant + helper; no internal package imports → zero circular
dependency risk.
"""
from __future__ import annotations

import re

# StageLabel is a plain str; this alias documents intent and can be used in
# downstream function signatures.
StageLabel = str

STAGE_BY_LAYER: dict[str, str] = {
    "l0":   "meta",
    "l1":   "data",
    "l2":   "clean",
    "l3":   "features",
    "l4":   "forecasts",
    "l5":   "evaluation",
    "l6":   "tests",
    "l7":   "interpretation",
    "l8":   "artifacts",
    "l1_5": "data_diagnostic",
    "l2_5": "clean_diagnostic",
    "l3_5": "features_diagnostic",
    "l4_5": "model_diagnostic",
}
"""Bijective mapping: LayerId value → StageLabel (13 entries, exhaustive)."""

_LAYER_PREFIX_RE = re.compile(r"^l\d+$")


def stage_of(
    *,
    layer_id: str | None = None,
    sink_name: str | None = None,
) -> str:
    """Resolve stage label from a layer ID or a sink contract name.

    Exactly one of *layer_id* or *sink_name* must be provided.
    Unknown inputs raise :exc:`ValueError` (consistent with
    :func:`~macroforecast.core.layers.registry.get_layer` pattern).

    Examples::

        >>> stage_of(layer_id="l3")
        'features'
        >>> stage_of(layer_id="l1_5")
        'data_diagnostic'
        >>> stage_of(sink_name="l3_features_v1")
        'features'
        >>> stage_of(sink_name="l1_5_diagnostic_v1")
        'data_diagnostic'
        >>> stage_of(sink_name="l7_transformation_attribution_v1")
        'interpretation'
    """
    if (layer_id is None) == (sink_name is None):
        raise ValueError(
            "stage_of: pass exactly one of layer_id or sink_name "
            f"(got layer_id={layer_id!r}, sink_name={sink_name!r})"
        )

    # --- Path A: layer_id provided ----------------------------------------
    if layer_id is not None:
        if layer_id not in STAGE_BY_LAYER:
            raise ValueError(
                f"stage_of: unknown layer_id={layer_id!r}. "
                f"Known layers: {sorted(STAGE_BY_LAYER)}"
            )
        return STAGE_BY_LAYER[layer_id]

    # --- Path B: sink_name provided ----------------------------------------
    # Prefix extraction rule:
    #   l1_5_diagnostic_v1  → parts = ["l1", "5", ...]  → candidate = "l1_5"
    #   l3_features_v1      → parts = ["l3", "features", ...] → candidate = "l3"
    parts = sink_name.split("_")  # type: ignore[union-attr]
    if not parts or not _LAYER_PREFIX_RE.match(parts[0]):
        raise ValueError(
            f"stage_of: cannot parse layer prefix from sink_name={sink_name!r}. "
            f"Expected format: l{{N}}[_5]_<description>_v{{version}}"
        )
    if len(parts) >= 2 and parts[1] == "5":
        candidate = f"{parts[0]}_5"
    else:
        candidate = parts[0]

    if candidate not in STAGE_BY_LAYER:
        raise ValueError(
            f"stage_of: cannot resolve stage for sink_name={sink_name!r}: "
            f"layer prefix {candidate!r} not in STAGE_BY_LAYER. "
            f"Known layers: {sorted(STAGE_BY_LAYER)}"
        )
    return STAGE_BY_LAYER[candidate]


__all__ = [
    "STAGE_BY_LAYER",
    "StageLabel",
    "stage_of",
]
