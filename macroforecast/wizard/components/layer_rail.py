"""Left navigation rail component for the Wizard.

Displays a vertical list of layer entries colour-coded by stage.
"""
from __future__ import annotations

from typing import Callable

try:
    import solara
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

from macroforecast.core.stages import STAGE_BY_LAYER

# ---------------------------------------------------------------------------
# Stage colour map (spec §7b)
# ---------------------------------------------------------------------------

STAGE_COLOR_MAP: dict[str, str] = {
    "meta":                 "#6366f1",   # indigo
    "data":                 "#0ea5e9",   # sky blue
    "clean":                "#10b981",   # emerald
    "features":             "#f59e0b",   # amber
    "forecasts":            "#ef4444",   # red
    "evaluation":           "#8b5cf6",   # violet
    "tests":                "#ec4899",   # pink
    "interpretation":       "#14b8a6",   # teal
    "artifacts":            "#78716c",   # stone
    "data_diagnostic":      "#7dd3fc",   # light sky
    "clean_diagnostic":     "#6ee7b7",   # light emerald
    "features_diagnostic":  "#fcd34d",   # light amber
    "model_diagnostic":     "#fca5a5",   # light red
}

# Layers shown in the rail (main layers l0..l8)
_RAIL_LAYERS: tuple[str, ...] = (
    "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8"
)

# DAG placeholder layers (Phase 3)
_DAG_LAYERS: frozenset[str] = frozenset({"l3", "l4", "l7"})


@solara.component
def LayerRail(
    selected_layer: str,
    on_select: Callable[[str], None],
) -> None:
    """Left navigation rail listing Overview + L0..L8 entries.

    Args:
        selected_layer: Currently active layer id or "overview".
        on_select: Callback invoked with the new layer id when user clicks.
    """
    with solara.Column(style="width:200px; min-width:200px; padding:8px; "
                              "background:#f8f9fa; border-right:1px solid #dee2e6; "
                              "overflow-y:auto;"):

        # Overview entry
        is_overview_active = (selected_layer == "overview")
        overview_style = (
            "cursor:pointer; padding:8px; border-radius:6px; margin-bottom:4px; "
            "font-weight:bold; background:#e0e7ff; border:2px solid #6366f1;"
            if is_overview_active
            else "cursor:pointer; padding:8px; border-radius:6px; margin-bottom:4px; "
                 "font-weight:bold; background:#ede9fe;"
        )
        with solara.v.Html(
            tag="div",
            style_=overview_style,
            on_click=lambda *_: on_select("overview"),
        ):
            solara.Text("Overview")

        solara.Text(
            "Layers",
            style="font-size:11px; color:#6b7280; text-transform:uppercase; "
                  "letter-spacing:0.05em; margin:8px 0 4px 4px;",
        )

        for layer_id in _RAIL_LAYERS:
            stage = STAGE_BY_LAYER.get(layer_id, "meta")
            color = STAGE_COLOR_MAP.get(stage, "#888888")
            is_active = (layer_id == selected_layer)

            row_style = (
                f"cursor:pointer; padding:8px 8px 8px 12px; border-radius:6px; "
                f"margin-bottom:2px; border-left:4px solid {color}; "
                f"background:{color}22; border:2px solid {color};"
                if is_active
                else
                f"cursor:pointer; padding:8px 8px 8px 12px; border-radius:6px; "
                f"margin-bottom:2px; border-left:4px solid {color}; background:{color}11;"
            )

            _layer_id = layer_id  # capture loop variable
            with solara.v.Html(
                tag="div",
                style_=row_style,
                on_click=lambda *_, lid=_layer_id: on_select(lid),
            ):
                with solara.Column(gap="0px"):
                    label_style = "font-weight:bold; font-size:13px;" if is_active else "font-size:13px;"
                    solara.Text(layer_id.upper(), style=label_style)
                    solara.Text(
                        stage,
                        style=f"font-size:10px; color:{color}; text-transform:uppercase;",
                    )
                    if layer_id in _DAG_LAYERS:
                        solara.Text(
                            "PLACEHOLDER",
                            style="font-size:9px; background:#fef9c3; color:#92400e; "
                                  "padding:1px 4px; border-radius:3px; margin-top:2px;",
                        )


__all__ = ["LayerRail", "STAGE_COLOR_MAP"]
