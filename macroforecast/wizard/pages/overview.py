"""Mosaic Cube overview page.

A 3x3 grid of clickable layer tiles, each coloured by stage.
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
from macroforecast.wizard.components.layer_rail import STAGE_COLOR_MAP
from macroforecast.wizard.state import RecipeState, current_recipe

# Main layers in display order
_GRID_LAYERS: tuple[str, ...] = (
    "l0", "l1", "l2",
    "l3", "l4", "l5",
    "l6", "l7", "l8",
)

# DAG placeholder layers
_DAG_LAYERS: frozenset[str] = frozenset({"l3", "l4", "l7"})


@solara.component
def MosaicCubeOverview(on_navigate: Callable[[str], None]) -> None:
    """3x3 grid of layer tiles for the overview page.

    Args:
        on_navigate: Callback invoked with layer_id when user clicks a tile.
    """
    recipe = solara.use_reactive(current_recipe)

    with solara.Column(style="padding:32px; flex:1;"):
        solara.Text(
            "Recipe Overview",
            style="font-size:24px; font-weight:bold; margin-bottom:8px;",
        )
        solara.Text(
            "Click any layer tile to configure it.",
            style="color:#6b7280; margin-bottom:24px;",
        )

        # 3-column grid
        with solara.Row(style="flex-wrap:wrap; gap:16px;"):
            for layer_id in _GRID_LAYERS:
                stage = STAGE_BY_LAYER.get(layer_id, "meta")
                color = STAGE_COLOR_MAP.get(stage, "#888888")
                is_dag = layer_id in _DAG_LAYERS

                # Check if this layer has any content in the recipe
                layer_key = RecipeState.LAYER_KEYS.get(layer_id, "")
                has_content = bool(recipe.value.get(layer_key))

                _lid = layer_id  # capture loop variable

                with solara.Column(
                    style=(
                        f"width:180px; min-height:120px; border-radius:8px; "
                        f"border:2px solid {color}; cursor:pointer; "
                        f"overflow:hidden; position:relative;"
                    ),
                    on_click=lambda lid=_lid: on_navigate(lid),
                ):
                    # Colour band at top
                    with solara.Column(
                        style=(
                            f"background:{color}; padding:8px 12px; "
                            f"border-bottom:1px solid {color};"
                        )
                    ):
                        solara.Text(
                            layer_id.upper(),
                            style="font-size:18px; font-weight:bold; color:white;",
                        )

                    # Content area
                    with solara.Column(style="padding:10px 12px; flex:1;"):
                        solara.Text(
                            stage,
                            style=f"font-size:11px; color:{color}; "
                                  f"text-transform:uppercase; letter-spacing:0.05em;",
                        )
                        if is_dag:
                            solara.Text(
                                "P3",
                                style=(
                                    "font-size:10px; background:#fef9c3; color:#92400e; "
                                    "padding:1px 6px; border-radius:3px; margin-top:4px; "
                                    "display:inline-block;"
                                ),
                            )

                    # Status indicator (bottom-right)
                    with solara.Row(
                        style=(
                            "padding:4px 10px; justify-content:flex-end;"
                        )
                    ):
                        if has_content:
                            solara.Text(
                                "✓",
                                style="color:#10b981; font-size:16px; font-weight:bold;",
                            )
                        else:
                            solara.Text(
                                "○",
                                style="color:#d1d5db; font-size:16px;",
                            )


__all__ = ["MosaicCubeOverview"]
