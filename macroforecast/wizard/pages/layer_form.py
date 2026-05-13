"""Layer form page.

For l0/l1/l2/l5/l6: renders a schema-driven form with OptionInput for
each operational axis.

For l3/l4/l7: delegates to LayerDagPlaceholder.
"""
from __future__ import annotations

from typing import Any

try:
    import solara
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

from macroforecast.core.stages import STAGE_BY_LAYER
from macroforecast.wizard.components.layer_rail import STAGE_COLOR_MAP
from macroforecast.wizard.components.option_input import OptionInput
from macroforecast.wizard.pages.layer_dag import LayerDagPlaceholder
from macroforecast.wizard.schema import FormField, layer_form_schema
from macroforecast.wizard.state import (
    RecipeState,
    _coerce_value,
    current_recipe,
    validation_errors,
)

# Layers that have form-driven configuration
_FORM_LAYERS: frozenset[str] = frozenset({"l0", "l1", "l2", "l5", "l6"})

# Layers that use the DAG placeholder
_DAG_LAYERS: frozenset[str] = frozenset({"l3", "l4", "l7"})


@solara.component
def LayerForm(layer_id: str) -> None:
    """Main workspace page for a layer.

    Renders either a form (for form layers) or the DAG placeholder.

    All hooks must be called unconditionally (React hooks rules) before
    any conditional returns.

    Args:
        layer_id: The layer to render (e.g. "l0").
    """
    # Hooks must be called unconditionally — before any early returns.
    recipe = solara.use_reactive(current_recipe)  # noqa: SH101
    errors = solara.use_reactive(validation_errors)  # noqa: SH101

    if layer_id in _DAG_LAYERS:
        LayerDagPlaceholder(layer_id=layer_id)
        return

    # --- Form layer ---
    stage = STAGE_BY_LAYER.get(layer_id, "meta")
    color = STAGE_COLOR_MAP.get(stage, "#888888")
    layer_key = RecipeState.LAYER_KEYS.get(layer_id, "")

    # Load schema (may raise if layer unknown)
    try:
        fields: list[FormField] = layer_form_schema(layer_id)
    except (KeyError, Exception) as exc:
        with solara.Column(style="padding:24px;"):
            solara.Text(
                f"Error loading schema for {layer_id}: {exc}",
                style="color:#dc2626;",
            )
        return

    def on_change(axis_name: str, raw_value: Any) -> None:
        """Handle axis value change: coerce and write to recipe."""
        field = next((f for f in fields if f.axis_name == axis_name), None)
        if field is not None:
            # Coerce only string values; booleans/numbers pass through
            if isinstance(raw_value, (bool, int, float)):
                coerced: Any = raw_value
            else:
                coerced = _coerce_value(str(raw_value), field.default)
            RecipeState.set_axis(layer_id, axis_name, coerced)
        else:
            RecipeState.set_axis(layer_id, axis_name, raw_value)

    with solara.Column(style="padding:24px; flex:1; overflow-y:auto;"):
        # Header
        with solara.Row(style="align-items:center; margin-bottom:20px; gap:12px;"):
            with solara.Column(
                style=(
                    f"width:12px; height:40px; border-radius:3px; "
                    f"background:{color};"
                )
            ):
                pass
            with solara.Column(gap="0px"):
                solara.Text(
                    layer_id.upper(),
                    style="font-size:22px; font-weight:bold;",
                )
                solara.Text(
                    stage,
                    style=(
                        f"font-size:13px; color:{color}; text-transform:uppercase; "
                        f"letter-spacing:0.05em;"
                    ),
                )

        if not fields:
            solara.Text(
                f"No operational axes found for {layer_id.upper()}.",
                style="color:#6b7280; font-style:italic;",
            )
        else:
            # Render each field
            for field in fields:
                current_val = RecipeState.get_axis(layer_id, field.axis_name)
                OptionInput(
                    field=field,
                    layer_id=layer_id,
                    on_change=on_change,
                    current_value=current_val,
                )

        # Validation errors for this layer
        layer_errors = [
            e for e in errors.value
            if e.startswith(f"{layer_key}:") or e.startswith(f"{layer_key}.")
        ]
        if layer_errors:
            with solara.Column(
                style=(
                    "background:#fef2f2; border:1px solid #fca5a5; border-radius:6px; "
                    "padding:12px; margin-top:16px;"
                )
            ):
                solara.Text(
                    "Validation errors:",
                    style="font-weight:bold; color:#dc2626; margin-bottom:8px;",
                )
                for err in layer_errors:
                    solara.Text(f"  • {err}", style="color:#991b1b; font-size:13px;")


__all__ = ["LayerForm"]
