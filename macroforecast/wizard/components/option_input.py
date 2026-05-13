"""OptionInput component — renders one FormField as a form widget.

For each axis the wizard shows a widget appropriate to the field's
widget_type, plus a "?" help button that surfaces the OptionDoc.
"""
from __future__ import annotations

from typing import Any, Callable

try:
    import solara
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

from macroforecast.wizard.schema import FormField, option_doc_for
from macroforecast.wizard.state import _coerce_value


@solara.component
def OptionInput(
    field: FormField,
    layer_id: str,
    on_change: Callable[[str, Any], None],
    current_value: Any = None,
) -> None:
    """Render one FormField as a form input widget.

    Args:
        field: The FormField describing the axis.
        layer_id: Layer this field belongs to (e.g. "l0").
        on_change: Callback with (axis_name, raw_value).  The parent is
            responsible for coercion via RecipeState.set_axis().
        current_value: Current value from the recipe (or None for default).
    """
    effective_value = current_value if current_value is not None else field.default
    show_help, set_show_help = solara.use_state(False)

    def handle_toggle_help() -> None:
        set_show_help(not show_help)

    with solara.Column(style="margin-bottom:12px;"):
        with solara.Row(style="align-items:flex-end; gap:8px;"):
            # ---- Widget by type ----
            wtype = field.widget_type

            # Fallback: empty options + "select" -> render as text
            if wtype == "select" and not field.options:
                wtype = "text"

            if wtype == "select":
                values = [v for v, _ in field.options]
                # Map display labels for tooltip but solara.Select takes
                # the raw value list; label is shown in Select itself.
                select_val = str(effective_value) if effective_value is not None else (values[0] if values else "")
                if select_val not in values and values:
                    select_val = values[0]

                solara.Select(
                    label=field.label,
                    values=values,
                    value=select_val,
                    on_value=lambda v: on_change(field.axis_name, v),
                )

            elif wtype == "bool":
                solara.Checkbox(
                    label=field.label,
                    value=bool(effective_value),
                    on_value=lambda v: on_change(field.axis_name, v),
                )

            elif wtype in ("int", "float"):
                str_val = str(effective_value) if effective_value is not None else str(field.default)
                solara.InputText(
                    label=field.label,
                    value=str_val,
                    on_value=lambda v: on_change(
                        field.axis_name,
                        _coerce_value(v, field.default),
                    ),
                    style="min-width:160px;",
                )

            else:  # text
                str_val = str(effective_value) if effective_value is not None else str(field.default or "")
                solara.InputText(
                    label=field.label,
                    value=str_val,
                    on_value=lambda v: on_change(field.axis_name, v),
                    style="min-width:200px;",
                )

            # Help button
            solara.Button(
                "?",
                on_click=handle_toggle_help,
                style=(
                    "font-size:11px; padding:2px 7px; border-radius:50%; "
                    "min-width:24px; background:#e0e7ff; color:#3730a3; "
                    "font-weight:bold;"
                ),
            )

        # Help panel (shown inline when "?" clicked)
        if show_help:
            _render_help_panel(field, layer_id, effective_value)


def _render_help_panel(
    field: FormField,
    layer_id: str,
    current_value: Any,
) -> None:
    """Render the inline help panel for a field."""
    # For select fields: look up doc for the *currently selected* option value
    if field.widget_type == "select" and field.options:
        cur_str = str(current_value) if current_value is not None else (field.options[0][0] if field.options else "")
        doc = option_doc_for(layer_id, field.sublayer, field.axis_name, cur_str)
    elif field.doc is not None:
        doc = field.doc
    else:
        doc = None

    panel_style = (
        "background:#eff6ff; border:1px solid #bfdbfe; border-radius:6px; "
        "padding:10px; margin-top:4px; font-size:12px;"
    )
    with solara.Column(style=panel_style):
        if doc is None:
            solara.Text(
                "No documentation registered yet.",
                style="color:#6b7280; font-style:italic;",
            )
        else:
            solara.Text(doc.summary, style="font-weight:bold; color:#1e40af;")
            if doc.description:
                solara.Text(doc.description, style="color:#374151; margin-top:4px;")
            if doc.when_to_use:
                solara.Text(
                    f"When to use: {doc.when_to_use}",
                    style="color:#065f46; margin-top:4px;",
                )
            if doc.when_not_to_use:
                solara.Text(
                    f"When NOT to use: {doc.when_not_to_use}",
                    style="color:#991b1b; margin-top:4px;",
                )


__all__ = ["OptionInput"]
