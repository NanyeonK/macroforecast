"""Right-pane YAML preview component.

Read-only by default (shows a Markdown fenced code block). Switches to
an editable textarea when the user clicks "Edit YAML".
"""
from __future__ import annotations

try:
    import solara
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

from macroforecast.wizard.state import (
    RecipeState,
    yaml_edit_mode,
    yaml_text,
)


@solara.component
def YamlPreview() -> None:
    """Right-pane YAML preview / editor.

    - Read-only mode: monospace Markdown code block.
    - Edit mode: raw textarea; changes are synced back on click of
      "Apply Changes".
    - "Edit YAML" button toggles between modes.
    - "Export" button calls RecipeState.export_to_path() or a download.
    """
    edit_mode = solara.use_reactive(yaml_edit_mode)
    text = solara.use_reactive(yaml_text)

    # Local staging value while in edit mode (so we don't hammer state on
    # every keystroke)
    local_text, set_local_text = solara.use_state(text.value)

    def on_toggle_edit() -> None:
        if edit_mode.value:
            # Leaving edit mode: discard local edits (or user clicked Apply first)
            yaml_edit_mode.value = False
        else:
            # Entering edit mode: seed the local buffer
            set_local_text(yaml_text.value)
            yaml_edit_mode.value = True

    def on_apply() -> None:
        yaml_text.value = local_text
        RecipeState.sync_yaml_to_recipe()
        yaml_edit_mode.value = False

    def on_export() -> None:
        import os
        import tempfile
        # Best-effort local export: write to a temp file and print path
        path = os.path.join(tempfile.gettempdir(), "recipe.yaml")
        RecipeState.export_to_path(path)
        # In a real browser environment a file-download link would be better;
        # for the local server use case a console message is sufficient.
        print(f"[macroforecast wizard] Recipe exported to: {path}")  # noqa: T201

    with solara.Column(
        style="width:360px; min-width:360px; padding:8px; "
              "background:#1e1e2e; border-left:1px solid #30304a; overflow-y:auto;"
    ):
        # Toolbar row
        with solara.Row(style="margin-bottom:8px; gap:8px;"):
            solara.Text(
                "YAML Preview",
                style="color:#cdd6f4; font-weight:bold; font-size:13px; flex:1;",
            )
            edit_label = "Done" if edit_mode.value else "Edit YAML"
            solara.Button(
                edit_label,
                on_click=on_toggle_edit,
                style="font-size:11px; padding:2px 8px;",
            )
            solara.Button(
                "Export",
                on_click=on_export,
                style="font-size:11px; padding:2px 8px;",
            )

        if edit_mode.value:
            # Editable mode
            solara.InputText(
                label="",
                value=local_text,
                on_value=set_local_text,
                multiline=True,
                style=(
                    "font-family:monospace; font-size:12px; width:100%; "
                    "min-height:400px; background:#181825; color:#cdd6f4; "
                    "border:1px solid #6c6f85; border-radius:4px; padding:8px;"
                ),
            )
            solara.Button(
                "Apply Changes",
                on_click=on_apply,
                style="margin-top:8px; width:100%;",
            )
        else:
            # Read-only mode: render as Markdown fenced block
            yaml_content = text.value or "(empty recipe)"
            md_src = f"```yaml\n{yaml_content}\n```"
            with solara.Column(
                style=(
                    "background:#181825; border-radius:4px; padding:8px; "
                    "font-family:monospace; font-size:12px; color:#cdd6f4; "
                    "overflow-x:auto; white-space:pre-wrap; min-height:200px;"
                )
            ):
                solara.Markdown(md_src)


__all__ = ["YamlPreview"]
