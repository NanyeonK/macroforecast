"""Root Solara component for the macroforecast Wizard.

3-pane layout:
    Left   (200px fixed): LayerRail
    Center (flex):        WorkspacePane (overview or layer form)
    Right  (360px fixed): YamlPreview
"""
from __future__ import annotations

try:
    import solara
except ImportError as exc:
    raise ImportError(
        "macroforecast wizard requires the [wizard] extra. "
        "Install with: pip install 'macroforecast[wizard]'"
    ) from exc

from macroforecast.wizard.components.layer_rail import LayerRail
from macroforecast.wizard.components.yaml_preview import YamlPreview
from macroforecast.wizard.pages.layer_form import LayerForm
from macroforecast.wizard.pages.overview import MosaicCubeOverview
from macroforecast.wizard.state import RecipeState, current_recipe, selected_layer

# Recipe templates (mirrors scaffold/wizard.py._TEMPLATES)
_TEMPLATES: dict[str, dict] = {
    "single_model_forecast": {},
    "model_horse_race": {},
    "regime_conditional": {},
    "blank": {},
}


@solara.component
def WizardApp() -> None:
    """Root 3-pane wizard component.

    Drives the selected_layer reactive variable to switch workspace content
    between the Mosaic Cube overview and individual layer forms.
    """
    layer = solara.use_reactive(selected_layer)
    show_new_dialog, set_show_new_dialog = solara.use_state(False)
    new_template_key, set_new_template_key = solara.use_state("blank")

    def on_select(layer_id: str) -> None:
        selected_layer.value = layer_id

    def on_navigate(layer_id: str) -> None:
        selected_layer.value = layer_id

    def on_new_recipe() -> None:
        set_show_new_dialog(True)

    def on_confirm_new(template_key: str) -> None:
        current_recipe.value = {}
        RecipeState.sync_recipe_to_yaml()
        selected_layer.value = "overview"
        set_show_new_dialog(False)

    def on_export_yaml() -> None:
        import os
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "recipe.yaml")
        RecipeState.export_to_path(path)
        print(f"[macroforecast wizard] Recipe exported to: {path}")  # noqa: T201

    # ---- Top bar ----
    with solara.Column(style="height:100vh; overflow:hidden;"):
        with solara.Row(
            style=(
                "background:#1e1e2e; padding:10px 16px; "
                "border-bottom:1px solid #30304a; align-items:center; "
                "flex-shrink:0; gap:12px;"
            )
        ):
            solara.Text(
                "macroforecast wizard",
                style="color:#cdd6f4; font-size:16px; font-weight:bold; flex:1;",
            )
            solara.Button(
                "New Recipe",
                on_click=on_new_recipe,
                style="font-size:12px;",
            )
            solara.Button(
                "Export YAML",
                on_click=on_export_yaml,
                style="font-size:12px;",
            )

        # ---- Main 3-pane body ----
        with solara.Row(style="flex:1; overflow:hidden; align-items:stretch;"):

            # Left pane: layer rail
            LayerRail(
                selected_layer=layer.value,
                on_select=on_select,
            )

            # Center pane: workspace
            with solara.Column(style="flex:1; overflow-y:auto; background:#ffffff;"):
                if layer.value == "overview":
                    MosaicCubeOverview(on_navigate=on_navigate)
                else:
                    LayerForm(layer_id=layer.value)

            # Right pane: YAML preview
            YamlPreview()

    # ---- New Recipe dialog ----
    if show_new_dialog:
        with solara.Card(
            title="New Recipe",
            style=(
                "position:fixed; top:50%; left:50%; transform:translate(-50%,-50%); "
                "z-index:1000; background:white; min-width:320px;"
            ),
        ):
            solara.Text("Choose a recipe template:", style="margin-bottom:12px;")
            template_options = list(_TEMPLATES.keys())
            solara.Select(
                label="Template",
                values=template_options,
                value=new_template_key,
                on_value=set_new_template_key,
            )
            with solara.Row(style="margin-top:16px; gap:8px;"):
                solara.Button(
                    "Create",
                    on_click=lambda: on_confirm_new(new_template_key),
                )
                solara.Button(
                    "Cancel",
                    on_click=lambda: set_show_new_dialog(False),
                )


__all__ = ["WizardApp"]
