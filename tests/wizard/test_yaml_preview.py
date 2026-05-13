"""Tests for YamlPreview component — Scenarios YP-01..YP-03."""
from __future__ import annotations

import reacton
import ipyvuetify as v

from macroforecast.wizard.state import (
    RecipeState,
    current_recipe,
    yaml_text,
    yaml_edit_mode,
    validation_errors,
)


def _reset_state():
    current_recipe.set({})
    validation_errors.set([])
    yaml_text.set("")
    yaml_edit_mode.set(False)


def _collect_vuetify_template_content(widget) -> str:
    """Collect rendered content from VuetifyTemplate (Solara Markdown renders to this)."""
    combined = ""
    if type(widget).__name__ == "VuetifyTemplate":
        template = getattr(widget, "template", "") or ""
        combined += template
    for child in getattr(widget, "children", []):
        if not isinstance(child, str):
            combined += _collect_vuetify_template_content(child)
    return combined


# ---------------------------------------------------------------------------
# YP-01: YAML preview displays current_recipe content
# ---------------------------------------------------------------------------

def test_yp01_yaml_preview_displays_recipe():
    """YP-01: YamlPreview renders yaml_text content (via Markdown / VuetifyTemplate)."""
    from macroforecast.wizard.components.yaml_preview import YamlPreview

    _reset_state()
    current_recipe.set({"0_meta": {"fixed_axes": {"failure_policy": "fail_fast"}}})
    RecipeState.sync_recipe_to_yaml()

    # Verify yaml_text contains expected string before rendering
    assert "failure_policy: fail_fast" in yaml_text.value, (
        f"yaml_text does not contain expected string: {yaml_text.value!r}"
    )

    el = YamlPreview()
    box, rc = reacton.render(el)

    # Solara Markdown renders to VuetifyTemplate with the HTML content in 'template' attr
    template_content = _collect_vuetify_template_content(box)
    assert "failure_policy" in template_content or "fail_fast" in template_content, (
        f"Expected YAML content not found in rendered VuetifyTemplate. "
        f"template_content[:200]: {template_content[:200]!r}"
    )


# ---------------------------------------------------------------------------
# YP-02: toggling edit mode switches yaml_edit_mode reactive
# ---------------------------------------------------------------------------

def test_yp02_edit_mode_toggle():
    """YP-02: yaml_edit_mode changes from False to True after Edit YAML button click."""
    from macroforecast.wizard.components.yaml_preview import YamlPreview

    _reset_state()
    assert yaml_edit_mode.value is False

    el = YamlPreview()
    box, rc = reacton.render(el)

    # Find the "Edit YAML" button specifically by its children label
    edit_finder = rc.find(v.Btn, children=["Edit YAML"])
    assert len(edit_finder) == 1, (
        f"Expected exactly 1 'Edit YAML' button, found: {len(edit_finder)}"
    )

    edit_btn = edit_finder.widget
    # Simulate click: toggles yaml_edit_mode from False to True
    edit_btn.fire_event("click", {})

    assert yaml_edit_mode.value is True, (
        f"yaml_edit_mode.value expected True after Edit YAML click, got: {yaml_edit_mode.value}"
    )


# ---------------------------------------------------------------------------
# YP-03: YAML edit triggers sync to current_recipe
# ---------------------------------------------------------------------------

def test_yp03_yaml_edit_triggers_sync():
    """YP-03: Setting yaml_text + calling sync_yaml_to_recipe updates current_recipe."""
    _reset_state()

    # Start in edit mode
    yaml_edit_mode.set(True)
    valid_yaml = "0_meta:\n  fixed_axes:\n    failure_policy: fail_fast\n"
    yaml_text.set(valid_yaml)

    # Trigger sync (as on_apply does in YamlPreview)
    RecipeState.sync_yaml_to_recipe()

    assert current_recipe.value == {
        "0_meta": {"fixed_axes": {"failure_policy": "fail_fast"}}
    }, f"Unexpected recipe after YAML sync: {current_recipe.value}"
