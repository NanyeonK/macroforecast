"""Tests for OptionInput component — Scenarios OI-01..OI-05.

Uses reacton.render for in-process headless widget rendering (no browser).
"""
from __future__ import annotations

import pytest
import reacton
import ipyvuetify as v

from macroforecast.wizard.schema import FormField


def _make_field(**kwargs):
    defaults = dict(
        axis_name="failure_policy",
        sublayer="l0_a",
        label="Failure Policy",
        widget_type="select",
        options=[("fail_fast", "Fail fast"), ("continue_on_failure", "Continue")],
        default="fail_fast",
        is_sweepable=False,
        leaf_config_keys=[],
        doc=None,
        status="operational",
    )
    defaults.update(kwargs)
    return FormField(**defaults)


def _collect_widget_types(widget) -> list[str]:
    """Recursively collect class names from a widget tree."""
    result = [type(widget).__name__]
    for child in getattr(widget, "children", []):
        if not isinstance(child, str):
            result.extend(_collect_widget_types(child))
    return result


# ---------------------------------------------------------------------------
# OI-01: select field renders a Select widget
# ---------------------------------------------------------------------------

def test_oi01_select_field_renders_select():
    """OI-01: FormField widget_type='select' renders a Select widget."""
    from macroforecast.wizard.components.option_input import OptionInput

    field = _make_field(widget_type="select")
    el = OptionInput(field=field, layer_id="l0", on_change=lambda *a: None, current_value="fail_fast")
    box, rc = reacton.render(el)

    # Use rc.find() for reliable widget discovery
    select_finder = rc.find(v.Select)
    assert len(select_finder) >= 1, (
        f"No Select widget found in rendered tree. "
        f"Found widget types: {_collect_widget_types(box)}"
    )


# ---------------------------------------------------------------------------
# OI-02: bool field renders Checkbox
# ---------------------------------------------------------------------------

def test_oi02_bool_field_renders_checkbox():
    """OI-02: FormField widget_type='bool' renders a Checkbox widget."""
    from macroforecast.wizard.components.option_input import OptionInput

    field = _make_field(
        axis_name="hln_correction",
        widget_type="bool",
        options=[],
        default=True,
    )
    el = OptionInput(field=field, layer_id="l6", on_change=lambda *a: None, current_value=True)
    box, rc = reacton.render(el)

    widget_types = _collect_widget_types(box)
    assert "Checkbox" in widget_types, (
        f"No 'Checkbox' widget found in rendered tree. Found: {widget_types}"
    )


# ---------------------------------------------------------------------------
# OI-03: text field renders InputText (TextField)
# ---------------------------------------------------------------------------

def test_oi03_text_field_renders_input():
    """OI-03: FormField widget_type='text' renders a text input widget."""
    from macroforecast.wizard.components.option_input import OptionInput

    field = _make_field(
        axis_name="some_text_field",
        widget_type="text",
        options=[],
        default="CPIAUCSL",
    )
    el = OptionInput(field=field, layer_id="l0", on_change=lambda *a: None, current_value="CPIAUCSL")
    box, rc = reacton.render(el)

    # Solara InputText renders as TextField (v-text-field in vuetify)
    text_finder = rc.find(v.TextField)
    widget_types = _collect_widget_types(box)
    assert len(text_finder) >= 1 or "TextField" in widget_types, (
        f"No 'TextField' widget found in rendered tree. Found: {widget_types}"
    )


# ---------------------------------------------------------------------------
# OI-04: help button click triggers doc display
# ---------------------------------------------------------------------------

def test_oi04_help_button_shows_doc():
    """OI-04: Clicking '?' help button makes doc panel visible."""
    from macroforecast.wizard.components.option_input import OptionInput
    from macroforecast.wizard.schema import option_doc_for

    # Verify doc is registered for this key
    doc = option_doc_for("l0", "l0_a", "failure_policy", "fail_fast")
    assert doc is not None, "Prerequisite: doc must be registered for l0/failure_policy/fail_fast"

    field = _make_field(widget_type="select")

    el = OptionInput(
        field=field,
        layer_id="l0",
        on_change=lambda *a: None,
        current_value="fail_fast",
    )
    box, rc = reacton.render(el)

    # Widget count before click
    types_before = _collect_widget_types(box)

    # Find the Btn widget (the '?' help button is the only Btn)
    btn_finder = rc.find(v.Btn)
    assert len(btn_finder) >= 1, "No Btn widget found (expected '?' help button)"

    btn_widget = btn_finder.widget  # first (and only) Btn
    assert btn_widget.children == ["?"], (
        f"Btn children should be ['?'], got: {btn_widget.children}"
    )

    # Simulate click: fires on_click callback which toggles show_help
    btn_widget.fire_event("click", {})

    # Widget count after click: doc panel should have been added (Html widgets for doc text)
    types_after = _collect_widget_types(box)
    assert len(types_after) > len(types_before), (
        f"Widget tree did not grow after help button click. "
        f"Before: {len(types_before)} widgets, After: {len(types_after)} widgets"
    )


# ---------------------------------------------------------------------------
# OI-05: on_change is called with correct axis_name and value
# ---------------------------------------------------------------------------

def test_oi05_on_change_called_with_correct_args():
    """OI-05: Setting Select v_model calls on_change with (axis_name, value)."""
    from macroforecast.wizard.components.option_input import OptionInput

    field = _make_field(
        axis_name="failure_policy",
        widget_type="select",
        options=[("fail_fast", "Fail fast"), ("continue_on_failure", "Continue")],
    )
    changes = []

    el = OptionInput(
        field=field,
        layer_id="l0",
        on_change=lambda name, val: changes.append((name, val)),
        current_value="fail_fast",
    )
    box, rc = reacton.render(el)

    # Find Select widget and trigger value change by setting v_model directly
    select_finder = rc.find(v.Select)
    assert len(select_finder) >= 1, "Select widget not found in rendered tree"

    select_w = select_finder.widget
    assert select_w.v_model == "fail_fast", (
        f"Select initial v_model expected 'fail_fast', got: {select_w.v_model!r}"
    )

    # Setting v_model triggers the on_value callback wired to on_change
    select_w.v_model = "continue_on_failure"

    assert len(changes) > 0, "on_change was not called after value change"
    assert changes[0][0] == "failure_policy", (
        f"Expected axis_name='failure_policy', got: {changes[0][0]!r}"
    )
    assert changes[0][1] == "continue_on_failure", (
        f"Expected value='continue_on_failure', got: {changes[0][1]!r}"
    )
