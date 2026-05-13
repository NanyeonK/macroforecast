"""Tests for macroforecast.wizard.schema — Scenarios SC-01..SC-10 + INV-03..INV-06."""
from __future__ import annotations

import pytest

from macroforecast.wizard.schema import FormField, layer_form_schema, option_doc_for


# ---------------------------------------------------------------------------
# SC-01: L0 schema produces exactly 3 operational FormFields
# ---------------------------------------------------------------------------

def test_sc01_l0_schema_three_fields():
    """SC-01: layer_form_schema('l0') returns exactly 3 operational fields."""
    fields = layer_form_schema("l0")
    assert len(fields) == 3
    axis_names = {f.axis_name for f in fields}
    assert axis_names == {"failure_policy", "reproducibility_mode", "compute_mode"}


# ---------------------------------------------------------------------------
# SC-02: L0 failure_policy field is a select widget
# ---------------------------------------------------------------------------

def test_sc02_l0_failure_policy_is_select():
    """SC-02: failure_policy field is widget_type='select' with >= 2 options."""
    fields = layer_form_schema("l0")
    f = next(x for x in fields if x.axis_name == "failure_policy")
    assert f.widget_type == "select"
    assert len(f.options) >= 2
    option_values = [v for v, _ in f.options]
    assert "fail_fast" in option_values


# ---------------------------------------------------------------------------
# SC-03: field default matches schema default
# ---------------------------------------------------------------------------

def test_sc03_failure_policy_default():
    """SC-03: failure_policy FormField.default == 'fail_fast'."""
    fields = layer_form_schema("l0")
    f = next(x for x in fields if x.axis_name == "failure_policy")
    assert f.default == "fail_fast"


# ---------------------------------------------------------------------------
# SC-04: L2 schema produces operational fields only
# ---------------------------------------------------------------------------

def test_sc04_l2_schema_all_operational():
    """SC-04: every FormField from layer_form_schema('l2') has status='operational'."""
    fields = layer_form_schema("l2")
    for f in fields:
        assert f.status == "operational", (
            f"Non-operational field found in L2: {f.axis_name!r} has status={f.status!r}"
        )


# ---------------------------------------------------------------------------
# SC-05: option_doc_for returns OptionDoc for known key
# ---------------------------------------------------------------------------

def test_sc05_option_doc_for_known_key():
    """SC-05: option_doc_for('l0','l0_a','failure_policy','fail_fast') returns a doc."""
    doc = option_doc_for("l0", "l0_a", "failure_policy", "fail_fast")
    assert doc is not None
    assert isinstance(doc.summary, str) and len(doc.summary) > 0
    assert isinstance(doc.description, str) and len(doc.description) > 0
    assert isinstance(doc.when_to_use, str) and len(doc.when_to_use) > 0


# ---------------------------------------------------------------------------
# SC-06: option_doc_for returns None for unregistered key
# ---------------------------------------------------------------------------

def test_sc06_option_doc_for_unknown_option():
    """SC-06: option_doc_for with nonexistent option returns None."""
    doc = option_doc_for("l0", "l0_a", "failure_policy", "nonexistent_option")
    assert doc is None


# ---------------------------------------------------------------------------
# SC-07: L1 schema includes dataset axis as select
# ---------------------------------------------------------------------------

def test_sc07_l1_dataset_is_select():
    """SC-07: L1 schema has a 'dataset' field with widget_type='select'."""
    fields = layer_form_schema("l1")
    f = next((x for x in fields if x.axis_name == "dataset"), None)
    assert f is not None, "No 'dataset' field found in L1 schema"
    assert f.widget_type == "select"


# ---------------------------------------------------------------------------
# SC-08: L5 schema includes primary_metric as select
# ---------------------------------------------------------------------------

def test_sc08_l5_primary_metric_present():
    """SC-08: L5 schema has a 'primary_metric' field."""
    fields = layer_form_schema("l5")
    f = next((x for x in fields if x.axis_name == "primary_metric"), None)
    assert f is not None, "No 'primary_metric' field found in L5 schema"


# ---------------------------------------------------------------------------
# SC-09: layer_form_schema("l3") does not raise
# ---------------------------------------------------------------------------

def test_sc09_l3_schema_no_exception():
    """SC-09: layer_form_schema('l3') returns a list without raising."""
    result = layer_form_schema("l3")
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# SC-10: unknown layer raises KeyError
# BUILDER BUG: introspect.axes('l99') silently returns () instead of raising
# KeyError. layer_form_schema() catches KeyError from introspect.axes but
# introspect.axes never raises — so 'l99' yields empty list, not KeyError.
# ---------------------------------------------------------------------------

def test_sc10_unknown_layer_raises_keyerror():
    """SC-10: layer_form_schema('l99') must raise KeyError containing 'l99'.

    BUILDER BUG: currently returns [] silently. This test FAILS, confirming
    the behavioral contract is violated. Route BLOCK to builder.
    """
    with pytest.raises(KeyError) as exc_info:
        layer_form_schema("l99")
    assert "l99" in str(exc_info.value), (
        f"KeyError should contain 'l99', got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# INV-03: STAGE_COLOR_MAP is exhaustive
# ---------------------------------------------------------------------------

def test_inv03_stage_color_map_exhaustive():
    """INV-03: STAGE_COLOR_MAP covers all stage labels in STAGE_BY_LAYER."""
    from macroforecast.wizard.components.layer_rail import STAGE_COLOR_MAP
    from macroforecast.core.stages import STAGE_BY_LAYER
    stage_values = set(STAGE_BY_LAYER.values())
    color_keys = set(STAGE_COLOR_MAP.keys())
    assert stage_values.issubset(color_keys), (
        f"Missing from STAGE_COLOR_MAP: {stage_values - color_keys}"
    )


# ---------------------------------------------------------------------------
# INV-04: layer_form_schema returns only operational fields
# ---------------------------------------------------------------------------

def test_inv04_layer_form_schema_operational_only():
    """INV-04: for l0,l1,l2,l5,l6 every FormField has status='operational'."""
    for layer_id in ["l0", "l1", "l2", "l5", "l6"]:
        fields = layer_form_schema(layer_id)
        for f in fields:
            assert f.status == "operational", (
                f"Non-operational field in {layer_id}: {f.axis_name!r} status={f.status!r}"
            )


# ---------------------------------------------------------------------------
# INV-05: All FormField option values are strings
# ---------------------------------------------------------------------------

def test_inv05_option_values_are_strings():
    """INV-05: every (value, label) in FormField.options consists of strings."""
    for layer_id in ["l0", "l1", "l2", "l5", "l6"]:
        fields = layer_form_schema(layer_id)
        for f in fields:
            for val, label in f.options:
                assert isinstance(val, str), (
                    f"{layer_id}/{f.axis_name}: option value {val!r} is not str"
                )
                assert isinstance(label, str), (
                    f"{layer_id}/{f.axis_name}: option label {label!r} is not str"
                )


# ---------------------------------------------------------------------------
# INV-06: FormField default type consistency
# ---------------------------------------------------------------------------

def test_inv06_default_type_consistency():
    """INV-06: bool fields have bool default, int fields have int default, etc."""
    for layer_id in ["l0", "l1", "l2", "l5", "l6"]:
        fields = layer_form_schema(layer_id)
        for f in fields:
            if f.widget_type == "bool":
                assert isinstance(f.default, bool), (
                    f"{layer_id}/{f.axis_name}: bool field default {f.default!r} is not bool"
                )
            elif f.widget_type == "int":
                assert isinstance(f.default, int) and not isinstance(f.default, bool), (
                    f"{layer_id}/{f.axis_name}: int field default {f.default!r} is not int"
                )
            elif f.widget_type == "float":
                assert isinstance(f.default, float), (
                    f"{layer_id}/{f.axis_name}: float field default {f.default!r} is not float"
                )
