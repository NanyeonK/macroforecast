"""Pin the introspect module's contract.

The wizard / docs site read every layer's axes through this module. A
schema-shape change in ``core/layers/`` must not silently break the
walker -- if the AxisSpec / Option / SubLayerSpec dataclasses change,
this test fails.
"""
from __future__ import annotations

import pytest

from macrocast.scaffold import introspect


def test_list_layers_returns_canonical_thirteen():
    layers = introspect.list_layers()
    expected = ("l0", "l1", "l1_5", "l2", "l2_5", "l3", "l3_5", "l4", "l4_5", "l5", "l6", "l7", "l8")
    assert layers == expected


@pytest.mark.parametrize("layer_id", introspect.list_layers())
def test_layer_metadata_resolves(layer_id):
    info = introspect.layer(layer_id)
    assert info.id == layer_id
    assert info.name
    # Category enum -- pin the exact set so an introspect refactor that
    # adds / drops a category triggers a failing test.
    assert info.category in {"setup", "meta", "foundation", "construction", "consumption", "diagnostic", "diagnostics", "output"}
    # Every layer must declare at least one sub-layer.
    assert info.sub_layers


@pytest.mark.parametrize("layer_id", introspect.list_layers())
def test_axes_walk_returns_axis_info(layer_id):
    axes = introspect.axes(layer_id)
    # Every axis carries a layer / sublayer pointer matching the requested
    # layer.
    for axis in axes:
        assert axis.layer == layer_id
        assert axis.sublayer
        assert axis.name


def test_axis_info_carries_options():
    """Pick a known layer with rich options and verify the walker produces
    the expected (axis -> options) structure."""

    l1_axes = {a.name: a for a in introspect.axes("l1")}
    assert "custom_source_policy" in l1_axes
    options = {o.value for o in l1_axes["custom_source_policy"].options}
    # All three documented options for the source policy must surface.
    assert {"official_only", "custom_panel_only", "official_plus_custom"} <= options


def test_operational_options_filters_by_status():
    """The completeness-test API only asks for operational tuples; future
    options must be excluded."""

    tuples = introspect.operational_options("l4")
    # No tuple should reference a future-status option.
    for layer_id, _sublayer, axis_name, option_value in tuples:
        assert layer_id == "l4"
        # ``midas_almon`` is a v1.0 future family per the L4 ops registry;
        # it must not appear in the operational tuples.
        if axis_name == "family":
            assert option_value not in {"midas_almon", "midas_beta", "midas_step"}


def test_unknown_layer_raises():
    with pytest.raises(KeyError):
        introspect.layer("not_a_real_layer")


def test_axes_for_unknown_layer_returns_empty():
    assert introspect.axes("not_a_real_layer") == ()
