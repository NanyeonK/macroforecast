"""Tests for the unified ``macrocast.core.status`` vocabulary (PR-A of the
honesty pass).

Pins:

* The 2-value ``ItemStatus`` literal (``operational`` / ``future``).
* ``normalize_status`` collapses every legacy alias (``planned``,
  ``approximation``, ``simplified``, ``registry_only``,
  ``contract_defined_gated``, ``stub``) to the canonical 2-value
  vocabulary.
* ``is_runnable`` / ``is_future`` agree with ``normalize_status``.
* ``register_op`` accepts the legacy strings and stores the normalised
  value on the ``OpSpec``.
* ``NodeStatus`` and ``OpStatus`` typing aliases resolve to ``ItemStatus``.
"""
from __future__ import annotations

from typing import get_args, get_origin

import pytest

from macrocast.core import (
    FUTURE,
    KNOWN_STATUSES,
    OPERATIONAL,
    ItemStatus,
    is_future,
    is_runnable,
    normalize_status,
)
from macrocast.core.dag import NodeStatus
from macrocast.core.ops.registry import OpStatus, OpSpec, _OPS, register_op
from macrocast.core.types import Panel


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_known_statuses_is_two_value():
    assert KNOWN_STATUSES == frozenset({"operational", "future"})


def test_constants_match_vocabulary():
    assert OPERATIONAL == "operational"
    assert FUTURE == "future"


def test_item_status_literal_has_two_args():
    args = get_args(ItemStatus)
    assert set(args) == {"operational", "future"}


def test_node_status_and_op_status_alias_to_item_status():
    # Whatever ``NodeStatus`` / ``OpStatus`` resolve to, runtime callers
    # use them interchangeably with ``ItemStatus``. The simplest pin is
    # that the literal args match the ``ItemStatus`` args.
    assert set(get_args(NodeStatus)) == set(get_args(ItemStatus))
    assert set(get_args(OpStatus)) == set(get_args(ItemStatus))


# ---------------------------------------------------------------------------
# normalize_status
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["operational", "future"])
def test_normalize_status_passthrough_for_canonical_values(value):
    assert normalize_status(value) == value


@pytest.mark.parametrize(
    "legacy",
    [
        "planned",
        "approximation",
        "simplified",
        "registry_only",
        "contract_defined_gated",
        "stub",
    ],
)
def test_normalize_status_collapses_legacy_aliases_to_future(legacy):
    assert normalize_status(legacy) == FUTURE


def test_normalize_status_none_is_operational_default():
    assert normalize_status(None) == OPERATIONAL
    assert normalize_status("") == OPERATIONAL


def test_normalize_status_unknown_strings_default_to_operational():
    # Lenient default: unknown strings don't crash. Strict callers that need
    # to reject unknown values should compare against KNOWN_STATUSES.
    assert normalize_status("totally_unknown") == OPERATIONAL


# ---------------------------------------------------------------------------
# is_runnable / is_future
# ---------------------------------------------------------------------------

def test_is_runnable_only_for_operational():
    assert is_runnable("operational")
    assert is_runnable(None)  # default
    assert not is_runnable("future")
    assert not is_runnable("planned")  # legacy alias collapses to future
    assert not is_runnable("registry_only")


def test_is_future_inverse_of_is_runnable_for_canonical_values():
    for status in ("operational", "future", "planned", "registry_only", None):
        assert is_runnable(status) is not is_future(status)


# ---------------------------------------------------------------------------
# register_op normalises legacy aliases at registration time
# ---------------------------------------------------------------------------

def test_register_op_normalises_legacy_status_string():
    name = "_status_test_legacy_alias_op"
    if name in _OPS:
        del _OPS[name]
    register_op(
        name=name,
        layer_scope=("l3",),
        input_types={"default": Panel},
        output_type=Panel,
        status="planned",  # legacy alias
    )(lambda inputs, params: inputs[0])
    spec = _OPS[name]
    assert isinstance(spec, OpSpec)
    assert spec.status == FUTURE  # collapsed
    del _OPS[name]


def test_register_op_passes_canonical_status_through():
    name = "_status_test_canonical_op"
    if name in _OPS:
        del _OPS[name]
    register_op(
        name=name,
        layer_scope=("l3",),
        input_types={"default": Panel},
        output_type=Panel,
        status="future",
    )(lambda inputs, params: inputs[0])
    assert _OPS[name].status == FUTURE
    del _OPS[name]
