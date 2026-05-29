"""Tests for ParameterDoc dataclass and its integration with OptionDoc.

Verifies:
- ParameterDoc can be constructed and is immutable (frozen=True).
- OptionDoc accepts and stores a parameters tuple.
- L0 option docs have the expected parameters populated.
- render_encyclopedia emits a ## Parameters table when parameters non-empty.
- Options with no parameters do not emit a Parameters table.
"""
from __future__ import annotations

import pytest

from tools.docgen.option_docs.types import CodeExample, OptionDoc, ParameterDoc, Reference, REQUIRED
from tools.docgen.render_encyclopedia import _render_option_body
from tools.docgen.introspect import OptionInfo


# ---------------------------------------------------------------------------
# ParameterDoc unit tests
# ---------------------------------------------------------------------------


def test_parameter_doc_construction():
    """ParameterDoc constructs with required fields and optional defaults."""
    p = ParameterDoc(name="random_seed", type="int", default=42)
    assert p.name == "random_seed"
    assert p.type == "int"
    assert p.default == 42
    assert p.constraint is None
    assert p.description == ""


def test_parameter_doc_full():
    """ParameterDoc stores all fields correctly."""
    p = ParameterDoc(
        name="parallel_unit",
        type="str enum {cells, models}",
        default=None,
        constraint="required when compute_policy=parallel",
        description="Parallelization granularity.",
    )
    assert p.name == "parallel_unit"
    assert p.type == "str enum {cells, models}"
    assert p.default is None
    assert "required" in p.constraint
    assert "Parallelization" in p.description


def test_parameter_doc_is_frozen():
    """ParameterDoc is immutable (frozen dataclass)."""
    p = ParameterDoc(name="x", type="int", default=0)
    with pytest.raises(Exception):  # FrozenInstanceError (AttributeError in older Python)
        p.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OptionDoc.parameters field
# ---------------------------------------------------------------------------


def test_option_doc_accepts_parameters():
    """OptionDoc stores a parameters tuple correctly."""
    p = ParameterDoc(name="random_seed", type="int", default=42)
    o = OptionDoc(
        layer="l0",
        sublayer="l0_a",
        axis="reproducibility_policy",
        option="seeded_reproducible",
        summary="Test summary for seeded option.",
        description="Test description with enough chars to pass the quality floor gate for v1.0 completeness.",
        when_to_use="Default for reproducible studies when you need exact bit-level replication.",
        references=(Reference(citation="macroforecast design Part 1, L0."),),
        parameters=(p,),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    assert len(o.parameters) == 1
    assert o.parameters[0].name == "random_seed"
    assert o.parameters[0].default == 42


def test_option_doc_default_parameters_empty():
    """OptionDoc.parameters defaults to empty tuple when not supplied."""
    o = OptionDoc(
        layer="l0",
        sublayer="l0_a",
        axis="failure_policy",
        option="fail_fast",
        summary="Stop on first error.",
        description="Raises on first cell exception, skipping remaining cells in the sweep loop.",
        when_to_use="Authoring iteration where fast feedback is preferred.",
        references=(Reference(citation="macroforecast design Part 1."),),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    assert o.parameters == ()


# ---------------------------------------------------------------------------
# render_encyclopedia: Parameters table rendering
# ---------------------------------------------------------------------------


def _make_option_info(value: str) -> OptionInfo:
    """Create a minimal OptionInfo for rendering tests."""
    from tools.docgen.introspect import OptionInfo
    return OptionInfo(value=value, label=value, status="operational", description="")


def test_render_option_body_emits_parameters_table():
    """_render_option_body emits ## Parameters table when doc.parameters non-empty."""
    p = ParameterDoc(
        name="random_seed",
        type="int",
        default=42,
        constraint=">=0",
        description="RNG seed.",
    )
    doc = OptionDoc(
        layer="l0",
        sublayer="l0_a",
        axis="reproducibility_policy",
        option="seeded_reproducible",
        summary="Test summary for the seeded reproducible option.",
        description="Test description with adequate length to satisfy quality floor constraints in v1.0.",
        when_to_use="Default. Use for studies requiring bit-exact replication across environments.",
        references=(Reference(citation="macroforecast design Part 1."),),
        parameters=(p,),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = _make_option_info("seeded_reproducible")
    rendered = _render_option_body(
        opt_info, doc, layer_id="l0", sublayer="l0_a", axis="reproducibility_policy"
    )
    assert "**Parameters**" in rendered
    assert "| name | type | default | constraint | description |" in rendered
    assert "| `random_seed` |" in rendered
    assert "`42`" in rendered
    assert ">=0" in rendered
    assert "RNG seed." in rendered


def test_render_option_body_no_parameters_table_when_empty():
    """_render_option_body does NOT emit Parameters table when parameters=()."""
    doc = OptionDoc(
        layer="l0",
        sublayer="l0_a",
        axis="failure_policy",
        option="fail_fast",
        summary="Stop on first cell error.",
        description="Raises immediately on the first cell exception with no manifest written.",
        when_to_use="Authoring iteration to get fast feedback on the first failure.",
        references=(Reference(citation="macroforecast design Part 1."),),
        parameters=(),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = _make_option_info("fail_fast")
    rendered = _render_option_body(
        opt_info, doc, layer_id="l0", sublayer="l0_a", axis="failure_policy"
    )
    assert "**Parameters**" not in rendered


def test_render_option_body_none_default_renders_dash():
    """Parameters with default=REQUIRED render as '—' in the table (C26: REQUIRED sentinel)."""
    p = ParameterDoc(
        name="parallel_unit",
        type="str",
        default=REQUIRED,
        constraint="required",
        description="The parallel unit.",
    )
    doc = OptionDoc(
        layer="l0",
        sublayer="l0_a",
        axis="compute_policy",
        option="parallel",
        summary="Distribute work across workers.",
        description="Activates the parallel cell loop; granularity set by parallel_unit leaf_config key.",
        when_to_use="Long sweeps on multi-core machines after serial validation.",
        references=(Reference(citation="macroforecast PR #173."),),
        parameters=(p,),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = _make_option_info("parallel")
    rendered = _render_option_body(
        opt_info, doc, layer_id="l0", sublayer="l0_a", axis="compute_policy"
    )
    assert "| `parallel_unit` |" in rendered
    assert "| — |" in rendered
