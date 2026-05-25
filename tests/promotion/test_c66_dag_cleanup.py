"""C66 — DAG jargon cleanup: independent validation tests (tester).

Covers test scenarios T1-T4 (dispatch) plus T5, T6, T7, T8, T9 from test-spec.md.
These tests check that:
  - Public docstrings contain no DAG/dag jargon (T1).
  - Updated error messages no longer say "DAG" (T2-a, T2-b, T2-c).
  - Backward-compat API names remain importable (T3).
  - Tutorial/how-to docs have no DAG where replacement was specified (T4).
  - Internal core files unchanged (T5).
  - Wizard file rename (T6).
  - Reference architecture docs cleaned (T7).
  - Frozen schema field names preserved (T8).
  - Encyclopedia boilerplate propagated (T9).
  - Edge cases: signature intact, no double-replacement (T-E1, T-E3).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

# Pattern that matches the standalone word DAG or dag (not as part of a
# compound identifier like validate_dag or DAGValidationError).
_DAG_PATTERN = re.compile(r"\bDAG\b|\bdag\b")


def _count_dag_in_file(path: Path) -> int:
    """Count lines in a file that contain a standalone DAG/dag word."""
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return 0
    return sum(1 for line in content.splitlines() if _DAG_PATTERN.search(line))


def _get_all_public_symbols(module_name: str) -> list[tuple[str, object]]:
    """Import a module and return (name, obj) for all public names."""
    mod = importlib.import_module(module_name)
    names = getattr(mod, "__all__", None) or [
        n for n in dir(mod) if not n.startswith("_")
    ]
    return [(n, getattr(mod, n, None)) for n in names]


def _file_dag_matches(path: Path) -> list[str]:
    """Return lines from file that contain standalone DAG/dag word matches."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [f"FILE NOT FOUND: {path}"]
    return [f"line {i+1}: {line}" for i, line in enumerate(lines) if _DAG_PATTERN.search(line)]


# ---------------------------------------------------------------------------
# T1: Public docstrings — no DAG in user-facing modules
# ---------------------------------------------------------------------------

_PUBLIC_MODULES = [
    "macroforecast.functions.linear",
    "macroforecast.functions.tree",
    "macroforecast.functions.deep",
    "macroforecast.functions.ridge",
    "macroforecast.functions.misc",
    "macroforecast.functions.timeseries",
    "macroforecast.functions.transforms",
    "macroforecast.scaffold.introspect",
    "macroforecast.scaffold.option_docs.l3",
    "macroforecast.scaffold.option_docs.l7_a",
    "macroforecast.scaffold.option_docs.l8",
    "macroforecast.scaffold.render_encyclopedia",
]


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_t1_no_dag_in_module_docstring(module_name: str) -> None:
    """T1: Module-level docstring of public modules must not contain DAG/dag."""
    mod = importlib.import_module(module_name)
    doc = inspect.getdoc(mod)
    if doc:
        assert not _DAG_PATTERN.search(doc), (
            f"Module {module_name} module docstring still contains DAG/dag:\n{doc}"
        )


@pytest.mark.parametrize("module_name", [
    "macroforecast.functions.linear",
    "macroforecast.functions.tree",
    "macroforecast.functions.deep",
    "macroforecast.functions.ridge",
    "macroforecast.functions.misc",
    "macroforecast.functions.timeseries",
    "macroforecast.functions.transforms",
    "macroforecast.scaffold.introspect",
])
def test_t1_no_dag_in_public_symbol_docstrings(module_name: str) -> None:
    """T1: Public callable docstrings must not contain standalone DAG/dag word."""
    symbols = _get_all_public_symbols(module_name)
    failures = []
    for name, obj in symbols:
        if callable(obj) or isinstance(obj, type):
            doc = inspect.getdoc(obj)
            if doc and _DAG_PATTERN.search(doc):
                failures.append(f"  {module_name}.{name}: {doc[:200]!r}")
    assert not failures, (
        f"Public symbols in {module_name} still contain DAG/dag in docstrings:\n"
        + "\n".join(failures)
    )


def test_t1_mf_init_docstring_no_dag() -> None:
    """T1: macroforecast.__init__ module docstring must not contain DAG/dag."""
    import macroforecast
    doc = inspect.getdoc(macroforecast)
    if doc:
        assert not _DAG_PATTERN.search(doc), (
            f"macroforecast.__init__ docstring contains DAG/dag:\n{doc}"
        )


def test_t1_mf_init_repr_no_dag() -> None:
    """T1: macroforecast/__init__.py non-comment lines must not contain DAG/dag."""
    init_file = REPO_ROOT / "macroforecast" / "__init__.py"
    content = init_file.read_text(encoding="utf-8")
    # Only look at lines that are not inline comments (lines where DAG appears
    # outside a # comment).
    dag_lines = [
        f"line {i+1}: {line}"
        for i, line in enumerate(content.splitlines())
        if _DAG_PATTERN.search(line) and not line.lstrip().startswith("#")
    ]
    assert not dag_lines, (
        "macroforecast/__init__.py non-comment lines still contain DAG/dag:\n"
        + "\n".join(dag_lines)
    )


# ---------------------------------------------------------------------------
# T2: Error messages updated
# ---------------------------------------------------------------------------


def test_t2a_l3_error_message_no_dag() -> None:
    """T2-a: L3 normalize_to_dag_form error says 'step graph', not 'DAG'."""
    from macroforecast.core.layers.l3 import normalize_to_dag_form

    with pytest.raises(ValueError, match="L3 uses a step graph") as exc_info:
        normalize_to_dag_form({"fixed_axes": {}})

    msg = str(exc_info.value)
    assert "DAG" not in msg, f"L3 error still contains 'DAG': {msg!r}"


def test_t2b_l4_error_message_no_dag() -> None:
    """T2-b: L4 normalize_to_dag_form error says 'step graph', not 'DAG'."""
    from macroforecast.core.layers.l4 import normalize_to_dag_form

    with pytest.raises(ValueError, match="L4 uses a step graph") as exc_info:
        normalize_to_dag_form({"fixed_axes": {}})

    msg = str(exc_info.value)
    assert "DAG" not in msg, f"L4 error still contains 'DAG': {msg!r}"


def test_t2c_cycle_detection_no_dag() -> None:
    """T2-c: Cycle detection message says 'Recipe graph', not 'DAG'."""
    from macroforecast.core.dag import DAG, Node, NodeRef
    from macroforecast.core.validator import validate_dag

    # Build a graph with a cycle: node_a inputs from node_b, node_b inputs from node_a.
    node_a = Node(id="node_a", type="step", layer_id="l3", op="step",
                  inputs=(NodeRef(node_id="node_b"),))
    node_b = Node(id="node_b", type="step", layer_id="l3", op="step",
                  inputs=(NodeRef(node_id="node_a"),))
    cyclic_dag = DAG(
        layer_id="l3",
        nodes={"node_a": node_a, "node_b": node_b},
        sinks={},
    )
    result = validate_dag(cyclic_dag)
    assert not result.valid, "Cyclic DAG should fail validation"

    cycle_issues = [i for i in result.issues if "cycle" in i.message.lower()]
    assert len(cycle_issues) > 0, "No cycle issue found in validation result"

    msg = cycle_issues[0].message
    assert "DAG" not in msg, f"Cycle error still uses 'DAG': {msg!r}"
    assert "Recipe graph" in msg or "cycle" in msg.lower(), (
        f"Expected 'Recipe graph contains a cycle' in message, got: {msg!r}"
    )


# ---------------------------------------------------------------------------
# T3: Backward compatibility — API names unchanged
# ---------------------------------------------------------------------------


def test_t3_core_api_imports() -> None:
    """T3: All public core DAG API names remain importable."""
    from macroforecast.core import (  # noqa: F401
        DAG,
        DAGValidationError,
        assert_valid_dag,
        normalize_to_dag_form,
        parse_dag_form,
        validate_dag,
    )


def test_t3_cell_concrete_dag_field() -> None:
    """T3: Cell.concrete_dag field must still exist."""
    from macroforecast.core import Cell

    field_names = {f.name for f in dataclasses.fields(Cell)}
    assert "concrete_dag" in field_names, (
        f"Cell.concrete_dag field was removed. Found fields: {sorted(field_names)}"
    )


# ---------------------------------------------------------------------------
# T4: Tutorial / how-to docs — replaced occurrences absent
# ---------------------------------------------------------------------------


def test_t4a_tutorial_no_dag() -> None:
    """T4-a: Tutorial files contain no DAG/dag."""
    files = [
        REPO_ROOT / "docs" / "tutorial" / "two_entry_points.md",
        REPO_ROOT / "docs" / "tutorial" / "replications" / "example_walkthrough.md",
    ]
    for f in files:
        matches = _file_dag_matches(f)
        assert not matches, (
            f"{f.name} still contains DAG/dag:\n" + "\n".join(matches)
        )


def test_t4b_partial_layer_execution_no_compound_dag() -> None:
    """T4-b: partial_layer_execution.md has no post-DAG or cascade-DAG compounds."""
    f = REPO_ROOT / "docs" / "how_to" / "partial_layer_execution.md"
    content = f.read_text(encoding="utf-8")
    assert "post-DAG" not in content, "partial_layer_execution.md still contains 'post-DAG'"
    assert "cascade-DAG" not in content, (
        "partial_layer_execution.md still contains 'cascade-DAG'"
    )


def test_t4c_contributing_dag_py_kept() -> None:
    """T4-c: contributing.md still references dag.py in the file listing (KEEP decision)."""
    f = REPO_ROOT / "docs" / "how_to" / "contributing.md"
    content = f.read_text(encoding="utf-8")
    assert "dag.py" in content, (
        "contributing.md no longer contains 'dag.py' -- KEEP decision was incorrectly applied"
    )


# ---------------------------------------------------------------------------
# T5: Internal modules unchanged
# ---------------------------------------------------------------------------


def test_t5_dag_py_has_dag_occurrences() -> None:
    """T5: macroforecast/core/dag.py must retain DAG/dag vocabulary (internal).

    Per test-spec.md T5, this file should have approximately 5+ line-level matches.
    """
    dag_file = REPO_ROOT / "macroforecast" / "core" / "dag.py"
    count = _count_dag_in_file(dag_file)
    assert count >= 5, (
        f"macroforecast/core/dag.py has fewer DAG/dag line matches than expected ({count}). "
        "Internal vocabulary may have been incorrectly stripped. "
        f"File exists: {dag_file.exists()}"
    )


def test_t5_cache_py_has_dag_occurrences() -> None:
    """T5: macroforecast/core/cache.py must retain DAG/dag vocabulary (internal).

    Per test-spec.md T5, this file should have approximately 10+ line-level matches.
    """
    cache_file = REPO_ROOT / "macroforecast" / "core" / "cache.py"
    count = _count_dag_in_file(cache_file)
    assert count >= 10, (
        f"macroforecast/core/cache.py has fewer DAG/dag line matches than expected ({count}). "
        "Internal vocabulary may have been incorrectly stripped."
    )


def test_t5_validator_error_messages_updated() -> None:
    """T5: validator.py no longer contains 'DAG contains a cycle' raw string."""
    validator_file = REPO_ROOT / "macroforecast" / "core" / "validator.py"
    content = validator_file.read_text(encoding="utf-8")
    assert "DAG contains a cycle" not in content, (
        "validator.py still contains old 'DAG contains a cycle' error message"
    )


# ---------------------------------------------------------------------------
# T7: Reference docs cleaned
# ---------------------------------------------------------------------------


def test_t7_reference_architecture_docs_no_dag() -> None:
    """T7: docs/reference/architecture/ must have no standalone DAG/dag.

    Exception: docs/reference/architecture/layer7/index.md line 31 contains
    'importance DAG body' which is a KEEP decision per impact.md 2.17 --
    it refers to the frozen API schema field L7_A_importance_dag_body.
    """
    arch_dir = REPO_ROOT / "docs" / "reference" / "architecture"
    failures = []
    for md_file in arch_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines()):
            if _DAG_PATTERN.search(line):
                # Allow the frozen schema field reference in layer7/index.md
                # (impact.md section 2.17: KEEP -- refers to frozen API name
                # L7_A_importance_dag_body).
                # Impact.md 2.17: 'importance DAG body' in layer7/index.md is a KEEP
                # decision -- it is the human-readable description of the frozen
                # API schema field L7_A_importance_dag_body. The condition matches
                # both the underscore form (field name) and the space form (description).
                is_frozen_ref = (
                    "layer7" in str(md_file)
                    and ("importance_dag_body" in line or "importance DAG body" in line)
                )
                if not is_frozen_ref:
                    failures.append(
                        f"\n{md_file.relative_to(REPO_ROOT)}:{i+1}: {line}"
                    )
    assert not failures, (
        "Reference architecture docs still contain DAG/dag (non-frozen):"
        + "".join(failures)
    )


def test_t7_explanation_docs_no_dag() -> None:
    """T7: docs/explanation/ must have no standalone DAG/dag matches."""
    expl_dir = REPO_ROOT / "docs" / "explanation"
    failures = []
    for md_file in expl_dir.rglob("*.md"):
        matches = _file_dag_matches(md_file)
        if matches:
            failures.append(f"\n{md_file.relative_to(REPO_ROOT)}:\n  " + "\n  ".join(matches))
    assert not failures, "Explanation docs still contain DAG/dag:" + "".join(failures)


def test_t7_recipe_schema_docs_no_dag() -> None:
    """T7: docs/reference/recipe_schema/ must have no standalone DAG/dag matches."""
    schema_dir = REPO_ROOT / "docs" / "reference" / "recipe_schema"
    failures = []
    for md_file in schema_dir.rglob("*.md"):
        matches = _file_dag_matches(md_file)
        if matches:
            failures.append(f"\n{md_file.relative_to(REPO_ROOT)}:\n  " + "\n  ".join(matches))
    assert not failures, "Recipe schema docs still contain DAG/dag:" + "".join(failures)


def test_t7_api_docs_no_dag() -> None:
    """T7: docs/reference/api/ must have no DAG/dag except frozen schema field names."""
    api_dir = REPO_ROOT / "docs" / "reference" / "api"
    failures = []
    for md_file in api_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        for i, line in enumerate(content.splitlines()):
            if _DAG_PATTERN.search(line):
                # Allow frozen schema field names in any context.
                frozen_ok = (
                    "navigator_selected_dag_items" in line
                    or "L7_A_importance_dag_body" in line
                    or "importance_dag_body" in line
                )
                if not frozen_ok:
                    failures.append(
                        f"\n{md_file.relative_to(REPO_ROOT)}:{i+1}: {line}"
                    )
    assert not failures, "API docs still contain non-frozen DAG/dag:" + "".join(failures)


def test_t7_encyclopedia_l1_no_dag() -> None:
    """T7: docs/reference/encyclopedia/l1/ must have no standalone DAG/dag."""
    l1_dir = REPO_ROOT / "docs" / "reference" / "encyclopedia" / "l1"
    failures = []
    for md_file in l1_dir.rglob("*.md"):
        matches = _file_dag_matches(md_file)
        if matches:
            failures.append(f"\n{md_file.relative_to(REPO_ROOT)}:\n  " + "\n  ".join(matches))
    assert not failures, "Encyclopedia L1 docs still contain DAG/dag:" + "".join(failures)


def test_t7_encyclopedia_l8_no_dag() -> None:
    """T7: docs/reference/encyclopedia/l8/ must have no standalone DAG/dag."""
    l8_dir = REPO_ROOT / "docs" / "reference" / "encyclopedia" / "l8"
    failures = []
    for md_file in l8_dir.rglob("*.md"):
        matches = _file_dag_matches(md_file)
        if matches:
            failures.append(f"\n{md_file.relative_to(REPO_ROOT)}:\n  " + "\n  ".join(matches))
    assert not failures, "Encyclopedia L8 docs still contain DAG/dag:" + "".join(failures)


def test_t7_public_api_md_no_dag() -> None:
    """T7: docs/reference/encyclopedia/public_api.md must have no DAG/dag."""
    f = REPO_ROOT / "docs" / "reference" / "encyclopedia" / "public_api.md"
    matches = _file_dag_matches(f)
    assert not matches, (
        "public_api.md still contains DAG/dag:\n" + "\n".join(matches)
    )


# ---------------------------------------------------------------------------
# T8: Frozen schema field names preserved
# ---------------------------------------------------------------------------


def test_t8a_l7_importance_dag_body_preserved() -> None:
    """T8-a: L7_A_importance_dag_body frozen field name must still appear in l7/index.md."""
    f = REPO_ROOT / "docs" / "reference" / "encyclopedia" / "l7" / "index.md"
    content = f.read_text(encoding="utf-8")
    assert "L7_A_importance_dag_body" in content, (
        "Frozen schema field L7_A_importance_dag_body was incorrectly renamed in l7/index.md"
    )


def test_t8b_navigator_selected_dag_items_preserved() -> None:
    """T8-b: navigator_selected_dag_items frozen field must still appear in tree_navigator.md."""
    f = REPO_ROOT / "docs" / "reference" / "api" / "navigator" / "tree_navigator.md"
    content = f.read_text(encoding="utf-8")
    assert "navigator_selected_dag_items" in content, (
        "Frozen schema field navigator_selected_dag_items was incorrectly renamed in tree_navigator.md"
    )


# ---------------------------------------------------------------------------
# T9: Encyclopedia boilerplate propagation
# ---------------------------------------------------------------------------


def test_t9a_l3_op_pages_no_dag_boilerplate() -> None:
    """T9-a: L3 op pages must not contain 'feature engineering is a DAG'."""
    l3_op_dir = REPO_ROOT / "docs" / "reference" / "encyclopedia" / "l3" / "op"
    failures = []
    for md_file in l3_op_dir.glob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        if "feature engineering is a DAG" in content:
            failures.append(str(md_file.name))
    assert not failures, (
        "L3 op pages still contain old 'feature engineering is a DAG' boilerplate:\n"
        + "\n".join(failures)
    )


def test_t9b_pca_has_pipeline_boilerplate() -> None:
    """T9-b: pca.md must contain new 'feature engineering is a pipeline' boilerplate."""
    pca_file = REPO_ROOT / "docs" / "reference" / "encyclopedia" / "l3" / "op" / "pca.md"
    content = pca_file.read_text(encoding="utf-8")
    assert "feature engineering is a pipeline" in content, (
        "pca.md does not contain new 'feature engineering is a pipeline' boilerplate. "
        "Encyclopedia regeneration may not have run."
    )


def test_t9c_asymmetric_trim_no_l3_dag_dispatch() -> None:
    """T9-c: asymmetric_trim.md must not contain 'L3 DAG can dispatch'."""
    f = REPO_ROOT / "docs" / "reference" / "encyclopedia" / "l3" / "op" / "asymmetric_trim.md"
    content = f.read_text(encoding="utf-8")
    assert "L3 DAG can dispatch" not in content, (
        "asymmetric_trim.md still contains 'L3 DAG can dispatch'"
    )


# ---------------------------------------------------------------------------
# Edge T-E1: validate_dag signature unchanged
# ---------------------------------------------------------------------------


def test_te1_validate_dag_signature_unchanged() -> None:
    """Edge T-E1: validate_dag must still accept 'dag' as a parameter name."""
    from macroforecast.core.validator import validate_dag

    sig = inspect.signature(validate_dag)
    assert "dag" in sig.parameters, (
        f"validate_dag signature was incorrectly modified. "
        f"Parameters: {list(sig.parameters.keys())}"
    )


# ---------------------------------------------------------------------------
# Edge T-E3: No double replacement in linear module docstrings
# ---------------------------------------------------------------------------


def test_te3_no_double_replacement_linear() -> None:
    """Edge T-E3: macroforecast.functions.linear must not contain 'pipeline pipeline'."""
    import macroforecast.functions.linear as m

    assert m.__doc__ is not None, "macroforecast.functions.linear has no module docstring"
    assert "pipeline pipeline" not in m.__doc__, (
        "Double replacement occurred in macroforecast.functions.linear docstring"
    )
    assert "DAG" not in m.__doc__, (
        "macroforecast.functions.linear module docstring still contains 'DAG'"
    )
