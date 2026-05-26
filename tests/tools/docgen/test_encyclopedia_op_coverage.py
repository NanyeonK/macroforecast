"""Encyclopedia op-coverage drift CI gate (Cycle 58).

Purpose
-------
Enumerate all operational ops/families from three authoritative sources
and assert that a checked-in encyclopedia reference page exists for each
one whose ``OptionDoc`` entry has ``op_page=True``.

Sources
-------
- **L3**: ``macroforecast.core.ops.registry._OPS`` — runtime registry
  populated by ``register_op()`` calls in layer ops modules. Filter:
  ``status == 'operational'`` AND ``'l3' in layer_scope``.
- **L4**: ``macroforecast.layers.l4_models.ops.OPERATIONAL_MODELS`` —
  explicit tuple of operational model names.
- **L7**: ``macroforecast.layers.l7_interpretation.ops.OPERATIONAL_OPS`` —
  derived tuple of operational importance ops.

Skip contract
-------------
For each op/family, the ``OptionDoc`` registry is queried at module
import time using the layer-specific key:

- L3: ``('l3', 'L3_A_step_op', 'op', name)``
- L4: ``('l4', 'L4_A_model_selection', 'family', name)``
- L7: ``('l7', 'L7_A_importance_dag_body', 'op', name)``

If the entry does **not** exist, or if ``entry.op_page is False``, the
op is excluded from the parametrized test list (no page is expected).
Only when an entry exists **and** ``entry.op_page is True`` is the
filesystem check performed.

Expected counts on HEAD b1babb65 (post-C57)
--------------------------------------------
- L3: 42 parametrized items (53 total ops - 11 op_page=False)
- L4: 43 parametrized items (47 total families - 4 op_page=False)
- L7: 10 parametrized items (36 total ops - 26 op_page=False)
- Total: 95 parametrized test items — all pass on HEAD.

Failure message format
----------------------
When a page is absent, ``pytest.fail`` is called with a message of the
form::

    {layer} op '{name}' is operational but has no encyclopedia page.
    Expected: {relative_path}
    Add the page or set op_page=False in its OptionDoc to intentionally opt out.

This test reads the checked-in ``docs/reference/encyclopedia/`` tree,
NOT a rendered tmp directory.  ``render_encyclopedia.write_all`` is
never called here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from macroforecast.layers.l4_models.ops import OPERATIONAL_MODELS
from macroforecast.layers.l7_interpretation.ops import OPERATIONAL_OPS
from macroforecast.core.ops.registry import _OPS
from tools.docgen.option_docs import OPTION_DOCS

# ---------------------------------------------------------------------------
# ENC_ROOT: path to the checked-in encyclopedia tree.
# From tests/tools/docgen/ we go up three levels to reach the repo root, then
# descend into docs/reference/encyclopedia/.
# ---------------------------------------------------------------------------
ENC_ROOT: Path = Path(__file__).parents[3] / "docs" / "reference" / "encyclopedia"


# ---------------------------------------------------------------------------
# Helper: build the list of L3 operational op names from the registry.
# ---------------------------------------------------------------------------

def _l3_op_names() -> list[str]:
    """Return sorted names of L3 operational ops from the runtime registry.

    Filters _OPS for entries where status == 'operational' AND the
    layer_scope includes 'l3'.  Sorts for stable parametrize ordering.
    """
    names: list[str] = []
    for name, spec in _OPS.items():
        if (
            getattr(spec, "status", None) == "operational"
            and "l3" in getattr(spec, "layer_scope", [])
        ):
            names.append(name)
    return sorted(names)


# ---------------------------------------------------------------------------
# Helper: decide whether to include an op in the parametrized test list.
# Returns True only when an OptionDoc entry exists AND op_page is True.
# ---------------------------------------------------------------------------

def _should_check(layer: str, sublayer: str, axis: str, name: str) -> bool:
    """Return True if this op should have its encyclopedia page checked.

    Looks up (layer, sublayer, axis, name) in OPTION_DOCS.  Returns
    True only when the entry exists and entry.op_page is True.
    No-entry and op_page=False both resolve to False (skip).
    """
    entry = OPTION_DOCS.get((layer, sublayer, axis, name))
    if entry is None:
        # No OptionDoc for this op — intent unknown, skip defensively.
        return False
    return entry.op_page is True


# ---------------------------------------------------------------------------
# Module-level parametrize list construction.
# Built at import time so pytest's collection phase sees all items and
# produces clean parametrized IDs without runtime filtering inside test bodies.
# ---------------------------------------------------------------------------

# L3: 53 operational ops in registry, 11 have op_page=False -> 42 checked.
_L3_CASES: list[tuple[str, Path]] = [
    (name, ENC_ROOT / "l3" / "op" / f"{name}.md")
    for name in _l3_op_names()
    if _should_check("l3", "L3_A_step_op", "op", name)
]

# L4: 47 OPERATIONAL_MODELS, 4 have op_page=False -> 43 checked.
_L4_CASES: list[tuple[str, Path]] = [
    (name, ENC_ROOT / "l4" / "model" / f"{name}.md")
    for name in sorted(OPERATIONAL_MODELS)
    if _should_check("l4", "L4_A_model_selection", "model", name)
]

# L7: 36 OPERATIONAL_OPS, 26 have op_page=False -> 10 checked.
_L7_CASES: list[tuple[str, Path]] = [
    (name, ENC_ROOT / "l7" / "op" / f"{name}.md")
    for name in sorted(OPERATIONAL_OPS)
    if _should_check("l7", "L7_A_importance_dag_body", "op", name)
]


# ---------------------------------------------------------------------------
# Test functions: one per layer, parametrized at collection time.
# Each test asserts the page file exists at the expected path and emits
# an informative failure message naming the missing page and the opt-out
# mechanism.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,page_path", _L3_CASES, ids=[c[0] for c in _L3_CASES])
def test_l3_op_has_encyclopedia_page(name: str, page_path: Path) -> None:
    """Assert the L3 op's encyclopedia page exists on disk."""
    if not page_path.exists():
        # Compute the relative path from the repo root for the error message.
        repo_root = Path(__file__).parents[3]
        try:
            relative = page_path.relative_to(repo_root)
        except ValueError:
            relative = page_path
        pytest.fail(
            f"L3 op '{name}' is operational but has no encyclopedia page.\n"
            f"Expected: {relative}\n"
            f"Add the page or set op_page=False in its OptionDoc to intentionally opt out."
        )


@pytest.mark.parametrize("name,page_path", _L4_CASES, ids=[c[0] for c in _L4_CASES])
def test_l4_family_has_encyclopedia_page(name: str, page_path: Path) -> None:
    """Assert the L4 model family's encyclopedia page exists on disk."""
    if not page_path.exists():
        repo_root = Path(__file__).parents[3]
        try:
            relative = page_path.relative_to(repo_root)
        except ValueError:
            relative = page_path
        pytest.fail(
            f"L4 family '{name}' is operational but has no encyclopedia page.\n"
            f"Expected: {relative}\n"
            f"Add the page or set op_page=False in its OptionDoc to intentionally opt out."
        )


@pytest.mark.parametrize("name,page_path", _L7_CASES, ids=[c[0] for c in _L7_CASES])
def test_l7_op_has_encyclopedia_page(name: str, page_path: Path) -> None:
    """Assert the L7 importance op's encyclopedia page exists on disk."""
    if not page_path.exists():
        repo_root = Path(__file__).parents[3]
        try:
            relative = page_path.relative_to(repo_root)
        except ValueError:
            relative = page_path
        pytest.fail(
            f"L7 op '{name}' is operational but has no encyclopedia page.\n"
            f"Expected: {relative}\n"
            f"Add the page or set op_page=False in its OptionDoc to intentionally opt out."
        )
