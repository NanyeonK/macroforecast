"""Phase 3e smoke tests: macroforecast.layers.l7_interpretation collocation.

Verifies that:
1. The collocated schema, ops, methods, and option_docs submodules are importable.
2. L7Interpretation from schema is the class registered in the layer registry.
3. PRE_DEFINED_BLOCKS and OPERATIONAL_OPS are accessible from the collocated ops module.
4. GIRF and LSTMHiddenState are accessible from methods (backward compat preserved).
5. The old macroforecast.core.ops.l7_ops path still resolves (backward compat).
"""
from __future__ import annotations


def test_collocated_schema_import() -> None:
    from macroforecast.layers.l7_interpretation.schema import (
        L7Interpretation,
        L7_OUTPUT_AXES,
        DEFAULT_AXES,
        L7_LAYER_SPEC,
    )
    assert "l7" in L7_LAYER_SPEC.layer_id
    assert len(L7_OUTPUT_AXES) == 8
    assert DEFAULT_AXES["enabled"] is False


def test_collocated_ops_import() -> None:
    from macroforecast.layers.l7_interpretation.ops import (
        OPERATIONAL_OPS,
        PRE_DEFINED_BLOCKS,
        DEFAULT_FIGURE_MAPPING,
        FUTURE_OPS,
    )
    assert "permutation_importance" in OPERATIONAL_OPS
    assert "mccracken_ng_md_groups" in PRE_DEFINED_BLOCKS
    assert len(DEFAULT_FIGURE_MAPPING) > 0


def test_collocated_methods_import() -> None:
    from macroforecast.layers.l7_interpretation.methods import GIRF, LSTMHiddenState
    assert GIRF is not None
    assert LSTMHiddenState is not None


def test_collocated_option_docs_import() -> None:
    import macroforecast.layers.l7_interpretation.option_docs  # noqa: F401
    # option_docs triggers registration; verify the docs are present
    from macroforecast.scaffold.option_docs import OPTION_DOCS
    l7_keys = [k for k in OPTION_DOCS if k[0] == "l7"]
    assert len(l7_keys) > 0


def test_l7_registry_uses_collocated_class() -> None:
    from macroforecast.core.layers.registry import get_layer
    from macroforecast.layers.l7_interpretation.schema import L7Interpretation

    spec = get_layer("l7")
    assert spec.cls is L7Interpretation


def test_backward_compat_core_ops_l7_ops() -> None:
    # The old import path must still work for existing test code.
    from macroforecast.core.ops.l7_ops import (
        OPERATIONAL_OPS,
        PRE_DEFINED_BLOCKS,
        HONESTY_DEMOTED_L7_OPS,
    )
    assert "permutation_importance" in OPERATIONAL_OPS
    assert isinstance(HONESTY_DEMOTED_L7_OPS, tuple)


def test_backward_compat_interpretation_module() -> None:
    from macroforecast.interpretation import GIRF, LSTMHiddenState
    assert GIRF is not None
    assert LSTMHiddenState is not None
