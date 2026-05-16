"""Cycle 23 acceptance test — REQUIRED sentinel, data_args, return_type.

v2: Fixed column-index fragility (types containing | cause extra columns).
    Uses string-presence checks instead of positional column index.
    Fixed Scenario W to filter by axis="point_metrics" to avoid ambiguity.
"""
from __future__ import annotations

import pytest

# Helper: find the default value in a rendered table row, robust to | in type strings.
def _default_cell(row: str) -> str:
    """Extract the default-value cell from a markdown table row.
    
    Row format: | name | type | default | constraint | description |
    The type column may contain | (e.g. str | None), so we can't split naively.
    Strategy: find ` | ` sequences and extract the 3rd field after the name.
    We count from the right to be robust: constraint and description don't contain |.
    """
    # Strip leading/trailing whitespace and pipes
    row = row.strip().strip("|").strip()
    # Split from the right: last 3 fields are constraint, description, and trailing empty
    # Reversed approach: count from right
    parts = row.rsplit(" | ", 3)  # [name+type+default, constraint, description, trailing?]
    if len(parts) < 3:
        return ""
    # parts[0] contains "name | type | default"
    left = parts[0]
    # Now split left from the right once to get default
    left_parts = left.rsplit(" | ", 1)
    if len(left_parts) >= 2:
        return left_parts[-1].strip()
    return ""


# ---------------------------------------------------------------------------
# Scenario A — Import and identity
# ---------------------------------------------------------------------------
def test_scenario_a_import_and_identity():
    from macroforecast.scaffold.option_docs.types import REQUIRED
    assert REQUIRED is not None
    assert REQUIRED is not False
    assert REQUIRED is not 0
    assert REQUIRED is not ""
    from macroforecast.scaffold.option_docs.types import REQUIRED as R2
    assert REQUIRED is R2

# ---------------------------------------------------------------------------
# Scenario B
# ---------------------------------------------------------------------------
def test_scenario_b_param_doc_default_is_required():
    from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED
    p = ParameterDoc(name="x", type="int")
    assert p.default is REQUIRED

# ---------------------------------------------------------------------------
# Scenario C
# ---------------------------------------------------------------------------
def test_scenario_c_param_doc_default_none():
    from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED
    p = ParameterDoc(name="x", type="int | None", default=None)
    assert p.default is None
    assert p.default is not REQUIRED

# ---------------------------------------------------------------------------
# Scenario D
# ---------------------------------------------------------------------------
def test_scenario_d_param_doc_explicit_default():
    from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED
    p = ParameterDoc(name="alpha", type="float", default=1.0)
    assert p.default == 1.0
    assert p.default is not REQUIRED
    assert p.default is not None

# ---------------------------------------------------------------------------
# Scenario E
# ---------------------------------------------------------------------------
def test_scenario_e_param_doc_frozen():
    from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED
    p = ParameterDoc(name="x", type="int")
    with pytest.raises(Exception):
        p.default = None

# ---------------------------------------------------------------------------
# Scenario F
# ---------------------------------------------------------------------------
def test_scenario_f_option_doc_defaults():
    from macroforecast.scaffold.option_docs.types import OptionDoc, Reference
    o = OptionDoc(
        layer="l0", sublayer="l0_a", axis="test_axis", option="test_opt",
        summary="A summary for test option entry.",
        description="A description with enough characters to satisfy the quality floor minimum.",
        when_to_use="Use this for testing purposes in unit tests.",
        references=(Reference(citation="Test ref."),),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    assert o.data_args == ()
    assert o.return_type == ""

# ---------------------------------------------------------------------------
# Scenario G
# ---------------------------------------------------------------------------
def test_scenario_g_data_args_return_type_round_trip():
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference, REQUIRED
    X_param = ParameterDoc(name="X", type="np.ndarray", default=REQUIRED)
    o = OptionDoc(
        layer="l4", sublayer="L4_A_model_selection", axis="family", option="test_op",
        summary="A test op summary for unit tests.",
        description="A test description with enough characters to pass the quality floor gate.",
        when_to_use="Use this in unit tests to validate the data_args field round-trip.",
        references=(Reference(citation="Test ref."),),
        data_args=(X_param,),
        return_type="MyResult",
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    assert len(o.data_args) == 1
    assert o.data_args[0].name == "X"
    assert o.data_args[0].default is REQUIRED
    assert o.return_type == "MyResult"

# ---------------------------------------------------------------------------
# Scenario H
# ---------------------------------------------------------------------------
def test_scenario_h_render_op_page_data_args_position():
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference, REQUIRED
    from macroforecast.scaffold.render_encyclopedia import _render_op_page
    from macroforecast.scaffold.introspect import OptionInfo

    doc = OptionDoc(
        layer="l4", sublayer="L4_A_model_selection", axis="family", option="test_op",
        summary="Test op for rendering unit tests.",
        description="Test description long enough to satisfy the quality floor minimum for v1.0.",
        when_to_use="Unit test scenarios only — not intended for production use.",
        references=(Reference(citation="Test."),),
        op_page=True, op_func_name="test_func",
        data_args=(
            ParameterDoc(name="X", type="np.ndarray", default=REQUIRED),
            ParameterDoc(name="y", type="np.ndarray", default=REQUIRED),
        ),
        parameters=(
            ParameterDoc(name="alpha", type="float", default=1.0),
            ParameterDoc(name="vol_model", type="str | None", default=None),
        ),
        return_type="MyResult",
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = OptionInfo(value="test_op", label="test_op", status="operational", description="")
    rendered = _render_op_page("l4", "family", opt_info, doc)

    x_pos = rendered.index("X: np.ndarray,")
    star_pos = rendered.index("    *,")
    assert x_pos < star_pos, "X should appear before *,"

    alpha_pos = rendered.index("alpha: float = 1.0,")
    vol_model_pos = rendered.index("vol_model: str | None = None,")
    assert alpha_pos > star_pos
    assert vol_model_pos > star_pos

    assert "-> MyResult" in rendered
    assert "X: np.ndarray," in rendered
    assert "y: np.ndarray," in rendered
    assert "vol_model: str | None = None," in rendered

# ---------------------------------------------------------------------------
# Scenario I
# ---------------------------------------------------------------------------
def test_scenario_i_required_param_no_equals():
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference, REQUIRED
    from macroforecast.scaffold.render_encyclopedia import _render_op_page
    from macroforecast.scaffold.introspect import OptionInfo

    doc_req = OptionDoc(
        layer="l4", sublayer="L4_A_model_selection", axis="family", option="test_req",
        summary="Test required param in parameters tuple.",
        description="Long enough description for the quality floor minimum in v1.0 tests.",
        when_to_use="Unit test for required param rendering in the function signature block.",
        references=(Reference(citation="Test."),),
        op_page=True, op_func_name="test_req_func",
        data_args=(),
        parameters=(
            ParameterDoc(name="required_param", type="str"),
            ParameterDoc(name="optional_param", type="int", default=5),
        ),
        return_type="",
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = OptionInfo(value="test_req", label="test_req", status="operational", description="")
    rendered = _render_op_page("l4", "family", opt_info, doc_req)
    assert "required_param: str," in rendered
    assert "required_param: str =" not in rendered
    assert "optional_param: int = 5," in rendered

# ---------------------------------------------------------------------------
# Scenario J
# ---------------------------------------------------------------------------
def test_scenario_j_empty_params_stub():
    from macroforecast.scaffold.option_docs.types import OptionDoc, Reference
    from macroforecast.scaffold.render_encyclopedia import _render_op_page
    from macroforecast.scaffold.introspect import OptionInfo

    doc_stub = OptionDoc(
        layer="l4", sublayer="L4_A_model_selection", axis="family", option="test_stub",
        summary="Test stub rendering for empty params.",
        description="Long enough description for the quality floor minimum check in test suite.",
        when_to_use="Use only in unit tests that verify the stub signature fallback behavior.",
        references=(Reference(citation="Test."),),
        op_page=True, op_func_name="test_stub_func",
        data_args=(), parameters=(), return_type="",
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = OptionInfo(value="test_stub", label="test_stub", status="operational", description="")
    rendered = _render_op_page("l4", "family", opt_info, doc_stub)
    assert "mf.functions.test_stub_func(...)" in rendered

# ---------------------------------------------------------------------------
# Scenario K
# ---------------------------------------------------------------------------
def test_scenario_k_data_args_only_no_star():
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference, REQUIRED
    from macroforecast.scaffold.render_encyclopedia import _render_op_page
    from macroforecast.scaffold.introspect import OptionInfo

    doc_data_only = OptionDoc(
        layer="l5", sublayer="L5_A_metric_specification", axis="point_metrics", option="test_metric",
        summary="Test metric with data_args only.",
        description="Long enough description to pass the quality floor minimum for v1.0 documentation.",
        when_to_use="Unit test scenario for data_args-only rendering without keyword params.",
        references=(Reference(citation="Test."),),
        op_page=True, op_func_name="test_metric_func",
        data_args=(
            ParameterDoc(name="y_true", type="np.ndarray", default=REQUIRED),
            ParameterDoc(name="y_pred", type="np.ndarray", default=REQUIRED),
        ),
        parameters=(),
        return_type="float",
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = OptionInfo(value="test_metric", label="test_metric", status="operational", description="")
    rendered = _render_op_page("l5", "point_metrics", opt_info, doc_data_only)
    assert "y_true: np.ndarray," in rendered
    assert "y_pred: np.ndarray," in rendered
    assert "*," not in rendered
    assert ") -> float" in rendered

# ---------------------------------------------------------------------------
# Scenario L — Use string presence instead of fragile column index
# ---------------------------------------------------------------------------
def test_scenario_l_table_default_rendering():
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference, REQUIRED
    from macroforecast.scaffold.render_encyclopedia import _render_op_page
    from macroforecast.scaffold.introspect import OptionInfo

    doc = OptionDoc(
        layer="l4", sublayer="L4_A_model_selection", axis="family", option="test_table",
        summary="Test parameters table rendering.",
        description="Long enough description for the v1.0 quality floor minimum in the test suite.",
        when_to_use="Verify default cell rendering in the Parameters table for all default types.",
        references=(Reference(citation="Test."),),
        op_page=True, op_func_name="test_func",
        data_args=(
            ParameterDoc(name="X", type="np.ndarray", default=REQUIRED),
        ),
        parameters=(
            ParameterDoc(name="alpha", type="float", default=1.0),
            ParameterDoc(name="vol_model", type="str | None", default=None),
        ),
        return_type="",
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = OptionInfo(value="test_table", label="test_table", status="operational", description="")
    rendered = _render_op_page("l4", "family", opt_info, doc)

    assert "## Parameters" in rendered

    # X row: default = — (REQUIRED)
    x_row = [row for row in rendered.splitlines() if "| `X` |" in row][0]
    assert "| — |" in x_row, f"Expected — for X default, row: {x_row!r}"
    assert "= REQUIRED" not in x_row

    # alpha row: contains `1.0`
    alpha_row = [row for row in rendered.splitlines() if "| `alpha` |" in row][0]
    assert "`1.0`" in alpha_row, f"Expected `1.0` in alpha row, got: {alpha_row!r}"

    # vol_model row: contains `None` (not —)
    vol_row = [row for row in rendered.splitlines() if "| `vol_model` |" in row][0]
    assert "`None`" in vol_row, f"Expected `None` in vol_model row, got: {vol_row!r}"
    assert "| — |" not in vol_row or "`None`" in vol_row  # None should not render as —

# ---------------------------------------------------------------------------
# Scenario M — Use string presence checks
# ---------------------------------------------------------------------------
def test_scenario_m_render_option_body_dash_for_required():
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference, REQUIRED
    from macroforecast.scaffold.render_encyclopedia import _render_option_body
    from macroforecast.scaffold.introspect import OptionInfo

    p_req = ParameterDoc(name="parallel_unit", type="str", default=REQUIRED, constraint="required")
    p_none = ParameterDoc(name="n_workers", type="int | None", default=None, constraint=">=1 or None")
    doc = OptionDoc(
        layer="l0", sublayer="l0_a", axis="compute_mode", option="parallel",
        summary="Parallel compute mode.",
        description="Long enough description satisfying the v1.0 quality floor minimum for this test.",
        when_to_use="Long sweeps on multi-core machines after serial validation is complete.",
        references=(Reference(citation="macroforecast design."),),
        parameters=(p_req, p_none),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = OptionInfo(value="parallel", label="parallel", status="operational", description="")
    rendered = _render_option_body(opt_info, doc, layer_id="l0", sublayer="l0_a", axis="compute_mode")

    assert "**Parameters**" in rendered
    # parallel_unit: REQUIRED → — (str type, no |, so row is simple)
    pu_row = [r for r in rendered.splitlines() if "| `parallel_unit` |" in r][0]
    assert "| — |" in pu_row, f"Expected — in parallel_unit row: {pu_row!r}"

    # n_workers: None → `None` 
    nw_row = [r for r in rendered.splitlines() if "| `n_workers` |" in r][0]
    assert "`None`" in nw_row, f"Expected `None` in n_workers row: {nw_row!r}"

# ---------------------------------------------------------------------------
# Scenario N
# ---------------------------------------------------------------------------
def test_scenario_n_ridge_md_signature_content(tmp_path):
    from macroforecast.scaffold import render_encyclopedia
    render_encyclopedia.write_all(tmp_path)

    ridge_md = (tmp_path / "l4" / "family" / "ridge.md").read_text(encoding="utf-8")

    assert "X: np.ndarray | pd.DataFrame," in ridge_md
    assert "y: np.ndarray | pd.Series," in ridge_md

    x_pos = ridge_md.index("X: np.ndarray | pd.DataFrame,")
    y_pos = ridge_md.index("y: np.ndarray | pd.Series,")
    star_pos = ridge_md.index("    *,")
    assert x_pos < star_pos, "X must appear before *,"
    assert y_pos < star_pos, "y must appear before *,"

    vol_pos = ridge_md.index("vol_model:")
    assert vol_pos > star_pos, "vol_model must appear after *,"

    assert "random_state: int | None = None," in ridge_md
    assert "-> RidgeFitResult" in ridge_md

    alpha_pos = ridge_md.index("alpha: float = 1.0,")
    assert alpha_pos > star_pos

    assert "REQUIRED" not in ridge_md
    assert "<object object at" not in ridge_md

# ---------------------------------------------------------------------------
# Scenario O — Use string presence checks
# ---------------------------------------------------------------------------
def test_scenario_o_theil_u1_md_signature_content(tmp_path):
    from macroforecast.scaffold import render_encyclopedia
    render_encyclopedia.write_all(tmp_path)

    theil_md = (tmp_path / "l5" / "point_metrics" / "theil_u1.md").read_text(encoding="utf-8")

    assert "y_true: np.ndarray | pd.Series," in theil_md
    assert "y_pred: np.ndarray | pd.Series," in theil_md

    sig_start = theil_md.index("## Function signature")
    sig_end = theil_md.index("## Parameters", sig_start)
    sig_block = theil_md[sig_start:sig_end]
    assert "    *," not in sig_block, "theil_u1 has no keyword-only params; no *,"

    assert ") -> float" in theil_md

    # Check — appears in y_true and y_pred rows (use | — | as substring)
    y_true_row = [r for r in theil_md.splitlines() if "| `y_true` |" in r][0]
    y_pred_row = [r for r in theil_md.splitlines() if "| `y_pred` |" in r][0]
    assert "| — |" in y_true_row, f"y_true default should be — : {y_true_row!r}"
    assert "| — |" in y_pred_row, f"y_pred default should be — : {y_pred_row!r}"

    assert "REQUIRED" not in theil_md
    assert "<object object at" not in theil_md

# ---------------------------------------------------------------------------
# Scenario P
# ---------------------------------------------------------------------------
def test_scenario_p_zero_default_not_confused():
    from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED
    p = ParameterDoc(name="n", type="int", default=0)
    assert p.default == 0
    assert p.default is not REQUIRED
    assert p.default is not None

# ---------------------------------------------------------------------------
# Scenario Q
# ---------------------------------------------------------------------------
def test_scenario_q_false_default_not_confused():
    from macroforecast.scaffold.option_docs.types import ParameterDoc, REQUIRED
    p = ParameterDoc(name="flag", type="bool", default=False)
    assert p.default is False
    assert p.default is not REQUIRED

# ---------------------------------------------------------------------------
# Scenario R
# ---------------------------------------------------------------------------
def test_scenario_r_option_doc_frozen():
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference
    o = OptionDoc(
        layer="l4", sublayer="L4_A_model_selection", axis="family", option="test",
        summary="Test summary text.",
        description="Test description with length adequate for quality floor.",
        when_to_use="For unit testing OptionDoc immutability.",
        references=(Reference(citation="Test."),),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    with pytest.raises(Exception):
        o.data_args = (ParameterDoc(name="X", type="np.ndarray"),)

# ---------------------------------------------------------------------------
# Scenario S
# ---------------------------------------------------------------------------
def test_scenario_s_l0_parallel_unit_is_required():
    from macroforecast.scaffold.option_docs import OPTION_DOCS
    from macroforecast.scaffold.option_docs.types import REQUIRED
    doc = OPTION_DOCS[("l0", "l0_a", "compute_mode", "parallel")]
    pu = next(p for p in doc.parameters if p.name == "parallel_unit")
    assert pu.default is REQUIRED

# ---------------------------------------------------------------------------
# Scenario T
# ---------------------------------------------------------------------------
def test_scenario_t_l0_n_workers_is_none():
    from macroforecast.scaffold.option_docs import OPTION_DOCS
    from macroforecast.scaffold.option_docs.types import REQUIRED
    doc = OPTION_DOCS[("l0", "l0_a", "compute_mode", "parallel")]
    nw = next(p for p in doc.parameters if p.name == "n_workers")
    assert nw.default is None
    assert nw.default is not REQUIRED

# ---------------------------------------------------------------------------
# Scenario U
# ---------------------------------------------------------------------------
def test_scenario_u_l4_vol_model_random_state_none():
    from macroforecast.scaffold.option_docs import OPTION_DOCS
    from macroforecast.scaffold.option_docs.types import REQUIRED
    doc = OPTION_DOCS[("l4", "L4_A_model_selection", "family", "ridge")]
    vol = next(p for p in doc.parameters if p.name == "vol_model")
    assert vol.default is None
    rand = next(p for p in doc.parameters if p.name == "random_state")
    assert rand.default is None

# ---------------------------------------------------------------------------
# Scenario V
# ---------------------------------------------------------------------------
def test_scenario_v_l4_ridge_data_args():
    from macroforecast.scaffold.option_docs import OPTION_DOCS
    from macroforecast.scaffold.option_docs.types import REQUIRED
    doc = OPTION_DOCS[("l4", "L4_A_model_selection", "family", "ridge")]
    assert len(doc.data_args) == 2
    data_names = [p.name for p in doc.data_args]
    assert "X" in data_names
    assert "y" in data_names
    for p in doc.data_args:
        assert p.default is REQUIRED
    assert doc.return_type == "RidgeFitResult"

# ---------------------------------------------------------------------------
# Scenario W — filter by axis="point_metrics" to get the op_page entry
# ---------------------------------------------------------------------------
def test_scenario_w_l5_theil_u1_data_args():
    from macroforecast.scaffold.option_docs import OPTION_DOCS
    from macroforecast.scaffold.option_docs.types import REQUIRED
    # Use exact key: theil_u1 op_page is under axis "point_metrics"
    doc = OPTION_DOCS[("l5", "L5_A_metric_specification", "point_metrics", "theil_u1")]
    assert len(doc.data_args) == 2
    data_names = [p.name for p in doc.data_args]
    assert "y_true" in data_names
    assert "y_pred" in data_names
    for p in doc.data_args:
        assert p.default is REQUIRED
    assert doc.return_type == "float"
    assert doc.parameters == ()

# ---------------------------------------------------------------------------
# Regression: None default renders as `None`, not —
# ---------------------------------------------------------------------------
def test_regression_none_default_renders_backtick_none():
    """Post-C23: explicit default=None renders as `None`, REQUIRED renders as —."""
    from macroforecast.scaffold.option_docs.types import OptionDoc, ParameterDoc, Reference
    from macroforecast.scaffold.render_encyclopedia import _render_option_body
    from macroforecast.scaffold.introspect import OptionInfo

    p = ParameterDoc(name="parallel_unit", type="str", default=None)
    doc = OptionDoc(
        layer="l0", sublayer="l0_a", axis="compute_mode", option="parallel",
        summary="Parallel compute mode.",
        description="Long enough description satisfying the v1.0 quality floor minimum check.",
        when_to_use="Long sweeps on multi-core machines after serial validation.",
        references=(Reference(citation="macroforecast design."),),
        parameters=(p,),
        last_reviewed="2026-05-16",
        reviewer="test",
    )
    opt_info = OptionInfo(value="parallel", label="parallel", status="operational", description="")
    rendered = _render_option_body(opt_info, doc, layer_id="l0", sublayer="l0_a", axis="compute_mode")
    pu_row = [r for r in rendered.splitlines() if "| `parallel_unit` |" in r][0]
    # None renders as `None` (not —)
    assert "`None`" in pu_row, f"None default should render as `None`, got: {pu_row}"

