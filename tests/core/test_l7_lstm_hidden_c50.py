"""test(c50): lstm_hidden_state L7 op -- 9 scenarios from test-spec.md.

Tests the behavioral contract for lstm_hidden_state promotion from 'future'
to 'operational' (Cycle 50).

Torch-dependent scenarios (3.5, 3.6, 3.7, 3.8) carry @pytest.mark.deep and
use pytest.importorskip("torch"). They are excluded by the scope-limited
execution command (-m "not slow and not heavy and not deep").

Scenario 3.9 (no-torch error hint) is skipped if torch is installed.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Scenario 3.1 -- lstm_hidden_state is NOT in FUTURE_OPS
# ---------------------------------------------------------------------------

def test_lstm_hidden_state_not_in_future_ops():
    """Regression guard: lstm_hidden_state must be absent from FUTURE_OPS after C50."""
    from macroforecast.core.ops.l7_ops import FUTURE_OPS

    assert "lstm_hidden_state" not in FUTURE_OPS, (
        f"lstm_hidden_state must be operational after C50; "
        f"still in FUTURE_OPS: {FUTURE_OPS}"
    )


# ---------------------------------------------------------------------------
# Scenario 3.2 -- FUTURE_OPS is empty after C50 (test-spec requirement)
# ---------------------------------------------------------------------------

def test_l7_future_ops_drops_lstm_hidden_state_after_c50():
    """C50 removes lstm_hidden_state from FUTURE_OPS. generalized_irf is C49 scope.

    This worktree branched from main c92c55a9 (pre-C49-merge), so generalized_irf
    is still in FUTURE_OPS in this worktree's view. After both C49 and C50 merge
    to main, FUTURE_OPS will be ().

    The C50-specific guarantee is: "lstm_hidden_state not in FUTURE_OPS".
    The final empty-state () is a cross-cycle property, not a C50-only guarantee.
    """
    from macroforecast.core.ops.l7_ops import FUTURE_OPS

    # C50 specifically removes lstm_hidden_state. generalized_irf is C49 scope —
    # in this worktree's view (branched from main c92c55a9, pre-C49-merge),
    # generalized_irf is still in FUTURE_OPS. After both C49 and C50 merge to main,
    # FUTURE_OPS will be ().
    assert "lstm_hidden_state" not in FUTURE_OPS
    assert FUTURE_OPS == () or FUTURE_OPS == ("generalized_irf",), (
        f"FUTURE_OPS unexpected state: {FUTURE_OPS}. "
        "Expected either () (both C49+C50 merged) or ('generalized_irf',) (C49 still open)."
    )


# ---------------------------------------------------------------------------
# Scenario 3.3 -- lstm_hidden_state in DEFAULT_FIGURE_MAPPING with heatmap
# ---------------------------------------------------------------------------

def test_lstm_hidden_state_in_default_figure_mapping():
    """Contract: lstm_hidden_state is registered in DEFAULT_FIGURE_MAPPING
    with figure type 'heatmap'."""
    from macroforecast.core.ops.l7_ops import DEFAULT_FIGURE_MAPPING

    assert "lstm_hidden_state" in DEFAULT_FIGURE_MAPPING, (
        "lstm_hidden_state must be in DEFAULT_FIGURE_MAPPING"
    )
    assert DEFAULT_FIGURE_MAPPING["lstm_hidden_state"] == "heatmap", (
        f"Expected 'heatmap', got {DEFAULT_FIGURE_MAPPING['lstm_hidden_state']!r}"
    )


# ---------------------------------------------------------------------------
# Scenario 3.4 -- Validator accepts lstm_hidden_state (schema contract)
# ---------------------------------------------------------------------------

def test_l7_lstm_hidden_state_now_accepted():
    """Contract: the L7 validator no longer rejects lstm_hidden_state as future."""
    from macroforecast.core.layers.l7 import parse_layer_yaml, validate_layer, make_l7_yaml

    layer = parse_layer_yaml(
        make_l7_yaml(op="lstm_hidden_state", model_family="lstm"), "l7"
    )
    report = validate_layer(layer)
    assert not any(
        "future" in issue.message.lower() for issue in report.hard_errors
    ), (
        f"lstm_hidden_state must not be rejected as future after C50; "
        f"hard_errors: {[i.message for i in report.hard_errors]}"
    )


# ---------------------------------------------------------------------------
# Scenario 3.5 -- _lstm_hidden_state_frame: shape and non-negativity (torch)
# ---------------------------------------------------------------------------

@pytest.mark.deep
def test_lstm_hidden_state_frame_shape_and_non_negative():
    """Requires torch ([deep] extra). Skipped if torch unavailable.

    Validates: output shape (hidden_size rows), required columns, non-negative
    importance values, correct method label, and feature name format.
    """
    torch = pytest.importorskip("torch")
    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _lstm_hidden_state_frame, _TorchSequenceModel

    n, k = 30, 5
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"x{i}" for i in range(k)])
    y = pd.Series(X.iloc[:, 0].values + rng.normal(size=n) * 0.1)

    model_obj = _TorchSequenceModel(kind="lstm", hidden_size=8, n_epochs=5, random_state=0)
    model_obj.fit(X, y)

    class MockArtifact:
        fitted_object = model_obj
        framework = "torch"

    frame = _lstm_hidden_state_frame(MockArtifact(), X)

    # Shape: one row per hidden unit.
    assert len(frame) == 8, f"Expected 8 rows (hidden_size=8), got {len(frame)}"
    # Required columns present.
    assert set(frame.columns) >= {"feature", "importance", "coefficient", "method"}, (
        f"Missing required columns. Got: {list(frame.columns)}"
    )
    # Non-negative importance values (mean absolute activations).
    assert (frame["importance"] >= 0).all(), (
        f"All importance values must be >= 0; got: {frame['importance'].tolist()}"
    )
    # Method label.
    assert (frame["method"] == "lstm_hidden_state").all(), (
        f"All method values must be 'lstm_hidden_state'"
    )
    # Feature name format: hidden_unit_<i>.
    assert all(frame["feature"].str.startswith("hidden_unit_")), (
        f"Feature names must start with 'hidden_unit_'; got: {frame['feature'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Scenario 3.6 -- _lstm_hidden_state_frame: determinism (torch)
# ---------------------------------------------------------------------------

@pytest.mark.deep
def test_lstm_hidden_state_deterministic():
    """Requires torch. Same model + same X must yield identical importance values."""
    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _lstm_hidden_state_frame, _TorchSequenceModel

    n, k = 20, 4
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"f{i}" for i in range(k)])
    y = pd.Series(rng.normal(size=n))

    model_obj = _TorchSequenceModel(kind="lstm", hidden_size=4, n_epochs=3, random_state=42)
    model_obj.fit(X, y)

    class MockArtifact:
        fitted_object = model_obj
        framework = "torch"

    r1 = _lstm_hidden_state_frame(MockArtifact(), X)
    r2 = _lstm_hidden_state_frame(MockArtifact(), X)

    assert r1["importance"].round(8).equals(r2["importance"].round(8)), (
        "lstm_hidden_state must be deterministic for same model and same X"
    )


# ---------------------------------------------------------------------------
# Scenario 3.7 -- GRU family supported (torch)
# ---------------------------------------------------------------------------

@pytest.mark.deep
def test_lstm_hidden_state_gru_supported():
    """Requires torch. GRU family must be supported (same hook logic as LSTM)."""
    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _lstm_hidden_state_frame, _TorchSequenceModel

    n, k = 20, 3
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"g{i}" for i in range(k)])
    y = pd.Series(rng.normal(size=n))

    model_obj = _TorchSequenceModel(kind="gru", hidden_size=6, n_epochs=3, random_state=0)
    model_obj.fit(X, y)

    class MockArtifact:
        fitted_object = model_obj
        framework = "torch"

    frame = _lstm_hidden_state_frame(MockArtifact(), X)

    assert len(frame) == 6, f"Expected 6 rows (hidden_size=6 GRU), got {len(frame)}"
    assert (frame["importance"] >= 0).all(), (
        "All importance values must be >= 0 for GRU"
    )


# ---------------------------------------------------------------------------
# Scenario 3.8 -- Transformer family raises NotImplementedError (torch)
# ---------------------------------------------------------------------------

@pytest.mark.deep
def test_lstm_hidden_state_transformer_raises():
    """Requires torch. Transformer family must raise NotImplementedError with
    'transformer' in the message."""
    pytest.importorskip("torch")
    import numpy as np
    import pandas as pd
    from macroforecast.core.runtime import _lstm_hidden_state_frame, _TorchSequenceModel

    n, k = 20, 4
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"t{i}" for i in range(k)])
    y = pd.Series(rng.normal(size=n))

    model_obj = _TorchSequenceModel(
        kind="transformer", hidden_size=8, n_epochs=2, random_state=0
    )
    model_obj.fit(X, y)

    class MockArtifact:
        fitted_object = model_obj
        framework = "torch"

    with pytest.raises(NotImplementedError, match="transformer"):
        _lstm_hidden_state_frame(MockArtifact(), X)


# ---------------------------------------------------------------------------
# Scenario 3.9 -- No-torch path raises NotImplementedError with [deep] hint
# ---------------------------------------------------------------------------

def test_lstm_hidden_state_no_torch_hint(monkeypatch):
    """Check that the no-torch code path raises NotImplementedError with a
    message referencing [deep]. Test is skipped if torch is installed."""
    try:
        import torch  # noqa: F401
        pytest.skip("torch is installed; cannot test no-torch error path")
    except ImportError:
        pass

    import pandas as pd
    from macroforecast.core.runtime import _lstm_hidden_state_frame

    class MockFitted:
        _model = None
        kind = "lstm"

    class MockArtifact:
        fitted_object = MockFitted()
        framework = "torch"

    with pytest.raises(NotImplementedError, match=r"\[deep\]"):
        _lstm_hidden_state_frame(MockArtifact(), pd.DataFrame())
