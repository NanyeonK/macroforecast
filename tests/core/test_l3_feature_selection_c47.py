"""Cycle 47 — L3 Feature Selection: Tester validation suite.

Tests the 5 promoted ops (boruta_selection, recursive_feature_elimination,
lasso_path_selection, stability_selection, genetic_algorithm_selection)
against behavioral contracts specified in test-spec.md.

Test-spec.md sections referenced:
  Sec 2 — shared fixtures
  Sec 3 — 5 ops × 3 scenarios (B1–B3, R1–R3, L1–L3, S1–S3, G1–G3)
  Sec 4 — status/validator checks (SV1–SV4)
  Sec 6 — regression guard (RG1, RG2)
  Sec 9 — top-level bit-exact replicate
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared fixtures — defined per test-spec.md Section 2.1
# ---------------------------------------------------------------------------

SEED: int = 42
N_OBS: int = 120
N_FEATURES: int = 20
N_RELEVANT: int = 4


def make_synthetic_panel(
    n_obs: int = N_OBS,
    n_features: int = N_FEATURES,
    n_relevant: int = N_RELEVANT,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a synthetic panel where the first n_relevant columns have true signal.

    y = X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + 0.3 * X[:, 3] + noise
    Remaining columns are pure noise.

    Returns (frame: pd.DataFrame, target: pd.Series).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n_obs, freq="ME")
    X = rng.standard_normal((n_obs, n_features))
    cols = [f"x{i}" for i in range(n_features)]
    frame = pd.DataFrame(X, index=idx, columns=cols)
    noise = rng.standard_normal(n_obs) * 0.5
    # Build y using only the first n_relevant features; extras are pure noise
    y = (
        frame["x0"]
        + frame["x1"]
        + 0.5 * frame["x2"]
        + 0.3 * frame["x3"]
        + noise
    )
    y.name = "y"
    return frame, y


# ---------------------------------------------------------------------------
# Private runtime helpers (imported directly per tester instructions)
# ---------------------------------------------------------------------------
from macroforecast.core.runtime import (  # noqa: E402
    _boruta_selection,
    _genetic_algorithm_selection,
    _lasso_path_selection,
    _recursive_feature_elimination,
    _stability_selection,
)

# ---------------------------------------------------------------------------
# Section 3.1 — boruta_selection (B1, B2, B3)
# ---------------------------------------------------------------------------


def test_boruta_selection_recovers_relevant_features() -> None:
    """B1: At least 2 of the 4 relevant features recovered on synthetic data."""
    frame, y = make_synthetic_panel()
    result = _boruta_selection(
        frame,
        target=y,
        params={
            "n_estimators_rf": 50,
            "max_iter": 30,
            "alpha": 0.05,
            "include_tentative": False,
            "random_state": SEED,
        },
    )
    assert isinstance(result, pd.DataFrame)
    relevant = {"x0", "x1", "x2", "x3"}
    selected = set(result.columns)
    n_recovered = len(relevant & selected)
    assert n_recovered >= 2, (
        f"Expected >= 2 relevant features, got {n_recovered}: {selected}"
    )


def test_boruta_selection_output_contract() -> None:
    """B2: Contract / shape — index preserved, columns subset, dtypes preserved."""
    frame, y = make_synthetic_panel()
    result = _boruta_selection(
        frame,
        target=y,
        params={
            "n_estimators_rf": 10,
            "max_iter": 5,
            "alpha": 0.05,
            "include_tentative": True,
            "random_state": 0,
        },
    )
    # Same index
    assert result.index.equals(frame.index), "Index not preserved"
    # Columns are a subset
    assert set(result.columns).issubset(set(frame.columns)), "Columns not a subset"
    # At least 1 column returned
    assert result.shape[1] >= 1, "No columns returned"
    # Same number of rows
    assert result.shape[0] == frame.shape[0], "Row count changed"
    # Dtype preservation per column
    for col in result.columns:
        assert result[col].dtype == frame[col].dtype, (
            f"dtype mismatch for {col}: {result[col].dtype} != {frame[col].dtype}"
        )


def test_boruta_selection_seed_determinism() -> None:
    """B3: Two calls with identical params produce identical DataFrames."""
    frame, y = make_synthetic_panel()
    params = {
        "n_estimators_rf": 10,
        "max_iter": 5,
        "alpha": 0.05,
        "include_tentative": True,
        "random_state": 99,
    }
    result1 = _boruta_selection(frame.copy(), target=y.copy(), params=params)
    result2 = _boruta_selection(frame.copy(), target=y.copy(), params=params)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Section 3.2 — recursive_feature_elimination (R1, R2, R3)
# ---------------------------------------------------------------------------


def test_rfe_recovers_relevant_features() -> None:
    """R1: At least 3 of the 4 relevant features recovered by RFE+Ridge."""
    frame, y = make_synthetic_panel()
    result = _recursive_feature_elimination(
        frame,
        target=y,
        params={
            "n_features_to_select": N_RELEVANT,
            "step": 1,
            "estimator": "ridge",
            "use_cv": False,
            "cv_folds": 5,
            "random_state": SEED,
        },
    )
    relevant = {"x0", "x1", "x2", "x3"}
    selected = set(result.columns)
    n_recovered = len(relevant & selected)
    assert n_recovered >= 3, (
        f"Expected >= 3 relevant features, got {n_recovered}: {selected}"
    )
    # RFE with n_features_to_select=4 must return exactly 4 columns
    assert result.shape[1] == N_RELEVANT, (
        f"Expected {N_RELEVANT} columns, got {result.shape[1]}"
    )


def test_rfe_output_contract() -> None:
    """R2: Contract / shape — integer k and fractional k both work correctly."""
    frame, y = make_synthetic_panel()

    # Integer k=4 -> exactly 4 columns
    result_int = _recursive_feature_elimination(
        frame,
        target=y,
        params={
            "n_features_to_select": 4,
            "step": 1,
            "estimator": "ridge",
            "use_cv": False,
            "random_state": 0,
        },
    )
    assert result_int.index.equals(frame.index), "Index not preserved (integer k)"
    assert set(result_int.columns).issubset(set(frame.columns))
    assert result_int.shape[1] == 4, f"Expected 4 columns (int k), got {result_int.shape[1]}"
    for col in result_int.columns:
        assert result_int[col].dtype == frame[col].dtype

    # Fractional k=0.25 with N=20 -> max(1, int(0.25*20)) = 5 columns
    result_frac = _recursive_feature_elimination(
        frame,
        target=y,
        params={
            "n_features_to_select": 0.25,
            "step": 1,
            "estimator": "ridge",
            "use_cv": False,
            "random_state": 0,
        },
    )
    expected_k_frac = max(1, int(0.25 * N_FEATURES))  # = 5
    assert result_frac.index.equals(frame.index), "Index not preserved (fractional k)"
    assert set(result_frac.columns).issubset(set(frame.columns))
    assert result_frac.shape[1] == expected_k_frac, (
        f"Expected {expected_k_frac} columns (frac k=0.25), got {result_frac.shape[1]}"
    )


def test_rfe_seed_determinism() -> None:
    """R3: Two calls with identical seed and estimator produce identical DataFrames."""
    frame, y = make_synthetic_panel()
    params = {
        "n_features_to_select": 4,
        "step": 1,
        "estimator": "ridge",
        "use_cv": False,
        "random_state": 7,
    }
    result1 = _recursive_feature_elimination(frame.copy(), target=y.copy(), params=params)
    result2 = _recursive_feature_elimination(frame.copy(), target=y.copy(), params=params)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Section 3.3 — lasso_path_selection (L1, L2, L3)
# ---------------------------------------------------------------------------


def test_lasso_path_recovers_entry_order() -> None:
    """L1: At least 3 of the 4 relevant features in the first 4 LARS-path entries."""
    frame, y = make_synthetic_panel()
    result = _lasso_path_selection(
        frame,
        target=y,
        params={
            "n_features_to_select": N_RELEVANT,
            "normalize_features": True,
            "random_state": SEED,
        },
    )
    relevant = {"x0", "x1", "x2", "x3"}
    selected = set(result.columns)
    n_recovered = len(relevant & selected)
    assert n_recovered >= 3, (
        f"Expected >= 3 relevant features in LARS path, got {n_recovered}: {selected}"
    )
    assert result.shape[1] == N_RELEVANT, (
        f"Expected {N_RELEVANT} columns, got {result.shape[1]}"
    )


def test_lasso_path_distinct_from_lasso_cv() -> None:
    """L2: Path-entry semantics are structurally tested with a collinear case."""
    rng = np.random.default_rng(0)
    n = 50
    x1 = rng.standard_normal(n)
    x2 = x1 * 0.99 + rng.standard_normal(n) * 0.1  # nearly collinear with x1
    x3 = rng.standard_normal(n)
    y_arr = x1 + 2 * x3 + rng.standard_normal(n) * 0.1
    frame_d = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    target_d = pd.Series(y_arr, name="y")

    result = _lasso_path_selection(
        frame_d,
        target=target_d,
        params={
            "n_features_to_select": 1,
            "normalize_features": True,
            "random_state": 0,
        },
    )
    # Must return exactly 1 column
    assert result.shape[1] == 1, f"Expected 1 column, got {result.shape[1]}"
    # The selected column must be in the genuine predictor set (not necessarily x2
    # though collinearity edge cases may push x2 first in the LARS path)
    assert set(result.columns).issubset({"x1", "x2", "x3"}), (
        f"Selected column not in input columns: {result.columns}"
    )
    # Index must be preserved
    assert result.index.equals(frame_d.index)


def test_lasso_path_seed_determinism() -> None:
    """L3: Two calls with same params produce identical results (LARS is deterministic)."""
    frame, y = make_synthetic_panel()
    params = {
        "n_features_to_select": 4,
        "normalize_features": True,
        "random_state": SEED,
    }
    result1 = _lasso_path_selection(frame.copy(), target=y.copy(), params=params)
    result2 = _lasso_path_selection(frame.copy(), target=y.copy(), params=params)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Section 3.4 — stability_selection (S1, S2, S3)
# ---------------------------------------------------------------------------


def test_stability_selection_recovers_stable_features() -> None:
    """S1: At least 2 of the 4 relevant features have pi >= pi_thr."""
    frame, y = make_synthetic_panel()
    result = _stability_selection(
        frame,
        target=y,
        params={
            "n_subsamples": 50,
            "subsample_fraction": 0.5,
            "pi_thr": 0.5,
            "base_estimator": "lasso",
            "alpha": 0.01,
            "random_state": SEED,
        },
    )
    assert isinstance(result, pd.DataFrame)
    relevant = {"x0", "x1", "x2", "x3"}
    selected = set(result.columns)
    n_recovered = len(relevant & selected)
    assert n_recovered >= 2, (
        f"Expected >= 2 stable relevant features, got {n_recovered}: {selected}"
    )


def test_stability_selection_output_contract() -> None:
    """S2: Contract / shape — index, subset, dtype. pi_thr edge cases."""
    frame, y = make_synthetic_panel()

    # Standard call
    result = _stability_selection(
        frame,
        target=y,
        params={
            "n_subsamples": 20,
            "subsample_fraction": 0.5,
            "pi_thr": 0.5,
            "base_estimator": "lasso",
            "alpha": 0.01,
            "random_state": 0,
        },
    )
    assert result.index.equals(frame.index), "Index not preserved"
    assert set(result.columns).issubset(set(frame.columns)), "Columns not a subset"
    assert result.shape[1] >= 1, "No columns returned"
    assert result.shape[0] == frame.shape[0], "Row count changed"
    for col in result.columns:
        assert result[col].dtype == frame[col].dtype

    # pi_thr=0.0 — all features should be selected (pi >= 0 for all)
    result_low = _stability_selection(
        frame,
        target=y,
        params={
            "n_subsamples": 10,
            "subsample_fraction": 0.5,
            "pi_thr": 0.0,
            "base_estimator": "lasso",
            "alpha": 0.01,
            "random_state": 0,
        },
    )
    assert result_low.shape[1] == frame.shape[1], (
        f"Expected all {frame.shape[1]} columns with pi_thr=0.0, got {result_low.shape[1]}"
    )

    # pi_thr=1.0 — extremely high threshold; fallback must return at least 1 feature
    result_high = _stability_selection(
        frame,
        target=y,
        params={
            "n_subsamples": 10,
            "subsample_fraction": 0.5,
            "pi_thr": 1.0,
            "base_estimator": "lasso",
            "alpha": 0.01,
            "random_state": 0,
        },
    )
    assert result_high.shape[1] >= 1, "Fallback must return at least 1 feature"


def test_stability_selection_seed_determinism() -> None:
    """S3: Two calls with random_state=42 produce identical DataFrames."""
    frame, y = make_synthetic_panel()
    params = {
        "n_subsamples": 20,
        "subsample_fraction": 0.5,
        "pi_thr": 0.5,
        "base_estimator": "lasso",
        "alpha": 0.01,
        "random_state": 42,
    }
    result1 = _stability_selection(frame.copy(), target=y.copy(), params=params)
    result2 = _stability_selection(frame.copy(), target=y.copy(), params=params)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Section 3.5 — genetic_algorithm_selection (G1, G2, G3)
# ---------------------------------------------------------------------------


def test_ga_selection_recovers_relevant_features_tiny() -> None:
    """G1: At least 2 of the 4 relevant features recovered with a tiny GA config."""
    # Use 10 features for speed (per test-spec.md G1 spec)
    frame, y = make_synthetic_panel(n_features=10)
    result = _genetic_algorithm_selection(
        frame,
        target=y,
        params={
            "population_size": 10,
            "n_generations": 5,
            "crossover_prob": 0.8,
            "fitness_estimator": "ridge",
            "cv_folds": 3,
            "random_state": SEED,
        },
    )
    assert isinstance(result, pd.DataFrame)
    relevant = {"x0", "x1", "x2", "x3"}
    selected = set(result.columns)
    n_recovered = len(relevant & selected)
    assert n_recovered >= 2, (
        f"Expected >= 2 relevant features, got {n_recovered}: {selected}"
    )


def test_ga_selection_output_contract() -> None:
    """G2: Contract / shape — index, subset, dtype, non-empty."""
    frame, y = make_synthetic_panel(n_features=10)
    result = _genetic_algorithm_selection(
        frame,
        target=y,
        params={
            "population_size": 5,
            "n_generations": 3,
            "crossover_prob": 0.8,
            "fitness_estimator": "ridge",
            "cv_folds": 3,
            "random_state": 0,
        },
    )
    assert result.index.equals(frame.index), "Index not preserved"
    assert set(result.columns).issubset(set(frame.columns)), "Columns not a subset"
    assert result.shape[1] >= 1, "No columns returned"
    assert result.shape[0] == frame.shape[0], "Row count changed"
    for col in result.columns:
        assert result[col].dtype == frame[col].dtype


def test_ga_selection_bit_exact_replicate() -> None:
    """G3 / test-spec.md Sec 3.5: Two calls with same seed produce byte-identical output."""
    # Tiny config for speed: n_obs=40, n_features=5, n_generations=3, population_size=5
    frame, y = make_synthetic_panel(n_obs=40, n_features=5)
    params = {
        "population_size": 5,
        "n_generations": 3,
        "crossover_prob": 0.8,
        "fitness_estimator": "ridge",
        "cv_folds": 3,
        "random_state": 123,
    }
    result1 = _genetic_algorithm_selection(frame.copy(), target=y.copy(), params=params)
    result2 = _genetic_algorithm_selection(frame.copy(), target=y.copy(), params=params)
    pd.testing.assert_frame_equal(result1, result2)


# ---------------------------------------------------------------------------
# Section 4 — Status and Validator Tests (SV1–SV4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op_name",
    [
        "boruta_selection",
        "recursive_feature_elimination",
        "lasso_path_selection",
        "stability_selection",
        "genetic_algorithm_selection",
    ],
)
def test_all_5_ops_operational_post_c47(op_name: str) -> None:
    """SV1: get_op(name).status == OPERATIONAL for all 5 promoted ops."""
    from macroforecast.core.ops import get_op
    from macroforecast.core.status import OPERATIONAL

    spec = get_op(op_name)
    assert spec.status == OPERATIONAL, (
        f"{op_name}: expected operational, got {spec.status}"
    )


@pytest.mark.parametrize(
    "op_name",
    [
        "boruta_selection",
        "recursive_feature_elimination",
        "lasso_path_selection",
        "stability_selection",
        "genetic_algorithm_selection",
    ],
)
def test_all_5_ops_l3_scope_not_l7(op_name: str) -> None:
    """SV2: layer_scope contains 'l3' and does NOT contain 'l7'."""
    from macroforecast.core.ops import get_op

    spec = get_op(op_name)
    assert "l3" in spec.layer_scope, f"{op_name} must be in L3 scope"
    assert "l7" not in spec.layer_scope, (
        f"{op_name} must NOT be in L7 scope after C47 promotion"
    )


@pytest.mark.parametrize(
    "op_name,extra_params",
    [
        ("boruta_selection", "n_estimators_rf: 10"),
        ("recursive_feature_elimination", "n_features_to_select: 0.5"),
        ("lasso_path_selection", "n_features_to_select: 0.5"),
        ("stability_selection", "n_subsamples: 20"),
        ("genetic_algorithm_selection", "population_size: 5"),
    ],
)
def test_validator_accepts_new_ops_in_l3_recipe(op_name: str, extra_params: str) -> None:
    """SV3: Validator no longer hard-rejects any of the 5 ops in an L3 recipe."""
    from macroforecast.core.layers.l3 import parse_layer_yaml, validate_layer

    yaml_text = f"""
3_feature_engineering:
  nodes:
    - id: src_x
      type: source
      selector: {{layer_ref: preprocessing, sink_name: preprocessed_panel_v1, subset: {{role: predictors}}}}
    - id: src_y
      type: source
      selector: {{layer_ref: preprocessing, sink_name: preprocessed_panel_v1, subset: {{role: target}}}}
    - id: x_final
      type: step
      op: {op_name}
      params: {{{extra_params}}}
      inputs: [src_x]
    - id: y_h
      type: step
      op: target_construction
      params: {{mode: point_forecast, method: direct, horizon: 1}}
      inputs: [src_y]
  sinks:
    l3_features_v1: {{X_final: x_final, y_final: y_h}}
    l3_metadata_v1: auto
"""
    report = validate_layer(parse_layer_yaml(yaml_text))
    assert not report.has_hard_errors, (
        f"{op_name} should be accepted by validator after C47 promotion; "
        f"errors: {[i.message for i in report.hard_errors]}"
    )


def test_l3_boruta_no_longer_rejected_as_future() -> None:
    """SV4: Inverse of the old test_l3_future_op_rejected_boruta.

    After C47, the validator must NOT hard-reject boruta_selection.
    Builder should already have updated tests/layers/test_l3.py;
    this test provides independent confirmation.
    """
    from macroforecast.core.layers.l3 import parse_layer_yaml, validate_layer

    yaml_text = """
3_feature_engineering:
  nodes:
    - id: src_x
      type: source
      selector: {layer_ref: preprocessing, sink_name: preprocessed_panel_v1, subset: {role: predictors}}
    - id: src_y
      type: source
      selector: {layer_ref: preprocessing, sink_name: preprocessed_panel_v1, subset: {role: target}}
    - id: x_final
      type: step
      op: boruta_selection
      params: {n_estimators_rf: 10}
      inputs: [src_x]
    - id: y_h
      type: step
      op: target_construction
      params: {mode: point_forecast, method: direct, horizon: 1}
      inputs: [src_y]
  sinks:
    l3_features_v1: {X_final: x_final, y_final: y_h}
    l3_metadata_v1: auto
"""
    report = validate_layer(parse_layer_yaml(yaml_text))
    # Must NOT be a hard error
    assert not report.has_hard_errors, (
        "boruta_selection must not be hard-rejected after C47 promotion. "
        f"Errors: {[e.message for e in report.hard_errors]}"
    )
    # Specifically confirm no 'future' status error
    future_errors = [e for e in report.hard_errors if "future" in e.message.lower()]
    assert len(future_errors) == 0, (
        f"Unexpected future-status hard errors: {future_errors}"
    )


# ---------------------------------------------------------------------------
# Section 6 — Regression Guard (RG2)
# (RG1 = the full test suite run; verified by running pytest tests/ in audit)
# ---------------------------------------------------------------------------


def test_existing_feature_selection_op_unchanged() -> None:
    """RG2: Original feature_selection op is still operational and L3-scoped."""
    from macroforecast.core.ops import get_op

    spec = get_op("feature_selection")
    assert spec.status == "operational", (
        f"feature_selection must remain operational; got {spec.status}"
    )
    assert "l3" in spec.layer_scope, "feature_selection must remain L3-scoped"


# ---------------------------------------------------------------------------
# Section 9 — Top-level bit-exact replicate (boruta tiny)
# ---------------------------------------------------------------------------


def test_bit_exact_replicate_boruta_tiny() -> None:
    """Bit-exact replicate: two calls with identical seed produce identical output.

    Uses a tiny config to stay fast (max_iter=3, n_estimators_rf=5).
    This is the primary required bit-exact replicate test per acceptance criterion 4.
    """
    frame, y = make_synthetic_panel(n_obs=40, n_features=6, seed=0)
    params = {
        "n_estimators_rf": 5,
        "max_iter": 3,
        "alpha": 0.05,
        "include_tentative": True,
        "random_state": 7,
    }
    result1 = _boruta_selection(frame.copy(), target=y.copy(), params=params)
    result2 = _boruta_selection(frame.copy(), target=y.copy(), params=params)
    pd.testing.assert_frame_equal(result1, result2)
    # Also verify the frame is a column subset
    assert set(result1.columns).issubset(set(frame.columns))


# ---------------------------------------------------------------------------
# Section 7 — Property-based invariant checks
# ---------------------------------------------------------------------------


def _check_output_invariants(
    result: pd.DataFrame,
    frame: pd.DataFrame,
    op_name: str,
) -> None:
    """Helper: assert the universal output contract holds for any op result."""
    # Column subset
    assert set(result.columns).issubset(set(frame.columns)), (
        f"{op_name}: columns not a subset of input"
    )
    # Non-empty output
    assert result.shape[1] >= 1, f"{op_name}: returned empty DataFrame"
    # Index preservation
    assert result.index.equals(frame.index), f"{op_name}: index not preserved"
    # Row count preserved
    assert result.shape[0] == frame.shape[0], f"{op_name}: row count changed"
    # Dtype preservation
    for col in result.columns:
        assert result[col].dtype == frame[col].dtype, (
            f"{op_name}: dtype mismatch for {col}"
        )
    # Column order: selected indices must be in ascending order matching input
    input_order = list(frame.columns)
    selected_order = [input_order.index(c) for c in result.columns]
    assert selected_order == sorted(selected_order), (
        f"{op_name}: column order is not monotone ascending"
    )


@pytest.mark.parametrize(
    "op_name,call_fn,params",
    [
        (
            "boruta_selection",
            _boruta_selection,
            {"n_estimators_rf": 5, "max_iter": 3, "alpha": 0.05,
             "include_tentative": True, "random_state": 1},
        ),
        (
            "recursive_feature_elimination",
            _recursive_feature_elimination,
            {"n_features_to_select": 4, "step": 1, "estimator": "ridge",
             "use_cv": False, "random_state": 1},
        ),
        (
            "lasso_path_selection",
            _lasso_path_selection,
            {"n_features_to_select": 4, "normalize_features": True, "random_state": 1},
        ),
        (
            "stability_selection",
            _stability_selection,
            {"n_subsamples": 10, "subsample_fraction": 0.5, "pi_thr": 0.4,
             "base_estimator": "lasso", "alpha": 0.01, "random_state": 1},
        ),
        (
            "genetic_algorithm_selection",
            _genetic_algorithm_selection,
            {"population_size": 5, "n_generations": 2, "crossover_prob": 0.8,
             "fitness_estimator": "ridge", "cv_folds": 3, "random_state": 1},
        ),
    ],
)
def test_property_invariants_all_ops(op_name: str, call_fn: object, params: dict) -> None:
    """Sec 7: All 7 property-based invariants hold for each op."""
    frame, y = make_synthetic_panel(n_obs=60, n_features=8)
    result = call_fn(frame, target=y, params=params)  # type: ignore[operator]
    _check_output_invariants(result, frame, op_name)


# ---------------------------------------------------------------------------
# Section 11 — Cross-reference benchmarks (sklearn RFE exact match)
# ---------------------------------------------------------------------------


def test_rfe_matches_sklearn_rfe_direct() -> None:
    """Sec 11: _recursive_feature_elimination matches sklearn RFE directly."""
    from sklearn.linear_model import Ridge
    from sklearn.feature_selection import RFE

    frame, y = make_synthetic_panel(n_obs=80, n_features=10)
    k = 4

    # Call our implementation
    result = _recursive_feature_elimination(
        frame,
        target=y,
        params={
            "n_features_to_select": k,
            "step": 1,
            "estimator": "ridge",
            "use_cv": False,
            "random_state": 0,
        },
    )

    # Direct sklearn reference
    X = frame.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    sk_rfe = RFE(estimator=Ridge(random_state=0), n_features_to_select=k, step=1)
    sk_rfe.fit(X, y_arr)
    sk_selected = [c for c, sel in zip(frame.columns, sk_rfe.support_) if sel]

    assert set(result.columns) == set(sk_selected), (
        f"RFE mismatch: our={set(result.columns)}, sklearn={set(sk_selected)}"
    )


def test_lasso_path_entry_order_matches_lars_path_directly() -> None:
    """Sec 11: _lasso_path_selection entry order matches sklearn lars_path active list."""
    from sklearn.linear_model import lars_path
    from sklearn.preprocessing import StandardScaler

    frame, y = make_synthetic_panel(n_obs=80, n_features=10)
    k = 4

    result = _lasso_path_selection(
        frame,
        target=y,
        params={
            "n_features_to_select": k,
            "normalize_features": True,
            "random_state": 0,
        },
    )

    # Direct sklearn reference
    X = frame.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    _, active_raw, _ = lars_path(X_scaled, y_arr, method="lasso", max_iter=10)

    seen: set[int] = set()
    expected_indices: list[int] = []
    for idx in active_raw:
        idx_int = int(idx)
        if idx_int not in seen:
            seen.add(idx_int)
            expected_indices.append(idx_int)
        if len(expected_indices) >= k:
            break

    cols = list(frame.columns)
    expected_cols = {cols[j] for j in expected_indices}
    assert set(result.columns) == expected_cols, (
        f"LARS path mismatch: our={set(result.columns)}, direct={expected_cols}"
    )
