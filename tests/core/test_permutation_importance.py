"""tests/core/test_permutation_importance.py

PR2 TDD tests: permutation importance RNG fix.

These tests were written BEFORE the fix was applied (TDD red phase) and
verify the three properties required by the Breiman (2001) / Fisher-Rudin-
Dominici (2019) definition:

  1. The permutation is random, not deterministic reversal.
  2. Same seed produces reproducible results; different seeds differ.
  3. n_repeats > 1 produces a (n_features, n_repeats) importance matrix.

The runtime helper _permutation_importance_frame is tested directly; the
public API mf.functions.permutation_importance is tested for consistency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macroforecast.core.runtime import _permutation_importance_frame
from macroforecast.core.types import ModelArtifact


# ---------------------------------------------------------------------------
# Helper: build a minimal ModelArtifact with a trivial predict-able model
# ---------------------------------------------------------------------------

class _LinearMock:
    """Minimal predict-compatible model: y_hat = X @ coef."""

    def __init__(self, coef: np.ndarray) -> None:
        self.coef_ = coef

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X.values @ self.coef_


def _make_artifact(coef: np.ndarray, feature_names: list[str]) -> ModelArtifact:
    mock = _LinearMock(coef)
    artifact = ModelArtifact.__new__(ModelArtifact)
    object.__setattr__(artifact, "fitted_object", mock)
    object.__setattr__(artifact, "feature_names", tuple(feature_names))
    object.__setattr__(artifact, "target_name", "y")
    object.__setattr__(artifact, "model_family", "ols")
    object.__setattr__(artifact, "params", {})
    object.__setattr__(artifact, "metadata", {})
    return artifact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def linear_dataset():
    """Small (n=30) dataset where x0 is the only relevant predictor."""
    rng = np.random.default_rng(42)
    n = 30
    X = pd.DataFrame(
        {
            "x0": np.arange(n, dtype=float),
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
        }
    )
    y = pd.Series(X["x0"] * 2.0 + 0.1 * rng.standard_normal(n), name="y")
    coef = np.array([2.0, 0.0, 0.0])
    artifact = _make_artifact(coef, ["x0", "x1", "x2"])
    return artifact, X, y


# ---------------------------------------------------------------------------
# Test 1: Permutation is random, not deterministic reversal
# ---------------------------------------------------------------------------

class TestPermutationIsRandom:
    """Two independent calls (no explicit seed) must produce different results.

    Under the old buggy code, `list(reversed(...))` is deterministic, so both
    calls return identical importances.  Under the fix, each call to
    _permutation_importance_frame uses np.random.default_rng(seed) where
    seed defaults to 0, but importantly the test verifies that two calls
    with different seeds produce different outputs.
    """

    def test_different_seeds_produce_different_importances(self, linear_dataset):
        """Two calls with different seeds must differ on at least one feature."""
        artifact, X, y = linear_dataset
        frame_s0 = _permutation_importance_frame(
            artifact, X, y, method="permutation_importance", n_repeats=1, seed=0
        )
        frame_s1 = _permutation_importance_frame(
            artifact, X, y, method="permutation_importance", n_repeats=1, seed=1
        )
        imp_s0 = frame_s0.set_index("feature")["importance"]
        imp_s1 = frame_s1.set_index("feature")["importance"]
        # At least one feature must have a different importance estimate
        assert not np.allclose(imp_s0.values, imp_s1.values), (
            "Two calls with different seeds must produce different importances "
            "(if they are identical, the implementation is deterministic/reversal, not RNG)"
        )

    def test_not_a_reversal(self, linear_dataset):
        """Verify the permuted column is NOT simply list(reversed(...)).

        The old bug: permuted[col] = list(reversed(permuted[col].tolist())).
        We can detect this by checking whether the permuted column in any
        repeat is the exact reverse of the original.
        """
        artifact, X, y = linear_dataset
        # Patch: call with n_repeats=5 and verify importance has nonzero std
        # (reversed gives identical loss on every repeat, so std=0)
        frame = _permutation_importance_frame(
            artifact, X, y, method="permutation_importance", n_repeats=5, seed=42
        )
        # importances_ column holds per-repeat values (list of floats)
        assert "importances_" in frame.columns, (
            "_permutation_importance_frame must return an 'importances_' column "
            "with per-repeat values when n_repeats > 1"
        )
        vals_x0 = frame.loc[frame["feature"] == "x0", "importances_"].values[0]
        # With a proper random permutation over 5 repeats, the values must not
        # all be identical (reversal always gives the same permutation)
        assert len(set(round(v, 12) for v in vals_x0)) > 1, (
            "All n_repeats importance values for x0 are identical — "
            "this indicates a deterministic permutation (e.g. reversal), not RNG"
        )


# ---------------------------------------------------------------------------
# Test 2: Reproducibility with seed
# ---------------------------------------------------------------------------

class TestReproducibilityWithSeed:
    """Same seed must produce identical results; different seeds must differ."""

    def test_same_seed_produces_identical_results(self, linear_dataset):
        """Two calls with the same seed and same n_repeats must be bit-exact."""
        artifact, X, y = linear_dataset
        f1 = _permutation_importance_frame(
            artifact, X, y, method="permutation_importance", n_repeats=5, seed=7
        )
        f2 = _permutation_importance_frame(
            artifact, X, y, method="permutation_importance", n_repeats=5, seed=7
        )
        np.testing.assert_array_equal(
            f1["importance"].values,
            f2["importance"].values,
            err_msg="Same seed must produce bit-exact importance values",
        )

    def test_public_api_same_seed_bit_exact(self, linear_dataset):
        """mf.functions.permutation_importance with same random_state is bit-exact."""
        import macroforecast as mf

        rng = np.random.default_rng(0)
        n, p = 50, 4
        X_arr = rng.standard_normal((n, p))
        y_arr = X_arr[:, 0] * 3.0 + X_arr[:, 1] * (-1.5) + 0.1 * rng.standard_normal(n)
        fit = mf.functions.ols_fit(X_arr, y_arr)

        imp_a = mf.functions.permutation_importance(fit, X_arr, y_arr, n_repeats=5, random_state=42)
        imp_b = mf.functions.permutation_importance(fit, X_arr, y_arr, n_repeats=5, random_state=42)

        np.testing.assert_array_almost_equal(
            imp_a.importances_mean_,
            imp_b.importances_mean_,
            decimal=10,
            err_msg="Same random_state must yield identical importances_mean_",
        )

    def test_public_api_different_seeds_differ(self, linear_dataset):
        """mf.functions.permutation_importance with different random_state must differ."""
        import macroforecast as mf

        rng = np.random.default_rng(0)
        n, p = 50, 4
        X_arr = rng.standard_normal((n, p))
        y_arr = X_arr[:, 0] * 3.0 + X_arr[:, 1] * (-1.5) + 0.1 * rng.standard_normal(n)
        fit = mf.functions.ols_fit(X_arr, y_arr)

        imp_a = mf.functions.permutation_importance(fit, X_arr, y_arr, n_repeats=5, random_state=42)
        imp_c = mf.functions.permutation_importance(fit, X_arr, y_arr, n_repeats=5, random_state=99)

        assert not np.allclose(imp_a.importances_mean_, imp_c.importances_mean_), (
            "Different random_state must produce different importances_mean_ "
            "(if identical, the seed is being ignored)"
        )


# ---------------------------------------------------------------------------
# Test 3: n_repeats affects importances_ shape and enables variance estimation
# ---------------------------------------------------------------------------

class TestNRepeatsShape:
    """n_repeats > 1 must produce a per-repeat importance matrix."""

    def test_importances_matrix_shape(self):
        """mf.functions.permutation_importance returns importances_ of shape (p, n_repeats)."""
        import macroforecast as mf

        rng = np.random.default_rng(0)
        n, p = 60, 3
        X_arr = rng.standard_normal((n, p))
        y_arr = X_arr[:, 0] * 2.0 + 0.5 * rng.standard_normal(n)
        fit = mf.functions.ols_fit(X_arr, y_arr)

        imp = mf.functions.permutation_importance(fit, X_arr, y_arr, n_repeats=10, random_state=0)

        assert imp.importances_.shape == (p, 10), (
            f"Expected importances_ shape ({p}, 10), got {imp.importances_.shape}"
        )

    def test_importances_mean_matches_importances_matrix(self):
        """importances_mean_ must equal importances_.mean(axis=1)."""
        import macroforecast as mf

        rng = np.random.default_rng(0)
        n, p = 60, 3
        X_arr = rng.standard_normal((n, p))
        y_arr = X_arr[:, 0] * 2.0 + 0.5 * rng.standard_normal(n)
        fit = mf.functions.ols_fit(X_arr, y_arr)

        imp = mf.functions.permutation_importance(fit, X_arr, y_arr, n_repeats=10, random_state=0)

        np.testing.assert_array_almost_equal(
            imp.importances_mean_,
            imp.importances_.mean(axis=1),
            decimal=10,
            err_msg="importances_mean_ must equal importances_.mean(axis=1)",
        )

    def test_n_repeats_one_returns_single_column(self, linear_dataset):
        """With n_repeats=1, importances_ column in frame has exactly one value."""
        artifact, X, y = linear_dataset
        frame = _permutation_importance_frame(
            artifact, X, y, method="permutation_importance", n_repeats=1, seed=0
        )
        assert "importances_" in frame.columns
        for _, row in frame.iterrows():
            assert len(row["importances_"]) == 1, (
                f"n_repeats=1 must produce exactly one value per feature, "
                f"got {len(row['importances_'])}"
            )


# ---------------------------------------------------------------------------
# Test 4: Recipe path and standalone API produce consistent rankings
# ---------------------------------------------------------------------------

class TestRecipeVsStandaloneConsistency:
    """The recipe path (_permutation_importance_frame) and the public API
    (mf.functions.permutation_importance) must produce consistent feature
    importance rankings when using the same seed.

    Before the fix: recipe used reversal, standalone used RNG — they diverged.
    After the fix: both use np.random.default_rng(seed) — rankings agree.
    """

    def test_ranking_agreement_same_seed(self, linear_dataset):
        """Top feature must agree between runtime helper and public API."""
        import macroforecast as mf

        artifact, X, y = linear_dataset
        # Runtime path
        frame = _permutation_importance_frame(
            artifact, X, y, method="permutation_importance", n_repeats=5, seed=0
        )
        top_runtime = frame.sort_values("importance", ascending=False).iloc[0]["feature"]

        # Public API path — same seed
        rng_outer = np.random.default_rng(42)
        n = 30
        X_api = pd.DataFrame(
            {
                "x0": np.arange(n, dtype=float),
                "x1": rng_outer.standard_normal(n),
                "x2": rng_outer.standard_normal(n),
            }
        )
        y_api = pd.Series(X_api["x0"] * 2.0 + 0.1 * rng_outer.standard_normal(n), name="y")
        fit = mf.functions.ols_fit(X_api.values, y_api.values)
        imp = mf.functions.permutation_importance(fit, X_api.values, y_api.values, n_repeats=5, random_state=0)

        # Both paths: x0 must be the most important feature
        api_top_idx = np.argmax(imp.importances_mean_)
        assert top_runtime == "x0", (
            f"Runtime helper top feature: expected x0, got {top_runtime}"
        )
        assert api_top_idx == 0, (
            f"Public API top feature index: expected 0 (x0), got {api_top_idx}"
        )
