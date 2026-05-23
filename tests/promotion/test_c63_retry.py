"""Tester-authored independent validation for Cycle 63 Retry.

Validates:
B. fit_transform exists and works for all 5 feature-selection classes
C. NotFittedError is raised on transform() before fit()
D. @pytest.mark.slow collection filter for C56 tests (D-1 done separately via CLI)

This file is authored independently from builder's tests and implementation.md.
Tester does NOT read implementation.md.

Synthetic data:
  _make_xy       — seed=42, n=80,  p=10, pure noise (for C-section shape/type tests)
  _make_xy_signal — seed=42, n=200, p=10, 2 true signals at x0 and x2
                   (for B-2 and B-3 where column count must be >= 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Shared synthetic data factories
# ---------------------------------------------------------------------------

def _make_xy(
    n: int = 80,
    p: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) with pure noise — used for C-section tests that only
    need shape / type checks and do not assert on column count."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, p)),
        columns=[f"x{i}" for i in range(p)],
    )
    y = pd.Series(rng.standard_normal(n), name="y")
    return X, y


def _make_xy_signal(
    n: int = 200,
    p: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) with 2 true signals at columns x0 (beta=1.5) and x2
    (beta=1.2), noise std=0.3.  Used for B-2 and B-3 where the assertion
    1 <= cols <= p requires at least one feature to be selected.

    n=200 gives Boruta adequate power to detect both signals reliably.
    """
    rng = np.random.default_rng(seed)
    X_arr = rng.standard_normal((n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 1.5
    true_beta[2] = 1.2
    y_arr = X_arr @ true_beta + 0.3 * rng.standard_normal(n)
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(p)])
    y = pd.Series(y_arr, name="y")
    return X, y


# ---------------------------------------------------------------------------
# Section B — fit_transform verification
# ---------------------------------------------------------------------------

class TestFitTransform:
    """Scenarios B-1, B-2, B-3 from test-spec-addendum.md Section 2."""

    CLASSES = [
        "Boruta",
        "RFE",
        "LassoPathSelector",
        "StabilitySelection",
        "GeneticSelection",
    ]

    def _import_cls(self, name: str):
        from macroforecast.feature_selection import (
            Boruta, RFE, LassoPathSelector, StabilitySelection, GeneticSelection,
        )
        return {
            "Boruta": Boruta,
            "RFE": RFE,
            "LassoPathSelector": LassoPathSelector,
            "StabilitySelection": StabilitySelection,
            "GeneticSelection": GeneticSelection,
        }[name]

    # --- B-1: fit_transform is callable ---

    @pytest.mark.parametrize("cls_name", CLASSES)
    def test_B1_fit_transform_callable(self, cls_name: str) -> None:
        """fit_transform must exist as a callable attribute on each class instance."""
        cls = self._import_cls(cls_name)
        instance = cls()
        assert callable(getattr(instance, "fit_transform", None)), (
            f"{cls_name}.fit_transform is not callable"
        )

    # --- B-2: fit_transform returns pd.DataFrame with correct shape ---

    @pytest.mark.parametrize("cls_name", CLASSES)
    def test_B2_fit_transform_returns_dataframe(self, cls_name: str) -> None:
        """fit_transform must return a pd.DataFrame with n rows and 1..p cols.

        Uses _make_xy_signal (n=200, 2 true signals at x0/x2) so that all
        selectors — including Boruta — have enough signal to select >= 1
        feature.  The null-hypothesis case (0 features) is a correct FP-
        control behaviour (C56/C59) but is not the contract tested here.
        """
        X, y = _make_xy_signal(n=200, p=10, seed=42)
        cls = self._import_cls(cls_name)
        result = cls().fit_transform(X, y)

        # Type check
        assert isinstance(result, pd.DataFrame), (
            f"{cls_name}.fit_transform returned {type(result)}, expected pd.DataFrame"
        )
        # Row count: must equal n
        assert result.shape[0] == 200, (
            f"{cls_name}.fit_transform: expected 200 rows, got {result.shape[0]}"
        )
        # Column count: must be in [1, p]
        assert 1 <= result.shape[1] <= 10, (
            f"{cls_name}.fit_transform: column count {result.shape[1]} not in [1, 10]"
        )
        # Columns must be a subset of X.columns
        assert set(result.columns).issubset(set(X.columns)), (
            f"{cls_name}.fit_transform: columns {list(result.columns)} not subset of X"
        )

    # --- B-3: fit_transform consistent with fit().transform() for Boruta ---

    def test_B3_boruta_fit_transform_consistent(self) -> None:
        """Boruta: fit_transform(X, y) columns == fit(X, y).transform(X) columns.

        Uses _make_xy_signal so Boruta selects >= 1 feature (deterministic
        consistency check requires a non-empty result set).
        """
        from macroforecast.feature_selection import Boruta

        X, y = _make_xy_signal(n=200, p=10, seed=42)

        # Path A: fit_transform
        inst_a = Boruta(random_state=42)
        result_a = inst_a.fit_transform(X, y)

        # Path B: fit then transform (same random_state for determinism)
        inst_b = Boruta(random_state=42)
        result_b = inst_b.fit(X, y).transform(X)

        assert result_a.columns.tolist() == result_b.columns.tolist(), (
            f"Boruta fit_transform columns {result_a.columns.tolist()} "
            f"!= fit+transform columns {result_b.columns.tolist()}"
        )


# ---------------------------------------------------------------------------
# Section C — NotFittedError verification
# ---------------------------------------------------------------------------

class TestNotFittedError:
    """Scenarios C-1 and C-2 from test-spec-addendum.md Section 3."""

    CLASSES = [
        "Boruta",
        "RFE",
        "LassoPathSelector",
        "StabilitySelection",
        "GeneticSelection",
    ]

    def _import_cls(self, name: str):
        from macroforecast.feature_selection import (
            Boruta, RFE, LassoPathSelector, StabilitySelection, GeneticSelection,
        )
        return {
            "Boruta": Boruta,
            "RFE": RFE,
            "LassoPathSelector": LassoPathSelector,
            "StabilitySelection": StabilitySelection,
            "GeneticSelection": GeneticSelection,
        }[name]

    # --- C-1: transform() raises NotFittedError on unfitted instance ---

    @pytest.mark.parametrize("cls_name", CLASSES)
    def test_C1_transform_raises_not_fitted_error(self, cls_name: str) -> None:
        """transform() before fit() must raise sklearn.exceptions.NotFittedError."""
        from sklearn.exceptions import NotFittedError

        X, _ = _make_xy(n=80, p=10, seed=42)
        cls = self._import_cls(cls_name)

        with pytest.raises(NotFittedError, match=cls_name):
            cls().transform(X)

    # --- C-2: transform() does NOT raise after fit() ---

    @pytest.mark.parametrize("cls_name", CLASSES)
    def test_C2_transform_ok_after_fit(self, cls_name: str) -> None:
        """transform() after fit() must not raise and must return pd.DataFrame."""
        X, y = _make_xy(n=80, p=10, seed=42)
        cls = self._import_cls(cls_name)

        instance = cls()
        instance.fit(X, y)
        result = instance.transform(X)  # must not raise

        assert isinstance(result, pd.DataFrame), (
            f"{cls_name}.transform after fit returned {type(result)}, expected pd.DataFrame"
        )
