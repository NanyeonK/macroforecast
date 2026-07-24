"""Regression tests for deterministic per-tree seeding of the Macro Random Forest.

Locks the objective-4 property: with random_state set, a serial fit is reproducible
and a parallel fit is BIT-IDENTICAL to it (parallelism is a pure speedup, never a
result change). random_state=None preserves the historical non-deterministic behavior.
"""
import numpy as np
import pandas as pd
import pytest

from macroforecast.models import macro_random_forest


def _toy():
    rng = np.random.default_rng(0)
    n = 90
    idx = pd.date_range("1990-01-01", periods=n, freq="QS")
    y = pd.Series(np.cumsum(rng.normal(size=n)) * 0.1, index=idx, name="y")
    cols = {f"s{j}": pd.Series(np.cumsum(rng.normal(size=n)), index=idx) for j in range(12)}
    X = pd.DataFrame(cols)
    X["y_lag1"] = y.shift(1)
    X["y_lag2"] = y.shift(2)
    keep = X.notna().all(axis=1) & y.notna()
    return X[keep], y[keep]


def _predict(X, y, **kw):
    fit = macro_random_forest(
        X, y, x_columns=["y_lag1", "y_lag2"], S_columns=[c for c in X.columns],
        B=8, minsize=8, mtry_frac=1 / 3, ridge_lambda=0.1, rw_regul=0.75,
        block_size=8, resampling_opt=2, **kw,
    )
    return np.asarray(fit.predict(X.tail(6)), dtype=float).ravel()


def test_mrf_seeded_fit_is_reproducible():
    X, y = _toy()
    a = _predict(X, y, random_state=42)
    b = _predict(X, y, random_state=42)
    np.testing.assert_array_equal(a, b)


def test_mrf_parallel_is_bit_identical_to_serial_when_seeded():
    X, y = _toy()
    serial = _predict(X, y, random_state=42)
    parallel = _predict(X, y, random_state=42, parallelise=True, n_cores=4)
    np.testing.assert_array_equal(serial, parallel)


def test_mrf_seed_actually_changes_the_forest():
    X, y = _toy()
    a = _predict(X, y, random_state=42)
    b = _predict(X, y, random_state=7)
    assert np.max(np.abs(a - b)) > 1e-9


def test_mrf_unseeded_preserves_nondeterministic_behavior():
    # No random_state -> historical behavior (fresh global np.random draws each fit).
    X, y = _toy()
    a = _predict(X, y)
    b = _predict(X, y)
    assert np.max(np.abs(a - b)) > 1e-12
