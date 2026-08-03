"""`ols` must say something when its design is numerically rank-deficient.

Fitting a rank-deficient design is not an error -- least squares still returns *a*
solution -- but it is not *the* solution: the answer is not unique, and which one
comes back is decided by LAPACK's truncation rather than by the data. When the
truncation does not happen, the coefficients come back as an exploded cancelling
pair and the prediction carries the floating-point residue of that cancellation.
That was measured at up to 0.56 in level on real data (issue #487), delivered with
no signal of any kind.

`rank_` cannot be the detector: in the observed failure LAPACK reported full rank
on a rank-one matrix, and that mis-report IS the failure. `singular_` -- which
sklearn already computes, so this costs nothing -- does expose it.

The warning changes no number. Every prediction is exactly what it was.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _y(n: int, rng: np.random.Generator) -> pd.Series:
    return pd.Series(rng.normal(size=n), name="y")


def test_exactly_duplicated_column_warns() -> None:
    """The shape that surfaced this: the same series entering twice."""
    rng = np.random.default_rng(0)
    n = 120
    a = rng.normal(size=n)
    X = pd.DataFrame({"x": a, "x_again": a})
    with pytest.warns(UserWarning, match="rank-deficient"):
        mf.models.ols(X, _y(n, rng))


def test_dummies_summing_to_the_intercept_warn() -> None:
    """The shape a user hits by accident: a complete dummy set plus an intercept."""
    rng = np.random.default_rng(1)
    n = 120
    even = (np.arange(n) % 2 == 0).astype(float)
    X = pd.DataFrame({"d_even": even, "d_odd": 1.0 - even})
    with pytest.warns(UserWarning, match="rank-deficient"):
        mf.models.ols(X, _y(n, rng))


def test_a_healthy_design_is_silent() -> None:
    """No false positives: an ordinary design must not warn."""
    rng = np.random.default_rng(2)
    n = 120
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails this test
        mf.models.ols(X, _y(n, rng))


def test_merely_ill_conditioned_is_silent() -> None:
    """Scoped deliberately: near-collinear is unique, just imprecise, so it is
    the caller's business. Only genuine rank deficiency is flagged."""
    rng = np.random.default_rng(3)
    n = 120
    a = rng.normal(size=n)
    X = pd.DataFrame({"x": a, "x_nudged": a + 1e-9 * rng.normal(size=n)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mf.models.ols(X, _y(n, rng))


def test_predictions_are_untouched() -> None:
    """The point of the fix: it warns, it does not change the answer."""
    rng = np.random.default_rng(4)
    n = 120
    a = rng.normal(size=n)
    X = pd.DataFrame({"x": a, "x_again": a})
    y = _y(n, rng)
    from sklearn.linear_model import LinearRegression

    reference = LinearRegression().fit(X, y).predict(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = np.asarray(mf.models.ols(X, y).predict(X), dtype=float).ravel()
    np.testing.assert_array_equal(got, np.asarray(reference, dtype=float).ravel())


def test_the_message_names_the_problem_and_a_remedy() -> None:
    rng = np.random.default_rng(5)
    n = 120
    a = rng.normal(size=n)
    X = pd.DataFrame({"x": a, "x_again": a})
    with pytest.warns(UserWarning) as caught:
        mf.models.ols(X, _y(n, rng))
    text = str(caught[0].message)
    assert "ols" in text
    assert "not unique" in text
    assert "ridge" in text.lower(), "should point at a regularized alternative"


def test_a_solver_without_singular_values_does_not_crash() -> None:
    """`positive=True` uses a different solver and exposes no `singular_`."""
    rng = np.random.default_rng(6)
    n = 120
    a = rng.normal(size=n)
    X = pd.DataFrame({"x": a, "x_again": a})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mf.models.ols(X, _y(n, rng), positive=True)
    assert fit.model == "ols"
