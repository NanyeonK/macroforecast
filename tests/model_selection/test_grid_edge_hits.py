"""Model selection must say when it picked a value at the edge of its own grid.

A search that keeps landing on the smallest or largest value it was offered is reporting
that the optimum may lie outside the grid -- usually a grid written on the wrong scale.
Returning that edge value silently is indistinguishable from a genuine interior optimum,
and a recursive search repeats the mistake at every origin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.model_selection import select_params


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    n = 240
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}
    )
    strong = pd.Series(3.0 * X["a"] - 2.0 * X["b"] + rng.normal(0.0, 0.1, n), name="y")
    weak = pd.Series(0.01 * X["a"] + rng.normal(0.0, 1.0, n), name="y")
    return X, strong, weak


def _select(X, y, alphas):
    spec = mf.model_selection.grid({"alpha": [float(a) for a in alphas]})
    return select_params("lasso", X, y, spec, fixed_params={"standardize": True})


def test_interior_optimum_reports_no_edge_hit(data) -> None:
    X, strong, _ = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _select(X, strong, np.logspace(-4, 0, 9))

    assert result.metadata["grid_edge_hits"] == {}
    assert not [w for w in caught if "edge of its search grid" in str(w.message)]


def test_grid_below_the_optimum_reports_an_upper_edge_hit(data) -> None:
    X, _, weak = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _select(X, weak, [1e-6, 1e-5, 1e-4])

    assert result.metadata["grid_edge_hits"] == {"alpha": "upper"}
    assert result.best_params["alpha"] == pytest.approx(1e-4)
    assert any("upper edge" in str(w.message) for w in caught)


def test_grid_above_the_optimum_reports_a_lower_edge_hit(data) -> None:
    X, strong, _ = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _select(X, strong, [0.3, 0.6, 1.0])

    assert result.metadata["grid_edge_hits"] == {"alpha": "lower"}
    assert result.best_params["alpha"] == pytest.approx(0.3)
    assert any("lower edge" in str(w.message) for w in caught)


def test_single_valued_parameter_has_no_edge(data) -> None:
    """A fixed parameter was never searched, so it cannot sit at an edge."""
    X, strong, _ = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _select(X, strong, [0.05])

    assert result.metadata["grid_edge_hits"] == {}
    assert not [w for w in caught if "edge of its search grid" in str(w.message)]
