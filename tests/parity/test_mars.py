"""``macroforecast.models.mars`` vs R ``earth`` -- work item 7 of the WP-V1
brief (lowest priority / time-permitting item, per the brief's own
ordering).

macroforecast's own docstring already disclaims exact parity: ``mars``'s
``implementation_note`` reads "Package-native Friedman/earth-aligned
hinge-basis MARS-style estimator; not a full R earth clone." Hyperparameters
do not map 1:1 either (macroforecast's ``max_terms``/``n_knots``/
``min_improvement``/``penalty`` vs earth's ``nprune``/``nk``/``thresh``/
``penalty`` use different internal search heuristics -- forward-pass knot
placement, pruning criterion, and termination rule all differ in detail).
Per the brief ("coefficients not required, prediction parity within
tolerance"), this file does NOT attempt a term-by-term structural match; it
checks that both algorithms, given the SAME deterministic known-hinge
fixture (no noise, so there is a single unambiguous ground truth), (a) each
independently recovers the true function to a tight RMSE, and (b) their
predictions therefore agree with each other to a loose, documented
tolerance (bounded by how well each independently tracks the true function,
not by any shared internal formula).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf

from tests.parity.conftest import parse_float_list, require_r, run_rscript, write_csv

pytestmark = [pytest.mark.rparity]


def _fixture(n: int = 300, seed: int = 4) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(0.0, 1.0, size=n)
    x2 = np.sin(np.linspace(0.0, 6.0, n))
    true_y = 1.0 + 3.0 * np.maximum(0.0, x1 - 0.45) - 0.25 * x2
    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(true_y, name="y")
    return X, y, true_y


def _r_earth_predict(X: pd.DataFrame, y: pd.Series, tmp_path) -> list[float]:
    x_path = tmp_path / "x.csv"
    y_path = tmp_path / "y.csv"
    X.to_csv(x_path, index=False)
    write_csv(y_path, {"y": y.to_numpy()})
    script = f'''
library(earth)
x <- read.csv("{x_path}")
y <- read.csv("{y_path}")$y
fit <- earth(x = x, y = y, degree = 1, nprune = 8, nk = 21, penalty = 2, thresh = 1e-6)
pred <- as.numeric(predict(fit, newdata = x))
emit("pred", pred)
'''
    result = run_rscript(script, timeout=120)
    return parse_float_list(result["pred"])


def test_mars_and_earth_each_recover_known_hinge_function(tmp_path) -> None:
    """Each side independently: does it get close to the true noiseless
    function? (Not a cross-language comparison yet -- a precondition for
    the cross-language check below to be meaningful at all.)
    """
    require_r("earth")
    X, y, true_y = _fixture()

    fit = mf.models.mars(X, y, max_terms=8, n_knots=6)
    py_pred = np.asarray(fit.predict(X))
    py_rmse = float(np.sqrt(np.mean((py_pred - true_y) ** 2)))

    r_pred = np.asarray(_r_earth_predict(X, y, tmp_path))
    r_rmse = float(np.sqrt(np.mean((r_pred - true_y) ** 2)))

    y_std = float(np.std(true_y))
    assert py_rmse < 0.1 * y_std, f"macroforecast mars did not recover the known hinge function: rmse={py_rmse}"
    assert r_rmse < 0.1 * y_std, f"R earth did not recover the known hinge function: rmse={r_rmse}"


def test_mars_predictions_agree_with_earth_within_loose_tolerance(tmp_path) -> None:
    require_r("earth")
    X, y, true_y = _fixture()

    fit = mf.models.mars(X, y, max_terms=8, n_knots=6)
    py_pred = np.asarray(fit.predict(X))
    r_pred = np.asarray(_r_earth_predict(X, y, tmp_path))

    max_abs_diff = float(np.max(np.abs(py_pred - r_pred)))
    corr = float(np.corrcoef(py_pred, r_pred)[0, 1])
    y_range = float(true_y.max() - true_y.min())

    # Loose, documented tolerance: two independent hinge-basis heuristics
    # fit to the same noiseless data should track each other closely, but
    # not to floating-point precision (different knot placement/pruning).
    assert max_abs_diff < 0.15 * y_range, (
        f"mars vs earth predictions differ by more than 15% of the y-range: "
        f"max_abs_diff={max_abs_diff!r}, y_range={y_range!r}"
    )
    assert corr > 0.98, f"mars vs earth predictions poorly correlated: corr={corr!r}"
