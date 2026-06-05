"""Regression test for MWC-01.

meta.configure(n_jobs=...) was stored but never consulted by any execution path.
The configured worker count must now actually reach the parallelizable tree
ensembles (random_forest / gradient_boosting) when the caller does not override
n_jobs explicitly, with 'auto' resolved to the CPU count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast import meta


def _xy(n: int = 40):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(X["a"] - 0.5 * X["b"] + rng.normal(scale=0.1, size=n), name="y")
    return X, y


def test_configured_n_jobs_reaches_random_forest():
    X, y = _xy()
    try:
        meta.configure(n_jobs=3)
        fit = mf.models.random_forest(X, y, n_estimators=10)
        assert fit.estimator.n_jobs == 3
        # Explicit override still wins.
        fit2 = mf.models.random_forest(X, y, n_estimators=10, n_jobs=1)
        assert fit2.estimator.n_jobs == 1
    finally:
        meta.reset_config()


def test_resolve_n_jobs_auto_is_cpu_count():
    import os
    try:
        meta.configure(n_jobs="auto")
        assert meta.resolve_n_jobs() == (os.cpu_count() or 1)
    finally:
        meta.reset_config()
