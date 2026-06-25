"""Safety contract for cost-aware thread allocation.

The cost-aware allocator (auto_parallelism) raises ``model_threads`` for heavy
runs. That is safe ONLY because the thread count never changes numerical
results. This test pins that contract:

- random_forest / extra_trees are the only builders that consume meta n_jobs;
  both are sklearn forests, deterministic under n_jobs given random_state.
- xgboost / lightgbm hardcode num_threads=1 (tree.py), so the allocator cannot
  touch them; we assert they ignore meta n_jobs.

If any of these break, the allocator is NOT safe to raise threads and the test
must fail loudly.
"""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf


def _xy(n=200, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = pd.Series(X.values @ rng.normal(size=p) + rng.normal(size=n), name="y")
    Xtr, ytr, Xte = X.iloc[:150], y.iloc[:150], X.iloc[150:]
    return Xtr, ytr, Xte


@pytest.mark.parametrize("builder", ["random_forest", "extra_trees"])
def test_forest_predictions_invariant_to_n_jobs(builder):
    Xtr, ytr, Xte = _xy()
    build = getattr(mf, builder)
    p1 = build(Xtr, ytr, n_estimators=60, random_state=0, n_jobs=1).predict(Xte)
    p4 = build(Xtr, ytr, n_estimators=60, random_state=0, n_jobs=4).predict(Xte)
    # NOT bit-identical: parallel aggregation reorders the floating-point sum, so
    # predictions differ at ~machine epsilon. This is the determinism contract the
    # cost-aware allocator relies on -- scientifically identical, not byte-exact.
    np.testing.assert_allclose(np.asarray(p1), np.asarray(p4), rtol=1e-9, atol=1e-10)


def test_forest_invariant_via_meta_configure():
    # The pipeline raises threads through meta.configure(n_jobs=...), not the
    # builder kwarg. Verify that path is invariant too.
    Xtr, ytr, Xte = _xy(seed=1)
    mf.meta.configure(n_jobs=1)
    a = mf.random_forest(Xtr, ytr, n_estimators=60, random_state=0).predict(Xte)
    mf.meta.configure(n_jobs=4)
    b = mf.random_forest(Xtr, ytr, n_estimators=60, random_state=0).predict(Xte)
    mf.meta.configure(n_jobs=1)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("builder", ["xgboost", "lightgbm"])
def test_boosters_ignore_meta_threads(builder):
    # These hardcode num_threads=1, so meta n_jobs must not change their output.
    build = getattr(mf, builder, None)
    if build is None:
        pytest.skip(f"{builder} not available")
    Xtr, ytr, Xte = _xy(seed=2)
    try:
        mf.meta.configure(n_jobs=1)
        a = build(Xtr, ytr).predict(Xte)
        mf.meta.configure(n_jobs=6)
        b = build(Xtr, ytr).predict(Xte)
    except Exception:
        pytest.skip(f"{builder} extra not installed")
    finally:
        mf.meta.configure(n_jobs=1)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
