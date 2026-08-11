"""The serial preprocessing cache holds one target at a time (#452).

Two things have to be true at once, and the second is what makes this a fix rather
than a regression:

1. A finished target's entries are released. Building a dict per target up front
   kept every finished target's prepared panels -- a transformed panel per
   (origin x horizon), plus each FittedPreprocessor's own fit panel -- reachable
   until the run ended, so peak memory scaled with the number of targets instead of
   with the largest one.

2. **No cache hit is lost.** Within a target the working set is the whole cache:
   the first arm fills every entry and every later arm reads all of them, so the
   reuse distance is a full pass over the origins. Any eviction narrower than the
   target boundary would evict each entry exactly before its reuse. The second test
   therefore counts the *work* -- adding arms must not add preprocessing fits.

The first test observes the dict objects the runner actually hands out, and asserts
the earlier target's dict is EMPTY once a later target starts. Asserting instead
that a new target begins with an empty cache would pass on the unfixed code, which
also gave each target its own (initially empty) dict -- the bug was never that the
dict was shared, it was that the finished one stayed reachable.
"""
from __future__ import annotations

import dataclasses as _dc

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

N_OBS = 90
N_TEST = 12


@pytest.fixture(scope="module")
def bundle_and_index():
    idx = pd.date_range("1990-01-31", periods=N_OBS, freq="ME", name="date")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"x{i}": rng.normal(size=N_OBS) for i in range(4)}, index=idx)
    panel["y1"] = 0.5 * panel["x0"] + rng.normal(size=N_OBS) * 0.3
    panel["y2"] = 0.4 * panel["x1"] + rng.normal(size=N_OBS) * 0.3
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    return idx, bundle


def _arms(n_ridge: int) -> list[Arm]:
    features = mf.feature_engineering.feature_spec(
        predictors=[f"x{i}" for i in range(4)], lags=0, target_lags=2
    )
    arms = [Arm("AR", model="ar", is_benchmark=True)]
    arms += [
        Arm(f"RIDGE{i}", model="ridge", features=features, params={"alpha": 0.1 * (i + 1)})
        for i in range(n_ridge)
    ]
    return arms


def _spec(idx, bundle, *, arms, horizons=(1, 3)):
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec("y1", transform="level"), TargetSpec("y2", transform="level")],
        horizons=list(horizons),
        window=mf.window.from_cutoffs(
            test_start=idx[N_OBS - N_TEST],
            horizon=max(horizons),
            embargo=0,
            val_method="expanding",
            val_min_train_size=20,
        ),
        arms=arms,
        preprocessing=mf.preprocessing.preprocess_spec(standardize="zscore"),
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse",), tests=()),
        save_models=False,
        n_jobs=1,
        seed=0,
    )


def test_a_finished_target_holds_nothing(bundle_and_index, monkeypatch):
    idx, bundle = bundle_and_index
    import macroforecast.pipeline.run as run_mod

    original = run_mod._execute_cell
    # (target name, the cache object handed to that cell), in visit order.
    handed: list[tuple[str, dict]] = []

    # Size is recorded when the cell FINISHES, i.e. while the cache is still live.
    # Reading it after the run would always see zero once the fix clears it, so the
    # "did sharing happen at all" guard has to be sampled in flight.
    sizes: list[tuple[str, int]] = []

    # ``**kwargs`` forwards whatever else the caller threads through (the cell's
    # pre-resolved tasks, today) so this probe measures the cache and nothing else:
    # spelling the full signature out here would make an unrelated argument added to
    # ``_execute_cell`` fail this test as a per-cell TypeError.
    def recording_execute(spec, cell, *, preprocessing_cache=None, **kwargs):
        name = spec.targets[cell.target_idx].name
        if preprocessing_cache is not None:
            handed.append((name, preprocessing_cache))
        out = original(spec, cell, preprocessing_cache=preprocessing_cache, **kwargs)
        if preprocessing_cache is not None:
            sizes.append((name, len(preprocessing_cache)))
        return out

    monkeypatch.setattr(run_mod, "_execute_cell", recording_execute)
    run_pipeline(_spec(idx, bundle, arms=_arms(2)))

    assert handed, "no cell received a cache; the probe measured nothing"
    names = [name for name, _ in handed]
    assert names[0] == "y1" and "y2" in names, f"unexpected visit order: {names}"

    y1_caches = [c for name, c in handed if name == "y1"]
    assert max((n for name, n in sizes if name == "y1"), default=0) > 0, (
        "no y1 cell ever populated its cache while running; the sharing this cache "
        "exists for is not happening, so the release assertion below is vacuous"
    )
    # After the run, every dict the first target was given must be empty.
    leftover = sum(len(c) for c in y1_caches)
    assert leftover == 0, (
        f"the first target's cache still holds {leftover} entries after the run "
        "moved on to the second target"
    )


def test_more_arms_do_not_cost_more_preprocessing_fits(bundle_and_index, monkeypatch):
    """The assertion a capacity-bounded LRU would fail."""
    idx, bundle = bundle_and_index
    original = mf.preprocessing.PreprocessSpec.fit

    def counting(self, *args, **kwargs):
        counting.n += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(mf.preprocessing.PreprocessSpec, "fit", counting)

    counting.n = 0
    run_pipeline(_spec(idx, bundle, arms=_arms(1)))
    two_arms = counting.n

    counting.n = 0
    run_pipeline(_spec(idx, bundle, arms=_arms(4)))
    five_arms = counting.n

    assert two_arms > 0, "no preprocessing happened; the probe measured nothing"
    assert five_arms == two_arms, (
        f"three extra arms cost {five_arms - two_arms} extra preprocessing fits; "
        "the per-origin cache is no longer shared across arms"
    )


def test_results_are_unchanged(bundle_and_index):
    """Scoping is a memory change, not a numerical one.

    Serial (shared cache, target-scoped) against parallel (no in-memory cache at
    all) is the strongest available control: had scoping dropped a still-live entry,
    the cached path would diverge from the path that never caches.
    """
    idx, bundle = bundle_and_index
    spec = _spec(idx, bundle, arms=_arms(2))
    serial = run_pipeline(spec).forecasts
    parallel = run_pipeline(_dc.replace(spec, n_jobs=2)).forecasts

    key = [c for c in ("contender", "target", "horizon", "date") if c in serial.columns]
    s = serial.sort_values(key).reset_index(drop=True)
    p = parallel.sort_values(key).reset_index(drop=True)
    np.testing.assert_allclose(
        s["prediction"].to_numpy(float),
        p["prediction"].to_numpy(float),
        rtol=0,
        atol=1e-12,
        err_msg="target-scoped caching changed a forecast",
    )


@pytest.mark.parametrize("n_jobs,result_store", [(1, None), (1, "set"), (2, None)])
def test_each_targets_cells_are_contiguous(bundle_and_index, tmp_path, n_jobs, result_store):
    """The invariant the target-scoped cache rests on.

    Dropping a target's cache when the loop reaches a different target is only
    sound if each target's cells form ONE contiguous block -- otherwise returning
    to an earlier target would find its entries gone and recompute them silently,
    which costs time and shows up nowhere. ``_enumerate_cells`` builds a nested
    loop with target outermost, and the two things that reshape it (``n_jobs > 1``
    and ``result_store``, which both split cells by horizon) reshape it WITHIN a
    target. This pins that, so a future change to the visit order fails here rather
    than turning into an unexplained slowdown.
    """
    from macroforecast.pipeline.run import _enumerate_cells

    idx, bundle = bundle_and_index
    spec = _spec(idx, bundle, arms=_arms(2))
    spec = _dc.replace(
        spec,
        n_jobs=n_jobs,
        result_store=str(tmp_path / "store") if result_store else None,
    )
    order = [cell.target_idx for cell in _enumerate_cells(spec)]

    assert order, "no cells enumerated"
    blocks = [k for i, k in enumerate(order) if i == 0 or order[i - 1] != k]
    assert len(blocks) == len(set(blocks)), (
        f"a target is visited in more than one block: {order}"
    )
