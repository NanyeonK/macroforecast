"""LPT (longest-processing-time-first) cell scheduling.

The parallel backend should dispatch the heaviest cells first so a heavy cell is
never stranded alone at the tail. Reordering the dispatch must NOT change the
master forecast frame (cells are independent; reassembly is canonical).
"""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline
from macroforecast.pipeline.run import _cell_cost, _enumerate_cells, _lpt_dispatch_order


def _toy_spec(n_jobs, horizons=(1, 3), policy="direct"):
    idx = pd.date_range("1990-01-01", periods=160, freq="MS")
    rng = np.random.default_rng(0)
    cols = {f"X{i}": rng.normal(size=160) for i in range(6)}
    cols["Y"] = np.cumsum(rng.normal(size=160))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    window = mf.window.from_cutoffs(
        test_start="2002-01-01", test_end="2004-12-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(target="Y", predictors="all", lags=range(1, 3))
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y", policy=policy)],
        horizons=list(horizons),
        window=window,
        arms=[
            Arm(name="AR", model="ar", is_benchmark=True),
            Arm(name="RF", model="random_forest", features=feats,
                params={"n_estimators": 20}),
        ],
        evaluation=EvalSpec(benchmark="AR", metrics=("rmse",)),
        n_jobs=n_jobs,
    )


def test_cell_cost_orders_heavy_first():
    spec = _toy_spec(n_jobs=2, horizons=(1, 3, 24), policy="path_average")
    cells = _enumerate_cells(spec)
    cost = {c: _cell_cost(spec, c) for c in cells}

    def find(arm_name, h):
        a_idx = next(i for i, a in enumerate(spec.arms) if a.name == arm_name)
        return next(c for c in cells if c.arm_idx == a_idx and max(c.horizons) == h)

    # A heavy tree model at the longest horizon must outrank a cheap AR at h=1.
    assert cost[find("RF", 24)] > cost[find("AR", 1)]
    # Within the same model, a longer horizon is heavier.
    assert cost[find("RF", 24)] > cost[find("RF", 1)]
    # The LPT order starts with the single most expensive cell.
    order = _lpt_dispatch_order(spec, cells)
    assert order[0] == max(cells, key=lambda c: cost[c])


def test_lpt_preserves_canonical_reassembly_order():
    spec = _toy_spec(n_jobs=2, horizons=(1, 3, 24), policy="path_average")
    cells = _enumerate_cells(spec)
    # Dispatch order is cost-sorted, but the canonical cell list is unchanged, so
    # reassembly (which iterates the canonical list) is unaffected.
    order = _lpt_dispatch_order(spec, cells)
    assert sorted(order) == sorted(cells)
    assert order != cells  # actually reordered for this heterogeneous spec


def test_parallel_matches_serial_after_lpt():
    serial = run_pipeline(_toy_spec(n_jobs=1)).forecasts
    parallel = run_pipeline(_toy_spec(n_jobs=2)).forecasts
    key = ["arm", "horizon", "date"]
    a = serial.sort_values(key).reset_index(drop=True)
    b = parallel.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[key + ["prediction"]], b[key + ["prediction"]],
                                  atol=1e-10, check_dtype=False)
