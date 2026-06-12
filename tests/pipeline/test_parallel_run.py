"""Native parallel fan-out: n_jobs>1 is numerically identical to n_jobs=1.

The parallel path splits the (arm x target x horizon) work units across a process
pool. Each unit is deterministic in ``spec.seed`` and independent of sibling units,
so the concatenated master frame -- and every downstream accuracy table -- must
match the sequential run row-for-row. The window is kept tiny so the test runs in
well under a minute and does not starve the long-running replication processes.
"""
import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, pipeline_spec, run_pipeline


def _bundle(n: int = 84) -> object:
    """A small two-target panel with a couple of predictors."""
    idx = pd.date_range("2000-01-31", periods=n, freq="ME", name="date")
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x + rng.standard_normal(n) * 0.05,
            "y2": 0.5 - 1.5 * x + rng.standard_normal(n) * 0.05,
            "x1": x,
            "x2": np.cos(np.linspace(0.0, 6.0, n)),
        },
        index=idx,
    )
    return mf.data.custom_dataset(
        frame, transform_codes={"y1": 1, "y2": 1, "x1": 1, "x2": 1}
    )


def _spec(n_jobs: int):
    """Two targets, three arms (incl. benchmark AR), horizons [1, 3], short window."""
    feats = mf.feature_engineering.feature_spec(
        target="y1", predictors=["x1", "x2"], lags=1, target_lags=(0, 1)
    )
    w = mf.window.spec(
        estimation=mf.window.estimation_expanding(min_size=36),
        val=mf.window.val_last_block(size=12),
        test=mf.window.test_origins(horizon=1, step=6),
    )
    return pipeline_spec(
        data=_bundle(),
        targets=["y1", "y2"],
        horizons=[1, 3],
        window=w,
        arms=[
            Arm("AR", model="ar", features=feats),
            Arm("OLS", model="ols", features=feats),
            Arm("RIDGE", model="ridge", features=feats),
        ],
        evaluation=EvalSpec(benchmark="AR"),
        save_models=False,
        n_jobs=n_jobs,
    )


_SORT_KEYS = ["target", "contender", "horizon", "origin"]


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in _SORT_KEYS if k in frame.columns]
    return (
        frame.sort_values(keys, kind="mergesort")
        .reset_index(drop=True)
        .reindex(sorted(frame.columns), axis=1)
    )


def test_parallel_forecasts_identical_to_sequential():
    seq = run_pipeline(_spec(n_jobs=1))
    par = run_pipeline(_spec(n_jobs=4))

    fs = _sorted(seq.forecasts)
    fp = _sorted(par.forecasts)

    # same rows
    assert len(fs) == len(fp)
    assert list(fs.columns) == list(fp.columns)

    # predictions numerically identical (zero tolerance)
    pd.testing.assert_series_equal(
        fs["prediction"].reset_index(drop=True),
        fp["prediction"].reset_index(drop=True),
        check_names=False,
    )
    # the full forecast frame matches (predictions, actuals, labels, metadata)
    pd.testing.assert_frame_equal(fs, fp, check_like=True)


def test_parallel_accuracy_tables_match():
    seq = run_pipeline(_spec(n_jobs=1))
    par = run_pipeline(_spec(n_jobs=4))

    sort_keys = ["target", "horizon", "contender"]

    def _norm(acc: pd.DataFrame) -> pd.DataFrame:
        keys = [k for k in sort_keys if k in acc.columns]
        return (
            acc.sort_values(keys, kind="mergesort")
            .reset_index(drop=True)
            .reindex(sorted(acc.columns), axis=1)
        )

    pd.testing.assert_frame_equal(_norm(seq.accuracy), _norm(par.accuracy))
