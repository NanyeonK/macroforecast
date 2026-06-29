"""Golden pin: serial and parallel pipeline backends must agree numerically.

This test guards an upcoming optimization (Part B) that introduces a shared,
content-addressed on-disk preprocessing cache. In the parallel backend each
worker process currently recomputes its own per-(spec, target, origin)
``FittedPreprocessor`` with no shared store. The optimization will compute each
such preprocessor once and reuse it across processes. This test pins that the
two backends already produce identical forecasts today, so the optimization can
be proven to preserve numerical identity rather than introduce drift.

The fixture uses a small, deterministic, stateful preprocessing chain
(mean imputation + zscore standardization) so that the serial/parallel sharing
of fitted preprocessing state is exercised, while staying cheap enough to run in
CI. (Note: ``impute="median"`` is not a valid imputation method in
``preprocess_spec`` -- the supported deterministic methods are ``mean``,
``forward_fill``, and ``none`` -- so ``mean`` is used here.)
"""

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)


def _toy(n_jobs: int):
    """Build a 2-arm, 2-horizon toy pipeline spec parameterized by n_jobs."""
    idx = pd.date_range("1990-01-01", periods=160, freq="MS")
    rng = np.random.default_rng(1)
    cols = {f"S{i}": rng.normal(size=160) for i in range(6)}
    cols["Y"] = np.cumsum(rng.normal(size=160))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    # Cheap, deterministic, stateful preprocessing: mean impute + zscore.
    prep = mf.preprocessing.preprocess_spec(
        transform="official", impute="mean", standardize="zscore"
    )
    win = mf.window.from_cutoffs(
        test_start="2002-01-01",
        test_end="2005-12-01",
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 4)
    )
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1, 3],
        window=win,
        arms=[
            Arm(
                name="RIDGE",
                model="ridge",
                preprocessing=prep,
                features=feats,
                is_benchmark=True,
            ),
            Arm(
                name="LASSO",
                model="lasso",
                preprocessing=prep,
                features=feats,
            ),
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        n_jobs=n_jobs,
    )


def test_pipeline_golden_serial_equals_parallel():
    a = (
        run_pipeline(_toy(1))
        .forecasts.sort_values(["arm", "horizon", "date"])
        .reset_index(drop=True)
    )
    b = (
        run_pipeline(_toy(2))
        .forecasts.sort_values(["arm", "horizon", "date"])
        .reset_index(drop=True)
    )
    # Guard against a silently empty run masking a no-op comparison.
    assert not a.empty
    assert not b.empty
    pd.testing.assert_frame_equal(
        a[["arm", "horizon", "date", "prediction"]],
        b[["arm", "horizon", "date", "prediction"]],
        atol=1e-10,
    )
