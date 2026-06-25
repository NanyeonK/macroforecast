"""At horizon 1, direct and path-average must give identical forecasts.

The forecast object is the same at h1 by construction (path-average over 1 step
IS the direct 1-step forecast). A plain linear model (ols) always satisfied this.
The information-criterion models (far, ar) did NOT: the direct path selected
their AR order by BIC/AIC on the full training sample, but the path-average
per-step block lacked the IC branch and instead ran CV/validation-split
selection on a truncated sample (the validation block held out), so it picked a
different order and diverged even at h1. The fix gives the path block the same
IC branch. This regression test pins the invariant.
"""
import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


def _data():
    idx = pd.date_range("1980-01-01", periods=320, freq="MS")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"x{i}": rng.normal(size=320) for i in range(5)}, index=idx)
    panel["Y"] = rng.normal(size=320) * 0.01  # stationary growth-like target
    panel.index.name = "date"
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _forecasts(policy, model):
    bundle = _data()
    win = mf.window.from_cutoffs(
        test_start="2005-01-01", test_end="2010-12-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 4),
        target_lags=range(0, 13), target_transform="value",
    )
    params = {"n_factors": 3, "n_lag": 12} if model == "far" else {}
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec(name="Y", transform="value", policy=policy)],
        horizons=[1], window=win,
        arms=[Arm(name="M", model=model, features=feats, is_benchmark=True, params=params)],
        evaluation=EvalSpec(benchmark="M", metrics=("rmse",)), n_jobs=1,
    )
    return run_pipeline(spec).forecasts


@pytest.mark.parametrize("model", ["ols", "far", "ar"])
def test_h1_direct_equals_path(model):
    d = _forecasts("direct_average", model).dropna(subset=["prediction"]).set_index("origin")["prediction"]
    p = _forecasts("path_average", model).dropna(subset=["prediction"]).set_index("origin")["prediction"]
    common = d.index.intersection(p.index)
    assert len(common) > 0
    max_abs = float(np.abs(d.loc[common] - p.loc[common]).max())
    assert max_abs < 1e-9, f"{model}: h1 direct != path (max abs diff {max_abs:.2e})"
