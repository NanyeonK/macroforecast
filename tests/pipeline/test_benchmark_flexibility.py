"""The evaluation benchmark can be any arm, including a same-model feature
variant or a user-defined custom model.

Covers the design where a base model is the benchmark and enhanced variants are
the contenders (e.g. a plain random forest vs random forests with extra feature
blocks), and confirms custom models work both as a contender and as the
benchmark.
"""
import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


def _bundle(n=120):
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(3)}, index=idx)
    panel["Y"] = 0.4 * panel["x0"] + rng.normal(size=n) * 0.5
    panel.index.name = "date"
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _feats(predictors):
    return mf.feature_engineering.feature_spec(
        target="Y", predictors=predictors, lags=range(0, 2),
        target_lags=range(0, 2), target_transform="value",
    )


def _window():
    return mf.window.from_cutoffs(
        test_start="2009-01-01", test_end="2009-03-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )


class _Fit:
    def __init__(self, est, cols):
        self.est, self.cols = est, cols

    def predict(self, X):
        Xc = X.reindex(columns=self.cols, fill_value=0.0).fillna(0.0)
        return self.est.predict(Xc.values)


def _ridge_fit(X, y, *, alpha=1.0):
    from sklearn.linear_model import Ridge

    Xc = X.fillna(0.0)
    est = Ridge(alpha=alpha).fit(Xc.values, np.asarray(y, dtype=float))
    return _Fit(est, list(Xc.columns))


def test_same_model_variants_with_base_as_benchmark():
    rf = {"n_estimators": 10, "max_features": 1 / 3, "min_samples_leaf": 5,
          "random_state": 0, "n_jobs": 1}
    spec = pipeline_spec(
        data=_bundle(), targets=[TargetSpec("Y", transform="value", policy="direct")],
        horizons=[1], window=_window(),
        arms=[
            Arm("RF_base", model="random_forest", features=_feats(["x0"]), params=rf, is_benchmark=True),
            Arm("RF_more", model="random_forest", features=_feats(["x0", "x1", "x2"]), params=rf),
        ],
        evaluation=EvalSpec(benchmark="RF_base", metrics=("rmse", "relative_mse", "r2_oos")),
        n_jobs=1,
    )
    acc = run_pipeline(spec).accuracy.dropna(subset=["relative_mse"])
    base = acc[acc.contender == "RF_base"].iloc[0]
    more = acc[acc.contender == "RF_more"].iloc[0]
    assert bool(base["is_benchmark"]) and abs(base["relative_mse"] - 1.0) < 1e-9
    assert not bool(more["is_benchmark"]) and np.isfinite(more["relative_mse"])


def test_custom_model_as_contender_and_benchmark():
    from types import SimpleNamespace

    from macroforecast.pipeline.evaluate import accuracy_table

    custom = mf.models.custom_model("my_ridge", _ridge_fit, default_params={"alpha": 0.5})
    spec = pipeline_spec(
        data=_bundle(), targets=[TargetSpec("Y", transform="value", policy="direct")],
        horizons=[1], window=_window(),
        arms=[
            Arm("my_ridge", model=custom, features=_feats(["x0", "x1"]), is_benchmark=True),
            Arm("OLS", model="ols", features=_feats(["x0", "x1"])),
        ],
        evaluation=EvalSpec(benchmark="my_ridge", metrics=("rmse", "relative_mse")),
        n_jobs=1,
    )
    report = run_pipeline(spec)
    assert "my_ridge" in set(report.forecasts["arm"])  # custom model ran as an arm

    # custom model as the benchmark
    acc = report.accuracy.dropna(subset=["relative_mse"])
    bench = acc[acc.contender == "my_ridge"].iloc[0]
    assert bool(bench["is_benchmark"]) and abs(bench["relative_mse"] - 1.0) < 1e-9
    assert np.isfinite(acc[acc.contender == "OLS"].iloc[0]["relative_mse"])

    # and the same forecasts re-scored against the OLS arm instead
    reb = accuracy_table(report.forecasts, SimpleNamespace(evaluation=SimpleNamespace(benchmark="OLS")))
    reb = reb.dropna(subset=["relative_mse"])
    assert abs(reb[reb.contender == "OLS"].iloc[0]["relative_mse"] - 1.0) < 1e-9
