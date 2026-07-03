"""Independent-reference oracle for the factor model (far/FM) THROUGH a policy.

Prior coverage anchored the far/FM factor family only at the model-callable level
(the scaled_pca reimplementations in test_models.py); no test checked a far forecast
produced by the runner under a forecast policy. These tests supply that anchor with a
ground-truth DGP.

Construction: a 4-dimensional latent state evolves by an exact linear map built from
two modulus-1 rotation blocks, so the state is bounded (non-decaying) and, crucially,
``x_{t+h} = A^h x_t`` EXACTLY. Eight predictors load on the state (rank 4) and the
target is a linear combination of the state, so the h-ahead target is an exact linear
function of the ORIGIN-time state -- perfectly recoverable from the predictor factors
with no noise. A correct far (n_factors=4) must therefore forecast the realised future
value to machine precision, under BOTH the direct and the path-average policy. An AR
given only two target lags cannot span the 4-D state and fails badly, which pins that
far genuinely uses the predictor factors rather than collapsing to an autoregression.
"""
import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

_N = 360
_A1, _A2 = np.pi / 7, np.pi / 5


def _rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _factor_dgp():
    """Return (bundle, target_series) for the noiseless rank-4 linear-state DGP."""
    A = np.zeros((4, 4))
    A[:2, :2] = _rotation(_A1)
    A[2:, 2:] = _rotation(_A2)
    state = np.zeros((_N, 4))
    state[0] = [1.0, 0.3, 0.5, -0.2]
    for t in range(1, _N):
        state[t] = A @ state[t - 1]
    rng = np.random.default_rng(1)
    loadings = rng.normal(size=(8, 4))
    predictors = state @ loadings.T                    # 8 series, exactly rank 4
    target = state @ np.array([0.7, -0.4, 0.9, 0.5])   # linear in the state
    idx = pd.date_range("1980-01-01", periods=_N, freq="MS")
    cols = {f"S{i}": predictors[:, i] for i in range(8)}
    cols["Y"] = target
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    return bundle, pd.Series(target, index=idx)


def _forecasts(bundle, *, policy, model, params):
    win = mf.window.from_cutoffs(
        test_start="2004-01-01", test_end="2005-06-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 2),
        target_lags=range(0, 2), target_transform="value",
    )
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec(name="Y", transform="value", policy=policy)],
        horizons=[2, 3], window=win,
        arms=[Arm(name="M", model=model, features=feats, is_benchmark=True, params=params)],
        evaluation=EvalSpec(benchmark="M", metrics=("rmse",)), n_jobs=1,
    )
    return run_pipeline(spec).forecasts


def _abs_errors_vs_truth(forecasts, series, horizon):
    f = forecasts[forecasts["horizon"] == horizon].dropna(subset=["prediction"])
    errs = []
    for _, r in f.iterrows():
        pos = series.index.get_loc(r["origin"])
        truth = float(series.iloc[pos + 1: pos + 1 + horizon].mean())
        errs.append(abs(float(r["prediction"]) - truth))
    return errs


def test_far_direct_recovers_factor_driven_future_exactly():
    """far under the direct policy forecasts the realised future mean to machine
    precision when the target is exactly factor-driven."""
    bundle, series = _factor_dgp()
    fc = _forecasts(bundle, policy="direct_average", model="far",
                    params={"n_factors": 4, "n_lag": 1})
    for h in (2, 3):
        errs = _abs_errors_vs_truth(fc, series, h)
        assert len(errs) > 3
        assert max(errs) < 1e-6, f"far direct h={h}: max err {max(errs):.2e}"


def test_far_path_average_recovers_factor_driven_future_exactly():
    """far under the PATH-AVERAGE policy must also be exact at h>=2. This fails if the
    per-step fit falls back to the iterated estimator (direct=False) instead of the
    direct s-step projection -- the bug this oracle was written to catch."""
    bundle, series = _factor_dgp()
    fc = _forecasts(bundle, policy="path_average", model="far",
                    params={"n_factors": 4, "n_lag": 1})
    for h in (2, 3):
        errs = _abs_errors_vs_truth(fc, series, h)
        assert len(errs) > 3
        assert max(errs) < 1e-6, f"far path_average h={h}: max err {max(errs):.2e}"


def test_far_uses_predictor_factors_not_just_autoregression():
    """far (predictor factors) must beat AR (target lags only) by a wide margin: the
    target is driven by a 4-D state that two target lags cannot span, so AR fails while
    far is exact. Guards against far silently collapsing to an autoregression."""
    bundle, series = _factor_dgp()
    for policy in ("direct_average", "path_average"):
        far_err = max(_abs_errors_vs_truth(
            _forecasts(bundle, policy=policy, model="far", params={"n_factors": 4, "n_lag": 1}),
            series, 2))
        ar_err = max(_abs_errors_vs_truth(
            _forecasts(bundle, policy=policy, model="ar", params={"n_lag": 1}),
            series, 2))
        assert far_err < 1e-6 < ar_err, (
            f"{policy}: far_err {far_err:.2e} should be tiny and ar_err {ar_err:.2e} large"
        )
        assert ar_err > 100 * far_err
