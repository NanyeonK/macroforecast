"""Independent-reference oracle for MULTI-STEP (h>=2) path-average forecasting.

The existing path-average coverage only ever pinned h=1, where path-average is
definitionally the direct 1-step forecast, so genuine multi-step averaging was
untested against any oracle. These tests supply that oracle two ways:

1. GROUND TRUTH (noiseless): on an exactly-linear AR(2) process the per-step direct
   OLS recovers the true linear map, so every s-step-ahead forecast equals the
   realised future value. A correct ``path_average`` at horizon h must therefore
   equal the mean of the realised future one-period path -- which we compute by hand
   from the raw series, independently of the runner. This discriminates against a
   wrong step alignment, averaging the wrong number of steps, an off-by-one in the
   forecast-target date, or the path silently collapsing to a single step.

2. CLEAN-ROOM REPRODUCTION (noisy): on a controlled noisy series we re-fit each
   per-step direct OLS with plain numpy (never calling the runner's path code) and
   average the step forecasts, then require the runner's ``path_average`` number to
   match, per origin, at h=2 and h=3.

A discrimination test additionally requires path-average != direct-average at h>=2,
so a future regression that collapses the h-step path into a single regression
cannot pass silently.
"""
import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


# ---- data generators -------------------------------------------------------

def _ar2_series(n, phi1, phi2, y0, y1, noise_sd=0.0, seed=0):
    """A length-n AR(2): Y_t = phi1*Y_{t-1} + phi2*Y_{t-2} (+ optional noise)."""
    rng = np.random.default_rng(seed)
    y = np.empty(n, dtype=float)
    y[0], y[1] = y0, y1
    eps = rng.normal(scale=noise_sd, size=n) if noise_sd else np.zeros(n)
    for t in range(2, n):
        y[t] = phi1 * y[t - 1] + phi2 * y[t - 2] + eps[t]
    return y


def _bundle(values):
    idx = pd.date_range("1980-01-01", periods=len(values), freq="MS")
    panel = pd.DataFrame({"Y": values}, index=idx)
    panel.index.name = "date"
    # transform_code 1 == no transformation (level), so Y is used as-is.
    return mf.data.custom_dataset(panel, transform_codes={"Y": 1})


def _spec(values, *, policy, horizons, n_target_lags=2):
    bundle = _bundle(values)
    win = mf.window.from_cutoffs(
        test_start="2004-01-01", test_end="2007-12-01", mode="expanding",
        val_method="last_block", retrain_every=1,
    )
    # OLS on the target's own lags only (no exogenous predictors): X = Y_lag0..lag{k-1}.
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors=[], lags=None,
        target_lags=range(0, n_target_lags), target_transform="value",
    )
    spec = pipeline_spec(
        data=bundle, targets=[TargetSpec(name="Y", transform="value", policy=policy)],
        horizons=list(horizons), window=win,
        arms=[Arm(name="M", model="ols", features=feats, is_benchmark=True)],
        evaluation=EvalSpec(benchmark="M", metrics=("rmse",)), n_jobs=1,
    )
    return run_pipeline(spec).forecasts


def _pred(forecasts, horizon):
    f = forecasts[forecasts["horizon"] == horizon].dropna(subset=["prediction"])
    return f.set_index("origin")["prediction"].sort_index()


def _actual(forecasts, horizon):
    f = forecasts[forecasts["horizon"] == horizon].dropna(subset=["actual"])
    return f.set_index("origin")["actual"].sort_index()


# ---- 1. ground-truth oracle (noiseless) ------------------------------------

def test_path_average_matches_realised_future_mean_on_noiseless_ar2():
    """Noiseless AR(2): path-average h>=2 forecast == mean of realised future path.

    With no noise the per-step direct OLS is an exact fit, so each step forecast
    equals the true future value and their average is the realised future one-period
    mean. We recompute that realised mean straight from the raw series (by hand) and
    require the runner's forecast to match it -- a real, external oracle.
    """
    # 2*cos(pi/6) ~ 1.7320508: complex roots on the unit circle -> bounded,
    # non-decaying oscillation, so values stay O(1) and tolerances are meaningful.
    values = _ar2_series(360, phi1=2 * np.cos(np.pi / 6), phi2=-1.0, y0=1.0, y1=0.7)
    series = pd.Series(values, index=pd.date_range("1980-01-01", periods=len(values), freq="MS"))

    for h in (2, 3):
        fc = _spec(values, policy="path_average", horizons=[h])
        pred = _pred(fc, h)
        assert len(pred) > 3, f"expected several origins at h={h}"
        # Hand-computed oracle: for origin date o the h-step path is Y_{o+1..o+h};
        # its mean is what a perfect per-step forecaster must predict.
        for origin in pred.index:
            pos = series.index.get_loc(origin)
            future = series.iloc[pos + 1: pos + 1 + h]
            assert len(future) == h
            oracle = float(future.mean())
            got = float(pred.loc[origin])
            assert abs(got - oracle) < 1e-6, (
                f"h={h} origin={origin}: path-average {got:.6g} != realised "
                f"future mean {oracle:.6g}"
            )


def test_direct_average_also_matches_realised_future_mean_on_noiseless_ar2():
    """Companion: direct-average (single regression of the pre-averaged target) must
    hit the same ground truth at h>=2, so the two policies agree on the truth even
    though they are different estimators (guards the h1 invariant's extension)."""
    values = _ar2_series(360, phi1=2 * np.cos(np.pi / 6), phi2=-1.0, y0=1.0, y1=0.7)
    series = pd.Series(values, index=pd.date_range("1980-01-01", periods=len(values), freq="MS"))
    for h in (2, 3):
        fc = _spec(values, policy="direct_average", horizons=[h])
        pred = _pred(fc, h)
        assert len(pred) > 3
        for origin in pred.index:
            pos = series.index.get_loc(origin)
            oracle = float(series.iloc[pos + 1: pos + 1 + h].mean())
            assert abs(float(pred.loc[origin]) - oracle) < 1e-6


# ---- 2. clean-room reproduction of the exact per-step OLS (noisy) ----------

def test_path_average_reproduces_per_step_ols_on_noisy_data():
    """The runner's path-average number == an independent per-step OLS, byte-close.

    With ``target_lags=range(0,2)`` and no exogenous predictors the design at origin
    position ``o`` is ``[1, Y_o, Y_{o-1}]``. For each step ``s`` the leak-free direct
    regression fits ``Y_{t+s} ~ [1, Y_t, Y_{t-1}]`` over the training rows whose
    target is observable at the origin (``t + s <= o``); the h-step forecast averages
    the ``s = 1..h`` step forecasts. We recompute this entirely in numpy -- never
    touching the runner's path code -- and require an exact match per origin. This
    pins the estimator, the leak-free training window, AND the averaging at once.
    """
    values = _ar2_series(360, phi1=0.5, phi2=0.3, y0=0.2, y1=-0.1, noise_sd=1.0, seed=7)
    y = np.asarray(values, dtype=float)
    idx = pd.date_range("1980-01-01", periods=len(values), freq="MS")
    pos_of = {d: i for i, d in enumerate(idx)}

    def _reference(origin_pos, h):
        step_preds = []
        for s in range(1, h + 1):
            # training rows t: need Y_{t-1} (so t >= 1) and Y_{t+s} observed at the
            # origin (t + s <= origin_pos), i.e. t in [1, origin_pos - s].
            rows = range(1, origin_pos - s + 1)
            X = np.array([[1.0, y[t], y[t - 1]] for t in rows])
            target = np.array([y[t + s] for t in rows])
            beta, *_ = np.linalg.lstsq(X, target, rcond=None)
            x_origin = np.array([1.0, y[origin_pos], y[origin_pos - 1]])
            step_preds.append(float(x_origin @ beta))
        return float(np.mean(step_preds))

    for h in (2, 3):
        pred = _pred(_spec(values, policy="path_average", horizons=[h]), h)
        assert len(pred) > 3
        for origin in pred.index:
            ref = _reference(pos_of[origin], h)
            got = float(pred.loc[origin])
            assert abs(got - ref) < 1e-9, (
                f"h={h} origin={origin}: runner {got:.10g} != clean-room per-step "
                f"OLS {ref:.10g} (diff {got - ref:.2e})"
            )


# ---- 3. discrimination: path != direct at h>=2 on noisy data ---------------

def test_path_average_differs_from_direct_average_at_multistep():
    """On noisy data the two multi-step constructions must genuinely differ at h>=2.

    path-average = mean of h separate per-step direct regressions; direct-average =
    one regression of the pre-averaged target. They coincide only at h=1. If a
    regression made path collapse to direct (or to h=1), this margin would vanish.
    """
    values = _ar2_series(360, phi1=0.5, phi2=0.3, y0=0.2, y1=-0.1, noise_sd=1.0, seed=7)
    for h in (2, 3):
        p = _pred(_spec(values, policy="path_average", horizons=[h]), h)
        d = _pred(_spec(values, policy="direct_average", horizons=[h]), h)
        common = p.index.intersection(d.index)
        assert len(common) > 3
        max_abs = float(np.abs(p.loc[common] - d.loc[common]).max())
        assert max_abs > 1e-4, (
            f"h={h}: path-average and direct-average are identical (max diff "
            f"{max_abs:.2e}) -- the multi-step path collapsed"
        )
