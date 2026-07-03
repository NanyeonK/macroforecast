"""Independent clean-room reference for ``adaptive_lasso``.

Before this, adaptive_lasso was the only regularised model with NO fit/predict
correctness test -- coverage checked its backend-string metadata but never a number it
produced. These tests reimplement the adaptive-lasso estimator from its definition
(standardise, ridge-initialised adaptive weights w_j = 1/|beta_init_j|^gamma with
mean-one normalisation, a weighted lasso, then map the coefficients back) using plain
numpy + sklearn, and require the package predictions to match. A separate test pins the
defining behaviour -- oracle-property support recovery -- on a sparse DGP.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge

import macroforecast as mf
from macroforecast.models import adaptive_lasso
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


def _design(n=200, p=8, seed=0, sparse_beta=None, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p) if sparse_beta is None else np.asarray(sparse_beta, dtype=float)
    y = X @ beta + noise * rng.normal(size=n)
    cols = [f"x{i}" for i in range(p)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def _reference_adaptive_lasso(X, y, *, alpha, gamma, initial_alpha, eps, max_iter, tol):
    """Reproduce _AdaptiveLinear.fit for kind='lasso' with normalize_weights=True."""
    values = X.astype(float).to_numpy()
    yv = y.astype(float).to_numpy()
    x_mean = values.mean(axis=0)
    x_scale = values.std(axis=0, ddof=1)
    x_scale = np.where(np.abs(x_scale) < 1e-12, 1.0, x_scale)  # _safe_array_scale floor
    y_mean = float(yv.mean())
    x_scaled = (values - x_mean) / x_scale
    y_centered = yv - y_mean

    init = Ridge(alpha=initial_alpha, fit_intercept=False).fit(x_scaled, y_centered)
    init_coef = np.asarray(init.coef_, dtype=float).reshape(-1)
    weights = 1.0 / np.power(np.abs(init_coef) + eps, gamma)
    weights = weights * (len(weights) / float(np.sum(weights)))  # mean-one normalisation

    weighted_x = x_scaled / weights
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=max_iter, tol=tol,
                  random_state=None).fit(weighted_x, y_centered)
    weighted_coef = np.asarray(lasso.coef_, dtype=float).reshape(-1)
    coef = (weighted_coef / weights) / x_scale
    intercept = y_mean - float(x_mean @ coef)
    return coef, intercept


def test_adaptive_lasso_predictions_match_clean_room_reference():
    X, y = _design(n=220, p=8, seed=1, sparse_beta=[3.0, 0.0, -2.0, 0.0, 1.5, 0, 0, 0])
    params = dict(alpha=0.05, gamma=1.0, initial_alpha=1.0, eps=1e-4,
                  max_iter=20000, tol=1e-4)
    fit = adaptive_lasso(X, y, initial="ridge", normalize_weights=True, **params)

    ref_coef, ref_intercept = _reference_adaptive_lasso(X, y, **params)
    X_test = X.iloc[-20:]
    got = np.asarray(fit.predict(X_test), dtype=float)
    expected = X_test.to_numpy() @ ref_coef + ref_intercept
    assert got.shape == (20,)
    np.testing.assert_allclose(got, expected, rtol=1e-7, atol=1e-8)


def test_adaptive_lasso_recovers_sparse_support():
    """Defining property: with a strong sparse signal the irrelevant predictors are
    driven to (near) zero while the true ones are retained -- an oracle property no
    self-consistency check would catch."""
    true_beta = [4.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0]
    X, y = _design(n=400, p=8, seed=3, sparse_beta=true_beta, noise=0.3)
    fit = adaptive_lasso(X, y, alpha=0.02, gamma=2.0, initial="ridge")
    coef = np.asarray(fit.coef_, dtype=float).reshape(-1)
    active = {i for i, b in enumerate(true_beta) if b != 0.0}
    inactive = set(range(len(true_beta))) - active
    # True predictors retained with the right sign and non-trivial magnitude.
    for i in active:
        assert np.sign(coef[i]) == np.sign(true_beta[i])
        assert abs(coef[i]) > 1.0
    # Irrelevant predictors shrunk far below the active ones.
    max_inactive = max(abs(coef[i]) for i in inactive)
    min_active = min(abs(coef[i]) for i in active)
    assert max_inactive < 0.2 * min_active


def test_adaptive_lasso_direct_policy_recovers_sparse_linear_signal():
    """adaptive_lasso THROUGH the runner's direct policy recovers a sparse-linear,
    predictor-driven target that an autoregression cannot.

    The target at date t is a sparse linear function of the predictors H periods
    earlier, so the direct H-step target Y_{t+H} is an EXACT sparse-linear function of
    the origin-time predictors. adaptive_lasso (with tiny alpha) must recover it to
    near machine precision, while AR -- which only sees the target's own lags -- has
    large error. This anchors AL through the policy path, not just at the model level.
    """
    n, H = 360, 2
    rng = np.random.default_rng(4)
    P = rng.normal(size=(n, 8))
    Y = np.zeros(n)
    for t in range(H, n):
        Y[t] = 3.0 * P[t - H, 0] - 2.0 * P[t - H, 3] + 0.8 * P[t - H, 5]
    idx = pd.date_range("1980-01-01", periods=n, freq="MS")
    cols = {f"x{i}": P[:, i] for i in range(8)}
    cols["Y"] = Y
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    series = pd.Series(Y, index=idx)

    def _run(model, params):
        win = mf.window.from_cutoffs(
            test_start="2004-01-01", test_end="2005-06-01", mode="expanding",
            val_method="last_block", retrain_every=1,
        )
        feats = mf.feature_engineering.feature_spec(
            target="Y", predictors="all", lags=range(0, 1),
            target_lags=range(0, 1), target_transform="value",
        )
        spec = pipeline_spec(
            data=bundle, targets=[TargetSpec(name="Y", transform="value", policy="direct")],
            horizons=[H], window=win,
            arms=[Arm(name="M", model=model, features=feats, is_benchmark=True, params=params)],
            evaluation=EvalSpec(benchmark="M", metrics=("rmse",)), n_jobs=1,
        )
        f = run_pipeline(spec).forecasts
        f = f[f["horizon"] == H].dropna(subset=["prediction"])
        return [abs(float(r["prediction"]) - float(series.iloc[series.index.get_loc(r["origin"]) + H]))
                for _, r in f.iterrows()]

    al_err = max(_run("adaptive_lasso", {"alpha": 0.001, "gamma": 1.0}))
    ar_err = max(_run("ar", {"n_lag": 1}))
    assert al_err < 1e-4, f"adaptive_lasso direct did not recover the signal (err {al_err:.2e})"
    assert ar_err > 100 * al_err
