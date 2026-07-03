"""Direct-mode AR/FAR: a one-shot projection of the (h-ahead) target onto the n
most recent OBSERVED one-period lags, NOT an iterated roll-forward from stale history.

Under the ``direct``/``direct_average`` forecast policy the regression target is a
pre-built h-ahead object, so the legacy ``_AR``/``_FAR`` (which autoregress the
target's own lags and iterate forward from the last training value) collapse to
persistence of a stale value. In ``direct`` mode the model must instead regress the
target on the n most recent observed one-period lags ``Y_lag0..Y_lag{n_lag-1}`` and
predict each test row independently. ``Y_lag0`` is the value AT the origin (the
decision time), which is observed and is NOT look-ahead for a future target -- it is
the standard Stock-Watson direct-projection regressor, the most informative one.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macroforecast.models.timeseries import _AR, _FAR


def _lag_frame(values, n_lags):
    """Build a Y_lag0..Y_lag{n_lags} design from a 1-D series (lag k = shift k)."""
    s = pd.Series(values, name="Y")
    cols = {f"Y_lag{k}": s.shift(k) for k in range(0, n_lags + 1)}
    return pd.DataFrame(cols, index=s.index)


def test_direct_ar_is_per_row_ols_on_the_n_most_recent_lags():
    rng = np.random.default_rng(0)
    n = 200
    series = np.cumsum(rng.normal(size=n))          # near-unit-root level
    target = pd.Series(series, name="target")        # stand-in h-ahead target
    X = _lag_frame(series, n_lags=4)

    fit = _AR(n_lag=3, direct=True).fit(X, target)

    # Reference: OLS of target on the 3 MOST RECENT observed lags Y_lag0..Y_lag2
    # (lag 0 is observed at the origin, not look-ahead), on the rows where all those
    # lags are observed.
    use_cols = ["Y_lag0", "Y_lag1", "Y_lag2"]
    design = X[use_cols]
    mask = design.notna().all(axis=1) & target.notna()
    ref = LinearRegression().fit(design[mask].to_numpy(), target[mask].to_numpy())

    X_test = X.iloc[-5:]
    got = np.asarray(fit.predict(X_test), dtype=float)
    expected = ref.predict(X_test[use_cols].fillna(0.0).to_numpy())

    # Per-row projection, NOT an iterated path: every test row maps through the
    # same fitted coefficients on its OWN fresh lags.
    assert got.shape == (5,)
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)

    # IC plumbing preserved.
    assert fit.ssr_ is not None and fit.nobs_ is not None
    assert fit.n_params_ == len(use_cols) + 1  # 3 lags (0,1,2) + intercept


def test_direct_ar_does_not_persist_stale_value():
    """The legacy bug: prediction ~ a stale constant. Direct mode must vary with X."""
    rng = np.random.default_rng(1)
    series = np.cumsum(rng.normal(size=160))
    target = pd.Series(series, name="target")
    X = _lag_frame(series, n_lags=2)
    fit = _AR(n_lag=2, direct=True).fit(X, target)
    preds = np.asarray(fit.predict(X.iloc[-10:]), dtype=float)
    # Distinct test rows -> distinct predictions (not a single persisted constant).
    assert np.std(preds) > 1e-6


def test_direct_far_projects_on_factors_plus_lags_no_iteration():
    rng = np.random.default_rng(2)
    n = 200
    # predictor block (drives factors) + the target's own lags
    P = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"P{i}" for i in range(6)])
    series = np.cumsum(rng.normal(size=n))
    lags = _lag_frame(series, n_lags=2)
    X = pd.concat([P, lags], axis=1)
    target = pd.Series(series, name="target")

    fit = _FAR(n_factors=3, n_lag=2, direct=True).fit(X, target)
    preds = np.asarray(fit.predict(X.iloc[-8:]), dtype=float)
    assert preds.shape == (8,)
    assert np.std(preds) > 1e-6          # varies per row, not a persisted constant
    assert fit.ssr_ is not None and fit.nobs_ is not None


def test_direct_far_uses_predictor_factors_when_predictors_are_lag_named():
    """FAR direct mode must build factors from the predictor block even when the
    predictor columns are lag-named (``P*_lag1``) -- the EXACT naming the pipeline's
    feature matrix uses. If the factor block excludes every ``*_lag*`` column (not just
    the target's own lags), the predictor lags are dropped, no factors are fit, and FAR
    collapses to plain AR. Here the predictors carry the target's signal, so a correct
    FAR must differ from AR.
    """
    rng = np.random.default_rng(3)
    n = 200
    # The h-ahead direct target is driven ONLY by the predictors' lag (as in the
    # pipeline, where the h-ahead average-growth target is a distinct object from the
    # one-period ``Y_lag*`` history). The target's own lag series ``z`` is INDEPENDENT
    # of the target, so AR (target lags only) cannot fit it while FAR's predictor
    # factors can -- the two must differ. ``Y_lag0`` here is the origin one-period
    # value, NOT the target, so there is no trivial identity fit.
    P = rng.normal(size=(n, 4))
    z = np.cumsum(rng.normal(size=n))  # target's own one-period history (independent)
    target_vals = 3.0 * np.r_[0.0, P[:-1, 0]] + 1.5 * np.r_[0.0, P[:-1, 1]] + rng.normal(scale=0.05, size=n)
    target = pd.Series(target_vals, name="Y")
    ylags = _lag_frame(z, n_lags=2)  # Y_lag0..Y_lag2 from the INDEPENDENT history z
    # Predictor LAG columns -- named exactly like the pipeline feature matrix so the
    # ``_lag(\d+)$`` pattern matches them (this is what regressed the bug).
    pred_lags = pd.DataFrame(
        {f"P{i}_lag1": np.r_[np.nan, P[:-1, i]] for i in range(4)}, index=ylags.index
    )
    X = pd.concat([pred_lags, ylags], axis=1)

    far_fit = _FAR(n_factors=3, n_lag=2, direct=True).fit(X, target)
    ar_fit = _AR(n_lag=2, direct=True).fit(X, target)
    far = np.asarray(far_fit.predict(X.iloc[-30:]), dtype=float)
    ar = np.asarray(ar_fit.predict(X.iloc[-30:]), dtype=float)

    # The predictor factors carry the target's whole signal, so a correct FAR fits far
    # better than AR (which only sees the independent ``Y_lag*`` history). If FAR drops
    # the lag-named predictor block it has the SAME regressors as AR and its residual
    # sum of squares matches AR's (up to solver noise) -- the regression signature of
    # the collapse. Require a real, large improvement so solver noise cannot mask it.
    assert far_fit._direct_pred_cols, "FAR built no predictor factor block"
    assert far_fit.ssr_ < 0.5 * ar_fit.ssr_, (
        f"FAR collapsed to AR: FAR ssr {far_fit.ssr_:.1f} not < half AR ssr {ar_fit.ssr_:.1f}"
    )
    # And the forecasts must differ substantially (not merely by solver noise).
    assert np.abs(far - ar).max() > 1.0
