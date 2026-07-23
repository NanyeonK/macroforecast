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


def test_far_predictions_match_regression_anchor():
    rng = np.random.default_rng(20260710)
    n = 48
    idx = pd.RangeIndex(n)
    base = rng.normal(size=(n, 5))
    X = pd.DataFrame(base, columns=[f"x{i}" for i in range(5)], index=idx)
    target = pd.Series(
        0.8 * X["x0"]
        - 0.35 * X["x1"]
        + 0.2 * np.r_[0.0, X["x2"].to_numpy()[:-1]]
        + 0.1 * rng.normal(size=n),
        index=idx,
        name="target",
    )

    recursive = _FAR(n_factors=2, n_lag=2, random_state=11, direct=False).fit(X, target)
    np.testing.assert_allclose(
        np.asarray(recursive.predict(X.iloc[-6:]), dtype=float),
        np.asarray(
            [
                0.0461667705814010,
                0.3831322291723367,
                -0.4661571237265418,
                0.1761409540412961,
                0.0271061955576718,
                0.0058917811944191,
            ],
            dtype=float,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(recursive.ssr_, 14.566995627997674, atol=1e-12)
    assert recursive.nobs_ == 46
    assert recursive.n_params_ == 5

    history = np.cumsum(rng.normal(size=n))
    lag_frame = pd.DataFrame(
        {f"target_lag{k}": pd.Series(history).shift(k) for k in range(3)},
        index=idx,
    )
    direct_X = pd.concat([X.add_prefix("p_"), lag_frame], axis=1)
    direct_target = pd.Series(
        1.2 * np.r_[0.0, base[:-1, 0]]
        - 0.7 * np.r_[0.0, base[:-1, 1]]
        + 0.05 * rng.normal(size=n),
        index=idx,
        name="target",
    )
    direct = _FAR(n_factors=2, n_lag=2, random_state=11, direct=True).fit(
        direct_X,
        direct_target,
    )
    np.testing.assert_allclose(
        np.asarray(direct.predict(direct_X.iloc[-6:]), dtype=float),
        np.asarray(
            [
                -0.2945239445418160,
                -0.0866913502773294,
                -0.2234713967527731,
                -0.0414753103557364,
                -0.1024913161471229,
                -0.1250310480699385,
            ],
            dtype=float,
        ),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(direct.ssr_, 62.28138785224115, atol=1e-12)
    assert direct.nobs_ == 48
    assert direct.n_params_ == 5


def test_direct_ar_ignores_predictor_lags_when_target_lag0_absent():
    """Regression (kitchen-sink benchmark bug): a direct AR whose feature matrix
    carries predictor ``*_lag0`` columns AND target lags that start at lag 1 (no
    ``*_lag0`` for the target) must NOT silently regress the target on the
    predictors' contemporaneous values when ``n_lag`` restricts the range to lag 0.
    It must select none of the target's own lags and fall back to the unconditional
    mean -- otherwise an "AR benchmark" becomes a p=N kitchen-sink OLS and every
    relative RMSE normalized against it is corrupted.
    """
    from macroforecast.models.timeseries import _select_lag_columns

    rng = np.random.default_rng(3)
    n = 160
    series = np.cumsum(rng.normal(size=n))
    target = pd.Series(series, name="UNRATE")
    s = pd.Series(series, name="UNRATE")
    cols = {f"UNRATE_lag{k}": s.shift(k) for k in (1, 2, 3, 4)}  # target lags: NO lag0
    for j in range(6):                                           # predictor lag0/lag1 block
        p = pd.Series(np.cumsum(rng.normal(size=n)))
        cols[f"P{j}_lag0"] = p.shift(0)
        cols[f"P{j}_lag1"] = p.shift(1)
    X = pd.DataFrame(cols)

    # n_lag=1 wants the target's lag0, which is absent -> select NOTHING (never the
    # predictors' *_lag0 columns).
    assert _select_lag_columns(X, 1, "UNRATE") == []

    fit = _AR(n_lag=1, direct=True).fit(X, target)
    preds = np.asarray(fit.predict(X.iloc[-10:]), dtype=float)
    # Mean fallback: a single constant near the training mean, NOT a wild fit.
    assert np.std(preds) < 1e-9
    assert abs(float(preds[0]) - fit._fallback) < 1e-9
    # IC plumbing exposed for the mean-only model so BIC/AIC order selection works.
    assert fit.ssr_ is not None and fit.nobs_ is not None and fit.n_params_ == 1


def test_select_lag_columns_matches_target_base_across_all_indices():
    """The target base is matched against EVERY lag column (any index), so a target
    whose lags start at 1 still resolves to its OWN base and returns [] when the
    requested range excludes them -- rather than falling through to other bases."""
    from macroforecast.models.timeseries import _select_lag_columns

    cols = [f"UNRATE_lag{k}" for k in (1, 2, 3, 4)] + ["GDP_lag0", "CPI_lag0"]
    X = pd.DataFrame({c: [0.0, 1.0, 2.0] for c in cols})
    assert _select_lag_columns(X, 1, "UNRATE") == []                 # no target lag0 -> []
    assert _select_lag_columns(X, 5, "UNRATE") == [
        "UNRATE_lag1", "UNRATE_lag2", "UNRATE_lag3", "UNRATE_lag4",  # target's own lags only
    ]
    # single-base name mismatch is benign: those ARE the target's lags (predictors=[])
    Y = pd.DataFrame({f"Y_lag{k}": [0.0, 1.0, 2.0] for k in range(0, 4)})
    assert _select_lag_columns(Y, 2, "target") == ["Y_lag0", "Y_lag1"]
