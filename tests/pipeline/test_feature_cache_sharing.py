"""Cross-arm reuse of the per-origin feature-builder FIT (Gap A promotion).

Before this module, the per-origin fitted feature builder (``FeatureSpec.fit()`` --
the PCA/MARX/SIR-style numerical state) lived in a ``run()``-LOCAL variable, so two
arms differing only in ``model`` -- the most common pipeline comparison -- each
refit the feature transform at every origin. ``forecasting/feature_stage.py``
promotes that fit into the SAME shared per-target cache dict
``_prepare_origin_panel`` already uses for preprocessing, so arms with a
content-identical ``features`` spec (and the same fit-sample row bounds) compute
the fit exactly once per origin.

These tests mirror the conventions of ``test_preprocessing_share.py`` and
``test_crosshorizon_transform_dedup.py``: monkeypatch ``FeatureSpec.fit`` as a
counting spy, and always cross-check the shared (pipeline) forecasts against an
UNSHARED baseline (independent ``forecasting.run()`` calls with no cache dict) for
byte-identity -- the sharing must be invisible to the numbers, only to the fit
count.
"""

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.feature_engineering.specs import FeatureSpec
from macroforecast.forecasting import runner as _runner
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)


def _bundle(n=140, seed=3):
    idx = pd.date_range("1990-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(seed)
    cols = {f"S{i}": rng.normal(size=n) for i in range(6)}
    cols["Y"] = np.cumsum(rng.normal(size=n))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    return mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})


def _window(step=1):
    return mf.window.from_cutoffs(
        test_start="2001-01-01",
        test_end="2001-06-01",
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )


def _feats(**overrides):
    kwargs = dict(target="Y", predictors="all", lags=range(1, 4), pca_components=2)
    kwargs.update(overrides)
    return mf.feature_engineering.feature_spec(**kwargs)


def _count_feature_fits(run_fn):
    """Run ``run_fn`` while counting real ``FeatureSpec.fit`` calls.

    Mirrors ``test_preprocessing_share.py::_count_fits`` and
    ``test_crosshorizon_transform_dedup.py::_run_counting``, but spies the
    feature-engineering fit entry point instead of the preprocessing one.
    """
    original = FeatureSpec.fit
    count = {"n": 0}

    def _counting_fit(self, *args, **kwargs):
        count["n"] += 1
        return original(self, *args, **kwargs)

    FeatureSpec.fit = _counting_fit  # type: ignore[method-assign]
    try:
        result = run_fn()
    finally:
        FeatureSpec.fit = original  # type: ignore[method-assign]
    return result, count["n"]


def _sorted_forecasts(report) -> pd.DataFrame:
    return (
        report.forecasts.sort_values(["arm", "date"])
        .reset_index(drop=True)[["arm", "date", "prediction"]]
    )


def _independent_runs(bundle, models, *, window, preprocessing, features) -> pd.DataFrame:
    """Two (or more) fully independent ``run()`` calls -- NO shared cache dict.

    The reference "sharing disabled" baseline: each call gets its own implicit
    ``preprocessing_cache=None``, so nothing is shared across models. Used to prove
    the pipeline's shared-cache path never moves a single forecast number.
    """
    frames = []
    for name, model in models:
        frame = _runner.run(
            bundle,
            model,
            window=window,
            preprocessing=preprocessing,
            features=features,
            target="Y",
            horizon=1,
            save_models=False,
        ).to_frame()
        frame["arm"] = name
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["arm", "date"])
        .reset_index(drop=True)[["arm", "date", "prediction"]]
    )


# --------------------------------------------------------------------------- #
# Core counter test (acceptance criterion 1)
# --------------------------------------------------------------------------- #


def test_model_comparison_reuses_feature_fit_across_arms():
    """2 arms x 6 origins, same features + spec-level preprocessing, differ only
    in `model`: the feature-builder fit runs exactly once per origin (6), not
    once per (arm, origin) (12).
    """
    bundle = _bundle()
    prep = mf.preprocessing.preprocess_spec(
        transform="official", impute="mean", standardize="zscore"
    )
    win = _window()
    feats = _feats()
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1],
        window=win,
        arms=[
            Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True),
            Arm(name="LASSO", model="lasso", features=feats),
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        preprocessing=prep,
        n_jobs=1,
    )
    report, n_fits = _count_feature_fits(lambda: run_pipeline(spec))
    assert not report.forecasts.empty
    assert report.forecasts["arm"].nunique() == 2
    n_origins = report.forecasts.groupby("arm").size().iloc[0]
    assert n_origins > 1  # a real dedup target
    assert n_fits == n_origins, (
        f"{n_fits} feature fits for {n_origins} origins x 2 arms "
        f"(no reuse would give {2 * n_origins})"
    )
    assert n_fits < 2 * n_origins


def test_feature_cache_preserves_forecasts():
    """Sharing must not move a single number: pipeline (shared) == independent
    run() calls (unshared).
    """
    bundle = _bundle()
    prep = mf.preprocessing.preprocess_spec(
        transform="official", impute="mean", standardize="zscore"
    )
    win = _window()
    feats = _feats()
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1],
        window=win,
        arms=[
            Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True),
            Arm(name="LASSO", model="lasso", features=feats),
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        preprocessing=prep,
        n_jobs=1,
    )
    shared = _sorted_forecasts(run_pipeline(spec))
    unshared = _independent_runs(
        bundle,
        [("RIDGE", "ridge"), ("LASSO", "lasso")],
        window=win,
        preprocessing=prep,
        features=feats,
    )
    assert not shared.empty
    assert not unshared.empty
    pd.testing.assert_frame_equal(shared, unshared, atol=1e-12)


# --------------------------------------------------------------------------- #
# Sharing works even with no spec-level preprocessing at all
# --------------------------------------------------------------------------- #


def test_feature_only_pipeline_still_shares_across_arms():
    """With NO ``preprocessing=`` set at all, feature-fit sharing must still
    kick in (this is what required unconditionally building ``target_caches``
    in ``pipeline/run.py``, not gating it on ``spec.preprocessing is not None``).
    """
    bundle = _bundle()
    win = _window()
    feats = _feats(pca_components=None, lags=range(1, 3))
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1],
        window=win,
        arms=[
            Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True),
            Arm(name="LASSO", model="lasso", features=feats),
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        n_jobs=1,
    )
    report, n_fits = _count_feature_fits(lambda: run_pipeline(spec))
    assert not report.forecasts.empty
    n_origins = report.forecasts.groupby("arm").size().iloc[0]
    assert n_fits == n_origins

    shared = _sorted_forecasts(report)
    unshared = _independent_runs(
        bundle,
        [("RIDGE", "ridge"), ("LASSO", "lasso")],
        window=win,
        preprocessing=None,
        features=feats,
    )
    pd.testing.assert_frame_equal(shared, unshared, atol=1e-12)


# --------------------------------------------------------------------------- #
# Safety: a per-arm window override must NOT wrongly share a different sample
# --------------------------------------------------------------------------- #


def test_divergent_arm_window_never_wrongly_shares():
    """An arm with its OWN window (a real, tested configuration -- see
    ``test_per_arm_window.py``) uses a genuinely different estimation strategy
    (rolling vs expanding), so its per-origin fit sample diverges from the
    shared-window arm's even at a matching ``origin_pos``. This must not be
    wrongly shared -- forecasts must stay byte-identical to two fully
    independent (unshared) run() calls regardless of whether sharing kicks in.
    """
    bundle = _bundle(seed=9)
    feats = _feats(pca_components=None, lags=range(1, 3))
    shared_win = _window()
    rolling_win = mf.window.spec(
        estimation=mf.window.estimation_rolling(size=24),
        val=shared_win.val,
        test=shared_win.test,
    )
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1],
        window=shared_win,
        arms=[
            Arm(name="ROLL", model="ridge", features=feats, window=rolling_win),
            Arm(name="EXPAND", model="ridge", features=feats),
        ],
        evaluation=EvalSpec(benchmark="EXPAND", metrics=("rmse",)),
        n_jobs=1,
    )
    report, n_fits = _count_feature_fits(lambda: run_pipeline(spec))
    n_origins = report.forecasts.groupby("arm").size()
    # A window-overriding arm opts out of sharing entirely -- the fit count is NOT
    # reduced below "once per (arm, origin)" the way the shared-window case is.
    assert n_fits == int(n_origins.sum())

    shared = _sorted_forecasts(report)
    unshared = _independent_runs(
        bundle,
        [("ROLL", "ridge")],
        window=rolling_win,
        preprocessing=None,
        features=feats,
    )
    unshared2 = _independent_runs(
        bundle,
        [("EXPAND", "ridge")],
        window=shared_win,
        preprocessing=None,
        features=feats,
    )
    unshared_all = (
        pd.concat([unshared, unshared2], ignore_index=True)
        .sort_values(["arm", "date"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(shared, unshared_all, atol=1e-12)


# --------------------------------------------------------------------------- #
# Retrain-cadence semantics (design constraint: never/interval must not change)
# --------------------------------------------------------------------------- #


def test_never_update_cadence_preserved_under_sharing():
    """``feature_policy`` update="never": the feature builder fits ONCE total
    (at the first origin) and is reused for every later origin BY THE SAME ARM
    (existing single-arm semantics) -- and, with sharing, the very same single
    fit is reused by the OTHER arm too. Total fit count across both arms and all
    origins must be exactly 1, and forecasts must stay byte-identical to the
    unshared baseline (which also fits once per arm under this cadence).
    """
    bundle = _bundle(seed=5)
    prep = mf.preprocessing.preprocess_spec(
        transform="official", impute="mean", standardize="zscore"
    )
    win = _window()
    feats = _feats(pca_components=None, lags=range(1, 3))
    never_policy = mf.window.stage_policy("fit_window", update="never")
    spec = pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1],
        window=win,
        arms=[
            Arm(
                name="RIDGE", model="ridge", features=feats, is_benchmark=True,
                feature_policy=never_policy,
            ),
            Arm(name="LASSO", model="lasso", features=feats, feature_policy=never_policy),
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        preprocessing=prep,
        n_jobs=1,
    )
    report, n_fits = _count_feature_fits(lambda: run_pipeline(spec))
    assert not report.forecasts.empty
    assert n_fits == 1

    shared = _sorted_forecasts(report)
    unshared = _independent_runs(
        bundle,
        [("RIDGE", "ridge"), ("LASSO", "lasso")],
        window=win,
        preprocessing=prep,
        features=feats,
    )
    # feature_policy is not threaded through _independent_runs' plain run() calls
    # above (they use the default "every_origin" cadence); rebuild them directly
    # with the never policy so the comparison is apples-to-apples.
    def _run_never(model):
        frame = _runner.run(
            bundle,
            model,
            window=win,
            preprocessing=prep,
            features=feats,
            feature_policy=never_policy,
            target="Y",
            horizon=1,
            save_models=False,
        ).to_frame()
        return frame

    r1 = _run_never("ridge")
    r1["arm"] = "RIDGE"
    r2 = _run_never("lasso")
    r2["arm"] = "LASSO"
    unshared = (
        pd.concat([r1, r2], ignore_index=True)
        .sort_values(["arm", "date"])
        .reset_index(drop=True)[["arm", "date", "prediction"]]
    )
    pd.testing.assert_frame_equal(shared, unshared, atol=1e-12)
