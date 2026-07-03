"""Cross-horizon reuse of the per-origin preprocessing TRANSFORM.

The per-origin spec-level preprocessing does two things at each origin: a ``.fit()``
(EM/factor parameter estimation) and a ``.transform()`` (applying the imputation +
factor projection to the panel). Both are horizon-INDEPENDENT for the rows observable
at the origin -- they depend only on the origin's ``origin_available`` window, never on
the forecast horizon. The per-origin ``.fit()`` is already shared across horizons via
the ``origin_pos``-keyed cache, but the (dominant-cost) ``.transform()`` of the
estimation-window block was re-executed once per horizon because the prepared-stage
cache key embedded the horizon-dependent target-row position.

These tests pin the desired behaviour: at each origin the heavy estimation-window
transform runs ONCE and is reused across every horizon, while the forecasts stay
byte-identical. They exercise the SPEC-LEVEL preprocessing path (``arm.preprocessing``
is ``None`` so the shared per-target cache is threaded in), which is the exact
configuration the GCLS replication pipeline uses.
"""

import numpy as np
import pandas as pd

import macroforecast as mf
from macroforecast.preprocessing.specs import FittedPreprocessor, PreprocessSpec
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)

HORIZONS = [1, 3]
_MAX_H = max(HORIZONS)


def _toy_spec(arm_models=("ridge",)):
    """A 2-horizon spec over ``arm_models``, all using SPEC-LEVEL preprocessing.

    Spec-level preprocessing (``arm.preprocessing is None``) is what routes the
    shared per-(target, origin) preprocessing cache into ``run()`` -- the path the
    cross-horizon transform reuse lives on. A short test window keeps several
    distinct origins while staying cheap.
    """
    idx = pd.date_range("1990-01-01", periods=140, freq="MS")
    rng = np.random.default_rng(7)
    cols = {f"S{i}": rng.normal(size=140) for i in range(8)}
    cols["Y"] = np.cumsum(rng.normal(size=140))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    prep = mf.preprocessing.preprocess_spec(
        transform="official", impute="mean", standardize="zscore"
    )
    pp_policy = mf.window.stage_policy("origin_available", update=1)
    win = mf.window.from_cutoffs(
        test_start="2001-01-01",
        test_end="2001-10-01",  # ~10 origins -- several, but cheap
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 4)
    )
    # One arm per model, ALL sharing the spec-level preprocessing (features=None means
    # each carries no per-arm preprocessing, so the shared per-origin cache is used).
    names = [m.upper() for m in arm_models]
    arms = [
        Arm(name=n, model=m, features=feats, is_benchmark=(i == 0))
        for i, (n, m) in enumerate(zip(names, arm_models))
    ]
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=HORIZONS,
        window=win,
        arms=arms,
        evaluation=EvalSpec(benchmark=names[0], metrics=("rmse",)),
        preprocessing=prep,
        preprocessing_policy=pp_policy,
        n_jobs=1,
    )


def _em_spec():
    """Like ``_toy_spec`` but with ``em_factor`` imputation over a RAGGED panel.

    ``em_factor`` is the default (and dominant-cost) imputer, and unlike mean
    imputation its transform is the one whose cross-row behaviour actually matters
    for the base/forward split. Missing values are injected so EM genuinely imputes
    from factors rather than trivially passing the panel through.
    """
    idx = pd.date_range("1990-01-01", periods=140, freq="MS")
    rng = np.random.default_rng(11)
    cols = {f"S{i}": rng.normal(size=140) for i in range(8)}
    cols["Y"] = np.cumsum(rng.normal(size=140))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    # Scatter missing entries across predictor columns so EM-factor imputation is
    # actually exercised (ragged-edge and interior gaps), leaving Y complete.
    miss = rng.random(panel.shape) < 0.08
    miss[:, panel.columns.get_loc("Y")] = False
    panel = panel.mask(miss)
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
    prep = mf.preprocessing.preprocess_spec(
        transform="official", impute="em_factor", standardize="zscore"
    )
    pp_policy = mf.window.stage_policy("origin_available", update=1)
    win = mf.window.from_cutoffs(
        test_start="2001-01-01", test_end="2001-10-01",
        mode="expanding", val_method="last_block", retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 4)
    )
    arms = [Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True)]
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=HORIZONS,
        window=win,
        arms=arms,
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        preprocessing=prep,
        preprocessing_policy=pp_policy,
        n_jobs=1,
    )


def test_crosshorizon_em_factor_split_equals_whole():
    """The base(<=origin)+forward(>origin) split must equal the whole-window transform
    UNDER em_factor imputation, not just mean.

    The serial backend threads the cross-horizon cache and reconstructs each origin's
    transformed panel from a cached base block plus the tiny forward block; the
    parallel backend passes ``preprocessing_cache=None`` and transforms every panel
    whole. Mean imputation is row-independent, so its split==whole identity is nearly
    tautological; em_factor's transform can couple across the applied rows, so this is
    the case that actually exercises the split. Byte-identity here is the real proof.
    """
    serial = _sorted(run_pipeline(_em_spec()))
    import dataclasses

    parallel = _sorted(run_pipeline(dataclasses.replace(_em_spec(), n_jobs=2)))
    assert not serial.empty
    assert not parallel.empty
    pd.testing.assert_frame_equal(serial, parallel, atol=1e-12)


def _run_counting(spec=None):
    """Run ``spec`` (default single-arm toy), counting fits and transform row sizes.

    Returns ``(report, n_fits, transform_sizes)`` where ``transform_sizes`` is the
    list of ``len(apply_panel)`` for every ``FittedPreprocessor.transform`` call. The
    per-origin ``.fit()`` is cached on ``origin_pos``, so ``n_fits`` equals the number
    of DISTINCT origins -- the number of heavy transforms a horizon-independent cache
    must perform.
    """
    fit_orig = PreprocessSpec.fit
    tr_orig = FittedPreprocessor.transform
    n_fits = {"n": 0}
    sizes: list[int] = []

    def _counting_fit(self, *a, **k):
        n_fits["n"] += 1
        return fit_orig(self, *a, **k)

    def _counting_transform(self, data, *a, **k):
        try:
            sizes.append(int(len(data)))
        except TypeError:
            sizes.append(-1)
        return tr_orig(self, data, *a, **k)

    PreprocessSpec.fit = _counting_fit  # type: ignore[method-assign]
    FittedPreprocessor.transform = _counting_transform  # type: ignore[method-assign]
    try:
        report = run_pipeline(spec if spec is not None else _toy_spec())
    finally:
        PreprocessSpec.fit = fit_orig  # type: ignore[method-assign]
        FittedPreprocessor.transform = tr_orig  # type: ignore[method-assign]
    return report, n_fits["n"], sizes


def test_heavy_transform_runs_once_per_origin_across_horizons():
    """The estimation-window transform executes once per origin, not once per (origin, horizon).

    A "heavy" transform is one whose apply panel spans the estimation window (far more
    rows than the tiny forward block of at most ``max_horizon`` rows). With the
    horizon-dependent prepared-stage key, one heavy transform runs per (origin, horizon)
    -- ``n_origins * n_horizons``. Keyed horizon-independently it runs once per origin.
    """
    report, n_origins, sizes = _run_counting()
    assert not report.forecasts.empty
    assert n_origins > 1  # several distinct origins -- a real dedup target

    heavy = [s for s in sizes if s > _MAX_H + 5]
    assert heavy, "expected some estimation-window transforms"
    # Heavy transforms must not scale with the horizon count.
    assert len(heavy) == n_origins, (
        f"{len(heavy)} heavy transforms for {n_origins} origins across "
        f"{len(HORIZONS)} horizons (horizon-dependent keying would give "
        f"{n_origins * len(HORIZONS)})"
    )
    assert len(heavy) < n_origins * len(HORIZONS)


def _sorted(report):
    return (
        report.forecasts.sort_values(["arm", "horizon", "date"])
        .reset_index(drop=True)[["arm", "horizon", "date", "prediction"]]
    )


def test_crosshorizon_reuse_preserves_forecasts():
    """Reusing the per-origin transform across horizons must not move a single number.

    The serial backend threads the in-memory cross-horizon preprocessing cache; the
    parallel backend passes ``preprocessing_cache=None`` so each (arm, horizon) cell
    recomputes its own transform from scratch. The two must be byte-identical -- so the
    parallel run is the reference that pins the serial cross-horizon reuse to the exact
    same forecasts.
    """
    serial = _sorted(run_pipeline(_toy_spec()))
    spec_par = _toy_spec()
    import dataclasses

    parallel = _sorted(run_pipeline(dataclasses.replace(spec_par, n_jobs=2)))
    assert not serial.empty
    assert not parallel.empty
    pd.testing.assert_frame_equal(serial, parallel, atol=1e-12)


def test_model_comparison_reuses_preprocessing_across_arms_and_horizons():
    """Model-comparison contract: preprocessing is HELD FIXED, only the model varies.

    With N models sharing one spec-level preprocessing over M horizons, the per-origin
    preprocessing FIT and the heavy per-origin TRANSFORM each run exactly ONCE per
    origin -- NOT once per (arm, horizon, origin). Comparing models must not recompute
    the shared preprocessing for every model. This pins both reuse axes at once:
    across arms (the shared per-target cache) and across horizons (the origin-keyed
    base-panel cache). If either axis regressed, the counts would scale with the arm
    or horizon count.
    """
    n_arms = 3
    spec = _toy_spec(arm_models=("ridge", "lasso", "elastic_net"))
    report, n_fits, sizes = _run_counting(spec)
    assert not report.forecasts.empty
    # Every arm produced rows (the comparison is real, not a silently-empty arm).
    assert report.forecasts["arm"].nunique() == n_arms

    heavy = [s for s in sizes if s > _MAX_H + 5]
    n_origins = n_fits  # fit is origin-keyed -> one fit per distinct origin
    assert n_origins > 1
    # One heavy transform and one fit per origin, independent of arms AND horizons.
    assert len(heavy) == n_origins, (
        f"{len(heavy)} heavy transforms for {n_origins} origins with {n_arms} arms x "
        f"{len(HORIZONS)} horizons (no reuse would give {n_origins * n_arms * len(HORIZONS)})"
    )
    assert len(heavy) < n_origins * n_arms * len(HORIZONS)
