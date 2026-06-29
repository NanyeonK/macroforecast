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
from macroforecast.forecasting import runner as _runner
from macroforecast.preprocessing.cache import PreprocessorStore
from macroforecast.preprocessing.specs import PreprocessSpec
from macroforecast.pipeline import (
    Arm,
    EvalSpec,
    TargetSpec,
    pipeline_spec,
    run_pipeline,
)


def _toy_run_inputs():
    """Build a single-target panel/spec/window/features for direct ``run()`` calls.

    Mirrors the pipeline fixture above but at the atomic ``run()`` level, so the
    on-disk store can be driven through two independent ``run()`` calls that share
    one ``PreprocessorStore`` -- the cross-process reuse the store exists for.
    """
    idx = pd.date_range("1990-01-01", periods=160, freq="MS")
    rng = np.random.default_rng(1)
    cols = {f"S{i}": rng.normal(size=160) for i in range(6)}
    cols["Y"] = np.cumsum(rng.normal(size=160))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(
        panel, transform_codes={c: 1 for c in panel.columns}
    )
    prep = mf.preprocessing.preprocess_spec(
        transform="official", impute="mean", standardize="zscore"
    )
    win = mf.window.from_cutoffs(
        test_start="2002-01-01",
        test_end="2003-12-01",
        mode="expanding",
        val_method="last_block",
        retrain_every=1,
    )
    feats = mf.feature_engineering.feature_spec(
        target="Y", predictors="all", lags=range(1, 4)
    )
    return bundle, prep, win, feats


def _run_once(*, preprocessing_store: PreprocessorStore | None = None) -> object:
    """Run one atomic forecast cell (single target/model/horizon)."""
    bundle, prep, win, feats = _toy_run_inputs()
    return _runner.run(
        bundle,
        "ridge",
        window=win,
        preprocessing=prep,
        features=feats,
        target="Y",
        horizon=1,
        save_models=False,
        preprocessing_store=preprocessing_store,
    )


def _toy(n_jobs: int, preprocessing_cache_dir: str | None = None):
    """Build a 2-arm, 2-horizon toy pipeline spec parameterized by n_jobs.

    When ``preprocessing_cache_dir`` is given, the spec carries a shared on-disk
    ``PreprocessorStore`` directory so parallel cells reuse each per-origin fit.
    """
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
        preprocessing_cache_dir=preprocessing_cache_dir,
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


def test_disk_store_preserves_numbers(tmp_path):
    """The on-disk tier must not change forecasts: store-OFF == store-ON."""
    off = (
        _run_once(preprocessing_store=None)
        .forecasts.sort_values(["horizon", "date"])
        .reset_index(drop=True)
    )
    store = PreprocessorStore(tmp_path)
    on = (
        _run_once(preprocessing_store=store)
        .forecasts.sort_values(["horizon", "date"])
        .reset_index(drop=True)
    )
    assert not off.empty
    assert not on.empty
    pd.testing.assert_frame_equal(
        off[["horizon", "date", "prediction"]],
        on[["horizon", "date", "prediction"]],
        atol=1e-10,
    )


def _count_fits(run_fn):
    """Run ``run_fn`` while counting real per-origin ``PreprocessSpec.fit`` calls.

    The spy wraps the EXACT fit call site that ``_prepare_origin_panel`` invokes
    to construct a ``FittedPreprocessor`` (``PreprocessSpec.fit``). Counting it
    measures how many per-(spec, target, origin) fits actually execute.
    """
    original = PreprocessSpec.fit
    count = {"n": 0}

    def _counting_fit(self, *args, **kwargs):
        count["n"] += 1
        return original(self, *args, **kwargs)

    PreprocessSpec.fit = _counting_fit  # type: ignore[method-assign]
    try:
        result = run_fn()
    finally:
        PreprocessSpec.fit = original  # type: ignore[method-assign]
    return result, count["n"]


def test_disk_store_dedupes_preprocessing_fit(tmp_path):
    """A shared store makes the second run() perform ZERO new per-origin fits.

    Two independent ``run()`` calls over the SAME target/window/origins/spec share
    one ``PreprocessorStore``. The first call fits every origin once and persists
    each fit; the second call finds every ``(spec, target, origin)`` on disk and
    fits NONE. The control -- two runs WITHOUT a shared store -- fits the full
    origin set twice, proving the dedupe is the store's doing.
    """
    store = PreprocessorStore(tmp_path)

    # --- Treatment: shared store across two run() calls --------------------
    _, first_fits = _count_fits(lambda: _run_once(preprocessing_store=store))
    assert first_fits > 0  # the first run actually computes the origins
    _, second_fits = _count_fits(lambda: _run_once(preprocessing_store=store))
    # Every origin the first run computed is now on disk, so the second run
    # recomputes nothing: the per-origin fit executes exactly once across the
    # two calls.
    assert second_fits == 0

    # --- Control: no shared store -> the second run refits everything -------
    _, ctrl_first = _count_fits(lambda: _run_once(preprocessing_store=None))
    _, ctrl_second = _count_fits(lambda: _run_once(preprocessing_store=None))
    assert ctrl_first == first_fits
    assert ctrl_second == ctrl_first  # without the store the work is repeated


# --------------------------------------------------------------------------- #
# Pipeline-level shared on-disk store (preprocessing_cache_dir)
# --------------------------------------------------------------------------- #


def _sorted_forecasts(report) -> pd.DataFrame:
    """The forecast frame sorted on the comparison keys for assert_frame_equal."""
    return (
        report.forecasts.sort_values(["arm", "horizon", "date"])
        .reset_index(drop=True)
    )


def test_pipeline_cache_dir_preserves_numbers(tmp_path):
    """``preprocessing_cache_dir`` must not change forecasts, serial or parallel.

    Three runs of the same toy spec -- (1) serial WITHOUT a cache dir, (2) serial
    WITH a shared on-disk store, (3) parallel WITH the same store -- must produce
    byte-identical forecasts. This pins that wiring the on-disk store into both
    backends preserves strict numerical identity (store-off == store-on and
    serial == parallel).
    """
    off = _sorted_forecasts(run_pipeline(_toy(1)))
    on_serial = _sorted_forecasts(
        run_pipeline(_toy(1, preprocessing_cache_dir=str(tmp_path / "serial")))
    )
    on_parallel = _sorted_forecasts(
        run_pipeline(_toy(2, preprocessing_cache_dir=str(tmp_path / "parallel")))
    )

    # Guard against a silently empty run masking a no-op comparison.
    assert not off.empty
    assert not on_serial.empty
    assert not on_parallel.empty

    cols = ["arm", "horizon", "date", "prediction"]
    pd.testing.assert_frame_equal(off[cols], on_serial[cols], atol=1e-10)
    pd.testing.assert_frame_equal(off[cols], on_parallel[cols], atol=1e-10)


def _distinct_origin_count(cache_dir: str) -> int:
    """Number of DISTINCT (spec, target, origin) triples the toy spec fits.

    Runs the toy spec SERIALLY with a shared store (so the in-process spy on
    ``PreprocessorStore.key`` observes every keyed fit) and counts the distinct
    keys. Both arms share one PreprocessSpec and one target, so the distinct-key
    count equals the number of distinct origins -- the number of files a correctly
    deduping store must hold, independent of arms and horizons.
    """
    keys: set[str] = set()
    original = PreprocessorStore.key

    def _spy_key(self, prep_spec, *, target, origin_pos):
        k = original(self, prep_spec, target=target, origin_pos=origin_pos)
        keys.add(k)
        return k

    PreprocessorStore.key = _spy_key  # type: ignore[method-assign]
    try:
        run_pipeline(_toy(1, preprocessing_cache_dir=cache_dir))
    finally:
        PreprocessorStore.key = original  # type: ignore[method-assign]
    return len(keys)


def test_pipeline_cache_dir_dedupes_across_cells(tmp_path):
    """A shared store holds ONE entry per distinct origin, not per arm x horizon.

    The toy spec has 2 arms, 2 horizons, and one shared PreprocessSpec + target, so
    a naive per-cell store would write (arms x horizons x origins) files. With the
    content-addressed store each per-(spec, target, origin) fit is persisted ONCE
    across all arms, horizons, and (in parallel) processes. The on-disk file count
    must therefore equal the number of DISTINCT origins, proving the cross-cell
    (and cross-process) EM dedup.
    """
    # Expected = distinct origins (computed independently via a serial control run).
    expected = _distinct_origin_count(str(tmp_path / "control"))
    assert expected > 1  # the window yields several origins; a real dedup target

    parallel_dir = tmp_path / "parallel"
    report = run_pipeline(_toy(2, preprocessing_cache_dir=str(parallel_dir)))
    assert not report.forecasts.empty

    entries = list(parallel_dir.iterdir())
    # No partial writes must remain: an atomic put renames the temp away.
    leftover_tmp = [p for p in entries if p.name.endswith(".tmp") or ".tmp" in p.name]
    assert leftover_tmp == [], f"leftover temp files: {leftover_tmp}"

    persisted = [p for p in entries if not (p.name.endswith(".tmp") or ".tmp" in p.name)]
    n_files = len(persisted)

    n_arms = len(_toy(2).arms)
    n_horizons = len(_toy(2).horizons)
    # One file per distinct origin -- NOT multiplied by arms or horizons.
    assert n_files == expected, (
        f"store holds {n_files} entries but expected {expected} distinct origins "
        f"(naive per-cell would write up to {expected * n_arms * n_horizons})"
    )
    # And strictly fewer than the naive per-cell count, proving the dedup happened.
    assert n_files < expected * n_arms * n_horizons
