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

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.forecasting import runner as _runner
from macroforecast.preprocessing import FittedPreprocessor
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


def _custom_marker_step(panel: pd.DataFrame, metadata=None) -> pd.DataFrame:
    out = panel.copy()
    out["CUSTOM_MARKER"] = 1.0
    return out


def _run_custom_once(*, preprocessing_store: PreprocessorStore | None = None) -> object:
    bundle, _prep, win, feats = _toy_run_inputs()
    prep = mf.preprocessing.preprocess_spec(
        transform="official",
        impute="mean",
        standardize="zscore",
        custom_steps=[
            mf.preprocessing.custom_preprocess_step(
                "marker",
                _custom_marker_step,
            )
        ],
    )
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


def _count_prepared_base_transforms(run_fn):
    """Count real horizon-independent prepared-base transform computations."""

    original = FittedPreprocessor.transform
    count = {"n": 0}

    def _counting_transform(self, panel, *args, **kwargs):
        available = kwargs.get("available")
        policy = kwargs.get("policy")
        if (
            policy == "origin_available"
            and available is not None
            and len(panel) == len(available)
        ):
            count["n"] += 1
        return original(self, panel, *args, **kwargs)

    FittedPreprocessor.transform = _counting_transform  # type: ignore[method-assign]
    try:
        result = run_fn()
    finally:
        FittedPreprocessor.transform = original  # type: ignore[method-assign]
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


def test_disk_store_custom_callable_requires_digest_for_reuse(tmp_path):
    if hasattr(_custom_marker_step, "__mf_digest__"):
        delattr(_custom_marker_step, "__mf_digest__")
    store = PreprocessorStore(tmp_path / "store")

    with pytest.warns(UserWarning, match="disk cache disabled"):
        _, first_fits = _count_fits(lambda: _run_custom_once(preprocessing_store=store))
    assert first_fits > 0
    with pytest.warns(UserWarning, match="disk cache disabled"):
        _, second_fits = _count_fits(lambda: _run_custom_once(preprocessing_store=store))
    assert second_fits == first_fits
    assert not list((tmp_path / "store").glob("*.pkl"))
    assert not list((tmp_path / "store").glob("*.parquet"))

    _custom_marker_step.__mf_digest__ = "marker-v1"
    digest_store = PreprocessorStore(tmp_path / "digest-store")
    _, digest_first = _count_fits(
        lambda: _run_custom_once(preprocessing_store=digest_store)
    )
    _, digest_second = _count_fits(
        lambda: _run_custom_once(preprocessing_store=digest_store)
    )
    assert digest_first > 0
    assert digest_second == 0

    _custom_marker_step.__mf_digest__ = "marker-v2"
    _, changed_digest = _count_fits(
        lambda: _run_custom_once(preprocessing_store=digest_store)
    )
    assert changed_digest == digest_first
    delattr(_custom_marker_step, "__mf_digest__")


def test_disk_store_dedupes_prepared_base_across_pipeline_runs(tmp_path):
    """A shared preprocessing_cache_dir persists prepared-base panels across runs."""

    cache_dir = tmp_path / "prepared"

    _, first_base = _count_prepared_base_transforms(
        lambda: run_pipeline(_toy(1, preprocessing_cache_dir=str(cache_dir)))
    )
    assert first_base > 0

    _, second_base = _count_prepared_base_transforms(
        lambda: run_pipeline(_toy(1, preprocessing_cache_dir=str(cache_dir)))
    )
    assert second_base == 0

    changed = _toy(1, preprocessing_cache_dir=str(cache_dir))
    changed_prep = mf.preprocessing.preprocess_spec(
        transform="official",
        impute="mean",
        standardize="none",
    )
    import dataclasses as _dc

    changed_arms = tuple(_dc.replace(arm, preprocessing=changed_prep) for arm in changed.arms)
    changed = _dc.replace(changed, arms=changed_arms)
    _, changed_base = _count_prepared_base_transforms(lambda: run_pipeline(changed))
    assert changed_base > 0


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

    fit_files = [p for p in entries if p.suffix == ".pkl"]
    frame_parquets = [p for p in entries if p.suffix == ".parquet"]
    frame_meta = [p for p in entries if p.suffix == ".json"]

    n_arms = len(_toy(2).arms)
    n_horizons = len(_toy(2).horizons)
    # One fitted-preprocessor file per distinct origin -- NOT multiplied by arms
    # or horizons. The prepared-base frame tier is also one parquet/json pair per
    # distinct origin.
    assert len(fit_files) == expected, (
        f"store holds {len(fit_files)} fit entries but expected {expected} distinct origins "
        f"(naive per-cell would write up to {expected * n_arms * n_horizons})"
    )
    assert len(frame_parquets) == expected
    assert len(frame_meta) == expected
    # And strictly fewer fit files than the naive per-cell count, proving the dedup happened.
    assert len(fit_files) < expected * n_arms * n_horizons


# --------------------------------------------------------------------------- #
# WP7: scope-namespace isolation (the footgun) + auto-managed cache dir (n_jobs>1)
# --------------------------------------------------------------------------- #


def test_store_namespace_isolates_cross_scope_writes(tmp_path):
    """A store's key must depend on its ``namespace``, closing the footgun where
    ``preprocessing_policy.scope`` was invisible to the cache key.

    Without a namespace (the pre-fix, and still the default, behavior), two
    different scopes sharing one store directory collide: the key depends only on
    ``(spec, target, origin_pos)``, so a fit_window fit could be served where an
    origin_available fit was expected. Constructing the store with a namespace
    that carries the effective policy (what ``pipeline/run.py`` does for every
    pipeline-driven store) makes the two scopes resolve to DISTINCT keys, so
    each writes/reads its own file and never collides with the other.
    """
    prep = PreprocessSpec(options={"impute": "mean", "standardize": "zscore"})

    # Pre-fix-equivalent: no namespace -> scope is invisible to the key.
    bare = PreprocessorStore(tmp_path)
    assert bare.key(prep, target="Y", origin_pos=5) == bare.key(prep, target="Y", origin_pos=5)

    store_origin_available = PreprocessorStore(tmp_path, namespace={"scope": "origin_available"})
    store_fit_window = PreprocessorStore(tmp_path, namespace={"scope": "fit_window"})
    key_origin_available = store_origin_available.key(prep, target="Y", origin_pos=5)
    key_fit_window = store_fit_window.key(prep, target="Y", origin_pos=5)
    assert key_origin_available != key_fit_window

    store_origin_available.put(key_origin_available, {"marker": "origin_available"})
    store_fit_window.put(key_fit_window, {"marker": "fit_window"})

    # Each store round-trips its OWN write, and (critically) its own .key() call
    # is what a real caller uses to look things up -- so the two scopes' fits
    # never occupy the same file, however many processes share this directory.
    assert store_origin_available.get(key_origin_available) == {"marker": "origin_available"}
    assert store_fit_window.get(key_fit_window) == {"marker": "fit_window"}
    assert len(list(tmp_path.iterdir())) == 2  # two distinct on-disk entries, not one


def _scope_toy(preprocessing_policy, cache_dir):
    """A single-arm toy pipeline spec parameterized by an explicit ARM-level
    ``preprocessing_policy``, sharing ``cache_dir`` as its ``preprocessing_cache_dir``.

    Arm-level preprocessing (not spec-level) exercises the exact override branch
    ``_effective_preprocessing_policy`` resolves (``arm.preprocessing_policy``
    when ``arm.preprocessing is not None``).
    """
    idx = pd.date_range("1990-01-01", periods=160, freq="MS")
    rng = np.random.default_rng(1)
    cols = {f"S{i}": rng.normal(size=160) for i in range(6)}
    cols["Y"] = np.cumsum(rng.normal(size=160))
    panel = pd.DataFrame(cols, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, transform_codes={c: 1 for c in panel.columns})
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
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y")],
        horizons=[1],
        window=win,
        arms=[
            Arm(
                name="RIDGE",
                model="ridge",
                preprocessing=prep,
                preprocessing_policy=preprocessing_policy,
                features=feats,
                is_benchmark=True,
            ),
        ],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        preprocessing_cache_dir=str(cache_dir),
    )


def test_shared_cache_dir_does_not_cross_contaminate_across_scopes(tmp_path):
    """Two specs sharing ONE ``preprocessing_cache_dir`` but differing ONLY in
    ``preprocessing_policy.scope`` must not silently serve one scope's fit to the
    other -- the exact cross-run footgun ``cache.py``'s ``key()`` docstring warns
    about, now closed by namespacing the store at construction (pipeline/run.py).

    Proof is by FIT-CALL COUNT (via ``_count_fits``, the same instrument
    ``test_disk_store_dedupes_preprocessing_fit`` above uses), not by forecast
    values: for this toy spec (no missing data, no model_selection carving out a
    distinct validation block) ``origin_available`` and ``fit_window`` happen to
    select the identical fit rows, so numeric equality would not distinguish
    "correctly recomputed" from "wrongly reused". Fit-call counting does not have
    that ambiguity -- under the pre-fix footgun, the second (different-scope) run
    would find the first run's entries already on disk under the SAME key and
    perform ZERO real fits (a silent wrong-scope cache hit); with the namespaced
    key, the second run's keys are misses against the first run's entries, so it
    must independently recompute every origin.
    """
    cache_dir = tmp_path / "shared"
    origin_available_spec = _scope_toy(mf.window.stage_policy("origin_available"), cache_dir)
    fit_window_spec = _scope_toy(mf.window.stage_policy("fit_window"), cache_dir)

    _, fits_first = _count_fits(lambda: run_pipeline(origin_available_spec))
    assert fits_first > 0  # the first scope actually computes its origins

    # A DIFFERENT scope sharing the same directory must NOT find any cache hits
    # from the first scope's entries -- it must recompute independently.
    _, fits_second = _count_fits(lambda: run_pipeline(fit_window_spec))
    assert fits_second == fits_first, (
        f"fit_window run performed {fits_second} fits but origin_available (same "
        f"origins, same directory) performed {fits_first} -- a shortfall would "
        "mean fit_window silently reused origin_available's cached fits"
    )

    # Within-scope dedup still works: re-running the FIRST scope again against the
    # same shared directory now finds its OWN entries and performs zero new fits.
    _, fits_first_again = _count_fits(lambda: run_pipeline(origin_available_spec))
    assert fits_first_again == 0


def test_auto_cache_dir_dedupes_across_cells_when_unset(monkeypatch, tmp_path):
    """``n_jobs>1`` with ``preprocessing_cache_dir`` left at its ``None`` default
    must still perform the expensive per-origin fit AT MOST ONCE per (target,
    origin) across the whole run -- proving the run-scoped auto-managed temp dir
    (WP7 fix 2) actually engages, not just the explicit ``preprocessing_cache_dir=``
    path the other tests in this file already pin.

    The auto-created directory is removed before ``run_pipeline`` returns, so it
    is snapshotted by wrapping ``shutil.rmtree`` (the exact call
    ``pipeline/run.py::_run_cells`` uses to clean it up) rather than inspected
    after the fact.
    """
    import shutil as _shutil_mod

    spec = _toy(2)
    assert spec.preprocessing_cache_dir is None  # sanity: truly left unset

    captured: dict[str, list] = {}
    real_rmtree = _shutil_mod.rmtree

    def _snapshotting_rmtree(path, *args, **kwargs):
        p = Path(path)
        if p.is_dir():
            captured["files"] = [
                entry for entry in p.iterdir()
                if entry.is_file() and not entry.name.endswith(".tmp")
            ]
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(_shutil_mod, "rmtree", _snapshotting_rmtree)

    report = run_pipeline(spec)
    assert not report.forecasts.empty
    assert "files" in captured, (
        "auto-managed preprocessing cache dir was never created/cleaned up -- "
        "the n_jobs>1 default-to-temp-dir path did not engage"
    )

    expected = _distinct_origin_count(str(tmp_path / "control"))
    assert expected > 1  # a real dedup target

    n_arms = len(spec.arms)
    n_horizons = len(spec.horizons)
    fit_files = [p for p in captured["files"] if p.suffix == ".pkl"]
    frame_parquets = [p for p in captured["files"] if p.suffix == ".parquet"]
    frame_meta = [p for p in captured["files"] if p.suffix == ".json"]
    assert len(fit_files) == expected, (
        f"auto-managed store holds {len(fit_files)} fit entries but expected "
        f"{expected} distinct origins (naive per-cell would write up to "
        f"{expected * n_arms * n_horizons})"
    )
    assert len(frame_parquets) == expected
    assert len(frame_meta) == expected
    assert len(fit_files) < expected * n_arms * n_horizons
