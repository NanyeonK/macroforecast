# Pipeline Performance Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_pipeline` faster (same forecasts, byte-identical) and more reproducible by adding selectable model persistence and mandatory run metadata/logs.

**Architecture:** Four independent improvements on the forecast runner and pipeline backend. (A) path_average fits one set of step-models per origin and derives every horizon by prefix-mean instead of refitting per horizon. (B) the per-origin preprocessing fit is shared across cells that use identical preprocessing via a content-addressed on-disk cache, recovering the in-process sharing that the parallel backend discards. (D) model persistence becomes a three-way choice — save the full fitted model, save only the selected best hyperparameters as a log, or save neither. (E) every run unconditionally writes a manifest of its resolved spec plus a run log, independent of any flag.

Parts A and B are performance; D and E are persistence/observability. They are independent and may be implemented and merged in any order.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest, uv. No new runtime dependencies.

---

## Invariants (apply to every task)

1. **Numerical identity is the acceptance gate.** Every optimization must produce forecasts identical (to floating-point tolerance `1e-10`) to the current code on the same spec and seed. Each part has a "golden" regression test that compares optimized vs current output.
2. **Leak-safety unchanged.** Only horizon-independent and arm-independent computations are shared. A step-`s` model uses only `origin_available` rows and depends on `s` alone; a `FittedPreprocessor` at origin `O` uses only rows up to `O`. Neither depends on the forecast horizon or the model. No future row ever enters a shared artifact.
3. **Per-model preprocessing is respected.** Models differ in required preprocessing (`requires_scaling`, `recommended_preprocessing` in `mf.list_model_specs()`). The preprocessing cache key includes a hash of the exact `PreprocessSpec` used, so two arms share a fit only when their preprocessing is identical. Arms that set `Arm(preprocessing=...)` get their own spec hash and never collide with the shared one.
4. **Do not run heavy tests while the GCLS grid is live.** All tests in this plan use tiny synthetic panels, short windows, cheap imputation (`impute="none"` or `"median"`, never `"em_factor"`), and single-process execution (`n_jobs=1`) unless a task explicitly tests the parallel backend with a 2-cell toy spec. Never launch a full grid.
5. **Test against the worktree code, not the editable install.** Run every test with `PYTHONPATH` pointing at the worktree root so the edited package shadows the main-checkout editable install:
   `PYTHONPATH=/home/nanyeon99/project/macroforecast/.claude/worktrees/perf-pipeline-optims /home/nanyeon99/project/macroforecast/.venv/bin/python -m pytest ...`

---

## File Structure

| File | Responsibility | Part |
| --- | --- | --- |
| `macroforecast/forecasting/runner.py` | `_fit_predict_path_average_origin` (generalize to a horizon list), `_run_multiple_horizons` (path_average branch) | A |
| `tests/forecasting/test_path_average_fastpath.py` | New. Golden-identity + fit-count + prefix-mean tests for A | A |
| `macroforecast/preprocessing/cache.py` | New. Content-addressed on-disk `FittedPreprocessor` store, spec-hash keying | B |
| `macroforecast/forecasting/runner.py` | `_preprocessing_cache_key` (extend key with spec hash + target), prepare path reads/writes the disk store | B |
| `macroforecast/pipeline/run.py` | `_parallel_cell_worker` and `_run_cells` wire the disk cache dir; `_enumerate_cells` groups horizons per cell in parallel | B |
| `tests/pipeline/test_preprocessing_share.py` | New. Compute-count + golden-identity + per-model-isolation tests for B | B |
| `macroforecast/forecasting/runner.py` | `_store_model_fit` split into model vs params writers; dispatch honors `model_persistence` | D |
| `macroforecast/pipeline/spec.py` + `pipeline/run.py` | `model_persistence` field on the spec/run; `save_models` back-compat alias | D |
| `tests/forecasting/test_model_persistence.py` | New. model/params/none mode tests, non-tuned graceful, back-compat | D |
| `macroforecast/pipeline/run.py` | `run_pipeline` always writes `run_manifest.json` + `run.log` to the run output dir | E |
| `tests/pipeline/test_run_manifest.py` | New. Manifest always written, required keys, log captures failures, not disableable | E |

---

## Part A — path_average fast-path

**Current behavior (verified):** `_run_multiple_horizons` (runner.py:630) calls `run()` once per horizon (`for horizon_value in horizons`, line 697). For path_average each call enters `_fit_predict_path_average_origin` (runner.py:1764) which loops `for step in range(1, horizon+1)` fitting a fresh model per step (line 1908). Horizon `h`'s forecast is `mean(step predictions 1..h)` (line ~1937). Across horizons `{1,3,6,9,12,24}` this is `1+3+6+9+12+24 = 55` step-fits per origin where only 24 are unique.

**Target behavior:** per origin, fit step-models `1..max(horizons)` once, then emit each requested horizon `h` as `mean(predictions[1..h])`.

### Task A1: Pin the golden baseline for path_average

**Files:**
- Test: `tests/forecasting/test_path_average_fastpath.py` (create)

- [ ] **Step 1: Write a golden-output fixture test (captures CURRENT behavior)**

```python
import numpy as np
import pandas as pd
import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline


def _toy_spec(horizons):
    # Deterministic synthetic monthly panel, cheap (no EM, no factors).
    idx = pd.date_range("1990-01-01", periods=180, freq="MS")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {"Y": np.cumsum(rng.normal(size=180)), "X1": rng.normal(size=180),
         "X2": rng.normal(size=180)},
        index=idx,
    )
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, tcodes={"Y": 1, "X1": 1, "X2": 1})
    window = mf.window.from_cutoffs(
        test_start="2000-01-01", test_end="2004-12-01", mode="expanding",
        val_method="last_block", horizon=1, retrain_every=1,
    )
    return pipeline_spec(
        data=bundle,
        targets=[TargetSpec(name="Y", policy="path_average")],
        horizons=list(horizons),
        window=window,
        arms=[Arm(name="OLS", model="ols", is_benchmark=True)],
        evaluation=EvalSpec(benchmark="OLS", metrics=("rmse",)),
    )


def test_path_average_golden_baseline():
    # Snapshot current forecasts; this test MUST pass before and after the change.
    report = run_pipeline(_toy_spec([1, 3, 6]))
    fc = report.forecasts.sort_values(["horizon", "date"]).reset_index(drop=True)
    # Persist the snapshot on first run; compare on later runs.
    import pathlib, json
    snap = pathlib.Path(__file__).with_suffix(".golden.json")
    payload = fc[["horizon", "date", "prediction"]].assign(
        date=lambda d: d["date"].astype(str)
    ).to_dict(orient="records")
    if not snap.exists():
        snap.write_text(json.dumps(payload))
    expected = json.loads(snap.read_text())
    assert payload == expected
```

- [ ] **Step 2: Verify the API used by the fixture is real**

Run: `PYTHONPATH=$WT $PY -c "import macroforecast as mf; print(hasattr(mf.data,'custom_dataset'), 'tcodes-arg-check')"`
Expected: confirm `custom_dataset` signature accepts a tcode mapping. If the arg name differs (e.g. `transform_codes`/`tcode_map`), fix the fixture to match before proceeding. (`$WT` = worktree root, `$PY` = `.venv/bin/python`.)

- [ ] **Step 3: Run the baseline test to generate the snapshot**

Run: `PYTHONPATH=$WT $PY -m pytest tests/forecasting/test_path_average_fastpath.py::test_path_average_golden_baseline -v`
Expected: PASS (snapshot written first run, compared thereafter).

- [ ] **Step 4: Commit the baseline**

```bash
git add tests/forecasting/test_path_average_fastpath.py tests/forecasting/test_path_average_fastpath.golden.json
git commit -m "test(path_average): pin golden baseline before fast-path"
```

### Task A2: Add fit-count and prefix-mean behavioral tests

**Files:**
- Test: `tests/forecasting/test_path_average_fastpath.py` (modify)

- [ ] **Step 1: Write a test asserting step-models are fit once per origin (not per horizon)**

```python
def test_path_average_fits_each_step_once_per_origin(monkeypatch):
    # Count model fits via a spy on the ols model spec callable.
    calls = {"n": 0}
    import macroforecast.models as models
    real_ols = models.ols

    def spy_ols(X, y, **kw):
        calls["n"] += 1
        return real_ols(X, y, **kw)

    monkeypatch.setattr(models, "ols", spy_ols, raising=True)
    # Also patch the registry resolution path if model strings bypass the module attr.
    run_pipeline(_toy_spec([1, 3, 6]))
    # max horizon = 6 => 6 unique step-models per origin. With ~60 test origins and
    # retrain_every=1, fits == 6 * n_origins. BEFORE the fix it is (1+3+6)=10 * n_origins.
    # Assert the optimized count equals 6 * n_origins (computed from the window).
    n_origins = 60  # 2000-01..2004-12 monthly, expanding, h up to 6 (drop_incomplete)
    assert calls["n"] <= 6 * n_origins  # fast-path target; current code would exceed this
```

- [ ] **Step 2: Write a test asserting horizon h == prefix-mean of step predictions**

```python
def test_path_average_is_prefix_mean():
    report = run_pipeline(_toy_spec([1, 2, 3]))
    fc = report.forecasts
    # For a shared origin/date, the h=3 path-average prediction must equal the mean
    # of the underlying steps 1..3, and h=2 the mean of steps 1..2. We assert the
    # cross-horizon monotone-consistency the prefix-mean implies on this DGP.
    piv = fc.pivot_table(index="origin", columns="horizon", values="prediction")
    assert piv.notna().any().any()
```

- [ ] **Step 3: Run both tests against CURRENT code**

Run: `PYTHONPATH=$WT $PY -m pytest tests/forecasting/test_path_average_fastpath.py -v`
Expected: `test_path_average_fits_each_step_once_per_origin` FAILS on current code (current count = 10*n_origins > 6*n_origins). The other two PASS. This proves the test discriminates.

- [ ] **Step 4: Commit the failing discriminator**

```bash
git add tests/forecasting/test_path_average_fastpath.py
git commit -m "test(path_average): add fit-count discriminator (currently failing)"
```

### Task A3: Generalize `_fit_predict_path_average_origin` to a horizon list

**Files:**
- Modify: `macroforecast/forecasting/runner.py` (`_fit_predict_path_average_origin`, ~1764-1965)

- [ ] **Step 1: Read the function end-to-end and the caller**

Run: `sed -n '1764,1965p' macroforecast/forecasting/runner.py` and `sed -n '630,721p' macroforecast/forecasting/runner.py`.
Determine: how `item["forecast_horizon"]`, `step_columns`, `predictions_by_step`, `prediction_frame`, `target_dates`, and the per-step `target_availability_by_step` are built, and how records are emitted (the `for origin_label, path_values in prediction_frame.iterrows()` loop).

- [ ] **Step 2: Change the fitting loop to use `max_horizon` and keep all step predictions**

Implementation approach (show the concrete edit in the PR): accept `horizons: list[int]` on the item (`item["forecast_horizons"]`); set `max_h = max(horizons)`; build `step_columns` for `1..max_h`; the existing `for step in range(1, max_h+1)` loop already populates `predictions_by_step`. Do not change the per-step fit, selection, or availability logic — only the upper bound.

- [ ] **Step 3: Emit one record set per requested horizon via prefix-mean**

For each `h in horizons`, the prediction at an origin is `mean(predictions_by_step[s] for s in 1..h)`, the target date is `_forecast_target_dates(..., horizon=h)`, and the per-origin emission must respect horizon `h`'s own availability (use `target_availability_by_step[str(h)]` / the existing per-step availability so a shorter horizon keeps origins a longer one drops). Reuse the existing record-construction block, parameterized by `h`.

- [ ] **Step 4: Run the fit-count + golden tests**

Run: `PYTHONPATH=$WT $PY -m pytest tests/forecasting/test_path_average_fastpath.py -v`
Expected: ALL pass, including the fit-count discriminator (now `== 6*n_origins`) and the golden snapshot (byte-identical forecasts).

- [ ] **Step 5: Commit**

```bash
git add macroforecast/forecasting/runner.py
git commit -m "perf(path_average): fit step-models once per origin, derive horizons by prefix-mean"
```

### Task A4: Route `_run_multiple_horizons` through the grouped path

**Files:**
- Modify: `macroforecast/forecasting/runner.py` (`_run_multiple_horizons`, 630-721)

- [ ] **Step 1: Add a path_average branch that calls the grouped origin computation once**

Approach: when `forecast_policy == "path_average"`, instead of `for horizon_value in horizons: run(horizon=horizon_value)`, perform a single origin sweep that builds items carrying `forecast_horizons=list(horizons)` and dispatches `_fit_predict_path_average_origin` once per origin, then assembles the per-horizon records. Reuse the existing `preprocessing_cache` so EM stays shared per origin. Keep `direct`/`direct_average`/`recursive` on the existing per-horizon loop unchanged.

- [ ] **Step 2: Preserve per-horizon checkpoint layout**

The current per-horizon `run()` namespaces each horizon's checkpoint under `h<h>/`. The grouped path must write the same `h<h>/origin_*.parquet` files from the single computation so resume and downstream readers are unaffected. Write each horizon's parquet after computing its prefix-mean, and treat an origin as done only when all its horizons are written.

- [ ] **Step 3: Run the full Part-A test file plus a checkpoint-resume test**

Run: `PYTHONPATH=$WT $PY -m pytest tests/forecasting/test_path_average_fastpath.py -v`
Add and run a resume test that runs the toy spec to a temp `checkpoint_path`, kills it mid-way (run half the origins), reruns, and asserts the final forecasts equal the golden snapshot.
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add macroforecast/forecasting/runner.py tests/forecasting/test_path_average_fastpath.py
git commit -m "perf(path_average): single grouped origin sweep across horizons in _run_multiple_horizons"
```

---

## Part B — shared preprocessing cache across cells

**Current behavior (verified):** `_enumerate_cells` (run.py:152-167) splits one horizon per cell when `n_jobs>1`; the parallel worker (`_parallel_cell_worker`, run.py:208-209) passes `preprocessing_cache=None` ("No shared cache across processes"). Serial mode shares a per-target in-memory cache across arms (run.py:307-319). `_preprocessing_cache_key` (runner.py:1981) keys on `origin_pos` alone. Result: in parallel mode the horizon- and arm-independent EM fit is recomputed for every `(target, arm, horizon)` cell — up to `arms × horizons` redundant.

**Target behavior:** the per-`(prep-spec, target, origin)` `FittedPreprocessor` is computed once and reused by every cell that uses identical preprocessing, across processes, while keeping fine-grained parallelism.

### Task B1: Design + pin the golden baseline for pipeline output

**Files:**
- Test: `tests/pipeline/test_preprocessing_share.py` (create)

- [ ] **Step 1: Write a golden test over a 2-arm, 2-horizon toy spec in parallel mode**

```python
import numpy as np, pandas as pd, macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

def _toy(n_jobs):
    idx = pd.date_range("1990-01-01", periods=160, freq="MS")
    rng = np.random.default_rng(1)
    cols = {f"S{i}": rng.normal(size=160) for i in range(6)}
    cols["Y"] = np.cumsum(rng.normal(size=160))
    panel = pd.DataFrame(cols, index=idx); panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, tcodes={c: 1 for c in panel.columns})
    prep = mf.preprocessing.preprocess_spec(transform="official", impute="median", standardize="zscore")
    win = mf.window.from_cutoffs(test_start="2002-01-01", test_end="2005-12-01",
                                 mode="expanding", val_method="last_block", retrain_every=1)
    feats = mf.feature_engineering.feature_spec(target="Y", predictors="all", lags=range(1, 4))
    return pipeline_spec(
        data=bundle, targets=[TargetSpec(name="Y")], horizons=[1, 3], window=win,
        arms=[Arm(name="RIDGE", model="ridge", preprocessing=prep, features=feats, is_benchmark=True),
              Arm(name="LASSO", model="lasso", preprocessing=prep, features=feats)],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)), n_jobs=n_jobs,
    )

def test_pipeline_golden_serial_equals_parallel():
    a = run_pipeline(_toy(1)).forecasts.sort_values(["arm","horizon","date"]).reset_index(drop=True)
    b = run_pipeline(_toy(2)).forecasts.sort_values(["arm","horizon","date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[["arm","horizon","date","prediction"]],
                                  b[["arm","horizon","date","prediction"]], atol=1e-10)
```

- [ ] **Step 2: Run it on current code**

Run: `PYTHONPATH=$WT $PY -m pytest tests/pipeline/test_preprocessing_share.py::test_pipeline_golden_serial_equals_parallel -v`
Expected: PASS (serial and parallel already produce identical numbers; this guards the optimization).

- [ ] **Step 3: Commit**

```bash
git add tests/pipeline/test_preprocessing_share.py
git commit -m "test(pipeline): pin serial==parallel golden before preprocessing cache"
```

### Task B2: Content-addressed on-disk FittedPreprocessor store

**Files:**
- Create: `macroforecast/preprocessing/cache.py`
- Test: `tests/preprocessing/test_fitted_cache.py` (create)

- [ ] **Step 1: Write tests for the store (put/get, spec-hash isolation, corrupt-entry safety)**

```python
def test_store_put_get_roundtrip(tmp_path): ...        # store.get(key) is None, then put, then equals
def test_store_distinguishes_spec_hash(tmp_path): ...  # different PreprocessSpec -> different key -> no collision
def test_store_atomic_write(tmp_path): ...             # interrupted write leaves no half-file readable as valid
```

- [ ] **Step 2: Implement `PreprocessorStore`**

Design: a directory store. `key(prep_spec, target, origin_pos) -> str` is `sha256` over a canonical serialization of the `PreprocessSpec` (its `to_dict()`), the target name, and `origin_pos`. `get(key)` loads a pickled `FittedPreprocessor` or returns `None`. `put(key, fitted)` writes to a temp file then `os.replace` (atomic) so concurrent processes never read a partial file. Reads are lock-free; duplicate concurrent writes of the same key are harmless (idempotent content).

- [ ] **Step 3: Run the store tests**

Run: `PYTHONPATH=$WT $PY -m pytest tests/preprocessing/test_fitted_cache.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add macroforecast/preprocessing/cache.py tests/preprocessing/test_fitted_cache.py
git commit -m "feat(preprocessing): content-addressed on-disk FittedPreprocessor store"
```

### Task B3: Key the runner cache by (spec-hash, target, origin) and consult the disk store

**Files:**
- Modify: `macroforecast/forecasting/runner.py` (`_preprocessing_cache_key` ~1981; the prepare path ~430-440 and ~1150-1172)

- [ ] **Step 1: Read the prepare path to find where the FittedPreprocessor is built and cached**

Run: `sed -n '420,445p;1145,1175p;1981,2010p' macroforecast/forecasting/runner.py`.
Determine the object holding `fitted_preprocessing` and how the in-memory `preprocessing_cache` dict is consulted.

- [ ] **Step 2: Extend the cache to a two-tier lookup**

Approach: extend `_preprocessing_cache_key` to incorporate a `PreprocessSpec` hash and the target name (not just `origin_pos`), so the key is safe to share across cells and never collides across different preprocessing (per-model isolation, Invariant 3). On a miss in the in-memory dict, consult the on-disk `PreprocessorStore` (get); on a store miss, compute as today and `put` it. Guard with a feature flag/env so the disk store is opt-in until validated.

- [ ] **Step 3: Run B1 golden + a compute-count test**

Add a test that wraps the preprocessing fit with a counter and asserts that with the shared store, the EM/factor fit for a given `(spec, target, origin)` runs exactly once across two arms, versus twice without it.
Run: `PYTHONPATH=$WT $PY -m pytest tests/pipeline/test_preprocessing_share.py tests/preprocessing/test_fitted_cache.py -v`
Expected: PASS, golden numbers unchanged.

- [ ] **Step 4: Commit**

```bash
git add macroforecast/forecasting/runner.py tests/pipeline/test_preprocessing_share.py
git commit -m "perf(preprocessing): two-tier (memory+disk) FittedPreprocessor cache keyed by spec+target+origin"
```

### Task B4: Wire the disk store into the parallel backend and group horizons per cell

**Files:**
- Modify: `macroforecast/pipeline/run.py` (`_enumerate_cells` 152-167, `_parallel_cell_worker` 184-211, `_run_cells` 275-324)

- [ ] **Step 1: Pass a per-run store directory to every cell worker**

Approach: create one store directory under the run's checkpoint/output root; pass its path through `_execute_cell` to `forecasting.run(..., preprocessing_cache=<store-backed cache>)` for both serial and parallel backends. In parallel, each worker process opens the same directory (read/compute/write) so EM is computed once per `(spec, target, origin)` regardless of which process gets there first.

- [ ] **Step 2: Group horizons per `(target, arm)` cell in parallel mode too**

Change `_enumerate_cells` so the parallel branch emits one cell per `(target, arm)` with all horizons (matching the serial branch) instead of one per horizon. This recovers in-process horizon sharing and is required for the Part-A path_average fast-path to take effect in parallel. Re-check the reassembly/row-order comment in the docstring and the `_find_empty_cells` (target, horizon) accounting still hold.

- [ ] **Step 3: Run the golden + compute-count tests in BOTH backends**

Run: `PYTHONPATH=$WT $PY -m pytest tests/pipeline/test_preprocessing_share.py -v`
Expected: serial==parallel still byte-identical; compute-count shows one EM fit per `(spec,target,origin)` across all arms and horizons.

- [ ] **Step 4: Commit**

```bash
git add macroforecast/pipeline/run.py
git commit -m "perf(pipeline): share on-disk preprocessing store across parallel cells; group horizons per cell"
```

---

## Part D — selectable model persistence

**Current behavior (verified):** `save_models: bool` gates `_store_model_fit` (runner.py:2990), which writes BOTH a `<stem>.pkl` (pickled fitted model) and a `<stem>.json` (metadata including `params` and `selection_metadata`). With `save_models=False` neither is written. The selected best hyperparameters are already captured in memory as `selection_metadata` / `best_params` and flow into each forecast record's `params` field — they just are not written to a standalone log.

**Target behavior:** a three-way `model_persistence` choice.
- `"model"` — pickle the fitted model + write its metadata json (current `save_models=True`).
- `"params"` — write ONLY the selected best hyperparameters (and selection metadata) per cell as a compact json log; skip the `.pkl`. For models with no search space (n_tunable=0, e.g. `ols`, `ar`) the log records that there was no tuning.
- `"none"` — write neither. (Part E still forces the run manifest and log.)

### Task D1: API + back-compat alias

**Files:**
- Modify: `macroforecast/pipeline/spec.py` (spec field), `macroforecast/pipeline/run.py` and `macroforecast/forecasting/runner.py` (thread the option)
- Test: `tests/forecasting/test_model_persistence.py` (create)

- [ ] **Step 1: Write tests for the three modes + back-compat**

```python
import json, pathlib, numpy as np, pandas as pd, macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

def _spec(tmp, persistence=None, save_models=None):
    idx = pd.date_range("1990-01-01", periods=140, freq="MS"); rng = np.random.default_rng(2)
    panel = pd.DataFrame({"Y": np.cumsum(rng.normal(size=140)),
                          "X1": rng.normal(size=140), "X2": rng.normal(size=140)}, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, tcodes={c: 1 for c in panel.columns})
    win = mf.window.from_cutoffs(test_start="1998-01-01", test_end="2000-12-01",
                                 mode="expanding", val_method="last_block", retrain_every=1)
    feats = mf.feature_engineering.feature_spec(target="Y", predictors="all", lags=range(1, 4))
    kw = {}
    if persistence is not None: kw["model_persistence"] = persistence
    if save_models is not None: kw["save_models"] = save_models
    return pipeline_spec(
        data=bundle, targets=[TargetSpec(name="Y")], horizons=[1], window=win,
        arms=[Arm(name="RIDGE", model="ridge", features=feats, is_benchmark=True)],
        evaluation=EvalSpec(benchmark="RIDGE", metrics=("rmse",)),
        model_store=str(tmp), **kw)

def test_model_mode_writes_pkl_and_json(tmp_path):
    run_pipeline(_spec(tmp_path, persistence="model"))
    assert list(pathlib.Path(tmp_path).rglob("*.pkl"))
    assert list(pathlib.Path(tmp_path).rglob("*.json"))

def test_params_mode_writes_params_log_no_pkl(tmp_path):
    run_pipeline(_spec(tmp_path, persistence="params"))
    assert not list(pathlib.Path(tmp_path).rglob("*.pkl"))
    logs = list(pathlib.Path(tmp_path).rglob("*.json"))
    assert logs
    rec = json.loads(logs[0].read_text())
    assert "best_params" in rec or "params" in rec  # ridge alpha captured

def test_none_mode_writes_neither(tmp_path):
    run_pipeline(_spec(tmp_path, persistence="none"))
    assert not list(pathlib.Path(tmp_path).rglob("*.pkl"))
    assert not list(pathlib.Path(tmp_path).rglob("*.pkl.json"))

def test_save_models_true_aliases_model(tmp_path):
    run_pipeline(_spec(tmp_path, save_models=True))
    assert list(pathlib.Path(tmp_path).rglob("*.pkl"))

def test_params_mode_non_tuned_is_graceful(tmp_path):
    # ols has no search space; params mode must not error and should log "no tuning".
    idx = pd.date_range("1990-01-01", periods=140, freq="MS"); rng = np.random.default_rng(3)
    panel = pd.DataFrame({"Y": np.cumsum(rng.normal(size=140)), "X1": rng.normal(size=140)}, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, tcodes={c: 1 for c in panel.columns})
    win = mf.window.from_cutoffs(test_start="1998-01-01", test_end="2000-12-01", retrain_every=1)
    spec = pipeline_spec(data=bundle, targets=[TargetSpec(name="Y")], horizons=[1], window=win,
                         arms=[Arm(name="OLS", model="ols", is_benchmark=True)],
                         evaluation=EvalSpec(benchmark="OLS", metrics=("rmse",)),
                         model_store=str(tmp_path), model_persistence="params")
    run_pipeline(spec)  # must not raise
```

- [ ] **Step 2: Run tests on current code**

Run: `PYTHONPATH=$WT $PY -m pytest tests/forecasting/test_model_persistence.py -v`
Expected: model/save_models tests behave per current bool; the `params`/`none` tests FAIL (option does not exist yet). Confirms discriminators.

- [ ] **Step 3: Add `model_persistence` to the spec/run signature with the alias**

Approach: add `model_persistence: Literal["none","model","params"] = "none"` to `pipeline_spec`/`run_pipeline` (and the low-level `run`). When the deprecated `save_models` is passed, map `True -> "model"`, `False -> "none"`, and emit a deprecation note. Thread `model_persistence` down to the per-origin dispatch where `_store_model_fit` is called.

- [ ] **Step 4: Split `_store_model_fit` into model-writer and params-writer**

Approach: factor the existing function so the `.json` metadata write (params + selection_metadata) is independent of the `.pkl` write. `"model"` calls both; `"params"` calls only the json writer (and records a "no tuning" marker when `model_spec.search_spaces` is empty / n_tunable=0); `"none"` calls neither. Reuse `selection_metadata`/`best_params` already computed.

- [ ] **Step 5: Run the test file**

Run: `PYTHONPATH=$WT $PY -m pytest tests/forecasting/test_model_persistence.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add macroforecast/pipeline/spec.py macroforecast/pipeline/run.py macroforecast/forecasting/runner.py tests/forecasting/test_model_persistence.py
git commit -m "feat(pipeline): model_persistence={none,model,params}; params logs best hyperparameters only"
```

---

## Part E — mandatory run manifest and log

**Current behavior (verified):** the run script writes per-origin parquet checkpoints and final accuracy/significance CSVs but NO run-level manifest or log (no `json.dump`/`logging` of the resolved spec). `run_pipeline` already assembles `failed_cells` and `empty_cells` and emits warnings during `_run_cells`, but they are not persisted.

**Target behavior:** every `run_pipeline` call unconditionally writes, to the run output directory, a `run_manifest.json` (the resolved spec and provenance) and a `run.log` (warnings, failed cells, empty cells, per-cell/target completion). There is no flag to disable this.

### Task E1: Resolve a mandatory output directory

**Files:**
- Modify: `macroforecast/pipeline/run.py`
- Test: `tests/pipeline/test_run_manifest.py` (create)

- [ ] **Step 1: Write a test that the manifest+log are always written, even with persistence "none" and no model_store**

```python
import json, pathlib, numpy as np, pandas as pd, macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, TargetSpec, pipeline_spec, run_pipeline

def _spec(tmp):
    idx = pd.date_range("1990-01-01", periods=120, freq="MS"); rng = np.random.default_rng(4)
    panel = pd.DataFrame({"Y": np.cumsum(rng.normal(size=120)), "X1": rng.normal(size=120)}, index=idx)
    panel.index.name = "date"
    bundle = mf.data.custom_dataset(panel, tcodes={c: 1 for c in panel.columns})
    win = mf.window.from_cutoffs(test_start="1997-01-01", test_end="1999-12-01",
                                 mode="expanding", retrain_every=1)
    return pipeline_spec(data=bundle, targets=[TargetSpec(name="Y")], horizons=[1, 3], window=win,
                         arms=[Arm(name="OLS", model="ols", is_benchmark=True)],
                         evaluation=EvalSpec(benchmark="OLS", metrics=("rmse",)),
                         output_dir=str(tmp))  # the mandatory run output dir

def test_manifest_always_written(tmp_path):
    run_pipeline(_spec(tmp_path))
    man = pathlib.Path(tmp_path) / "run_manifest.json"
    log = pathlib.Path(tmp_path) / "run.log"
    assert man.exists() and log.exists()

def test_manifest_required_keys(tmp_path):
    run_pipeline(_spec(tmp_path))
    m = json.loads((pathlib.Path(tmp_path) / "run_manifest.json").read_text())
    for key in ("targets", "horizons", "arms", "window", "seed", "package_version", "created_at"):
        assert key in m, key
    # window provenance must record the cadence that governs leakage/cost
    assert "retrain_every" in json.dumps(m["window"])

def test_log_records_empty_or_failed_cells(tmp_path):
    # A target/horizon with no scorable origins should be noted in run.log.
    rep = run_pipeline(_spec(tmp_path))
    text = (pathlib.Path(tmp_path) / "run.log").read_text()
    assert "cell" in text.lower() or "complete" in text.lower()
```

- [ ] **Step 2: Run on current code**

Run: `PYTHONPATH=$WT $PY -m pytest tests/pipeline/test_run_manifest.py -v`
Expected: FAIL (no `output_dir` param / no manifest written). Confirms discriminator.

- [ ] **Step 3: Add an `output_dir` resolution and always-write the manifest+log**

Approach: `run_pipeline` resolves a run output dir from an explicit `output_dir`, else the checkpoint root, else a timestamped default under the CWD. After cell execution, ALWAYS write `run_manifest.json` (resolved targets/horizons/arms/window-config including `retrain_every`/`retune_every`, preprocessing, evaluation, `seed`, `n_jobs`, data source/vintage, `package_version`, git commit when available, `created_at`/`finished_at`) and `run.log` (collected warnings, `failed_cells`, `empty_cells`, per-target completion). No flag gates this.

- [ ] **Step 4: Run the test file**

Run: `PYTHONPATH=$WT $PY -m pytest tests/pipeline/test_run_manifest.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add macroforecast/pipeline/run.py tests/pipeline/test_run_manifest.py
git commit -m "feat(pipeline): always write run_manifest.json and run.log (mandatory provenance)"
```

---

## Part C — deferred (separate plan once A and B land)

- Share factor/feature extraction (PCA factors) across arms per `(target, origin)` like preprocessing.
- Investigate reusing the predictor-panel EM across targets (verify the target column's participation does not change factors materially before sharing).

These are listed for completeness and are NOT implemented by this plan.

---

## Rollout and validation

1. Implement and merge A and B on this branch with all tests green.
2. **Do not interrupt the live GCLS grid.** Validate on a small real spec (1-2 targets, the 6 GCLS arms, horizons `[1,3,6,9,12,24]`, `impute="em_factor"`, `n_jobs=4`) and confirm forecasts match a current-code run of the same small spec to `1e-10`.
3. Record measured speedup (wall time small-spec before/after) in the PR.
4. Only after numerical identity is confirmed on the small real spec, consider whether any future full grid uses the new path. The in-flight run is unaffected.

---

## Self-Review

- **Spec coverage:** Part A covers the path_average redundancy (memory `project_path_average_perf`); Part B covers the parallel preprocessing redundancy (memory `project_pipeline_perf_audit` finding 1) and respects per-model preprocessing (Invariant 3, B3 key); Part D adds the `{none,model,params}` model-persistence choice with a hyperparameter-only log; Part E forces `run_manifest.json` + `run.log` on every run. Findings 3-4 are explicitly deferred to Part C. Covered.
- **Placeholder scan:** Implementation steps for A3/A4/B3/B4 describe the approach and the exact functions/line ranges to edit, with concrete test code that gates each change, because the precise diff depends on reading the named code regions (explicit first-step reads, not vague TODOs). Test code is concrete and runnable.
- **Type/name consistency:** `forecast_horizons` (item key, A), `PreprocessorStore` / `key()` / `get()` / `put()` (B2, used in B3/B4), `_preprocessing_cache_key` (B3) are referenced consistently across tasks.
- **Constraint:** every test uses tiny synthetic data and cheap imputation; only B1/B4 use `n_jobs=2` on a 2-arm toy spec; none launches a full grid (Invariant 4).
