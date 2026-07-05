# `macroforecast.pipeline`

A comprehensive pseudo-out-of-sample (POOS) forecasting pipeline that orchestrates
the existing `macroforecast` building blocks. You declare a set of **arms** (each a
full preprocessing/features/model configuration) and the pipeline runs them through
`forecasting.run`, evaluates every contender against a benchmark with relative RMSE,
Diebold-Mariano, Clark-West and the Model Confidence Set, builds forecast
combinations, and assembles a single `PipelineReport`. Targets are resolved from the
FRED-MD/QD t-code, and ML arms can be interpreted (SHAP/ALE/PDP) without re-running.

## Terminology

The pipeline uses four terms consistently.

- **cell** -- the execution unit the pipeline manages: one `arm` applied to one
  `target` over the window for a horizon-group. The pipeline enumerates and
  schedules cells, and each cell is executed by exactly one `run()` call. The
  serial backend groups all horizons of a (target, arm) into one cell; the
  parallel backend splits one horizon per cell.
- **run()** -- the atomic forecasting function (`forecasting.run`) that executes
  one cell (one model, one target, over the window). "run" is the function and the
  verb; the execution unit it produces is a cell, not "a run".
- **arm** -- a target-agnostic configuration (preprocessing + features + a single
  model). It is not a cell by itself; applied to a target and a horizon it forms a
  cell, and in the evaluation it appears as one contender.
- **contender** -- an arm as it appears in the evaluation comparison, the named
  entity compared via DM/CW/MCS. One arm is one contender; forecast combinations
  are additional contenders.

In one line: **arm × target × horizon → cell = one run(); arm = contender.**

## Entry points

| Symbol | Summary |
| --- | --- |
| `pipeline_spec(...)` | Validating generator that builds a `PipelineSpec`. |
| `model_arms(models, ...)` | Build one `Arm` per model for a pure model comparison. |
| `run_pipeline(spec)` | Run arms, evaluate, and return a `PipelineReport`. |
| `rescore(checkpoint_dir, spec)` | Re-score a checkpointed run from disk alone (no refitting); returns a `PipelineReport`. |
| `interpret_pipeline(report, *, methods, which_fit, arms)` | Deferred multi-method ML interpretation. |
| `run_arms(spec)` | Execute every cell into the master forecast frame (lower-level; name retained for back-compat). |
| `evaluate(master, spec)` | Accuracy + DM/CW + MCS + combinations on a master frame. |
| `apply_combinations(master, spec)` | Append cross-arm combination contenders. |
| `resolve_target(target, *, data, tcode, tcode_map, reduce_i2)` | Resolve a target's (policy, transform). |
| `contender_names(arm)` | The contender label for an arm. A contender IS exactly an arm (one model per arm), so this returns `[arm.name]`. |

## Configuration objects

- `PipelineSpec` -- validated, frozen run configuration.
- `Arm` -- a target-agnostic configuration with exactly ONE model: name, model, preprocessing, features, params, interpret. An arm is one contender in the evaluation; applied to a target and horizon it forms a cell. Comparing models means multiple arms identical except for `model`; comparing feature cases means arms differing in features. A list/mapping of models in one arm is rejected.
- `TargetSpec` -- a target and how its forecast object is defined (transform, policy, reduce_i2).
- `ResolvedTarget` -- a target with its forecast policy and transform resolved.
- `InterpretSpec` -- ML interpretation request (methods, which_fit, background, top_k).
- `EvalSpec` -- benchmark, metrics, tests, loss, primary_axis, MCS settings, subsamples. `metrics` and `tests` are live configuration (see "Custom metrics, significance tests, and loss" below), not documentation-only fields.
- `CombinationContender` -- a forecast combination that becomes an additional contender.
- `PipelineReport` -- output: forecasts, accuracy, significance, mcs, provenance, leakage_audit, interpretation, failed_cells. `failed_cells` lists any `(target, arm, horizon-group)` cell whose `run()` raised; the rest of the cells still ran and the failures are also mirrored into `leakage_audit["failed_cells"]`. `provenance` is self-certifying by default (`pipeline_spec(..., provenance_level="full")`, the default) -- see "Provenance" below.
- `TCODE_TARGET_MAP` -- FRED t-code to (forecast_policy, target_transform) mapping.

## t-code to target mapping

The forecast object is the h-period cumulation (`direct_average`) of the
transformation implied by the target's t-code, not the raw single-period transform.
I(2) price/level series are reduced to the first-difference object (`reduce_i2`).

| t-code | forecast policy | transform |
| --- | --- | --- |
| 1 | direct | level |
| 2 | direct_average | change |
| 3 | direct_average | change |
| 4 | direct | level |
| 5 | direct_average | log_growth |
| 6 | direct_average | log_growth |
| 7 | direct_average | growth |

## Example

```python
import macroforecast as mf
from macroforecast.pipeline import Arm, EvalSpec, CombinationContender, pipeline_spec, run_pipeline

spec = pipeline_spec(
    data=bundle,                                  # carries FRED transform_codes
    targets=["INDPRO", "CPIAUCSL"],               # resolved from t-codes
    horizons=[1, 3, 12],
    window=mf.window.from_cutoffs(test_start="1990-01", horizon=1, embargo=0),
    arms=[
        Arm("AR", model="ar"),                    # benchmark
        Arm("RF", model="random_forest", interpret=("shap", "ale")),
        Arm("MARX", model="ridge", preprocessing=marx_spec),   # transformation comparison
    ],
    evaluation=EvalSpec(benchmark="AR", tests=["dm", "cw", "mcs"]),
    combinations=[CombinationContender("POOL", method="constrained_ls")],
    n_jobs=8,                                     # fan (arm × target × horizon) cells across 8 processes
)
report = run_pipeline(spec)
report.accuracy        # contender x target x horizon relative RMSE (common sample)
report.significance    # Diebold-Mariano and Clark-West p-values vs the benchmark
report.mcs             # Model Confidence Set membership
report.leakage_audit   # window.validate() warnings (per horizon)

from macroforecast.pipeline import interpret_pipeline
interpret_pipeline(report, methods=("shap", "ale"))   # deferred, no re-forecast
```

## Comparing models (`model_arms`)

Since an `Arm` is exactly ONE model, a pure model comparison is "several arms
identical except `model`". Writing one `Arm` per model by hand is verbose, so
`model_arms(...)` builds them for you:

```python
from macroforecast.pipeline import model_arms, pipeline_spec, EvalSpec

arms = model_arms(["ar", "random_forest", "far"], features=feats, preprocessing=pp)
spec = pipeline_spec(..., arms=arms, evaluation=EvalSpec(benchmark="ar"))
```

Each model becomes one `Arm` (one contender; each of its (target, horizon) is one
cell run by one `run()`), and the contender is the arm name. Arm names default to the model name (`str(model)` /
`ModelSpec.name` / `callable.__name__`); pass a `Mapping[name -> model]` or
`names=[...]` to label them explicitly. All arms share the given `preprocessing`,
`features`, and evaluation config.

`params` and `model_selection` are shared by every arm unless given as a Mapping
whose keys are exactly the arm names, in which case each entry is applied to its
own arm. A plain shared dict of hyperparameters (whose keys are hyperparameter
names, not arm names) is therefore unambiguously shared. `nested_in_benchmark` is
a bool shared by all, or a set/sequence of arm names that nest the benchmark.

This shares all config, so it is for **pure model comparison**. Comparing feature
cases (arms differing in `features` or `preprocessing`) still needs explicit
`Arm` objects built by hand.

### What is held fixed vs varied (the reuse contract)

A controlled comparison changes ONE stage and holds the rest fixed. The pipeline
reuses the fixed stages instead of recomputing them per variant, and the boundary is
determined by where the varied stage sits in the `preprocessing -> features -> model`
chain.

| Comparison | Computed once and REUSED | Recomputed per variant |
| --- | --- | --- |
| **Model** (arms differ in `model` only) | data load, window/origin schedule, the per-origin preprocessing `fit` **and** `transform` (EM/factor imputation), **and** the per-origin feature-builder `fit` (PCA/MARX/SIR/etc. numerical state) | the feature `transform` (building X/y from the fitted builder) and the model fit/predict |
| **Feature** (arms differ in `features`, same `preprocessing`) | data, window/origins, per-origin preprocessing | feature build (fit + transform) + model |
| **Preprocessing** (arms differ in `preprocessing`) | data, window/origin schedule | the preprocessing itself (it is the thing being compared) + everything downstream |

The load-bearing case is **model comparison**: the dominant per-origin costs -- the
`FittedPreprocessor` fit and its EM/factor transform of the panel, AND the
feature-builder fit -- are each computed once per origin and reused across every arm
AND every horizon of the same target. So comparing `N` models over `M` horizons runs
preprocessing and feature-fitting once per origin, not `N x M` times. This reuse is
real only when the arms **share one spec-level preprocessing** -- i.e. `preprocessing=`
is set on `pipeline_spec` (or `model_arms`) and each `Arm.preprocessing is None` --
AND do not override `Arm.window` (a window override changes which rows fall in the
per-origin fit sample, so it opts out of both the preprocessing and feature-fit
reuse). An arm that carries its OWN `preprocessing` or `window` opts out and
recomputes both stages (correct for a preprocessing comparison, or a genuinely
independent per-arm window, where each variant must differ). The preprocessing reuse
is keyed on `(PreprocessSpec, target, origin_pos)` and is horizon-independent; the
feature-fit reuse is keyed on a content digest of the `FeatureSpec` (which already
captures horizon/forecast-policy-dependence -- e.g. a supervised
`sliced_inverse_regression` feature step) plus the EXACT fit-sample row-position
bounds for that origin, so it never leaks a different spec's, origin's, or arm's fit
even when the position-bounds-based key alone is asked to arbitrate (see
`macroforecast/forecasting/feature_stage.py`). The feature *transform* step (applying
the shared fit to build X/y) is NOT shared -- it is cheap relative to the fit for the
supported feature methods, and doing so safely needs additional per-arm bookkeeping
that duplicates most of the fit-sharing complexity for comparatively little gain, so
it is left recomputed per arm.

For **preprocessing comparison** the preprocessing necessarily differs per arm and is
recomputed -- that is the comparison. Everything upstream (the same `data` bundle and
`window`/origin schedule) is shared trivially, and two variants that share an
identical `(spec, target, origin)` sub-fit can still be deduplicated across processes
and runs via the on-disk `preprocessing_cache_dir` (`PreprocessorStore`).

These invariants are pinned by `tests/pipeline/test_preprocessing_share.py`
(cross-arm and on-disk dedup, serial==parallel),
`tests/pipeline/test_crosshorizon_transform_dedup.py` (the per-origin fit and heavy
transform each run exactly once per origin regardless of arm and horizon count), and
`tests/pipeline/test_feature_cache_sharing.py` (the per-origin feature-builder fit
runs exactly once per origin across arms, including with no spec-level preprocessing
at all; a per-arm `window`/`preprocessing` override correctly opts out and still
produces byte-identical forecasts; `never`/interval feature-update cadences keep
their exact single-fit semantics under sharing).

## Parallel execution (`n_jobs`)

`pipeline_spec(..., n_jobs=N)` parallelises the pseudo-out-of-sample replication
natively, so a fan-out across cores no longer needs hand-rolled shell processes.

- `n_jobs=1` (default) keeps the sequential path byte-for-byte: each `(target, arm)`
  is one cell run as a consolidated multi-horizon `run()` that shares a single
  per-origin preprocessing cache across horizons (EM imputation is computed once
  per origin).
- `n_jobs>1` splits the work into `(arm × target × horizon)` cells and runs them
  across a `ProcessPoolExecutor` with `min(n_jobs, n_cells)` workers. Each cell is a
  single-horizon `run()`, so it trades the cross-horizon EM-sharing for wall-clock
  parallelism: every cell recomputes its own preprocessing. On a many-core machine
  this finishes in a fraction of the sequential wall-clock.
- The parallel path is **deterministic**: every cell uses `spec.seed` and the
  per-cell computation is independent of sibling cells, so the assembled forecast
  frame and every downstream accuracy table are numerically identical to `n_jobs=1`.
- Workers coordinate through the existing per-`(target, arm, horizon)` checkpoints
  (`checkpoint_dir`), which are already namespaced so parallel processes never
  collide. Each worker caps nested BLAS/OpenMP threads to one to avoid
  oversubscribing cores.
- **Memory scales with `n_jobs`**: each worker holds its own copy of the data panel,
  so peak memory is roughly `n_jobs ×` the single-process footprint.

### `n_jobs="auto"`

`pipeline_spec(..., n_jobs="auto")` removes the need to hand-tune the worker count.
It inspects the core budget (the affinity count, `len(os.sched_getaffinity(0))`, which
respects cgroup/taskset pinning) and the work structure
(`len(targets) × len(arms) × len(horizons)` cells), then splits the cores between
**cell workers** and **per-cell model-internal threads** so the CPU is saturated
without oversubscription (`cell_workers × model_threads ≤ cores`). Cell-level
parallelism comes first (one worker per cell up to the core count); leftover cores
become model-internal threads handed to the parallelizable models
(`random_forest`, `gradient_boosting`, `xgboost`, `lightgbm`) inside each worker.
The single-threaded models (`ar`, `ols`, `ridge`, `lasso`, `elastic_net`, `far`)
ignore the thread budget and are unaffected. The resolved cell-worker count is stored
as `PipelineSpec.n_jobs` and the per-cell thread budget as `PipelineSpec.model_threads`.

This also fixes a latent oversubscription: previously a tree model inside a parallel
worker defaulted its internal `n_jobs` to `'auto'` (= full CPU count), so `N` workers
each spawned `cpu_count` threads. Each worker now pins its model-internal threads to
`model_threads`. The split changes only the number of internal threads, **not** the
numerical result (tree training is deterministic in `random_state` regardless of the
thread count), so a `n_jobs="auto"` run is byte-for-byte identical to `n_jobs=1`.

## Provenance (what a referee can verify from the report)

`PipelineReport.provenance` is self-certifying by default: everything a referee
needs to verify and attempt to reproduce a result lives in the report object
itself, without re-running anything or having access to the original script.

By default (`pipeline_spec(..., provenance_level="full")`, the default) the
dict has these keys:

| Key | Contents |
| --- | --- |
| `package_version`, `seed`, `targets`, `horizons`, `arms`, `benchmark`, `combinations` | The original, pre-existing fields: what ran. |
| `environment` | Git commit/branch/dirty, Python version/executable, platform string, and pinned `numpy`/`pandas`/`scipy`/`scikit-learn`/`statsmodels` versions -- **where** it ran from. |
| `data` | Dataset name/source family/declared vintage (from the `DataBundle` metadata), panel shape, date range, and a content fingerprint -- **what data** it ran on. |
| `spec_echo` | A plain, JSON-able snapshot of the resolved spec's key choices: targets/policies, horizons, window cutoffs, arms/models, benchmark, evaluation config, seed, `n_jobs`/`model_threads`, cache/checkpoint dirs -- **what was asked for**. |

```python
report = run_pipeline(spec)
report.provenance["environment"]["git"]           # {"commit": ..., "branch": ..., "dirty": ...}
report.provenance["data"]["fingerprint"]["value"]  # sha256 over the panel's index+columns+values
report.provenance["spec_echo"]["window"]["test"]   # resolved test-window cutoffs
```

`environment` reuses `output.collect_provenance` (the same probe the opt-in
save path has always used, see `docs/reference/output.md`), pointed at the
**running macroforecast package's own checkout** -- not the caller's current
working directory. From a source/editable install this resolves to that
checkout's commit/branch/dirty state regardless of what directory the
analysis script runs from; from a wheel install (no `.git` above
site-packages) the git probe fails gracefully and `commit`/`branch` are `None`
(`dirty` is not meaningful in that case -- it is not a tri-state and reports
`False` by default, matching `collect_provenance`'s existing behavior on the
save path, which this reuse does not change).

`data.fingerprint` is a sha256 digest over the panel's `DatetimeIndex` (as
int64 nanosecond timestamps), column names in order, and values cast to
explicit little-endian float64 bytes -- stable across runs and across
platforms/byte orders, and it changes if a single cell, column, or date
changes. It is computed over the FULL panel by default: measured at ~1ms for a
FRED-MD-sized panel (780 rows x 130 columns) and ~8ms for a much larger 2,000 x
400 panel, both far under the ~0.5s budget this design targets. Only above
20,000,000 cells (a size no FRED-MD/QD/SD panel macroforecast loads
approaches -- a deliberately huge synthetic 50,000 x 2,000 panel, 100,000,000
cells, took ~1.3s to fingerprint in full) does it fall back to a deterministic
strided subsample instead, and `fingerprint["method"]` says
`"strided_subsample"` (with `row_stride`/`col_stride`/`sampled_shape`) rather
than silently returning a partial-content digest labeled as full.

Pass `provenance_level="basic"` to `pipeline_spec(...)` to keep exactly the
pre-existing dict shape (`package_version`/`seed`/`targets`/`horizons`/`arms`/
`benchmark`/`combinations`, no `environment`/`data`/`spec_echo`) -- for callers
who assemble their own environment/data documentation elsewhere, or who want
to skip the git/environment probe and the one panel-fingerprint pass. This is
independent of the pre-existing `provenance=` keyword (caller-supplied notes,
e.g. the `save_models=False` + `interpret` warning merged into whichever shape
results): `provenance=` is the payload, `provenance_level` is only the
additive-blocks toggle. Forecasts and accuracy are byte-identical regardless
of `provenance_level` -- this only changes what is attached to the report, never
what is computed.

`rescore(...)` reports carry the same `environment` block (when
`provenance_level="full"`, respecting the spec's setting) plus the existing
`rescored_from` marker; `data`/`spec_echo` are not attached to a rescored
report (rescoring does not re-touch the original data or re-derive execution
choices -- the `rescored_from`/`rescore_note` fields already document that this
is a reassembly, not a live run).

## Re-scoring saved forecasts (`rescore`)

A checkpointed pipeline run (`pipeline_spec(..., checkpoint_dir=<dir>)`) persists
each cell's lean forecast records under
`<checkpoint_dir>/<target>__<arm>/h<h>/origin_<pos>.parquet` as origins complete.
Because the evaluation stage is pure-frame (`evaluate(master, spec)` needs the
forecasts, not the fitted models), those saved records are sufficient to rebuild
the full evaluation -- accuracy, DM/CW significance, MCS, and combination
contenders -- without re-running a single fit. `rescore` is that glue:

```python
spec = mf.pipeline.pipeline_spec(..., checkpoint_dir="ckpt")   # the original spec
report = mf.pipeline.rescore("ckpt", spec)                     # no refitting
report.accuracy                                                # same as the live run's
```

`rescore(checkpoint_dir, spec)` walks every `(target, arm, horizon)` cell the
spec describes, loads the persisted records, reassembles the master forecast
frame (re-attaching `arm`/`contender` labels from the spec -- the lean checkpoint
schema does not store them), and runs the standard evaluation. It returns the
same `PipelineReport` type as `run_pipeline`, with the evaluation fields
(`forecasts`, `accuracy`, `significance`, `mcs`) populated identically to a live
run over the same forecasts. The directory argument wins over whatever
`spec.checkpoint_dir` says, so a spec built without checkpointing (or a copied
checkpoint directory) re-scores fine.

Fields that require having actually executed the run are absent or best-effort:

- `interpretation` is always `None` (needs fitted models; use
  `interpret_pipeline` on a live run).
- `failed_cells` is always empty -- a cell that failed during the original run
  wrote no checkpoint files and cannot be distinguished from one that never ran.
- `empty_cells` lists arms with zero checkpoint rows across all horizons
  (failed, interrupted, or never run -- indistinguishable from disk alone).
- `provenance` / `leakage_audit` carry a `rescored_from` marker instead of a
  recomputed live audit.

An empty or wrong `checkpoint_dir` raises a `ValueError` naming the problem
(no cell directories found / directories present but all empty) rather than
returning a silently empty report.

## Custom metrics, significance tests, and loss (`EvalSpec`)

`EvalSpec.metrics` and `EvalSpec.tests` are consumed, not decorative: only the
metrics and tests actually named are computed, and unsupported names fail fast.

- **`metrics: tuple[str | Callable, ...]`** (default `("rmse", "relative_mse",
  "r2_oos")`) -- each entry is either a name resolved through
  `macroforecast.metrics.get_metric` (e.g. `"mae"`, `"mape"`, `"bias"`) or a
  callable `metric(y_true, y_pred) -> float`, named by its `__name__` for the
  accuracy-table column. `accuracy_table` computes exactly the listed metrics,
  per contender, on the same pairwise-vs-benchmark sample it always used. The
  three defaults keep their existing benchmark-relative formulas (`relative_mse`/
  `r2_oos` need the benchmark's MSE; a same-named custom callable does not
  override this) regardless of how they are requested. Passing `metrics=("mae",)`
  now returns only an `mae` column -- prior to this, `metrics` was parsed but
  never read, so every spec silently got `rmse`/`relative_mse`/`r2_oos` no matter
  what was requested.
- **`tests: tuple[str, ...]`** (default `("dm", "cw", "mcs")`) -- only the named
  tests run. `"cw"` is additionally gated by `cw_for_nested` as before (Clark-West
  only for contenders whose `Arm.nested_in_benchmark=True`). `tests=("dm",)` runs
  DM only (`significance` carries no `cw_*` columns, `mcs` is empty);
  `tests=()` yields `accuracy` only. An unsupported name (anything other than
  `"dm"`/`"cw"`/`"mcs"`) raises `ValueError` at `pipeline_spec(...)` build time
  rather than being silently dropped. `"spa"`/`"gr"` (superior predictive
  ability / Giacomini-Rossi) are implemented in `macroforecast.tests` but not yet
  wired into the pipeline evaluator -- a follow-up, not supported here.
- **`loss: Callable[[y_true, y_pred], ndarray] | None`** (default `None` = squared
  error) -- a per-observation loss threaded into the Diebold-Mariano loss
  differential and the Model Confidence Set's loss matrix, enabling asymmetric-
  loss horse races (e.g. linex, or an absolute-error DM/MCS instead of squared
  error). **Clark-West is derived under quadratic loss** and is not a valid test
  for an arbitrary loss function: when `loss` is set and CW would otherwise run
  (`"cw"` in `tests`, `cw_for_nested` true, and at least one nested contender),
  `significance_table` skips CW entirely and emits a `UserWarning` explaining why,
  instead of silently computing it against the wrong loss. DM and MCS are
  loss-agnostic and are unaffected.

```python
from macroforecast.pipeline import EvalSpec

def linex_loss(y_true, y_pred, a=1.0):
    err = y_pred - y_true
    return (np.exp(a * err) - a * err - 1.0) / (a ** 2)

evaluation = EvalSpec(
    benchmark="AR",
    metrics=("rmse", "mae", my_custom_metric),   # my_custom_metric(y_true, y_pred) -> float
    tests=("dm", "mcs"),                          # skip cw explicitly (or let the loss rule skip it)
    loss=linex_loss,                              # asymmetric-loss DM/MCS horse race
)
```

`rescore(checkpoint_dir, spec)` honors all of this automatically -- it calls the
same `evaluate(master, spec)` the live run does.

## Notes

- Accuracy uses a **common sample**: every contender is scored on the origins where
  all contenders and the realised target are observed, consistent with the MCS.
- The leakage audit validates the window for each horizon, surfacing the multi-horizon
  pseudo-out-of-sample embargo convention.
- Models are saved by default so `interpret_pipeline` runs without re-forecasting.
- `Arm.nested_in_benchmark` (default `False`) marks an arm whose model nests the
  benchmark, for example a data-poor autoregressive benchmark nested in a
  factor-augmented or penalised model. The evaluator reports a Clark-West
  statistic only for such contenders, since Clark-West is valid only when the
  benchmark is nested within the contender. Arms that do not nest the benchmark
  receive Diebold-Mariano only, and forecast combinations are never treated as
  nested.
