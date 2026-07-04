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
- `EvalSpec` -- benchmark, metrics, tests, primary_axis, MCS settings, subsamples.
- `CombinationContender` -- a forecast combination that becomes an additional contender.
- `PipelineReport` -- output: forecasts, accuracy, significance, mcs, provenance, leakage_audit, interpretation, failed_cells. `failed_cells` lists any `(target, arm, horizon-group)` cell whose `run()` raised; the rest of the cells still ran and the failures are also mirrored into `leakage_audit["failed_cells"]`.
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
| **Model** (arms differ in `model` only) | data load, window/origin schedule, and the per-origin preprocessing `fit` **and** `transform` (EM/factor imputation) | only the model fit/predict |
| **Feature** (arms differ in `features`, same `preprocessing`) | data, window/origins, per-origin preprocessing | feature build + model |
| **Preprocessing** (arms differ in `preprocessing`) | data, window/origin schedule | the preprocessing itself (it is the thing being compared) + everything downstream |

The load-bearing case is **model comparison**: the dominant per-origin cost (the
`FittedPreprocessor` fit and its EM/factor transform of the panel) is computed once
per origin and reused across every arm AND every horizon of the same target. So
comparing `N` models over `M` horizons runs the preprocessing once per origin, not
`N x M` times. This reuse is real only when the arms **share one spec-level
preprocessing** -- i.e. `preprocessing=` is set on `pipeline_spec` (or `model_arms`)
and each `Arm.preprocessing is None`. An arm that carries its OWN `preprocessing`
opts out and recomputes it (correct for a preprocessing comparison, where each
variant must differ). The reuse is keyed on `(PreprocessSpec, target, origin_pos)`
and is horizon-independent, so it never leaks a different origin's or horizon's fit.

For **preprocessing comparison** the preprocessing necessarily differs per arm and is
recomputed -- that is the comparison. Everything upstream (the same `data` bundle and
`window`/origin schedule) is shared trivially, and two variants that share an
identical `(spec, target, origin)` sub-fit can still be deduplicated across processes
and runs via the on-disk `preprocessing_cache_dir` (`PreprocessorStore`).

These invariants are pinned by `tests/pipeline/test_preprocessing_share.py`
(cross-arm and on-disk dedup, serial==parallel) and
`tests/pipeline/test_crosshorizon_transform_dedup.py` (the per-origin fit and heavy
transform each run exactly once per origin regardless of arm and horizon count).

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
