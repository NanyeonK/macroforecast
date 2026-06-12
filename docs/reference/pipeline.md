# `macroforecast.pipeline`

A comprehensive pseudo-out-of-sample (POOS) forecasting pipeline that orchestrates
the existing `macroforecast` building blocks. You declare a set of **arms** (each a
full preprocessing/features/model configuration) and the pipeline runs them through
`forecasting.run`, evaluates every contender against a benchmark with relative RMSE,
Diebold-Mariano, Clark-West and the Model Confidence Set, builds forecast
combinations, and assembles a single `PipelineReport`. Targets are resolved from the
FRED-MD/QD t-code, and ML arms can be interpreted (SHAP/ALE/PDP) without re-running.

## Entry points

| Symbol | Summary |
| --- | --- |
| `pipeline_spec(...)` | Validating generator that builds a `PipelineSpec`. |
| `model_arms(models, ...)` | Build one `Arm` per model for a pure model comparison. |
| `run_pipeline(spec)` | Run arms, evaluate, and return a `PipelineReport`. |
| `interpret_pipeline(report, *, methods, which_fit, arms)` | Deferred multi-method ML interpretation. |
| `run_arms(spec)` | Execute arms into the master forecast frame (lower-level). |
| `evaluate(master, spec)` | Accuracy + DM/CW + MCS + combinations on a master frame. |
| `apply_combinations(master, spec)` | Append cross-arm combination contenders. |
| `resolve_target(target, *, data, tcode, tcode_map, reduce_i2)` | Resolve a target's (policy, transform). |
| `contender_names(arm)` | The contender label for an arm. A contender IS exactly an arm (one model per arm), so this returns `[arm.name]`. |

## Configuration objects

- `PipelineSpec` -- validated, frozen run configuration.
- `Arm` -- one comparison unit and exactly ONE model: name, model, preprocessing, features, params, interpret. Comparing models means multiple arms identical except for `model`; comparing feature cases means arms differing in features. A list/mapping of models in one arm is rejected.
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
    n_jobs=8,                                     # fan (arm × target × horizon) units across 8 processes
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

Each model becomes one atomic `Arm` (one `run`, one contender), and the contender
is the arm name. Arm names default to the model name (`str(model)` /
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

## Parallel execution (`n_jobs`)

`pipeline_spec(..., n_jobs=N)` parallelises the pseudo-out-of-sample replication
natively, so a fan-out across cores no longer needs hand-rolled shell processes.

- `n_jobs=1` (default) keeps the sequential path byte-for-byte: each `(arm, target)`
  is run as one consolidated multi-horizon call that shares a single per-origin
  preprocessing cache across horizons (EM imputation is computed once per origin).
- `n_jobs>1` splits the work into `(arm × target × horizon)` units and runs them
  across a `ProcessPoolExecutor` with `min(n_jobs, n_units)` workers. Each unit is a
  single-horizon run, so it trades the cross-horizon EM-sharing for wall-clock
  parallelism: every unit recomputes its own preprocessing. On a many-core machine
  this finishes in a fraction of the sequential wall-clock.
- The parallel path is **deterministic**: every unit uses `spec.seed` and the
  per-cell computation is independent of sibling cells, so the assembled forecast
  frame and every downstream accuracy table are numerically identical to `n_jobs=1`.
- Workers coordinate through the existing per-`(target, arm, horizon)` checkpoints
  (`checkpoint_dir`), which are already namespaced so parallel processes never
  collide. Each worker caps nested BLAS/OpenMP threads to one to avoid
  oversubscribing cores.
- **Memory scales with `n_jobs`**: each worker holds its own copy of the data panel,
  so peak memory is roughly `n_jobs ×` the single-process footprint.

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
