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
| `run_pipeline(spec)` | Run arms, evaluate, and return a `PipelineReport`. |
| `interpret_pipeline(report, *, methods, which_fit, arms)` | Deferred multi-method ML interpretation. |
| `run_arms(spec)` | Execute arms into the master forecast frame (lower-level). |
| `evaluate(master, spec)` | Accuracy + DM/CW + MCS + combinations on a master frame. |
| `apply_combinations(master, spec)` | Append cross-arm combination contenders. |
| `resolve_target(target, *, data, tcode, tcode_map, reduce_i2)` | Resolve a target's (policy, transform). |
| `contender_names(arm)` | Display contender labels for an arm. |

## Configuration objects

- `PipelineSpec` -- validated, frozen run configuration.
- `Arm` -- one comparison unit: name, model, preprocessing, features, params, interpret.
- `TargetSpec` -- a target and how its forecast object is defined (transform, policy, reduce_i2).
- `ResolvedTarget` -- a target with its forecast policy and transform resolved.
- `InterpretSpec` -- ML interpretation request (methods, which_fit, background, top_k).
- `EvalSpec` -- benchmark, metrics, tests, primary_axis, MCS settings, subsamples.
- `CombinationContender` -- a forecast combination that becomes an additional contender.
- `PipelineReport` -- output: forecasts, accuracy, significance, mcs, provenance, leakage_audit, interpretation.
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
)
report = run_pipeline(spec)
report.accuracy        # contender x target x horizon relative RMSE (common sample)
report.significance    # Diebold-Mariano and Clark-West p-values vs the benchmark
report.mcs             # Model Confidence Set membership
report.leakage_audit   # window.validate() warnings (per horizon)

from macroforecast.pipeline import interpret_pipeline
interpret_pipeline(report, methods=("shap", "ale"))   # deferred, no re-forecast
```

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
