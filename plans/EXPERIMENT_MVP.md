# Experiment MVP

Purpose: make the user-facing `Experiment` path real without deciding every internal axis.

## Scope

The MVP should prove this public workflow:

```python
import macrocast as mc

result = mc.forecast(
    dataset="fred_md",
    target="INDPRO",
    horizons=[1, 3, 6],
)
```

```python
exp = (
    mc.Experiment(dataset="fred_md", target="INDPRO", horizons=[1, 3, 6])
    .compare_models(["ar", "ridge", "lasso"])
)
result = exp.run()
```

## Decisions Fixed Now

- Public object name: `Experiment`
- Default profile name: `macrocast-default-v1`
- User-facing docs tracks: `docs/simple/` and `docs/detail/`
- YAML recipe/compiler/registry remain advanced/internal infrastructure
- Custom model and X-only custom preprocessor APIs are implemented
- Target transformer API is executable for `autoreg_lagged_target` and the first raw-panel slice (`ols`, `ridge`, `lasso`, `elasticnet`, or a registered custom model); broader exogenous/factor paths are deferred

## MVP Defaults

`macrocast-default-v1` currently lowers to the smallest stable runtime path:

- `research_design`: `single_path_benchmark` unless a sweep is requested
- `dataset`: user supplied
- `information_set_type`: `revised`
- `task`: `single_target_point_forecast`
- `target`: user supplied
- `horizons`: user supplied, default `[1]`
- `framework`: `expanding`
- `benchmark_family`: `zero_change`
- `feature_builder`: `autoreg_lagged_target`
- `model_family`: `ar`
- `primary_metric`: `msfe`
- `stat_test`: `none`
- `importance_method`: `none`
- preprocessing: explicit no-op contract
- `reproducibility_mode`: `seeded_reproducible`
- `failure_policy`: `fail_fast`
- `compute_mode`: `serial`
- `random_seed`: `42`

The profile is recorded in recipe leaf config and execution manifest.

## Deferred

- Complete registry axis audit
- Full result facade
- Broader target transformer runtime for exogenous/factor feature builders
- All simple sweep aliases
- Support-status rewrite
- Execution engine split
- Old docs deletion

## Next Passes

1. Add a small result facade over `ExecutionResult` and `SweepResult`.
2. Extend target transformer runtime beyond the first raw-panel slice to factor/exogenous feature builders.
3. Audit Layer 0 and Layer 1 against `Experiment`.
4. Expand simple sweep aliases only after layer audit.
5. Split model dispatch out of `execution/build.py`.
