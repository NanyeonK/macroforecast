# 4.1.5 compute_mode

- Parent: [4.1 Layer 0: Study Setup](index.md)
- Previous: [4.1.4 reproducibility_mode](reproducibility_mode.md)
- Current: `compute_mode`
- Next: none inside Layer 0

`compute_mode` requests where parallel work should happen.

It is an execution-layout contract, not a modeling choice. It does not change target construction, features, models, metrics, or statistical tests. It only changes whether a supported runner uses serial loops or thread pools over a specific work unit.

## Where It Lives In Code

| Purpose | Function or object |
|---|---|
| Registry entries | `macrocast.registry.stage0.compute_mode.COMPUTE_MODE_ENTRIES` |
| Compiler runtime support gate | `macrocast.compiler.build._execution_status` |
| Manifest payload | `macrocast.compiler.build.compiled_spec_to_dict` |
| Execution payload reader | `macrocast.execution.build._compute_mode_spec` |
| Direct recipe runtime | `macrocast.execution.build.execute_recipe` |
| Horizon / OOS-date parallelism | `macrocast.execution.build._build_predictions` |
| Sweep parent policy reader | `macrocast.execution.sweep_runner._extract_parent_compute_mode` |
| Sweep model-variant parallelism | `macrocast.execution.sweep_runner.execute_sweep` |

## Choices

| Choice | Status | Work Unit | Current Runtime Behavior |
|---|---:|---|---|
| `serial` | operational | none | Default. Runs one unit at a time. |
| `parallel_by_model` | operational | sweep variants whose swept axis includes `model_family` | `execute_sweep()` uses `ThreadPoolExecutor`, capped at 4 workers, when the parent recipe selected `parallel_by_model` and there is more than one model-family variant. |
| `parallel_by_horizon` | operational | forecast horizons | `execute_recipe()` parallelizes horizon rows in `_build_predictions()` when there is more than one horizon. Capped at 4 workers. |
| `parallel_by_target` | operational | targets | `execute_recipe()` parallelizes target jobs when the recipe has more than one target. Capped at 4 workers. |
| `parallel_by_oos_date` | operational | OOS origin dates | `_build_predictions()` parallelizes origin-date model/benchmark fits after refit-policy state is computed serially. Capped at 4 workers. |
| `parallel_by_trial` | registry_only | tuning or trial units | Reserved. Not supported by the current runtime slice. |
| `distributed_cluster` | registry_only | external cluster tasks | Reserved. Not supported by the current runtime slice. |

## Compiler Contract

The compiler currently accepts only these runtime-supported values:

- `serial`;
- `parallel_by_model`;
- `parallel_by_horizon`;
- `parallel_by_target`;
- `parallel_by_oos_date`.

Selecting `parallel_by_trial` or `distributed_cluster` produces a not-supported compiler status.

The compiler writes:

```json
"compute_mode_spec": {
  "compute_mode": "serial"
}
```

The runtime reads this with `_compute_mode_spec()`.

## YAML Examples

Serial default:

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: serial
```

Model sweep parallelism:

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: controlled_variation
      compute_mode: parallel_by_model
  3_training:
    sweep_axes:
      model_family: [ridge, lasso, random_forest]
```

Multi-horizon parallelism:

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_horizon
```

Multi-target parallelism:

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_target
```

## No-Op Cases

Parallel modes are requests, not guarantees. They become no-ops when the selected work unit has only one item.

Examples:

- `parallel_by_horizon` with one horizon behaves like serial execution.
- `parallel_by_target` with one target behaves like serial execution.
- `parallel_by_model` without a model-family sweep behaves like serial sweep execution.

## Guidance

Use `serial` for debugging and replication.

Use `parallel_by_model` for model comparison sweeps.

Use `parallel_by_horizon` when horizons are independent and numerous.

Use `parallel_by_target` for multi-target recipes.

Use `parallel_by_oos_date` for long pseudo-OOS loops where origin-date fits dominate runtime.
