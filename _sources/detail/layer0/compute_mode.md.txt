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

Read this axis as an execution layout request. Parallel modes only help when the selected work unit has more than one item.

### Quick Map

| Choice | Work Unit | Current State |
|---|---|---|
| `serial` | none | runnable |
| `parallel_by_model` | model variants | runnable for model-family sweeps |
| `parallel_by_horizon` | horizons | runnable for multi-horizon runs |
| `parallel_by_target` | targets | runnable for multi-target runs |
| `parallel_by_oos_date` | OOS origin dates | runnable for long pseudo-OOS loops |
| `parallel_by_trial` | tuning trials | reserved |
| `distributed_cluster` | external cluster tasks | reserved |

### `serial`

Use this for debugging, replication, and ordinary small runs.

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: serial
```

Runtime behavior:

```text
runner = execute one unit at a time
parallel workers = none
```

### `parallel_by_model`

Use this when the recipe sweeps model families.

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

Runtime behavior:

```text
runner = execute_sweep
work unit = model-family variant
executor = ThreadPoolExecutor, capped at 4 workers
```

If there is no `model_family` sweep, this behaves like serial sweep execution.

### `parallel_by_horizon`

Use this when horizons are independent and numerous.

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_horizon
  1_data_task:
    leaf_config:
      horizons: [1, 3, 6, 12]
```

Runtime behavior:

```text
runner = execute_recipe
work unit = forecast horizon
executor = ThreadPoolExecutor, capped at 4 workers
```

If there is only one horizon, this is a no-op.

### `parallel_by_target`

Use this for multi-target recipes.

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_target
  1_data_task:
    fixed_axes:
      target_structure: multi_target
    leaf_config:
      targets: [INDPRO, RPI]
```

Runtime behavior:

```text
runner = execute_recipe
work unit = target
executor = ThreadPoolExecutor, capped at 4 workers
```

If there is only one target, this is a no-op.

### `parallel_by_oos_date`

Use this when pseudo-OOS origin-date fits dominate runtime.

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_oos_date
```

Runtime behavior:

```text
runner = execute_recipe
work unit = OOS origin date
executor = ThreadPoolExecutor, capped at 4 workers
```

Refit-policy state is computed serially before origin-date model/benchmark fits are parallelized.

### `parallel_by_trial`

Reserved for tuning or trial-level parallelism.

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: parallel_by_trial
```

Current status:

```text
status = registry_only
runtime = not supported by current slice
```

### `distributed_cluster`

Reserved for external cluster execution.

```yaml
path:
  0_meta:
    fixed_axes:
      compute_mode: distributed_cluster
```

Current status:

```text
status = registry_only
runtime = not supported by current slice
```

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
