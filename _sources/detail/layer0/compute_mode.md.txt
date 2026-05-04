# Layer 0 Axis: `compute_mode`

- Parent: [Layer 0](index.md)
- Current: `compute_mode`

`compute_mode` decides whether execution runs serially or in parallel. It is a
runtime setup choice, not a scientific treatment.

## Values

| Value | Status | Meaning |
|---|---|---|
| `serial` | operational, default | run one work unit at a time |
| `parallel` | operational | parallelize over a leaf-config work unit |

This axis is not sweepable. Do not compare `serial` and `parallel` as model
alternatives.

## Leaf Config

`compute_mode: parallel` requires:

| Key | Values | Meaning |
|---|---|---|
| `parallel_unit` | `models`, `horizons`, `targets`, `oos_dates` | unit over which workers are split |
| `n_workers` | positive integer or `auto` | worker count |

`compute_mode: serial` should omit `parallel_unit` and `n_workers` unless a
wrapper explicitly ignores them.

## Work Units

| `parallel_unit` | Use When |
|---|---|
| `models` | the same data/task is evaluated across model alternatives |
| `horizons` | the same method is evaluated across forecast horizons |
| `targets` | multi-target studies can run targets independently |
| `oos_dates` | rolling/recursive origin cells dominate runtime |

Parallelism changes scheduling only. It should not change the compiled recipe,
feature construction, fitted model definition, or metric definitions.

## YAML

Serial default:

```yaml
0_meta:
  fixed_axes:
    compute_mode: serial
```

Parallel over targets:

```yaml
0_meta:
  fixed_axes:
    compute_mode: parallel
  leaf_config:
    parallel_unit: targets
    n_workers: 4
```

Parallel over out-of-sample origins:

```yaml
0_meta:
  fixed_axes:
    compute_mode: parallel
  leaf_config:
    parallel_unit: oos_dates
    n_workers: auto
```

## Invalid Patterns

| Invalid Pattern | Use Instead |
|---|---|
| `compute_mode: parallel_by_model` | `compute_mode: parallel`, `parallel_unit: models` |
| `compute_mode: parallel_by_horizon` | `compute_mode: parallel`, `parallel_unit: horizons` |
| `compute_mode: parallel_by_target` | `compute_mode: parallel`, `parallel_unit: targets` |
| `compute_mode: parallel_by_oos_date` | `compute_mode: parallel`, `parallel_unit: oos_dates` |
| sweeping `compute_mode` | keep fixed; sweep scientific axes only |

## Notes

- The public registry intentionally has only `serial` and `parallel`.
- Worker count belongs in `leaf_config`, not in the axis value.
- If parallel execution fails, failure handling is still controlled by
  `failure_policy`.
