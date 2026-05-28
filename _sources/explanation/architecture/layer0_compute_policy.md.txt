# Layer L0 Axis: `compute_policy`

- Parent: [L0 - Meta / Study Setup](layer0.md)
- Current: `compute_policy`

`compute_policy` controls whether the execution runner processes work serially or parallelizes it. It is fixed for the whole study and is not a scientific comparison axis.

## Values

| Value | Status | Meaning |
|---|---|---|
| `serial` | default | run one work unit at a time |
| `parallel` | operational | parallelize over `leaf_config.parallel_unit` |

`serial` should omit `parallel_unit` and `n_workers`. `parallel` requires both.

## Parallel Leaf Config

| Key | Values | Rule |
|---|---|---|
| `parallel_unit` | `cells`, `models`, `horizons`, `targets`, `oos_dates` | required when `compute_policy=parallel` |
| `n_workers` | positive integer or `auto` | required when `compute_policy=parallel` |

| `parallel_unit` | Use When |
|---|---|
| `cells` | independent sweep cells can run in separate processes |
| `models` | model fits inside a cell can run concurrently |
| `horizons` | horizon-specific fit/evaluation branches can run concurrently |
| `targets` | target-specific branches can run concurrently |
| `oos_dates` | walk-forward origins can run concurrently |

## YAML Examples

Serial default:

```yaml
0_meta:
  fixed_axes:
    compute_policy: serial
```

Cell-level parallelism:

```yaml
0_meta:
  fixed_axes:
    compute_policy: parallel
  leaf_config:
    parallel_unit: cells
    n_workers: 4
```

OOS-date parallelism:

```yaml
0_meta:
  fixed_axes:
    compute_policy: parallel
  leaf_config:
    parallel_unit: oos_dates
    n_workers: auto
```

## Callable Equivalent

```python
import macroforecast as mf

block = mf.l0(
    compute_policy="parallel",
    parallel_unit="cells",
    n_workers=4,
)
```

## Retired Patterns

| Retired Pattern | Current Pattern |
|---|---|
| `compute_policy: parallel_by_model` | `compute_policy: parallel`, `leaf_config.parallel_unit: models` |
| `compute_policy: parallel_by_horizon` | `compute_policy: parallel`, `leaf_config.parallel_unit: horizons` |
| `compute_policy: parallel_by_target` | `compute_policy: parallel`, `leaf_config.parallel_unit: targets` |
| `compute_policy: parallel_by_oos_date` | `compute_policy: parallel`, `leaf_config.parallel_unit: oos_dates` |
| `parallel_unit` under `fixed_axes` | `parallel_unit` under `leaf_config` |
