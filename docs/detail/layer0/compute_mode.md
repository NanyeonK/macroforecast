# Layer 0 Axis: `compute_mode`

- Parent: [Layer 0](index.md)
- Current: `compute_mode`

`compute_mode` decides whether the run is serial or parallel.

## Values

| Value | Status | Meaning |
|---|---|---|
| `serial` | operational, default | run one unit at a time |
| `parallel` | operational | parallelize over a leaf-config work unit |

This axis is not sweepable.

## Leaf Config

`compute_mode: parallel` requires:

| Key | Values |
|---|---|
| `parallel_unit` | `models`, `horizons`, `targets`, `oos_dates` |
| `n_workers` | positive integer or `auto` |

## YAML

```yaml
0_meta:
  fixed_axes:
    compute_mode: parallel
  leaf_config:
    parallel_unit: horizons
    n_workers: 4
```

## Notes

- Parallelization type is not encoded as separate top-level values.
- Use `compute_mode: parallel` plus `leaf_config.parallel_unit` rather than encoding the work unit in the axis value.
