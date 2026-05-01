# Layer 0 Axis: `failure_policy`

- Parent: [Layer 0](index.md)
- Current: `failure_policy`

`failure_policy` decides how runtime handles failed cells.

## Values

| Value | Status | Meaning |
|---|---|---|
| `fail_fast` | operational, default | stop on the first cell failure |
| `continue_on_failure` | operational | record failed cells and continue remaining cells |

This axis is not sweepable. A study cannot compare failure policies as a scientific axis.

## YAML

```yaml
0_meta:
  fixed_axes:
    failure_policy: continue_on_failure
```

## Notes

- `continue_on_failure` is for large sweeps where partial completion is useful.
- Failed cells must remain visible in the manifest; do not silently drop them.
- The current contract has one continuation mode; model-level and cell-level details belong in the failure manifest.
