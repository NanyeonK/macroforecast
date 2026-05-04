# Layer 0 Axis: `failure_policy`

- Parent: [Layer 0](index.md)
- Current: `failure_policy`

`failure_policy` decides how execution handles failed cells. It is a runtime
control for robustness and auditability; it is not a research design axis.

## Values

| Value | Status | Meaning |
|---|---|---|
| `fail_fast` | operational, default | stop on the first cell failure |
| `continue_on_failure` | operational | record failed cells and continue remaining cells |

This axis is not sweepable. A study cannot compare failure policies as a
scientific alternative.

## Runtime Semantics

| Policy | Runtime Behavior | Typical Use |
|---|---|---|
| `fail_fast` | raise the error and stop the run | recipe development, debugging, CI |
| `continue_on_failure` | keep failed-cell metadata and run remaining cells | large sweeps, long multi-target jobs |

Failed cells must remain visible in manifests and output summaries. Continuing
after failure must not silently remove cells from denominators, rankings, or
provenance.

## YAML

Fail fast:

```yaml
0_meta:
  fixed_axes:
    failure_policy: fail_fast
```

Continue a large sweep:

```yaml
0_meta:
  fixed_axes:
    failure_policy: continue_on_failure
```

## Interaction With Other Layers

| Layer | Interaction |
|---|---|
| L4 model fitting | failed model cells are recorded rather than hidden when continuation is enabled |
| L5 evaluation | metrics should distinguish missing forecasts from valid bad forecasts |
| L8 output | exported manifests should preserve failed-cell records |

## Invalid Patterns

| Invalid Pattern | Reason |
|---|---|
| sweeping `failure_policy` | failure behavior is not a scientific treatment |
| silently dropping failed cells | breaks auditability and denominator interpretation |
| using retired values such as `warn_only` in new recipes | public L0 value is `continue_on_failure` |

## Notes

- Use `fail_fast` until the recipe is known to compile and run.
- Use `continue_on_failure` when the cost of restarting a large run is high.
- Continuation policy does not make failed outputs valid; it only preserves the
  rest of the run.
