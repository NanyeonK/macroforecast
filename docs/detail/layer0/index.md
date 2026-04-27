# 4.1 Layer 0: Study Setup

Layer 0 decides the shape of the study before data or models are chosen. It answers: what kind of run is this, which runner owns it, how failures behave, how deterministic it should be, and how work may be parallelized.

## Decision order

| Step | Axis | Role |
|---|---|---|
| 4.1.1 | [research_design](research_design.md) | User-facing study route. |
| 4.1.2 | [experiment_unit](experiment_unit.md) | Runner unit, usually derived. |
| 4.1.3 | [failure_policy](failure_policy.md) | Runtime failure behavior. |
| 4.1.4 | [reproducibility_mode](reproducibility_mode.md) | Seed and determinism policy. |
| 4.1.5 | [compute_mode](compute_mode.md) | Serial or parallel work layout. |

`axis_type` is internal path grammar for fixed/sweep/conditional/derived choices. It is documented as an appendix because users normally express it by placing values under `fixed_axes`, `sweep_axes`, or `conditional_axes`.

## Layer contract

Input:
- user intent for study shape;
- target structure and sweep shape from the recipe when the compiler derives a runner unit.

Output:
- resolved `research_design`;
- resolved or derived `experiment_unit`;
- failure/reproducibility/compute policies recorded in manifest and runner context.

## YAML shape

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_path_benchmark
      failure_policy: fail_fast
      reproducibility_mode: seeded_reproducible
      compute_mode: serial
```

## Related reference

- [Layer 0 Meta Audit](../layer0_meta_audit.md)
- [Layer Boundary Contract](../layer_boundary_contract.md)
- [Layer Contract Ledger](../layer_contract_ledger.md)

```{toctree}
:maxdepth: 1

research_design
experiment_unit
failure_policy
reproducibility_mode
compute_mode
axis_type
```
