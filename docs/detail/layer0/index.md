# 4.1 Layer 0: Study Setup

- Parent: [4. Detail (code): Full](../index.md)
- Current: Layer 0
- Next: [4.2 Layer 1: Data Task](../layer1/index.md)

Layer 0 decides the execution grammar before data or models are chosen. It answers: what unit of work is being compared or repeated, how failures behave, how deterministic execution should be, and how work may be parallelized.

## How many steps?

Layer 0 has **four user-facing steps**. Select them in this order:

1. `experiment_unit` — choose the work unit: one cell, one controlled grid, multi-target shared run, wrapper handoff, or replication handoff.
2. `failure_policy` — decide how failed variants or cells are handled.
3. `reproducibility_mode` — decide how strictly stochastic components are pinned.
4. `compute_mode` — decide whether execution is serial or parallelized over a work unit.

There is also an internal `axis_type` grammar, but it is not a sixth user step. Users express it by placing choices under `fixed_axes`, `sweep_axes`, or `conditional_axes`.

## Decision order

| Step | Axis | Role |
|---|---|---|
| 4.1.1 | [experiment_unit](experiment_unit.md) | Execution unit and runner ownership. |
| 4.1.2 | [failure_policy](failure_policy.md) | Runtime failure behavior. |
| 4.1.3 | [reproducibility_mode](reproducibility_mode.md) | Seed and determinism policy. |
| 4.1.4 | [compute_mode](compute_mode.md) | Serial or parallel work layout. |

## Selection logic

Start with `experiment_unit`. A one-path forecast is not a separate route; it is the one-cell case of the same `comparison_sweep` execution grammar.

- `single_target_single_generator` runs one target through one forecasting path.
- `single_target_generator_grid` opens a controlled comparison where one or more downstream axes are swept.
- `multi_target_shared_design` runs several targets under one shared design.
- `multi_target_separate_runs`, `benchmark_suite`, and `ablation_study` are wrapper handoff units.
- `replication_recipe` is a replication handoff and preserves source recipe provenance.

Then set policies:

- `failure_policy` matters most when a sweep or bundle can produce invalid cells.
- `reproducibility_mode` should be stricter for replication and looser for exploratory work.
- `compute_mode` is an execution request. It only has an effect when the selected run has multiple units of the chosen type.

## Layer contract

Input:
- user intent for execution unit;
- target structure and sweep shape from the recipe when the compiler derives a runner unit.

Output:
- resolved or derived `experiment_unit`;
- `execution_route=comparison_sweep` for direct one-cell and sweep-grid work;
- failure/reproducibility/compute policies recorded in manifest and runner context.

## Canonical names

Layer 0 is canonical-only in generated recipes and Navigator paths. The compiler validates the route, runner, and policy IDs listed in this section; retired IDs are rejected instead of silently rewritten.

## YAML shape

```yaml
path:
  0_meta:
    fixed_axes:
      experiment_unit: single_target_single_generator
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

experiment_unit
failure_policy
reproducibility_mode
compute_mode
axis_type
```
