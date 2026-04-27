# 4.1 Layer 0: Study Setup

- Parent: [4. Detail (code): Full](../index.md)
- Current: Layer 0
- Next: [4.2 Layer 1: Data Task](../layer1/index.md)

Layer 0 decides the shape of the study before data or models are chosen. It answers: what kind of run is this, which runner owns it, how failures behave, how deterministic it should be, and how work may be parallelized.

## How many steps?

Layer 0 has **five user-facing steps**. Select them in this order:

1. `research_design` — choose the study route.
2. `experiment_unit` — confirm the runner unit, usually derived by the compiler.
3. `failure_policy` — decide how failed variants or cells are handled.
4. `reproducibility_mode` — decide how strictly stochastic components are pinned.
5. `compute_mode` — decide whether execution is serial or parallelized over a work unit.

There is also an internal `axis_type` grammar, but it is not a sixth user step. Users express it by placing choices under `fixed_axes`, `sweep_axes`, or `conditional_axes`.

## Decision order

| Step | Axis | Role |
|---|---|---|
| 4.1.1 | [research_design](research_design.md) | User-facing study route. |
| 4.1.2 | [experiment_unit](experiment_unit.md) | Runner unit, usually derived. |
| 4.1.3 | [failure_policy](failure_policy.md) | Runtime failure behavior. |
| 4.1.4 | [reproducibility_mode](reproducibility_mode.md) | Seed and determinism policy. |
| 4.1.5 | [compute_mode](compute_mode.md) | Serial or parallel work layout. |

## Selection logic

Start with `research_design`. That decision constrains or derives the runner shape:

- `single_forecast_run` normally derives `experiment_unit=single_target_single_generator` unless the target/sweep shape requires another supported unit.
- `controlled_variation` is used when one or more axes are swept while the rest of the path is held fixed.
- `study_bundle` is a wrapper route and only opens when a concrete wrapper runner contract exists.
- `replication_recipe` routes to a replication-style runner and should preserve recipe provenance.

Then set policies:

- `failure_policy` matters most when a sweep or bundle can produce invalid cells.
- `reproducibility_mode` should be stricter for replication and looser for exploratory work.
- `compute_mode` is an execution request. It only has an effect when the selected run has multiple units of the chosen type.

## Layer contract

Input:
- user intent for study shape;
- target structure and sweep shape from the recipe when the compiler derives a runner unit.

Output:
- resolved `research_design`;
- resolved or derived `experiment_unit`;
- failure/reproducibility/compute policies recorded in manifest and runner context.

## Naming migration

Layer 0 now uses clearer canonical IDs. Older recipe IDs are still accepted at
compile time and normalized before registry validation:

| Legacy ID | Canonical ID |
|---|---|
| `single_path_benchmark` | `single_forecast_run` |
| `orchestrated_bundle` | `study_bundle` |
| `replication_override` | `replication_recipe` |
| `single_target_single_model` | `single_target_single_generator` |
| `single_target_model_grid` | `single_target_generator_grid` |

New docs and generated Navigator data use the canonical IDs. Keep legacy IDs
only for old recipes that have not yet been rewritten.

## YAML shape

```yaml
path:
  0_meta:
    fixed_axes:
      research_design: single_forecast_run
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
