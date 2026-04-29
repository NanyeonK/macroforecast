# 4.0 Layer 0: Study Scope

- Parent: [4. Detail (code): Full](../index.md)
- Current: Layer 0
- Next: [4.1 Layer 1: Data Source, Target y, Predictor x](../layer1/index.md)

Layer 0 defines the study scope and execution discipline before data or models are chosen. It answers: how many targets are in the study, whether the downstream method path is fixed or compared, how failures behave, how deterministic execution should be, and how work may be parallelized.

## Simple vs Full

Simple exposes only the first Layer 0 decision: `study_scope`. A single-method run maps to `one_target_one_method`; a Simple model comparison maps to `one_target_compare_methods`. The remaining Layer 0 policies use defaults and are still written to manifests for auditability.

Full shows all four user-facing Layer 0 decisions. You should review them in order, but you do not need to write every defaulted policy into YAML. If omitted, the compiler/runtime apply the defaults below.

## How many steps?

Layer 0 has **four user-facing steps**. Select them in this order:

1. `study_scope` — choose target cardinality and whether the method path is fixed or compared.
2. `failure_policy` — decide how failed variants or cells are handled; default is `fail_fast`.
3. `reproducibility_mode` — decide how strictly stochastic components are pinned; default is `seeded_reproducible` with seed `42`.
4. `compute_mode` — decide how execution work is laid out; default is `serial`.

There is also an internal `axis_type` grammar, but it is not a sixth user step. Users express it by placing choices under `fixed_axes`, `sweep_axes`, or `conditional_axes`.

## Decision order

| Step | Axis | Role |
|---|---|---|
| 4.0.1 | [Study Scope](study_scope.md) | Target/method cardinality for the study. |
| 4.0.2 | [Failure Handling](failure_policy.md) | Runtime failure behavior. |
| 4.0.3 | [Reproducibility](reproducibility_mode.md) | Seed and determinism policy. |
| 4.0.4 | [Compute Layout](compute_mode.md) | Serial by default, or local parallelism over a supported work unit. |

## Defaults When Omitted

| Axis | Default if omitted | Notes |
|---|---|---|
| `study_scope` | Derived when the recipe shape is sufficient; minimal one-cell recipes fall back to `one_target_one_method`. | Full users should normally set this explicitly because it drives Navigator compatibility and runner contracts. |
| `failure_policy` | `fail_fast` | Defaulted execution policy. |
| `reproducibility_mode` | `seeded_reproducible` | Defaulted execution policy; omitted seed falls back to `42`. Explicit strict or seeded modes must carry `leaf_config.random_seed`. |
| `compute_mode` | `serial` | Defaulted execution policy. |

## Selection logic

Start with `study_scope`. A one-path forecast is not a separate route; it is the one-cell case of the same `comparison_sweep` execution grammar.

- `one_target_one_method` runs one target through one fixed forecasting method path.
- `one_target_compare_methods` runs one target across one or more method alternatives.
- `multiple_targets_one_method` runs several targets through one fixed forecasting method path.
- `multiple_targets_compare_methods` runs several targets across one or more method alternatives.

Replication Library entries are ordinary YAML recipes with one of these four scopes; replication is not a Study Scope branch.

Then set policies:

- `failure_policy` is already `fail_fast` by default. Change it only when a sweep or bundle should continue after invalid cells.
- `reproducibility_mode` is already `seeded_reproducible` with seed `42` by default. Change it only for strict replication or intentionally exploratory work.
- `compute_mode` is already `serial` by default. Change it only when the selected run has multiple units of the chosen type.

## Layer contract

Input:
- user intent for study scope;
- target structure and sweep shape from the recipe when the compiler derives the scope.

Output:
- resolved or derived `study_scope`;
- `execution_route=comparison_sweep` for direct one-cell and sweep-grid work;
- failure/reproducibility/compute policies recorded in manifest and runner context.

## Canonical names

Layer 0 is canonical-only in generated recipes and Navigator paths. The compiler validates the route, runner, and policy IDs listed in this section; retired IDs are rejected instead of silently rewritten.

## YAML shape

Minimal one-cell Full YAML can specify only Study Scope:

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_one_method
```

The fully explicit equivalent records the same defaults directly:

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_one_method
      failure_policy: fail_fast
      reproducibility_mode: seeded_reproducible
      compute_mode: serial
    leaf_config:
      random_seed: 42
```

## Related reference

- [Layer 0 Meta Audit](../layer0_meta_audit.md)
- [Layer Boundary Contract](../layer_boundary_contract.md)
- [Layer Contract Ledger](../layer_contract_ledger.md)

```{toctree}
:maxdepth: 1

study_scope
failure_policy
reproducibility_mode
compute_mode
axis_type
```
