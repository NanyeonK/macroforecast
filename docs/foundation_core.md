# Foundation Core

`macrocast.core` is the Phase 0 foundation for the next DAG-based execution
surface. It is intentionally introduced beside the current public runtime.

## Current Authority

The existing public recipe/compiler/runtime stack remains authoritative:

- `macrocast/registry` defines public axes used by Navigator and current YAML
  recipes.
- `macrocast/compiler` compiles current recipes.
- `macrocast/execution` runs current forecasting studies.

`macrocast.core` does not replace these modules in this PR.

## New Core Contract

The foundation layer provides:

- universal DAG schema with five node types: `source`, `axis`, `step`,
  `combine`, `sink`;
- typed artifacts for cross-layer sink contracts;
- layer registration for `L0` through `L8` plus diagnostics;
- operation registration for DAG step libraries;
- sweep expansion, validation, cache hashing, YAML normalization, recipe, and
  manifest scaffolding.

## Registry Migration Plan

The legacy axis registry and the new op registry serve different roles.

- `macrocast/registry`: user-facing axis choices and current runtime
  compatibility.
- `macrocast/core/ops`: typed operation registry for future DAG execution.

Phase 1 layer work should add adapters from existing `AxisDefinition` entries
to `AxisSpec` where possible. Until those adapters exist, public axes must not
be duplicated manually in `macrocast.core`.

After L0-L8 coverage lands, the project should choose one source of truth:

1. generate `macrocast.core` layer specs from legacy registry entries, or
2. generate legacy registry entries from `macrocast.core` layer specs.

Before that decision, the safe rule is: existing public behavior stays in
`macrocast/registry`; new DAG execution contracts live in `macrocast.core`.

## Capability Counts

README capability counts refer to the current runtime, not `macrocast.core`.
Registry inspection currently shows:

- `model_family`: 30 registered values, 27 `operational`, 3
  `operational_narrow`;
- `importance_method`: 13 operational values including `none`, so 12 actual
  importance methods excluding `none`;
- Layer 6 statistical tests are split across several axes, so headline counts
  should be maintained by a dedicated capability-audit test.

The Foundation Core should not change headline runtime counts until it is wired
to execution.
