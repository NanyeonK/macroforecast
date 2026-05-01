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

`macrocast.core` is the source of truth for the new DAG execution system.
`macrocast/registry` will be regenerated from `macrocast.core` entries once
L0-L8 coverage lands.

Until that point, `macrocast/registry` stays frozen for backward compatibility:

- DO NOT add new axes to `macrocast/registry`.
- DO NOT modify existing axes in `macrocast/registry`.
- DO NOT remove existing axes from `macrocast/registry`.

Phase 1 layer work adds new axes only to `macrocast.core`. If current public
recipes need backward-compatible behavior during the transition, implement an
adapter at the boundary instead of changing legacy registry entries.

## Capability Counts

README capability counts refer to the legacy registry runtime, not
`macrocast.core`. Phase 0 Foundation work must not modify headline capability
counts.

The Foundation Core should not change runtime capability counts until it is
wired to execution and a dedicated capability-audit test owns those numbers.
