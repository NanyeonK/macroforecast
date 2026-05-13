# Foundation Core

`macroforecast.core` is the Phase 0 foundation for the next DAG-based execution
surface. It is intentionally introduced beside the current public runtime.

## Current Authority

The public runtime stack is:

- `macroforecast.api` — public entry points (`mf.run`, `mf.forecast`, `mf.Experiment`).
- `macroforecast.core` — recipe schema, DAG, layer contracts, and execution runtime that backs every public entry.
- `macroforecast.scaffold` — option-doc registry and encyclopedia generator that drives Navigator and `for_researchers/encyclopedia/`.

There is no separate `macroforecast.compiler` or `macroforecast.execution` module in v0.9.0; recipe parsing, validation, and execution are unified under `macroforecast.core`.

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

`macroforecast.core` is the source of truth for the new DAG execution system.
`macroforecast/registry` will be regenerated from `macroforecast.core` entries once
L0-L8 coverage lands.

Until that point, `macroforecast/registry` stays frozen for backward compatibility:

- DO NOT add new axes to `macroforecast/registry`.
- DO NOT modify existing axes in `macroforecast/registry`.
- DO NOT remove existing axes from `macroforecast/registry`.

Phase 1 layer work adds new axes only to `macroforecast.core`. If current public
recipes need backward-compatible behavior during the transition, implement an
adapter at the boundary instead of changing legacy registry entries.

## Capability Counts

README capability counts refer to the legacy registry runtime, not
`macroforecast.core`. Phase 0 Foundation work must not modify headline capability
counts.

The Foundation Core should not change runtime capability counts until it is
wired to execution and a dedicated capability-audit test owns those numbers.
