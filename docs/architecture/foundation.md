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

## Source of Truth

`macroforecast.core` is the live source of truth for all layer specifications
and op registries. The migration from a legacy registry subsystem is complete.
Layer contracts, op enumerations, and sweep machinery all live under
`macroforecast.core`.
