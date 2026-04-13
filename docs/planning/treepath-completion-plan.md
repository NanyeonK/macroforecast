# Tree-Path Completion Plan

## Objective

Finish the remaining work needed to align macrocast with the architecture intent of `archive/legacy-plans/source/plan_2026_04_09_2358.md`.

## Current position

Already implemented:
- architecture spine
- recipe-native compile path
- shared constructor layer
- benchmark family/options structure
- path-aware runs
- CLSS as recipe/example only

Still incomplete:
- full registries truth migration
- deeper runtime provenance semantics
- final compiler/runtime cleanup for direct tree-path semantics
- docs/plan consolidation after stabilization

## Remaining streams

1. SP-C1 registries truth completion
2. SP-C2 runtime provenance / contract deepening
3. SP-C3 final tree-path compiler cleanup
4. SP-C4 docs and migration plan consolidation
5. SP-C5 commit hygiene / checkpoint packaging

## Completion criteria

Tree-path completion will be considered sufficient when:
- registries are canonical truth for active live domains
- recipe compile path no longer materially depends on legacy grammar internals
- manifests/results encode richer tree context including fixed/sweep semantics where relevant
- docs/plans reflect active state without major drift
- package identity remains generic tree-path forecasting package first
