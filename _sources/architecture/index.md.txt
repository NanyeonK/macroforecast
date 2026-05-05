# Architecture

The architecture pages document **what each layer owns** and **how layers
hand artifacts to each other**. Read these when you want to understand
the canonical contracts before authoring recipes, extending the runtime,
or auditing manifests.

## Canonical layer flow

```text
L0 -> L1 -> L2 -> L3(DAG) -> L4(DAG) -> L5 -> L6 -> L7(DAG) -> L8
        |      |      |       |
       L1.5   L2.5   L3.5    L4.5 diagnostics
```

- L0-L2 and L5-L8 are list-style layers.
- L3, L4, and L7 are graph-style layers.
- L6 and L7 are default off.
- L8 is always the external export boundary.

## Diagnostic hooks

```text
L1.5 <- L1
L2.5 <- L1 + L2
L3.5 <- L1 + L2 + L3
L4.5 <- L4 + L3
```

Diagnostics are default off. With `enabled: false`, they create no DAG
nodes and no sink. With `enabled: true`, they emit diagnostic artifacts
that L8 can include through `diagnostics_l1_5`, `diagnostics_l2_5`,
`diagnostics_l3_5`, `diagnostics_l4_5`, or `diagnostics_all`.

## Foundation and philosophy

- [Foundation core](foundation.md) — `macroforecast.core` Phase 0 contract
  that the layered runtime extends.
- [Philosophy](philosophy.md) — design intent that the layer contracts
  encode.
- [Layer boundary contract](layer_boundary_contract.md) — what each layer
  may consume from upstream and what it must emit downstream.
- [Recipe layers](recipe_layers.md) — YAML key naming, DAG/list shape,
  diagnostic-layer semantics.
- [Artifacts and manifest](artifacts_and_manifest.md) — sink names,
  manifest fields, on-disk layout.
- [Reproducibility](reproducibility.md) — bit-exact replicate contract.
- [Terminology](terminology.md) — glossary.

## Per-layer pages

```{toctree}
:maxdepth: 1
:caption: Layers

layer0/index
layer1/index
layer2/index
layer3/index
layer4/index
layer5/index
layer6/index
layer7/index
layer8/index
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Cross-cutting

foundation
philosophy
layer_boundary_contract
recipe_layers
artifacts_and_manifest
reproducibility
terminology
```
