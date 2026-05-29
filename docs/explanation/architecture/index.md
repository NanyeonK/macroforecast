# Architecture

These pages document what each layer owns, how layers hand artifacts to each other,
and the design rationale behind each boundary. Read them to understand the canonical
contracts before authoring recipes, extending the runtime, or auditing manifests.

> **Looking for option definitions?** Use the
> [Encyclopedia](../../reference/index.md) — one page per axis option
> with description, when to use, when NOT to use, and references.

## Canonical layer flow

```text
L0 -> L1 -> preprocessing -> L3(graph) -> L4(graph) -> L5 -> L6 -> L7(graph) -> L8
              |      |       |
             L1.5   Data diagnostic   L3.5    L4.5 diagnostics
```

- Meta, data, preprocessing, and L5-L8 are list-style stages.
- L3, L4, and L7 are graph-style layers.
- L6 and L7 are default off.
- L8 is always the external export boundary.

## Diagnostic hooks

```text
L1.5 <- L1
Data diagnostic <- L1 + preprocessing
L3.5 <- L1 + preprocessing + L3
L4.5 <- L4 + L3
```

Diagnostics are default off. With `enabled: false`, they create no graph
nodes and no sink. With `enabled: true`, they emit diagnostic artifacts
that L8 can include through `diagnostics_l1_5`, `diagnostics_l3_5`,
`diagnostics_l4_5`, or `diagnostics_all`.

## Foundation and philosophy

- [Foundation core](../foundation.md)
- [Philosophy](../philosophy.md)
- [Layer boundary contract](../layer_boundary_contract.md)
- [Recipe layers](../recipe_layers.md)
- [Artifacts and manifest](../artifacts_and_manifest.md)
- [Reproducibility](../reproducibility.md)
- [Terminology](../terminology.md)

## Per-layer pages

```{toctree}
:maxdepth: 1
:caption: Layers

layer0
layer1
layer2
layer3
layer4
layer5
layer6
layer7
layer8
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: L0 sub-pages

layer0_derived_study_scope
layer0_failure_policy
layer0_reproducibility_policy
layer0_compute_policy
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: L1 sub-pages

layer1_availability_timing
layer1_frame_availability
layer1_fred_sd_source_selection
layer1_source_frame
layer1_target_universe
```
