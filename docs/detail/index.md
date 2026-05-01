# Detail: Layer Contracts

Detail docs describe the Full interface for macrocast recipes. Full means that layer keys, axes, DAG nodes, sink names, runtime artifacts, and L8 manifests follow the same contract.

## Main Flow

```text
L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> L7 -> L8
```

- L0-L2 and L5-L8 are list-style layers.
- L3, L4, and L7 are graph-style layers.
- L6 and L7 are default off.
- L8 is always the external export boundary.

## Diagnostic Hooks

```text
L1.5 <- L1
L2.5 <- L1 + L2
L3.5 <- L1 + L2 + L3
L4.5 <- L4 + L3
```

Diagnostics are default off. With `enabled: false`, they create no DAG nodes and no sink. With `enabled: true`, they emit diagnostic artifacts that L8 can include through `diagnostics_l1_5`, `diagnostics_l2_5`, `diagnostics_l3_5`, `diagnostics_l4_5`, or `diagnostics_all`.

## Full Layer Documents

```{toctree}
:maxdepth: 1
:caption: Full Layers

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

## Related Reference

- [Contract Source of Truth](contract_source_of_truth.md)
- [Layer Boundary Contract](layer_boundary_contract.md)
- [Artifacts and Manifest](artifacts_and_manifest.md)
- [Recipe Layers](recipe_layers.md)
