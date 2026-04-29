# 4. Detail (code): Full

Detail (code) is the **Full** interface for macrocast. Use it when Simple code is not enough and you need exact control over the layer path, YAML, runtime contracts, artifacts, or custom researcher methods.

Full means:

- every layer decision is explicit or compiler-derived from an explicit rule;
- disabled choices and forced choices are part of the contract;
- YAML paths, Python builders, runtime artifacts, and manifests must agree;
- custom methods enter through documented Layer 2 or Layer 3 contracts.

Layer 0 is the bridge between Simple and Full. Simple asks for Study Scope only, then applies default execution policies. Full shows all four user-facing Layer 0 axes. When a Full recipe omits defaulted policy axes, the compiler/runtime use the documented defaults and record the resolved values in manifests.

Read the layers in order. Earlier layers define the data and representation contract that later layers consume.

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
```

## Navigation Rule

Each layer page shows only local context:

- parent: Detail (code);
- current layer;
- previous layer when one exists;
- next layer when one exists;
- lower-level axis pages only for the current layer.

Legacy audit and compatibility pages remain linkable from relevant layer pages, but they are not part of the main Full navigation spine.
