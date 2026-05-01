# Tree Navigator

The Navigator has two surfaces.

## 1. Canonical Layer/DAG Map

This is the primary UI. It mirrors the registered runtime architecture:

- L0-L8 main flow;
- L1.5-L4.5 diagnostic side branches;
- graph mode for L3, L4, and L7;
- list mode for setup, construction list layers, diagnostics, evaluation, tests, and output;
- sink handoffs between registered layers.

Click a layer card to inspect:

- role and category;
- expected upstream sinks;
- produced sinks;
- sub-layers;
- layer globals;
- axes or output controls.

Graph/DAG layers do not have one flat axis list. Their decisions live in YAML `nodes`, `params`, `inputs`, and `sinks`.

## 2. Compatibility Axis Explorer

The lower explorer is retained for option-level compatibility checks. It shows enabled/disabled choices, disabled reasons, path effects, and YAML preview. It still contains some historical axis-group labels because it is a compatibility surface, not the canonical architecture view.

Use it after choosing a canonical layer when you need to know why a specific option is blocked.

## Reading The UI

| UI element | Meaning |
|---|---|
| Main flow cards | Canonical L0-L8 execution order. |
| Diagnostic side branches | Default-off hooks that consume construction sinks and produce diagnostics. |
| `DAG` badge | Layer is configured as a graph of source/step/sink nodes. |
| `List` badge | Layer is configured by ordered sub-layers and axes. |
| Sink handoffs | Registry-derived edges from `produces` to `expected_inputs`. |
| Runtime support | Current runtime support class for selected options. |

## Boundary Rule

The layer map is authoritative for architecture. The compatibility explorer is useful for individual option constraints, but old labels in that explorer should not override the canonical layer IDs.
