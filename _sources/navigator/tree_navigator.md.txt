# Tree Navigator

The Tree Navigator turns a recipe into a decision tree view. Each axis is shown with its current selection, all known options, disabled reasons, and the canonical path effect of selecting each option.

## CLI

```bash
macrocast-navigate tree examples/recipes/model-benchmark.yaml
```

Show only the upstream forecast-construction layers:

```bash
macrocast-navigate tree examples/recipes/model-benchmark.yaml --upstream-only
```

`--upstream-only` shows Layers 0-3:

- Layer 0: study design and execution grammar;
- Layer 1: data task and official data frame;
- Layer 2: representation construction and researcher preprocessing;
- Layer 3: forecast generation.

The default view shows Layers 0-7. Downstream layers are part of the same path contract:

| Layer | Tree axes now surfaced |
|---|---|
| Layer 4 | metric families, benchmark comparison scope, aggregation, ranking, reporting style, regimes, decomposition, OOS period |
| Layer 5 | export format, saved objects, provenance fields, artifact granularity |
| Layer 6 | legacy `stat_test`, split test-family axes, test scope, dependence correction, overlap handling |
| Layer 7 | importance method, scope, model-native/agnostic families, SHAP, local surrogate, partial dependence, grouped/stability/temporal/gradient outputs |

The interactive app shows these as separate layer tabs. Layer 4, Layer 5, Layer 6, and Layer 7 are not collapsed into one downstream view because each layer has its own order and its own contracts.

## Layer Order and Tree Path

Within a layer, the app follows the canonical axis order from `navigator_ui_data.tree_axes`. The Tree Path panel should be read from top to bottom:

1. the step number is the decision order within the selected layer;
2. the title is a human label, not just the registry key;
3. the selected branch is shown with a human label plus the raw `axis = value`;
4. contract notes explain when a value is user-selected, derived, or constrained by another layer;
5. disabled or blocked branches stay visible with the reason;
6. the YAML effect shows the exact canonical path field that will change;
7. the detailed-docs link opens the source documentation for that axis.

Layer 0 currently exports the first presentation contract:

| Step | Registry axis | Human label | Contract |
|---|---|---|---|
| 1 | `study_scope` | Study Scope | Target cardinality and method-comparison scope; derives Layer 1 `target_structure` and keeps downstream paths fixed or sweep-aware. |
| 2 | `failure_policy` | Failure Handling | Runtime failure behavior for comparison cells and sweeps. |
| 3 | `reproducibility_mode` | Reproducibility | Seed and determinism policy applied before stochastic model code runs. |
| 4 | `compute_mode` | Compute layout | Serial or parallel execution request; unsupported modes and singleton work units degrade to serial/no-op behavior. |

The same presentation contract should be extended layer by layer so each choice has a label, question, contract note, detailed-docs route, and value descriptions.

## Python

```python
from macrocast.navigator import build_navigation_view, load_recipe

recipe = load_recipe("examples/recipes/model-benchmark.yaml")
view = build_navigation_view(recipe)
```

## Interactive App

The static Navigator App ships the exported `navigator_state_engine_v1` payload. This lets the browser update the current path without a Python process:

- clicking an enabled option changes the current axis selection;
- disabled options remain visible with the reason they are unavailable;
- the full Tree Path shows Layers 0-7 as vertical blocks, with the active
  layer expanded and other layers summarized;
- YAML preview is regenerated from the selected path and changed axes;
- generated YAML can be downloaded and existing recipe YAML can be imported;
- `browser_preview` means the edited path is internally compatible in the browser;
- `browser_blocked` means the current browser path contains a selected value that violates an active rule.

The browser state engine is an exploration surface. `compile_preview` is still the snapshot from the exported sample recipe, and `macrocast-navigate resolve` remains the authoritative compiler check before running a recipe. The static app keeps the resolver, compatibility, and replication explanations in the docs pages rather than as separate in-app panels.

## Output Shape

The payload has four top-level fields:

| Field | Meaning |
|---|---|
| `canonical_path` | Current Layer 0-7 path split into `fixed_axes`, `sweep_axes`, and `leaf_config`. |
| `tree` | One entry per axis with option status and disabled reasons. |
| `compatibility` | Active compatibility rules and recommendations. |
| `compile_preview` | Compiler status, warnings, blocked reasons, tree context, and Layer 3 capability matrix. |

Every option in `tree` has:

| Field | Meaning |
|---|---|
| `value` | Candidate value. |
| `status` | Registry/runtime status such as `operational`, `registry_only`, `future`, or `gated_named`. |
| `enabled` | Whether the current path can select this value. |
| `disabled_reason` | Human-readable reason when `enabled=false`. |
| `canonical_path_effect` | Where the value lands in YAML, for example `path.3_training.fixed_axes.model_family = 'ridge'`. |

## Example Use

Use this page when a researcher asks a path question such as:

- Can I use `tree_shap` with `ridge`?
- Why is `sequence_tensor` not open?
- If I choose `forecast_object=quantile`, which model family remains valid?
- If I choose raw-panel iterated forecasting, which future-X path assumptions are available?
- Which evaluation, export, statistical-test, and importance branches are named versus executable?

The tree view should be the first diagnostic, not the API reference.
