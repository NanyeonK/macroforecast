# Navigator UI Redesign Plan

This plan replaces the removed static Navigator app with a new recipe IDE for
the current L0-L8 layer contract system.

## Goal

Build a UI where non-technical researchers can construct a broad, valid
macroforecast recipe while technical users can inspect and edit the exact layer
contracts, DAG nodes, sinks, and generated YAML.

The UI must make L3, L4, and L7 DAG layers first-class. Users should be able to
click, drag, connect, edit, and revisit graph choices without hand-writing YAML.

## Product Shape

The app is a dark, Cursor-inspired workspace:

```text
top bar
left layer rail | main workspace | right inspector
bottom panel
```

It is not a dashboard and not a marketing page. It is a research workflow IDE.

## Required Layers

All layers are visible in the left rail:

| Layer | UI mode | Default | Role |
|---|---|---|---|
| L0 | form | on | setup, failure policy, reproducibility, compute |
| L1 | form | on | data source, targets, horizons, predictor universe |
| L1.5 | form/diagnostic | off | raw data diagnostics |
| L2 | form | on | preprocessing and clean panel |
| L2.5 | form/diagnostic | off | pre/post preprocessing diagnostics |
| L3 | DAG | on | feature engineering and target construction |
| L3.5 | form/diagnostic | off | feature diagnostics |
| L4 | DAG | on | model fitting, forecasts, benchmarks, ensembles |
| L4.5 | form/diagnostic | off | generator and model diagnostics |
| L5 | form | on | evaluation metrics, aggregation, ranking |
| L6 | form | off | statistical tests |
| L7 | DAG | off | interpretation and importance |
| L8 | form | on | output, provenance, saved objects |

Diagnostic `.5` layers are default off, but they are never hidden.

## Default Preset

Primary preset: `Broad Multi-Target FRED-MD`.

Defaults are intentionally broad and non-technical.

```yaml
1_data:
  fixed_axes:
    dataset: fred_md
    target_structure: multi_target
    variable_universe: all_variables
  leaf_config:
    targets: [INDPRO, PAYEMS, UNRATE, CPIAUCSL, RPI]
    horizons: [1, 3, 6, 12]
```

Other defaults:

| Area | Default |
|---|---|
| L0 failure | `fail_fast` |
| L0 reproducibility | `seeded_reproducible`, seed 42 |
| L0 compute | `serial` |
| L2 | minimal preprocessing; official transforms where supported |
| L3 | broad source passthrough plus target construction |
| L4 model | `ridge` |
| L4 benchmark | `autoregressive_bic` |
| L5 metrics | MSFE, RMSE, MAE, ranking |
| L6 | off |
| L7 | off |
| L1.5-L4.5 | off but visible |
| L8 | forecasts, metrics, ranking, manifest, recipe YAML |

Open question for implementation: the exact runtime-valid L3/L4 starter DAG
should be generated from the layer modules rather than hard-coded in the UI.

## Main Screens

### Workspace

The first screen is the editable workspace.

- left layer rail;
- center contract map or selected layer editor;
- right inspector;
- bottom YAML/validation panel.

### Layer Contract Map

The map shows:

- L0-L8 main flow;
- L1.5-L4.5 diagnostic side branches;
- layer mode: form or DAG;
- produced sinks;
- consumed upstream sinks;
- status: default, edited, invalid, disabled, active.

Clicking any layer opens it.

### List Layer Editor

Used by L0, L1, L2, L5, L6, L8, and diagnostics.

Requirements:

- dense grouped controls;
- visible defaults;
- edited markers;
- reset section/reset layer;
- inactive axis explanation;
- validation inline and in bottom panel.

### DAG Layer Editor

Used by L3, L4, and L7.

Required interactions:

- add source node;
- add step node;
- add combine node;
- add sink node or bind required sink;
- drag node;
- connect edge;
- delete node/edge;
- duplicate node;
- auto-layout;
- reset layout;
- click node/edge/sink to edit in inspector.

Graph constraints:

- L3 must produce `l3_features_v1` with `X_final` and `y_final`.
- L3 must produce or auto-produce `l3_metadata_v1`.
- Forecast combination is forbidden in L3.
- L4 consumes L3 features and owns forecasts, models, benchmarks, ensembles,
  and tuning.
- L4 must produce `l4_forecasts_v1`, `l4_model_artifacts_v1`, and
  `l4_training_metadata_v1`.
- L7 is disabled by default and only validates sinks when enabled.

### Diagnostics Editor

Each `.5` layer has:

- on/off toggle;
- preset selector: `minimal`, `full`, `custom`;
- active axes;
- validation issues;
- downstream L8 saved-object effect.

When off, the UI states: `No nodes, no sink`.

### Inspector

The inspector edits the current selection.

Selection types:

- layer;
- diagnostic layer;
- node;
- edge;
- sink binding;
- validation issue;
- saved object.

The inspector always shows:

- id;
- role;
- params;
- default value;
- current value;
- reset action;
- validation messages.

### Bottom Panel

Tabs:

- YAML preview;
- validation;
- contract;
- run command;
- debug JSON.

YAML is generated from state. Importing YAML reconstructs state.

## State Model

The UI should store one normalized recipe workspace state:

```text
workspace
  metadata
  preset
  layerSelections
  diagnosticSelections
  dagLayers
    l3
    l4
    l7
  layout
  validation
  yamlProjection
```

Important rules:

- UI state is canonical while editing.
- YAML is a projection of UI state.
- Import parses YAML into UI state.
- Manual DAG layout is stored separately from recipe semantics.
- Validation errors should reference both UI location and YAML path.

## Technical Architecture

Recommended app stack:

- Vite;
- React;
- TypeScript;
- React Flow for DAG canvas;
- Zustand or equivalent small store;
- CodeMirror for YAML preview;
- Zod or generated schemas for client-side validation;
- Playwright for visual and interaction tests.

Suggested source layout:

```text
apps/navigator/
  src/
    app/
    components/
    dag/
    layers/
    state/
    validation/
    yaml/
    styles/
```

Docs build should link to the app only after the app is built and verified.

## Data Contracts

The UI needs generated JSON from Python:

- layer topology;
- layer ids and YAML keys;
- layer modes;
- axes and values;
- defaults;
- diagnostic defaults;
- DAG op registry;
- required sinks;
- source selectors;
- validation rules;
- starter presets.

Do not reuse the deleted static app JSON as the new source of truth. Generate a
new compact schema from the current `macrocast.core.layers` modules.

## Validation Model

Validation runs in two levels:

1. client-side immediate validation for missing fields, bad edges, invalid
   required sinks, inactive diagnostics, and obvious boundary violations;
2. Python validation through layer parsers and runtime validators before export
   or run handoff.

The UI should classify messages:

- hard error;
- warning;
- inactive choice;
- missing default;
- boundary violation;
- runtime unsupported.

## MVP

MVP scope:

- app shell;
- layer rail with L0-L8 and L1.5-L4.5;
- broad multi-target FRED-MD preset;
- form editors for L0, L1, L2, L5, L6, L8, diagnostics;
- DAG editors for L3 and L4 with starter nodes;
- L7 disabled but visible;
- YAML preview;
- validation panel;
- import/export YAML;
- Playwright desktop screenshot and core interaction tests.

Out of MVP:

- direct server-side run execution;
- results dashboard;
- collaborative editing;
- mobile DAG editing;
- full custom plugin marketplace.

## Implementation Phases

### Phase 1: Contracts And Skeleton

- Add `DESIGN.md`.
- Add this redesign plan.
- Define new UI JSON contract.
- Create app shell and layer rail.
- Render default preset state.

### Phase 2: Forms And Defaults

- Implement form layers.
- Implement diagnostics toggles.
- Show defaults, edited markers, reset actions.
- Generate YAML from form state.

### Phase 3: DAG Editors

- Implement L3 and L4 canvas.
- Add node palette.
- Add inspector param editing.
- Add required sink binding.
- Add boundary validation.

### Phase 4: Import, Validation, Export

- Import YAML into state.
- Run Python validators.
- Show validation issue locations.
- Export YAML and run command.

### Phase 5: Verification

- Add unit tests for state projection.
- Add Playwright tests for layer edit, DAG edit, import/export.
- Add screenshot checks for desktop and narrow desktop.
- Remove or rewrite remaining static-app-only tests.

## Acceptance Criteria

- User can start with `Broad Multi-Target FRED-MD`.
- User can see and toggle `.5` diagnostics.
- User can edit a prior layer choice after moving forward.
- User can build and edit L3/L4 DAGs through click/drag/connect.
- UI prevents forecast combination in L3.
- UI shows required sinks and missing sink errors.
- YAML preview updates after each edit.
- Exported YAML validates through current layer modules.
- Sphinx docs build without static Navigator app assets.
- Tests do not depend on deleted static app files.
