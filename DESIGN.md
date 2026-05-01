---
name: macroforecast-ui
inspiration: cursor
purpose: Design rules for the macroforecast recipe IDE and navigator rewrite.
---

# macroforecast UI Design System

macroforecast UI is a research workflow IDE for building macro forecasting
recipes from layer contracts and DAGs. It should feel closer to a code editor
than a dashboard: dense, dark, precise, inspectable, and built for repeated
editing.

## Product Principles

- The first screen is the working surface, not a landing page.
- Every L0-L8 layer is visible.
- L1.5, L2.5, L3.5, and L4.5 diagnostics are always visible as side branches.
- L3, L4, and L7 are graph-first. The canvas is the primary editor.
- YAML is generated from UI state. Users can inspect it, but should not need to
  hand-write it.
- Every prior selection is editable after creation.
- Defaults are broad, non-technical, and safe for a first research run.

## Visual Direction

Use a Cursor-inspired dark IDE style:

- near-black shell;
- compact panels;
- thin borders;
- subtle blue/violet accent;
- restrained gradients only for active focus, validation status, or selected
  graph nodes;
- monospace text for YAML, node ids, sink names, and validation codes.

Avoid:

- marketing hero layouts;
- oversized cards;
- decorative gradients or blobs;
- vague dashboard charts before a recipe exists;
- hiding layer structure behind generic tabs.

## Color Tokens

| Token | Value | Usage |
|---|---:|---|
| `bg.app` | `#0b0d10` | application background |
| `bg.panel` | `#111318` | sidebars, inspector, bottom panel |
| `bg.raised` | `#171a21` | selected panels, popovers |
| `bg.input` | `#0f1217` | text inputs, selects |
| `border.default` | `#2a2f3a` | panel and control borders |
| `border.subtle` | `#1d222b` | low-emphasis separators |
| `text.primary` | `#f4f7fb` | main labels |
| `text.secondary` | `#9aa4b2` | descriptions |
| `text.muted` | `#6f7785` | metadata |
| `accent.primary` | `#7c8cff` | active layer, selected node |
| `accent.secondary` | `#4cc9f0` | links, live preview accents |
| `status.success` | `#35c98b` | valid contract |
| `status.warning` | `#f0b84b` | warning, default override |
| `status.error` | `#ff5c7a` | hard validation error |
| `status.off` | `#4a5260` | disabled diagnostics |

## Typography

- UI font: Inter, system-ui, sans-serif.
- Code font: JetBrains Mono, SFMono-Regular, Menlo, monospace.
- Base size: 13px.
- Dense labels: 12px.
- Section headings: 14px, semibold.
- Page/workspace headings: 16px, semibold.
- Do not scale type with viewport width.
- Letter spacing is 0.

## Spacing And Shape

- Base spacing unit: 4px.
- Control height: 28-32px.
- Toolbar height: 40px.
- Left rail width: 248px.
- Inspector width: 360px.
- Bottom panel height: 240px default, resizable.
- Border radius: 6px for controls, 8px max for panels.
- Cards are only for repeated items or popovers. Do not nest cards.

## Layout

```text
top bar
left layer rail | main workspace | right inspector
bottom panel
```

### Top Bar

Contains recipe name, preset, validation state, undo/redo, save, export YAML,
and command palette. Keep it compact.

### Left Layer Rail

Shows the full layer map:

- L0 Setup
- L1 Data
- L1.5 Data Diagnostics
- L2 Preprocessing
- L2.5 Pre/Post Diagnostics
- L3 Feature DAG
- L3.5 Feature Diagnostics
- L4 Forecast DAG
- L4.5 Generator Diagnostics
- L5 Evaluation
- L6 Statistical Tests
- L7 Interpretation DAG
- L8 Output

Each item shows status: default, edited, invalid, disabled, or active.

### Main Workspace

The selected layer controls the main editor:

- list layers use dense forms;
- graph layers use a DAG canvas;
- no selected layer shows the full contract map.

### Inspector

The inspector edits the selected layer, node, edge, sink, or diagnostic toggle.
It must always show:

- selected object id;
- contract role;
- editable params;
- default value;
- reset-to-default action;
- validation issues.

### Bottom Panel

Tabs:

- YAML;
- Validation;
- Contract;
- Run Command;
- Debug JSON.

The YAML tab is read-only by default with an explicit "edit YAML" escape hatch
for advanced users.

## Component Rules

### Layer Row

- icon or compact glyph for layer type;
- layer id and name;
- small status pill;
- modified marker when user changed defaults;
- diagnostic rows are indented side branches but always visible.

### DAG Node

Node classes:

- source;
- step;
- combine;
- sink;
- diagnostic.

Node states:

- default;
- selected;
- invalid;
- missing input;
- inactive;
- output consumed.

Use colored left strips instead of large colored fills.

### Edge

- thin neutral line by default;
- accent line when selected;
- error line when invalid;
- arrowheads point downstream.

### Forms

- labels above controls for dense technical fields;
- inline help only when a label is ambiguous;
- defaults visible in muted text;
- changed values show a small warning-colored dot.

### Toggles

Use toggles for default-off layers:

- L1.5-L4.5 diagnostics;
- L6;
- L7.

When off, show "no nodes, no sink" in muted text.

## Default Preset

Default preset: `Broad Multi-Target FRED-MD`.

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

Additional defaults:

- L0: `fail_fast`, `seeded_reproducible`, `serial`;
- L2: minimal preprocessing, official transforms when supported;
- L3: source predictors and target construction with broad passthrough/lag
  baseline;
- L4: `ridge` model plus `autoregressive_bic` benchmark;
- L5: MSFE, RMSE, MAE, ranking;
- L6: off;
- L7: off;
- L1.5-L4.5: off but visible;
- L8: forecasts, metrics, ranking, manifest, recipe YAML.

## Interaction Rules

- Users can add, drag, connect, duplicate, and delete DAG nodes.
- Users can click any layer, diagnostic branch, node, edge, or sink to edit it.
- Users can import YAML and reconstruct UI state.
- Auto-layout is available but never overwrites manual layout without user
  action.
- Validation runs on every meaningful edit.
- Hard errors block export/run handoff. Warnings do not.

## Accessibility

- Minimum contrast ratio 4.5:1 for text.
- Keyboard access for layer navigation, command palette, node selection, and
  inspector fields.
- Focus rings use `accent.primary`.
- Do not encode validation state by color alone.

## Responsive Behavior

Desktop is primary. Tablet and narrow desktop should retain functionality:

- left rail can collapse to icons;
- inspector can dock below canvas;
- bottom panel remains resizable;
- graph canvas keeps stable controls and minimap.

Mobile is read/review only for MVP; full DAG editing is desktop-first.
