# API Reference

Curated reference for the public surface of the current public surface of macroforecast.

## Top-level API

| Symbol | Description |
|--------|-------------|
| `macroforecast.run(recipe, output_directory=...)` | Execute a recipe end-to-end (L1->L8). Iterates every `{sweep: [...]}` cell, applies L0 failure_policy + seed, returns a `ManifestExecutionResult`. |
| `macroforecast.replicate(manifest_path)` | Re-execute a stored recipe and verify per-cell sink hashes match bit-for-bit. |
| `macroforecast.ManifestExecutionResult` | Per-cell `RuntimeResult` + `sink_hashes`; serializes to `manifest.json`. |
| `macroforecast.ReplicationResult` | `recipe_match`, `sink_hashes_match`, `per_cell_match`. |

## Submodule surfaces

| Module | Purpose |
|--------|---------|
| `macroforecast.core` | 12-layer DAG runtime (foundation, layers, ops, runtime, execution, figures) |
| `macroforecast.raw` | FRED-MD/QD/SD adapters, vintage manager, manifest |
| `macroforecast.preprocessing` | Preprocessing contract helpers |
| `macroforecast.tuning` | Hyperparameter search engines |
| `macroforecast.custom` | User-defined model/preprocessor/feature registration |
| `macroforecast.defaults` | Default profile dict template |

## Layer modules

`macroforecast.core.layers.l{0..8}` (plus `l{1,2,3,4}_5` diagnostics) hold the
canonical schema (`LayerImplementationSpec`) for each layer. Runtime
materialization helpers live in `macroforecast.core.runtime`:

- `materialize_l1`, `materialize_l2`, `materialize_l3_minimal`,
  `materialize_l4_minimal`, `materialize_l5_minimal`,
  `materialize_l6_runtime`, `materialize_l7_runtime`,
  `materialize_l8_runtime`
- `materialize_l1_5_diagnostic` ... `materialize_l4_5_diagnostic`
- `execute_minimal_forecast(recipe)` -- single-cell convenience wrapper

Sweep loop + bit-exact replicate are in `macroforecast.core.execution`:
`execute_recipe`, `replicate_recipe`, `CellExecutionResult`,
`ManifestExecutionResult`, `ReplicationResult`.

Figure rendering (matplotlib + stylized US state choropleth) is in
`macroforecast.core.figures`: `render_bar_global`, `render_heatmap`,
`render_pdp_line`, `render_us_state_choropleth`,
`render_default_for_op`.

## Operational coverage

See [`CLAUDE.md`](../../CLAUDE.md) at the repo root for the operational
matrix: 30+ model families, 37 L3 ops, 7 L6 sub-layers, 29 L7 importance
ops, FRED-SD US state choropleth, parquet/latex/markdown export.
