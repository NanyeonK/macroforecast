# Public Python API

[Back to reference](index.md)

Curated reference for the importable macroforecast surface. The generated layer pages document recipe axes and options; this page documents Python imports and semantic package ownership.

## Top-level API

| Symbol | Description |
|--------|-------------|
| `macroforecast.forecast(...)` | One-shot helper that assembles a default recipe and runs it. |
| `macroforecast.Experiment(...)` | Builder with `compare_models`, `compare`, `sweep`, `run`, `replicate`, `to_yaml`, and `validate`. |
| `macroforecast.ForecastResult` | Thin facade over a recipe execution result. |
| `macroforecast.run(recipe, output_directory=...)` | Execute a recipe end-to-end and return `ManifestExecutionResult`. |
| `macroforecast.run_file(path, output_directory=...)` | Execute a recipe file. |
| `macroforecast.replicate(manifest_path)` | Re-execute a stored manifest and verify sink hashes. |
| `macroforecast.ManifestExecutionResult` | Per-cell `RuntimeResult` plus sink hashes. |
| `macroforecast.ReplicationResult` | Bit-exact replication comparison result. |
| `macroforecast.meta.configure(...)` | Update package-wide execution defaults for direct Python calls. |

## Callable Recipe Blocks

| Symbol | Description |
|--------|-------------|
| `macroforecast.meta.configure(...)` | Update package-wide execution defaults. |
| `macroforecast.meta.get_config()` | Return a copy of the active defaults. |
| `macroforecast.meta.use_config(...)` | Temporarily override defaults in a context. |

## Submodule Surfaces

| Module | Purpose |
|--------|---------|
| `macroforecast.recipes` | Recipe orchestration namespace; top-level `run`, `replicate`, `Experiment`, and `forecast` route here. |
| `macroforecast.meta` | Package-wide execution settings. |
| `macroforecast.data` | Canonical pandas panels, metadata, FRED/custom loaders, and data specs. |
| `macroforecast.preprocessing` | Callable pandas preprocessing helpers and transformation utilities. |
| `macroforecast.features` | L3 feature engineering ops, transforms, and selectors. |
| `macroforecast.models` | L4 model classes, model ops, paper helpers, and tuning. |
| `macroforecast.evaluation` | L5 metrics and evaluation ops. |
| `macroforecast.stat_tests` | L6 forecast-comparison statistical tests. |
| `macroforecast.interpretation` | L7 interpretation schemas, ops, and methods. |
| `macroforecast.output` | L8 artifact, provenance, and export ops. |
| `macroforecast.diagnostics` | Legacy recipe diagnostics retained for L3.5/L4.5 runtime compatibility. |
| `macroforecast.core` | Cross-layer runtime, registry, manifest, cache, validation, execution, and figures. |
| `macroforecast.api.functions` | Canonical standalone callable namespace; also available as `macroforecast.functions`. |
| `macroforecast.api.defaults` | Canonical default profile helpers; also available through top-level lazy exports and `macroforecast.defaults`. |
| `macroforecast.api.custom` | Custom model, preprocessor, feature, and target-transform registration. |
| `macroforecast.feature_selection` | Promoted compatibility namespace for selector classes. |
| `macroforecast.transforms` | Promoted compatibility namespace for transform callables. |

## Layer Module Ownership

Canonical implementation now lives in semantic packages: `meta`, `data`, `preprocessing`, `features`, `models`, `evaluation`, `stat_tests`, `interpretation`, `output`, and `diagnostics.*`.

`macroforecast.layers.*` is compatibility-only. `macroforecast.core.layers` owns registry-facing compatibility modules and runtime glue, not the primary public implementation surface.

## Runtime Helpers

Runtime materialization helpers live in `macroforecast.core.runtime`:

- `materialize_l1`, `materialize_preprocessing`, `materialize_l3_minimal`, `materialize_l4_minimal`, `materialize_l5_minimal`, `materialize_l6_runtime`, `materialize_l7_runtime`, `materialize_l8_runtime`
- `materialize_l3_5_diagnostic`, `materialize_l4_5_diagnostic`
- `execute_minimal_forecast(recipe)`

Sweep execution and bit-exact replication live in `macroforecast.core.execution`: `execute_recipe`, `replicate_recipe`, `CellExecutionResult`, `ManifestExecutionResult`, and `ReplicationResult`.
