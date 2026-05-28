# Public Python API

[Back to reference](index.md)

Curated reference for the importable macroforecast surface. The generated pages document recipe axes and options; this page documents Python imports and semantic package ownership.

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
| `macroforecast.l0(...)` | Callable L0/meta recipe block builder; equivalent to authoring `0_meta` in YAML. |

## Callable Recipe Blocks

| Symbol | Description |
|--------|-------------|
| `macroforecast.meta.configure(...)` | Build and validate a canonical `0_meta` block. |
| `macroforecast.meta.l0(...)` | Alias for `configure(...)`; also exported as `macroforecast.l0(...)`. |
| `macroforecast.data.data(...)` | Build and validate the canonical `data` block without loading a dataset. |
| `macroforecast.data.load_fred_md(...)`, `load_fred_qd(...)`, `load_fred_sd(...)` | Load cached or downloaded official datasets as pandas data frames. |
| `macroforecast.data.metadata(frame_or_result)` | Return dataset metadata/provenance stored on a loaded frame or raw load result. |
| `macroforecast.data.load_fred_md_result(...)`, `load_fred_qd_result(...)`, `load_fred_sd_result(...)` | Advanced raw load envelope with metadata, artifact record, and transform codes. |
| `macroforecast.preprocessing.preprocessing(...)` | Build and validate the canonical `preprocessing` block without executing cleaning. |
| `macroforecast.preprocessing.configure(...)` | Alias for `preprocessing(...)`. |

## Submodule Surfaces

| Module | Purpose |
|--------|---------|
| `macroforecast.recipes` | Recipe orchestration namespace; top-level `run`, `replicate`, `Experiment`, and `forecast` route here. |
| `macroforecast.meta` | L0 study setup, failure policy, reproducibility, and compute policy. |
| `macroforecast.data` | Data recipe authoring, FRED-MD/QD/SD adapters, vintage manager, manifests, and cache helpers. |
| `macroforecast.preprocessing` | Preprocessing recipe authoring, cleaning schemas, transformations, and contract helpers. |
| `macroforecast.features` | L3 feature engineering ops, transforms, and selectors. |
| `macroforecast.models` | L4 model classes, model ops, paper helpers, and tuning. |
| `macroforecast.evaluation` | L5 metrics and evaluation ops. |
| `macroforecast.stat_tests` | L6 forecast-comparison statistical tests. |
| `macroforecast.interpretation` | L7 interpretation schemas, ops, and methods. |
| `macroforecast.output` | L8 artifact, provenance, and export ops. |
| `macroforecast.diagnostics` | L1.5/L2.5/L3.5/L4.5 diagnostic packages. |
| `macroforecast.core` | Cross-layer runtime, registry, manifest, cache, validation, execution, and figures. |
| `macroforecast.api.functions` | Canonical standalone callable namespace; also available as `macroforecast.functions`. |
| `macroforecast.api.defaults` | Canonical default profile helpers; also available through top-level lazy exports and `macroforecast.defaults`. |
| `macroforecast.api.custom` | Custom model, preprocessor, feature, and target-transform registration. |
| `macroforecast.feature_selection` | Promoted compatibility namespace for selector classes. |
| `macroforecast.transforms` | Promoted compatibility namespace for transform callables. |

## Package Ownership

Canonical implementation now lives in semantic packages: `meta`, `data`, `preprocessing`, `features`, `models`, `evaluation`, `stat_tests`, `interpretation`, `output`, and `diagnostics.*`.

`macroforecast.layers.*` is compatibility-only. `macroforecast.core.layers` owns registry-facing compatibility modules and runtime glue, not the primary public implementation surface.

## Runtime Helpers

Runtime materialization helpers live in `macroforecast.core.runtime`:

- `materialize_l1`, `materialize_l2`, `materialize_l3_minimal`, `materialize_l4_minimal`, `materialize_l5_minimal`, `materialize_l6_runtime`, `materialize_l7_runtime`, `materialize_l8_runtime`
- `materialize_l1_5_diagnostic` through `materialize_l4_5_diagnostic`
- `execute_minimal_forecast(recipe)`

Sweep execution and bit-exact replication live in `macroforecast.core.execution`: `execute_recipe`, `replicate_recipe`, `CellExecutionResult`, `ManifestExecutionResult`, and `ReplicationResult`.
