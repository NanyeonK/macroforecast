# macroforecast — Architecture

Version: **v0.9.2b1**. Package root: `macroforecast/`.

---

## Top-level module layout

| Module | Role |
|--------|------|
| `macroforecast.api` | Public entry points: `mf.run`, `mf.replicate`, `mf.forecast` |
| `macroforecast.api_high` | High-level `Experiment` class and `ForecastResult` |
| `macroforecast.core` | DAG runtime, layer schemas, op registries, sweep, cache, manifest |
| `macroforecast.scaffold` | Option-doc registry and encyclopedia generator |
| `macroforecast.functions` | 118 standalone callables organized by layer (L2–L7) |
| `macroforecast.raw` | FRED-MD/QD/SD adapters, vintage manager, raw manifest |
| `macroforecast.preprocessing` | Preprocessing contract helpers |
| `macroforecast.custom` | User-defined model, preprocessor, and target transformer registration |
| `macroforecast.defaults` | Default profile dict template |
| `macroforecast.tuning` | Hyperparameter search engines (optional, integrated via L4) |

---

## 12-layer canonical design (L0–L8 + diagnostic half-layers)

| Layer | Purpose | Primary module paths |
|-------|---------|----------------------|
| L0 | Study setup: failure policy, seed, compute mode, study scope | `core/layers/l0.py` |
| L1 | Data definition: FRED-MD/QD/SD source, target y, predictor x, regime, availability | `core/layers/l1.py` |
| L1.5 | Diagnostic hook (default-off): stationarity tests, raw data summaries | `core/layers/l1_5.py` |
| L2 | Preprocessing: transform, outlier, imputation, frame edge, mixed-frequency alignment | `core/layers/l2.py` |
| L2.5 | Diagnostic hook (default-off): preprocessed panel summaries | `core/layers/l2_5.py` |
| L3 | Feature engineering DAG: 36 ops (lags, factors, filters, selection, targets) | `core/layers/l3.py`, `core/ops/l3_ops.py` |
| L3.5 | Diagnostic hook (default-off): feature distribution summaries | `core/layers/l3_5.py` |
| L4 | Forecasting model + tuning: 35+ families, 5 combine ops | `core/layers/l4.py`, `core/ops/l4_ops.py` |
| L4.5 | Diagnostic hook (default-off): residual diagnostics, fitted-vs-actual | `core/layers/l4_5.py` |
| L5 | Evaluation: metrics, benchmarks, decomposition, aggregation, ranking | `core/layers/l5.py`, `core/ops/l5_ops.py` |
| L6 | Statistical tests: DM/HLN, CW, MCS/SPA/RC/StepM, PT/HM, residual battery, density tests | `core/layers/l6.py`, `core/ops/l6_ops.py` |
| L7 | Interpretation: 29 importance ops, group aggregate, lineage, transformation attribution | `core/layers/l7.py`, `core/ops/l7_ops.py` |
| L8 | Output and provenance: json/csv/parquet/latex/markdown, manifest, saved objects | `core/layers/l8.py`, `core/ops/l8_ops.py` |

Layers L6 and L7 are default-off and require explicit `enabled: true`. Layer L8
is always on. The diagnostic half-layers L1.5, L2.5, L3.5, L4.5 are
default-off and non-blocking.

---

## 2-paradigm model

macroforecast exposes two complementary access patterns.

**Recipe DSL** (`mf.run`). A YAML recipe fully specifies a study as an
end-to-end DAG from L0 through L8. Sweep markers expand the recipe into
independent cells. Every run writes a manifest with per-cell sink hashes and
supports bit-exact replication via `mf.replicate(manifest_path)`. This
paradigm is appropriate for reproducible comparative studies.

**Standalone callables** (`mf.functions.*`). Individual operations from L2
through L7 are also available as direct Python callables requiring no YAML.
They accept NumPy arrays or DataFrames and return frozen dataclasses with typed
result attributes. This paradigm is appropriate for exploratory analysis,
notebook workflows, and integration into custom pipelines.

See [docs/two_entry_points.md](docs/two_entry_points.md) for a decision guide
on when to choose each paradigm.

---

## Core runtime module map

| Module | Role |
|--------|------|
| `core/execution.py` | Cell loop (`execute_recipe`), seed propagation, `replicate_recipe` |
| `core/runtime.py` | Per-layer `materialize_l*` helpers; layer artifact construction |
| `core/layers/l0.py`–`l8.py` | Layer schema definitions (axes, gates, defaults, status) |
| `core/layers/l1_5.py`, `l2_5.py`, `l3_5.py`, `l4_5.py` | Diagnostic half-layer schemas |
| `core/ops/l3_ops.py`–`l8_ops.py` | Op registries per layer |
| `core/dag.py` | Universal DAG schema: 5 node types (source, axis, step, combine, sink) |
| `core/sweep.py` | Sweep expansion: param-level, recipe-level (external axis), node-level (sweep_groups) |
| `core/cache.py` | Content-addressed artifact cache; SHA-256 sink hash generation |
| `core/manifest.py` | Manifest read/write; provenance record schema |

Foundation contracts (typed artifacts, validator, YAML normalizer, recipe
schema, selector types) live in `core/types.py`, `core/validator.py`,
`core/yaml.py`, `core/recipe.py`, and `core/selectors.py`.

---

## Standalone functions summary

`macroforecast.functions` contains **118** standalone callables organized by
layer. Per-layer counts (verified via `tools/gen_standalone_docs.py` against
live `macroforecast.functions.__all__` at HEAD afc28282):

| Layer | Count | Module |
|-------|-------|--------|
| L2 | 14 | `functions/clean.py` |
| L3 | 36 | `functions/transforms.py` |
| L4 | 38 | `functions/fit.py` |
| L5 | 15 | `functions/metrics.py` |
| L6 | 7 | `functions/tests.py` |
| L7 | 8 | `functions/importance.py` |

For the full per-callable reference (signatures, result attributes,
examples), see [docs/standalone_functions/](docs/standalone_functions/index.md).
