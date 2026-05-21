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
| L3 | Feature engineering DAG: 41 ops (lags, factors, filters, selection, targets) | `core/layers/l3.py`, `core/ops/l3_ops.py` |
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

---

## Cycle 47 — L3 feature-selection honesty pass (2026-05-21)

Cycle 47 promoted five L3 feature-selection ops from `status="future"` to
`status="operational"`. This is the fourth honesty pass in the package's
history (following v0.1, v0.25, and v0.3 passes) and the first to target
the L3 selection sub-family exclusively.

### Promoted ops

Each op implements the published algorithm exactly. No approximations are
accepted under the `operational` label.

| Op | Reference | Procedure |
|-----|-----------|-----------|
| `boruta_selection` | Kursa & Rudnicki (2010), JSS 36(11) | Shadow-feature permutation test with RF mean-decrease-impurity importance; Bonferroni-corrected per-feature binomial acceptance/rejection iterated until all features decided or `max_iter` reached. |
| `recursive_feature_elimination` | Guyon, Weston, Barnhill & Vapnik (2002), Machine Learning 46 | Recursive backward elimination by squared coefficient magnitude; optional CV step for automatic `n_features_to_select` selection (RFECV extension). |
| `lasso_path_selection` | Efron, Hastie, Johnstone & Tibshirani (2004), AoS 32(2) | LARS path entry order: features selected in the order they first enter the active set as the regularization parameter decreases from infinity. Distinct from `feature_selection(method="lasso")`, which uses LassoCV coefficient magnitude ranking. |
| `stability_selection` | Meinshausen & Bühlmann (2010), JRSS-B 72(4) | Subsample-based selection probability: lasso/elastic-net fitted on `n_subsamples` random half-samples; features retained where selection probability exceeds threshold `pi_thr`. |
| `genetic_algorithm_selection` | Goldberg (1989), Addison-Wesley | Binary-chromosome GA with tournament selection, single-point crossover, bit-flip mutation at rate 1/N, and elitism; fitness evaluated via CV neg-MSE. Pure NumPy implementation; no `deap` dependency. |

### L3 op count change

| Metric | Pre-C47 | Post-C47 |
|--------|---------|---------|
| L3 operational ops | >= 32 | >= 37 |
| L3 future ops | >= 6 | >= 3 (remaining: `chow_lin_disaggregation` L2-family, `lstm_hidden_state` L7-only, `generalized_irf` L7-only) |
| L7 future ops count | >= 6 | >= 2 (`lstm_hidden_state`, `generalized_irf`) |

### Implementation notes

All five helpers live in `macroforecast/core/runtime.py` immediately after
`_feature_selection`. Each is private (`_boruta_selection` etc.) and is
dispatched via `_execute_l3_op`. The validator no longer hard-rejects any of
the five op names in L3 recipes; all five return a `Panel` whose column set
is the selected subset of the input.

Random-state propagation follows the established #215 / #279 contracts: L0
`random_seed` is injected into `params["random_state"]` at materialize time,
and per-iteration seeds within stochastic ops (Boruta, stability, GA) are
derived as `(seed + iteration_index) % (2**31 - 1)`, guaranteeing bit-exact
replication across identical calls.

The `boruta` and `deap` packages are optional accelerators and are NOT
required. All default code paths use pure NumPy and scikit-learn primitives.
`scipy.stats.binom` (already a transitive dependency) is used in
`_boruta_selection` for the Bonferroni-corrected binomial test.

Four of the five ops (`boruta_selection`, `recursive_feature_elimination`,
`lasso_path_selection`, `stability_selection`) were previously listed in L7's
`FUTURE_OPS` tuple, which would have extended their `layer_scope` to include
`"l7"` via the tail registration loop. This extension was unintended. Cycle 47
removed these four names from `FUTURE_OPS` so all five ops remain
`layer_scope=("l3",)` exclusively.
