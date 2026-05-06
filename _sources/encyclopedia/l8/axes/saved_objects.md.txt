# `saved_objects`

[Back to L8](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``saved_objects`` on sub-layer ``L8_B_saved_objects`` (layer ``l8``).

## Sub-layer

**L8_B_saved_objects**

## Axis metadata

- Default: `None`
- Sweepable: False
- Status: operational

## Operational status summary

- Operational: 20 option(s)
- Future: 0 option(s)

## Options

### `clean_panel`  --  operational

Cleaned L2 panel (post tcode / outlier / imputation / frame edge).

The output of the L2 pipeline. Useful when downstream re-runs need to skip the (potentially expensive) cleaning stages.

**When to use**

When downstream re-runs without re-cleaning are needed.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts), [`forecast_intervals`](#forecast-intervals)

_Last reviewed 2026-05-05 by macroforecast author._

### `combination_weights`  --  operational

Ensemble weights from L4 combine ops.

Per-origin per-member weights produced by L4 combine ops (equal_weighted / dmsfe / inverse_msfe / mallows_cp / etc.). Active when ensemble combine ops are in the L4 DAG.

**When to use**

Active when ensemble combine ops are in the L4 DAG.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `decomposition`  --  operational

L5.D decomposition tables (per-period / per-block / Shapley).

Variance / loss decomposition outputs. Default-on when L5.D decomposition is active.

**When to use**

Default-on when decomposition is active.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `diagnostics_all`  --  operational

Every active diagnostic layer's output (convenience option).

Convenience flag: enables ``diagnostics_l{1..4}_5`` simultaneously when the corresponding diagnostic layer is active. Recommended default for first-time runs.

**When to use**

Default convenience option for full-diagnostic runs.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `diagnostics_l1_5`  --  operational

L1.5 diagnostic outputs (sample coverage / stationarity / outlier audit).

JSON + figures from the L1.5 sub-layer. Active when L1.5 is enabled in the recipe.

**When to use**

Active when L1.5 is enabled. Selecting ``diagnostics_l1_5`` on ``l8.saved_objects`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `diagnostics_l2_5`  --  operational

L2.5 diagnostic outputs (cleaning effect summaries).

JSON + figures from the L2.5 sub-layer. Active when L2.5 is enabled.

Configures the ``saved_objects`` axis on ``L8_B_saved_objects`` (layer ``l8``); the ``diagnostics_l2_5`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Active when L2.5 is enabled. Selecting ``diagnostics_l2_5`` on ``l8.saved_objects`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `diagnostics_l3_5`  --  operational

L3.5 diagnostic outputs (factor / lag / selection inspection).

JSON + figures from the L3.5 sub-layer. Active when L3.5 is enabled.

Configures the ``saved_objects`` axis on ``L8_B_saved_objects`` (layer ``l8``); the ``diagnostics_l3_5`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Active when L3.5 is enabled. Selecting ``diagnostics_l3_5`` on ``l8.saved_objects`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `diagnostics_l4_5`  --  operational

L4.5 diagnostic outputs (in-sample fit / window stability / tuning history).

JSON + figures from the L4.5 sub-layer. Active when L4.5 is enabled.

Configures the ``saved_objects`` axis on ``L8_B_saved_objects`` (layer ``l8``); the ``diagnostics_l4_5`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Active when L4.5 is enabled. Selecting ``diagnostics_l4_5`` on ``l8.saved_objects`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `feature_metadata`  --  operational

L3 column lineage + pipeline definitions.

The L3 metadata sink containing per-feature lineage, transformation chain, and pipeline ID. Default-on when L7 ``lineage_attribution`` or ``transformation_attribution`` is active -- those ops require this metadata to function.

**When to use**

Default-on when L7 lineage / transformation_attribution is active.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`forecasts`](#forecasts), [`forecast_intervals`](#forecast-intervals)

_Last reviewed 2026-05-05 by macroforecast author._

### `forecast_intervals`  --  operational

Per-cell prediction intervals (when forecast_object = quantile / density).

Quantile forecasts at the user-specified ``α`` levels (default 5% / 50% / 95%). Default-on when L4 emits ``forecast_object = quantile`` or ``density``.

**When to use**

Default-on when forecast_object = quantile / density.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `forecasts`  --  operational

Per-cell point forecasts.

The headline output: per (cell, target, horizon, origin) forecast. Default-on; required for replication and for every downstream L5 / L6 / L7 op.

**When to use**

Default-on; required for replication.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecast_intervals`](#forecast-intervals)

_Last reviewed 2026-05-05 by macroforecast author._

### `importance`  --  operational

L7 importance outputs.

Tables and figures from every L7.A op in the recipe's interpretation DAG. Default-on when L7 is enabled.

**When to use**

Default-on when L7 is active. Selecting ``importance`` on ``l8.saved_objects`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `metrics`  --  operational

L5 metric tables.

Per-cell per-metric scores aggregated by the L5.C configuration. Default-on; the standard headline output for every horse-race study.

**When to use**

Default-on. Selecting ``metrics`` on ``l8.saved_objects`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `model_artifacts`  --  operational

Pickled / joblib model objects.

Serialised fitted estimators (one per (model, origin) pair). Default-off because model objects can be large; enable for downstream prediction without re-fitting.

**When to use**

When downstream prediction without re-fitting is needed.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `ranking`  --  operational

L5.E ranking tables.

Models ranked by primary metric / MCS inclusion / Borda count / etc. Default-on when L5.E ranking is active.

**When to use**

Default-on when ranking is active.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `raw_panel`  --  operational

Raw L1 panel before any L2 cleaning.

The original raw FRED-MD / -QD / -SD / custom panel. Default-off because raw FRED panels are large; enabling this makes the run fully self-contained -- a downstream user can re-run the entire pipeline from the manifest alone without internet access.

**When to use**

Default-off for size; enable for fully self-contained runs.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts), [`forecast_intervals`](#forecast-intervals)

_Last reviewed 2026-05-05 by macroforecast author._

### `regime_metrics`  --  operational

Regime-conditional metrics.

Metric breakdowns by L1.G regime classification. Default-on when L1.G regime is non-pooled (i.e. regime-conditional analysis is intended).

**When to use**

Active when L1.G regime is non-pooled.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `state_metrics`  --  operational

State-level metrics for FRED-SD geographic studies.

Per-state metric breakdowns. Default-on when L1.D geography is state-level (FRED-SD pipelines).

**When to use**

Active when L1.D geography is state-level (FRED-SD).

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `tests`  --  operational

L6 test outputs (DM / GW / MCS / SPA / RC / StepM / PT / residual / density).

Test statistics, p-values, kernel choices, and lag-truncation parameters for every L6 sub-layer that is enabled. Default-on when any L6 sub-layer is active.

**When to use**

Default-on when L6 sub-layers are active.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._

### `transformation_attribution`  --  operational

L7 transformation_attribution Shapley table.

Per-pipeline Shapley contributions to forecast skill. Active when ``transformation_attribution`` is in the L7 DAG (typically alongside multi-cell sweeps over alternative L3 transforms).

**When to use**

Active when transformation_attribution op is in the L7 DAG.

**References**

* macroforecast design Part 3, L8: 'reproducibility = manifest + provenance + bit-exact replicate.'

**Related options**: [`raw_panel`](#raw-panel), [`clean_panel`](#clean-panel), [`feature_metadata`](#feature-metadata), [`forecasts`](#forecasts)

_Last reviewed 2026-05-05 by macroforecast author._
