# macrocast

> Fair, reproducible macro forecasting benchmarking package.
> Version 0.1.0 (12-layer canonical design — see `plans/design/part1-4`).

## Quick start

```bash
python3 -m pytest tests/ -x -q                     # ~470 tests, <10s on a laptop
python3 -c "import macrocast; print(macrocast.__version__)"
python3 -c "import macrocast; macrocast.run('examples/recipes/l4_minimal_ridge.yaml')"
```

## Public API

```python
import macrocast

result = macrocast.run("path/to/recipe.yaml", output_directory="out/")
# result.cells -> per-sweep-cell RuntimeResult + sink_hashes
# manifest.json + per-cell artifacts written to out/

replication = macrocast.replicate("out/manifest.json")
# bit-exact: replication.sink_hashes_match  -> True when artifacts identical
```

## Architecture (12-layer design, canonical order)

| Layer | Purpose | Module |
|-------|---------|--------|
| L0 | Study setup (failure_policy, seed, compute_mode) | `core/layers/l0.py` |
| L1 | Data definition (FRED-MD/QD/SD, target, geography, regime) | `core/layers/l1.py` |
| L2 | Preprocessing (transform / outlier / imputation / frame edge) | `core/layers/l2.py` |
| L3 | Feature engineering DAG (37 ops + cascade β) | `core/layers/l3.py`, `core/ops/l3_ops.py` |
| L4 | Forecasting model + tuning (30+ families, 5 combine ops) | `core/layers/l4.py`, `core/ops/l4_ops.py` |
| L5 | Evaluation (metrics × benchmark × aggregation × decomposition × ranking) | `core/layers/l5.py`, `core/ops/l5_ops.py` |
| L6 | Statistical tests (DM/HLN, CW, MCS bootstrap, PT/HM, residual battery) | `core/layers/l6.py`, `core/ops/l6_ops.py` |
| L7 | Interpretation (29 importance ops, group_aggregate, lineage, US choropleth) | `core/layers/l7.py`, `core/ops/l7_ops.py` |
| L8 | Output / provenance (json/csv/parquet/latex/markdown, manifest) | `core/layers/l8.py`, `core/ops/l8_ops.py` |
| L1.5 / L2.5 / L3.5 / L4.5 | Diagnostic hooks (default-off, non-blocking) | `core/layers/l{1,2,3,4}_5.py` |

Foundation contracts (5 node types, sweep machinery, cache/hash, manifest schema)
live in `core/{dag, sweep, cache, manifest, types, validator, yaml}.py`. The
end-to-end runtime (cell loop, seed propagation, bit-exact replicate) is
`core/execution.py`; per-layer materialization helpers are in `core/runtime.py`.

## Operational coverage (v0.1)

- **Models (30+):** ar_p, ols, ridge, lasso, elastic_net, lasso_path,
  bayesian_ridge, huber, glmboost, var, factor_augmented_ar,
  principal_component_regression, decision_tree, random_forest, extra_trees,
  gradient_boosting, xgboost / lightgbm / catboost (try-import),
  svr_linear/rbf/poly, mlp, lstm/gru/transformer (torch try-import + sklearn
  fallback), knn, macroeconomic_random_forest, bvar_minnesota,
  bvar_normal_inverse_wishart, dfm_mixed_mariano_murasawa.
- **L3 ops (37):** lag/seasonal_lag, ma_window/ma_increasing_order (MARX),
  pca/sparse_pca/scaled_pca/dfm/varimax/partial_least_squares/random_projection,
  wavelet/fourier, hp_filter/hamilton_filter, polynomial/interaction/kernel/
  nystroem, regime_indicator/season_dummy/time_trend/holiday,
  feature_selection (variance/correlation/lasso), cumulative_average target,
  hierarchical_pca, weighted_concat, simple_average.
- **L6 tests:** Diebold-Mariano with Newey-West HAC + HLN correction,
  Clark-West with MSE adjustment, MCS / SPA / Reality Check / StepM via
  bootstrap, Pesaran-Timmermann + Henriksson-Merton, statsmodels Ljung-Box /
  ARCH-LM / Jarque-Bera / Breusch-Godfrey / Durbin-Watson.
- **L7 importance:** SHAP family (shap try-import → linear/permutation
  proxy), partial_dependence, accumulated_local_effect, friedman_h_interaction,
  lasso_inclusion_frequency, cumulative_r2_contribution, bootstrap_jackknife,
  rolling_recompute, forecast_decomposition, fevd / historical_decomposition /
  generalized_irf, group_aggregate, lineage_attribution,
  transformation_attribution, mrf_gtvp.
- **L7 figures:** matplotlib bar_global, heatmap, pdp_line, plus a stylized
  US state choropleth (`render_us_state_choropleth`) for FRED-SD geographic
  visualization. Default mappings auto-render per importance op.
- **L8 export:** json / csv / parquet / latex / markdown / json_csv /
  json_parquet / all + per-cell granularity + descriptive naming.
- **Sweep:** param-level + external-axis sweeps; grid (default) and zip
  combination modes; cell loop iterates `{sweep: [...]}` markers in the
  recipe root.
- **Replication:** `macrocast.replicate(manifest_path)` re-runs the stored
  recipe and verifies per-cell sink hashes match bit-for-bit.

## Module layout

```
macrocast/
  __init__.py             # lazy-export top-level surface
  api.py                  # macrocast.run / macrocast.replicate
  core/
    execution.py          # execute_recipe (cell loop) + replicate_recipe
    runtime.py            # per-layer materialize_l{1..8}_minimal helpers
    figures.py            # matplotlib backend + US state choropleth
    cache.py, dag.py, sweep.py, manifest.py, validator.py, yaml.py, types.py
    layer_specs.py, recipe.py, selectors.py
    layers/               # l0..l8 + l1_5/l2_5/l3_5/l4_5 schema definitions
    ops/                  # universal/l3/l4/l5/l6/l7/l8/diagnostic op registry
  raw/                    # FRED-MD/QD/SD adapters, vintage manager, manifest
  preprocessing/          # preprocessing contract helpers (legacy support)
  custom.py               # user-defined model / preprocessor registration
  defaults.py             # default profile dict template
  tuning/                 # HP search engines (optional, integrated via L4)
plans/design/             # 4-part design document (canonical source of truth)
tests/                    # 474 tests (core, layers, integration)
examples/recipes/         # YAML recipe examples per layer
```

## Key design principles

- **Schema before runtime:** every layer declares an axis × option × gate
  contract in `LayerImplementationSpec`; runtime helpers in `core/runtime.py`
  materialize that contract.
- **One recipe = one study:** a YAML recipe fully specifies the DAG; sweep
  markers expand into independent cells.
- **Bit-exact replication:** seed propagation + canonical key ordering +
  per-cell sink hashes guarantee that `replicate(manifest_path)` produces
  identical artifacts.
- **Default-off diagnostics:** L1.5/L2.5/L3.5/L4.5 + L6/L7 require explicit
  `enabled: true` so a minimal recipe stays fast.
- **Cross-layer references:** `is_benchmark` (L4 → L5/L6), `mcs_inclusion`
  (L6.D → L7), `lineage` (L3.metadata → L7), `regime` (L1.G → L3/L4/L5/L6.C).

## Status levels

`operational` (runtime executes) > `planned` (next milestone) > `future`
(schema-only, runtime raises). v0.1 collapsed `planned` and `registry_only`
into `operational` for the design's main axes; only deep-NN families and the
DFM-MM mixed-frequency model fall back to lighter implementations when the
optional extras (`torch`, `xgboost`, etc.) are not installed.
