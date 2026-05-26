# docs/recipe-snippets/

이 폴더의 yaml 파일은 **standalone executable이 아니다.**

## 목적

These recipes are partial-layer documentation snippets. Each file demonstrates
the syntax for a single layer (e.g., `3_feature_engineering:`, `5_evaluation:`,
`8_output:`) without providing the full runnable recipe skeleton. They fail
`mf.run()` with `single_target requires leaf_config.target string` or similar
schema errors because they intentionally omit the `1_data` block with a
`target` and `custom_panel_inline`.

They exist to illustrate layer-specific configuration options clearly without
the surrounding boilerplate of a complete end-to-end recipe. Docs pages
reference them as syntax demos.

## Usage

These files are cited in `docs/` pages via narrative quotation or `:include:`
directives. They are not intended for direct execution.

If you want to understand the syntax for a specific layer, read the snippet.
If you want to run a recipe end-to-end, see `examples/recipes/`.

## Cross-link

Runnable examples (pass `mf.run()` smoke tests):
- `examples/recipes/` — all 6 PASS recipes are end-to-end runnable
- `examples/recipes/l4_minimal_ridge.yaml` — canonical minimal recipe
- `examples/recipes/l4_bagging.yaml`, `l4_ensemble_ridge_xgb_vs_ar1.yaml`,
  `l4_quantile_regression_forest.yaml`, `l6_standard.yaml`,
  `l6_full_replication.yaml`

## File inventory

| File | Layer demonstrated |
|------|--------------------|
| `l0_minimal.yaml` | L0 study setup |
| `l1_5_full.yaml` | L1.5 stationarity diagnostics (full) |
| `l1_5_minimal.yaml` | L1.5 stationarity diagnostics (minimal) |
| `l2_5_full.yaml` | L2.5 cleaning diagnostics (full) |
| `l2_5_minimal.yaml` | L2.5 cleaning diagnostics (minimal) |
| `l3_5_full.yaml` | L3.5 feature diagnostics (full) |
| `l3_5_minimal.yaml` | L3.5 feature diagnostics (minimal) |
| `l3_cascade_pca_on_marx.yaml` | L3 DAG: MARX lag cascade into PCA |
| `l3_mccracken_ng_baseline.yaml` | L3 McCracken-Ng block feature spec |
| `l3_minimal_lag_only.yaml` | L3 single-lag pipeline, no combiner |
| `l3_multi_pipeline_F_MARX.yaml` | L3 multi-pipeline Factor + MARX |
| `l4_5_full.yaml` | L4.5 residual diagnostics (full) |
| `l4_5_minimal.yaml` | L4.5 residual diagnostics (minimal) |
| `l4_mrf_placeholder.yaml` | L4 Coulombe (2024) GTVP MRF config |
| `l4_regime_separate_fit.yaml` | L4 per-regime model fitting |
| `l5_full_reporting.yaml` | L5 full evaluation + decomposition |
| `l5_latex_export.yaml` | L5 LaTeX table export config |
| `l5_minimal.yaml` | L5 minimal evaluation block |
| `l7_coulombe_groups.yaml` | L7 Coulombe group-aggregate importance |
| `l7_minimal_shap.yaml` | L7 SHAP global importance (minimal) |
| `l7_multi_method.yaml` | L7 multiple importance methods |
| `l7_temporal.yaml` | L7 rolling importance over time |
| `l7_transformation_attribution.yaml` | L7 transformation attribution |
| `l8_compact_mode.yaml` | L8 compact output mode |
| `l8_latex_paper.yaml` | L8 LaTeX paper export profile |
| `l8_paper_replication.yaml` | L8 paper replication provenance config |
