# Recipe Gallery

Curated index of every recipe under
[`examples/recipes/`](https://github.com/NanyeonK/macroforecast/tree/main/examples/recipes)
with one-line purpose, required extras, and smoke-test status.

**38 recipes total** — 32 run on a stock `pip install macroforecast`;
6 require optional extras.

## How to read the table

- **Stock install** = `pip install macroforecast` (no extras). All
  recipes in this column run end-to-end on a fresh venv with only the
  baseline dependencies (pandas / numpy / scikit-learn / statsmodels /
  scipy / matplotlib / openpyxl / PyYAML).
- **Extras required** = pin the relevant extra. e.g.
  `pip install "macroforecast[xgboost,shap]"`.
- **Smoke-tested** = covered by `tests/test_examples_smoke.py`. The
  smoke suite parses + canonical-schema-validates every recipe and
  end-to-end runs the curated runnable subset.

## L0 / L1 / L2 layer fragments (6)

Recipes that exercise only one layer at a time. Useful for unit work;
not standalone runnable (they expect upstream sinks).

| Recipe | Purpose | Extras |
|---|---|---|
| `l0_minimal.yaml` | L0 study setup defaults | — |
| `l1_minimal.yaml` | L1 with custom panel | — |
| `l1_with_regime.yaml` | L1 + NBER regime | — |
| `l1_estimated_markov_switching.yaml` | L1 + Hamilton MS regime estimator | — |
| `l2_minimal.yaml` | L2 transform + outlier + impute pass-through | — |
| `l2_fred_sd_alignment.yaml` | L2 with FRED-SD frequency filter | — |

## L3 feature engineering DAG (4)

| Recipe | Purpose | Extras |
|---|---|---|
| `l3_minimal_lag_only.yaml` | Single lag step | — |
| `l3_mccracken_ng_baseline.yaml` | McCracken-Ng PCA + lag baseline | — |
| `l3_cascade_pca_on_marx.yaml` | Cascade β: PCA over MARX (`pipeline_id` cascade) | — |
| `l3_multi_pipeline_F_MARX.yaml` | Multi-pipeline DAG: factors + MARX combine | — |

## L4 forecasting model — runnable end-to-end (8)

| Recipe | Purpose | Extras | Smoke |
|---|---|---|---|
| `l4_minimal_ridge.yaml` | **Quick start** — minimal ridge on inline panel | — | ✅ |
| `l4_bagging.yaml` | Bagging meta-estimator (Breiman 1996) | — | ✅ |
| `l4_quantile_regression_forest.yaml` | QRF (Meinshausen 2006) with quantile bands | — | ✅ |
| `l4_regime_separate_fit.yaml` | Per-regime separate fit | — | — |
| `l4_mrf_placeholder.yaml` | Coulombe (2024) MRF GTVP | — | — |
| `l4_5_minimal.yaml` | L4 + L4.5 generator diagnostics | — | — |
| `l4_5_full.yaml` | L4.5 full diagnostic battery | — | — |
| `l4_ensemble_ridge_xgb_vs_ar1.yaml` | Horse race: AR(1) bench vs ridge + xgb / lightgbm / catboost | `[xgboost,lightgbm,catboost]` | — |

## L5 evaluation (3)

| Recipe | Purpose | Extras |
|---|---|---|
| `l5_minimal.yaml` | MSE + RMSE + MAE table | — |
| `l5_full_reporting.yaml` | Full reporting: ranking + decomposition + per-target/horizon | — |
| `l5_latex_export.yaml` | LaTeX table emit for paper appendix | — |

## L6 statistical tests (2)

| Recipe | Purpose | Extras | Smoke |
|---|---|---|---|
| `l6_standard.yaml` | DM (HLN) + Clark-West + residual battery | — | ✅ |
| `l6_full_replication.yaml` | Full L6: DM/CW/GR/MCS/PT/residual; 3 L4 models | `[xgboost,lightgbm,catboost]` | ✅ |

## L7 interpretation (5)

| Recipe | Purpose | Extras |
|---|---|---|
| `l7_minimal_shap.yaml` | SHAP TreeExplainer on tree model | `[shap,xgboost]` |
| `l7_multi_method.yaml` | SHAP + permutation + lasso_inclusion together | `[shap,xgboost]` |
| `l7_coulombe_groups.yaml` | McCracken-Ng group-aggregate SHAP | `[shap,xgboost]` |
| `l7_temporal.yaml` | Rolling SHAP for time-varying importance | `[shap,xgboost]` |
| `l7_transformation_attribution.yaml` | Real Shapley over (cell × pipeline) | — |

## L8 export (3)

| Recipe | Purpose | Extras |
|---|---|---|
| `l8_compact_mode.yaml` | Single-cell artifact (no per-cell directory) | — |
| `l8_latex_paper.yaml` | LaTeX table + figure export for paper appendix | — |
| `l8_paper_replication.yaml` | Full provenance manifest + lockfiles | — |

## Diagnostics — L1.5 / L2.5 / L3.5 (6)

| Recipe | Purpose | Extras |
|---|---|---|
| `l1_5_minimal.yaml` | L1.5 raw data summary, default views | — |
| `l1_5_full.yaml` | L1.5 full battery (ADF/PP/KPSS + correlation pre-cleaning) | — |
| `l2_5_minimal.yaml` | L2.5 pre-vs-post preprocessing comparison | — |
| `l2_5_full.yaml` | L2.5 full distribution shift + cleaning effect | — |
| `l3_5_minimal.yaml` | L3.5 feature comparison (cleaned vs features) | — |
| `l3_5_full.yaml` | L3.5 factor block + lag block + selection diagnostics | — |

## Replications (1, more in `replications/`)

| Recipe | Purpose | Extras | Smoke |
|---|---|---|---|
| [`goulet_coulombe_2021_replication.yaml`](https://github.com/NanyeonK/macroforecast/blob/main/examples/recipes/goulet_coulombe_2021_replication.yaml) | Goulet-Coulombe (2021) "The Macroeconomy as a Random Forest" — ridge baseline (paper-ported from old 8-layer schema) | — | — |

See [docs/replications/](../replications/index.md) for end-to-end
walkthroughs and the maintainer's research replications.

## Run any recipe

```bash
# Stock install
pip install macroforecast

# Or with extras
pip install "macroforecast[xgboost,lightgbm,catboost,shap]"

# Run
macroforecast run examples/recipes/<recipe>.yaml -o out/
macroforecast replicate out/manifest.json
```

Or programmatic:

```python
import macroforecast as mf
result = mf.run("examples/recipes/l4_minimal_ridge.yaml", output_directory="out/")
```

## Browse by axis / option

To find recipes that exercise a specific axis or option, browse the
[encyclopedia](../encyclopedia/index.md) and follow the "see also"
cross-references back to example recipes.
