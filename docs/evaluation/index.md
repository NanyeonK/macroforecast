# Evaluation Layer

The Evaluation Layer (Layer 3) assesses the OOS forecast accuracy of all models produced by `ForecastExperiment`. All functions in this layer operate on the tidy DataFrame returned by `ResultSet.to_dataframe()`, which has one row per forecast cell (model, horizon, date).

The benchmark throughout is the AR(p) model with lag order selected by BIC, identified by `model_id="linear__none__bic__l2"`. All relative accuracy measures are defined with respect to this benchmark.

---

## Modules

| Module | Key Exports | Purpose |
|--------|-------------|---------|
| `metrics` | `msfe`, `mae`, `relative_msfe`, `csfe`, `oos_r2` | Scalar and time-series forecast accuracy metrics |
| `decomposition` | `decompose_treatment_effects`, `DecompositionResult` | OLS decomposition of OOS-R² gains by treatment (CLSS 2022) |
| `mcs` | `mcs`, `MCSResult` | Model Confidence Set (Hansen, Lunde, Nason 2011) |
| `dm` | `dm_test`, `DMResult` | Diebold-Mariano test with HLN small-sample correction |
| `regime` | `regime_conditional_msfe`, `RegimeResult` | Regime-conditional MSFE by quantile splits of a regime indicator |
| `dual` | `krr_dual_weights`, `tree_dual_weights`, `nn_dual_weights`, `effective_history_length`, `top_analogies` | Dual observation weight matrices for KRR, tree, and NN models |
| `pbsv` | `oshapley_vi`, `compute_pbsv`, `model_accordance_score` | oShapley variable importance and PBSV time-varying contributions |

---

## Benchmark Convention

The baseline model is AR(p) with p selected by BIC over the inner CV loop. Its `model_id` is `"linear__none__bic__l2"`. Every relative metric in this layer divides the candidate model's loss by the AR benchmark's loss. Values below 1.0 indicate improvement over the benchmark.

---

## Input Format

All evaluation functions that accept a `result_df` argument expect a DataFrame with at minimum the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `model_id` | str | Unique model identifier |
| `forecast_date` | datetime | Date of the forecast target |
| `horizon` | int | Forecast horizon h |
| `y_true` | float | Realized value |
| `y_hat` | float | Point forecast |

This format is produced directly by `ResultSet.to_dataframe()`.

---

## Sub-pages

- [Metrics](metrics.md)
- [Treatment Effect Decomposition](decomposition.md)
- [Model Confidence Set](mcs.md)
- [Diebold-Mariano Test](dm.md)
- [Regime-Conditional Evaluation](regime.md)
- [Dual Observation Weights](dual.md)
- [PBSV and oShapley Variable Importance](pbsv.md)
- [Full API Reference](../api/evaluation.md)
