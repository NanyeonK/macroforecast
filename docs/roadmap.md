# Roadmap

---

## v0.1.0 — Data Layer (Current)

Released: March 2026

**Completed:**

- `load_fred_md()` — FRED-MD monthly loader with caching, vintage support, and tcode parsing
- `load_fred_qd()` — FRED-QD quarterly loader
- `load_fred_sd()` — FRED-SD state-level loader (Excel, openpyxl)
- `MacroFrame` — immutable panel container with fluent API
- `TransformCode` enum and `apply_tcode` / `apply_tcodes` — all seven McCracken-Ng transformations
- `classify_missing` / `handle_missing` — three missing-type classification and five treatment methods
- `list_available_vintages` / `load_vintage_panel` / `RealTimePanel` — vintage enumeration and multi-vintage loading
- `macrocast.utils.cache` — local file caching with age-based expiry

**Test coverage:** 107 tests passing. All Layer 1 modules covered.

---

## v0.2.0 — Forecasting Pipeline (Complete)

Released: March 2026

**Completed:**

- `ForecastExperiment` — outer pseudo-OOS loop orchestrator with expanding and rolling window strategies
- `MacrocastEstimator` and `SequenceEstimator` — abstract base classes for cross-sectional and sequence models
- Component definitions: `Nonlinearity`, `Regularization`, `CVScheme`, `LossFunction`, `Window` — enum-like objects, never plain string flags
- Python model zoo: `KRRModel`, `SVRRBFModel`, `SVRLinearModel`, `RFModel`, `XGBoostModel`, `NNModel`, `LSTMModel`
- `macrocastR` linear model suite: Ridge, LASSO, Adaptive LASSO, Group LASSO, Elastic Net, ARDI (R-side via parquet exchange)
- `FeatureBuilder` — PCA diffusion index factors (ARDI mode) and AR-only mode, strict pseudo-OOS discipline
- `ModelSpec` and `FeatureSpec` — dataclass configuration bundles
- `ForecastRecord` and `ResultSet` — result containers with parquet serialisation
- Direct multi-step forecasting only (one model per horizon h); iterated forecasting out of scope
- Parallel execution via `joblib` (`n_jobs` parameter)
- YAML-based experiment configuration via CLI

---

## v0.3.0 — Evaluation Layer (Complete)

Released: March 2026

**Completed:**

- `decompose_treatment_effects()` — four-component OLS decomposition (CLSS 2022, Eq. 11) with HC3 standard errors
- `regime_conditional_msfe()` — regime-conditional evaluation via quantile or binary indicator splits
- `mcs()` — Model Confidence Set (Hansen, Lunde, Nason 2011) with stationary block bootstrap
- `dm_test()` — Diebold-Mariano (1995) test with Harvey-Leybourne-Newbold (1997) finite-sample correction
- Core metrics: `msfe`, `mae`, `relative_msfe`, `csfe`, `oos_r2`
- Dual weight representations: `krr_dual_weights`, `tree_dual_weights`, `nn_dual_weights` (CGK 2024)
- `effective_history_length`, `top_analogies` — dual weight diagnostics
- `oshapley_vi`, `compute_pbsv`, `model_accordance_score` — PBSV / oShapley-VI (CBRSS 2022)

---

## v0.4.0 — Paper Submission (Planned)

Target: August 2026

**Planned:**

- Full empirical illustration replicating Coulombe et al. (2022) Table 3–5
- Supplementary visualisations: waterfall plots, CSFE paths, regime heatmaps, horizon profiles
- Submission to *International Journal of Forecasting* Special Issue

---

## Paper Submission

Target: August 2026

Submission to *International Journal of Forecasting* Special Issue "Advances in Open Source Forecasting Software".

---

## Known Limitations in v0.1

- EM-based imputation (`method="em"`) raises `NotImplementedError`. Not required for the core decomposition analysis and will not be implemented.
- `RealTimePanel` provides basic access only. Pseudo-out-of-sample alignment utilities are planned for v0.2.
- The `target` parameter in `load_vintage_panel` is accepted but not yet applied.
