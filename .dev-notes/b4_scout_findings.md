# B4 Scout Findings — GCLS 2022 replication feasibility (READ-ONLY)

Scout date: 2026-07-16. Worktree: `/home/nanyeon99/project/mf-b4-gcls2022` (branch `repro/gcls-2022`, HEAD `5c625368`). Package `macroforecast` 0.9.5 imports clean.
Scope: data verdict + 46-model/API delta + first blocking dependency. NO package patch, NO full run, NO downloads, NO push.
Basis: design `phaseB_design_b4_gcls2022.md` (2026-07-08); official archive; live `inspect.signature` on current tree. NOTE: the design's file:line refs (`search.py:477`, `specs.py:1415/1596/3798`, `spec.py:120`, `window/core.py:296`) are STALE — package reorganized into subpackages since 07-08; verified against current API below.

## TASK 1 — DATA VERDICT

Official ZBW/JAE archive located and integrity-stamped:
`/home/nanyeon99/second_brain/00_wiki/raw/paper_code/jae_2910_glss_replication_20260602/`
- `glss-files.zip` (68,486,664 B, SHA256 `64c63aece…81e4c` per MANIFEST.md), `readme.glss.txt`, `MANIFEST.md`. Data-only (no .m/.R/.py/.do/.ipynb) — matches design assumption of no official code.

### Main panel — BUILDABLE NOW
`MainAnalysis/2018-01.csv` inside the zip (591,557 B).
- Format: genuine McCracken-Ng FRED-MD — row1 header `sasdate,RPI,W875RX1,DPCERA3M086SBEA,CMRMTSPLx,RETAILx,…`; row2 `Transform:` tcodes (5,5,5,5,5…).
- Columns: 129 (= sasdate + 128 series). Rows: 711 (2 header + 709 data). Dates: 1959M01 → 2017M12.
- [GAP-minor / reconcile] Paper text (tex L533) claims "134 monthly … indicators, 1960M01–2017M12". Actual vintage file has 128 series and starts 1959M01. The 1960M01 start is post-transformation (McCracken-Ng transforms consume initial obs); 134-vs-128 is the usual FRED-MD nominal-vs-vintage gap. Reconcile retained-series count vs the paper's variable table during G1; not a blocker.
- An identical `2018-01.csv` also sits under `SupplementaryMaterial/FredMD_Vintages/` (same 591,557 B). Pre-existing cache copy at `~/project/macroforecast_replication_cache/fred_md/vintages/2018-01.csv` (129 cols, 1959M01–2017M12) — usable, but prefer extracting from the integrity-stamped archive.

### 4 interaction series (Table 3) — BUILDABLE NOW (all present in `MainAnalysis/`)
- MacroUncertainty (Jurado-Ludvigson) → `MacroUncertaintyToCirculate.xlsx`
- NFCI / ANFCI → `National_Financial_Conditions_index.xls`
- CSUSHPINSA → `CSUSHPINSA.xls`
- UMCSENT → `UMCSENT.xls`
- (bonus `PMI.xlsx`). All .xls/.xlsx binary — parse with pandas/openpyxl at build time.

### Wiley supplement S5/S6 (CV algorithm + HP grids) — PARTIAL GAP, mostly recovered
- The file flagged in the design, `GCLSS1_v20200217_SuppMaterial_WP_title.pdf`, is a 1-page title page only (pdftotext yields none of CV/λ/σ/C/grid). It is NOT the CV/HP appendix.
- BUT the supplementary material is embedded in the main paper TeX `…/GCLSS_JAE_TexFiles/GCLSS_JAE_13527_2.tex` (from L1258; `\label{CVdetails}` L1644, `\label{sec:models}` L1674). From it, fully specified:
  - POOS-CV = last 25% of in-sample; K-fold k=5; retune every 24 months (L572, L1668).
  - Discrete grids: p_y,p_f ∈ {1,3,6,12}, n_f/K ∈ {3,6,10} (L1742, L1784, supp L510/L556).
  - Random Forest: 500 trees, mtry = 1/3 (L1715–1718).
  - Elastic-Net ζ ∈ {0, ζ̂, 1}; 9 shrinkage specs × 2 CV = 18 (L383–397).
  - Density (deferred): QARDI grid p_y,p_f∈{1,3,6,12}, n_f∈{3,6,10}, K=5, 24-mo (L1197); skewed-t τ∈{0.1,.25,.5,.75,.9} (L1194).
- [GAP-real, LOW severity] Continuous penalty grids are never given numerically: λ (Ridge/KRR/EN), σ (RBF scale), C and ε̄ (SVR). The TeX only says "the Ridge hyperparameter is selected … by CV" (L1701-1706) etc. Reconstruct from GCLS-2021 public R code (`macrorf`/`tvpridge` on disk under `…/coulombe_site_github_20260530/`) or standard kernlab/e1071/sklearn defaults, and document as a labelled deviation. Design R4 stands; treatment-effect signs (primary parity target) are robust to grid choice, so this does not block start.

Verdict: main panel + all 4 interaction series + real-time vintages (B6, deferred) + FREDQD/Canadian (S3/S4, excluded) = buildable-now from on-disk archive. Only continuous-penalty grids need reconstruction (needs-user judgement / borrow 2021 R code), not acquisition.

## TASK 2 — 46-MODEL REGISTRY & API DELTAS (verified live on current tree)

### Native model builders — ALL POINT-FORECAST MODELS PRESENT (no custom_model)
`mf.ar`, `mf.ar_bic(criterion in {bic,aic,aicc})`, `mf.far(n_factors,n_lag,direct)`, `mf.ridge(alpha)`, `mf.random_forest(n_estimators=500, max_features=1/3)` ← defaults already = GCLS, `mf.kernel_ridge(alpha,kernel,gamma)`, `mf.svr(kernel,C,epsilon,gamma,degree)` (+`linear_svr`,`nu_svr`), `mf.elastic_net(alpha,l1_ratio)`, `mf.lasso`, `mf.quantile_regression_forest(quantile_levels)`.
- 46 point-forecast models = native builders + feature pipelines. custom_model NOT required — confirmed.
- `far` exposes single `n_lag` (+`n_factors`); GCLS ARDI needs separate (p_y, p_f, n_f). needs-config: build ARDI arms via feature pipeline (`target_lags=p_y`, factor `lags=p_f`, `pca_components=n_f`) + linear/ridge, or minor `far` extension. Not absent.
- QARDI (linear quantile regression + ARDI) = ABSENT — no bare `quantile_regression` atom (only `quantile_regression_forest`). Density-lane, deferred per design §1c; add a `quantile_regression` atom later or defer. Not blocking point-forecast replication.

### Delta A2 — IC path: PRESENT (design "post-AICc" note confirmed)
`SearchSpec(method="information_criterion")` exists (`model_selection/search.py:125,649`); criterion in {aic, aicc, bic} enforced (search.py:654); `select_by_information_criterion` exported; `_gaussian_information_criterion` has aicc branch. AR,BIC/AIC also via `ar_bic` directly. IC scores over the (p_y,p_f,n_f) `param_grid`.

### Delta A3 — axis_contribution: PRESENT, all 3 deltas already shipped (`analysis/contribution.py:14`)
`axis_contribution(master, *, features, outcome='r2', fixed_effects=('target','horizon','date'), interactions=None, hac_lags=None, vcov='driscoll_kraay', cluster_by='date', reference=None)`. Docstring explicitly cites GCLS 2022 — purpose-built.
- (1) e²→R² util: `outcome="r2"` = 1 − e²/MSE(reference) GCLS pseudo-R².
- (2) ψ(t,v,h) FE: `fixed_effects=('target','horizon','date')` joint FE.
- (3) interaction series (Table 3): `interactions: Mapping[str,pd.Series]`.
- BONUS: HAC / Driscoll-Kraay `vcov`,`hac_lags`,`cluster_by` = "SEs are HAC". Design listed these as deltas-to-build; they are already native.

### Delta A4 — NBER mask: PRESENT
`_NAMED_SUBSAMPLE_MASKS={"nber_recession","nber_expansion"}` (`pipeline/spec.py:344`), wired to USREC(monthly)/USRECQ(quarterly) with invert (`pipeline/evaluate.py:120-121`). Use `SubsampleWindow(mask="nber_recession")`. [minor] confirm USREC resolves offline before G3 (not in the GCLS archive).

### Window / retune API (design §1a, R2 trap): PRESENT & decoupled
`from_cutoffs(…, retrain_every=1, retune_every=1, retune_on_retrain=True, val_method, val_ratio=0.2, val_n_splits=5,…)`. GCLS config: `retrain_every=1` (monthly refit), `retune_every=24`, `retune_on_retrain=False` — all three knobs native (`window/core.py:301,389,390`). `poos(validation_ratio=0.25)` = POOS-CV last 25% (native default). k=5 via `val_n_splits=5` + `random_kfold`. R2 refit-count assert expressible.

### Delta A6 — DM/MCS/relative-RMSPE tables: NATIVE in EvalSpec (adapter mostly moot)
`EvalSpec(benchmark, metrics=('rmse','relative_mse','r2_oos'), tests=('dm','cw','mcs'), subsamples: Mapping[str,SubsampleWindow], mcs_alpha, mcs_method,…)`. DM+CW+MCS+relative-MSE all built-in; full/NBER via `subsamples={'full':SubsampleWindow(),'nber_rec':SubsampleWindow(mask='nber_recession')}`. Remaining A6 work = the 46×5×2 table formatter (cosmetic/reporting), not a capability gap.

### Feature pipelines (B1/B2/B3 shrinkage × feature-scope): PRESENT
`FeatureSpec(predictors='all'|tuple, lags, target_lags, pca_components, pca_columns, feature_steps, include_original,…)`; builders `pca_features`, `lags_then_pca` (B3: lag→PCA), `pca_then_lags`, `group_pca`. `fit_policy='expanding'` = recursive factors. B1=EN/lasso/ridge on identity X; B2=pca_features (rotation, keep-all → set `n_components=min(N,T)`, needs-config, no 'all' sentinel); B3=lags_then_pca. EN `l1_ratio`=ζ (1→lasso,0→ridge,ζ̂→CV).

### SVR kernels spot-check: PRESENT
`svr(kernel in {linear,rbf}, C, epsilon(=ε̄ tube), gamma(=σ), degree)`. All GCLS SVR arms (Lin/RBF × POOS/K-fold) expressible.

### Pipeline skeleton names (design §3): PRESENT under `mf.pipeline`
`pipeline_spec(data,targets,horizons,window,arms,evaluation,…,result_store=,n_jobs=,seed=42)`, `PipelineSpec`, `run_pipeline(spec)`, `EvalSpec`, `SubsampleWindow`, `TargetSpec`, `Arm`. [minor] these live in `mf.pipeline.*` (not top-level `mf.*`) — design skeleton's `mf.EvalSpec`/`mf.pipeline_spec` need `mf.pipeline.` prefix or import.

## TASK 3 — RECOMMENDED PLAN + FIRST BLOCKING DEPENDENCY

Stage gates G1(smoke, 2 arm×1 target×h1, 456 origins + refit-count assert) → G2(Table A1 INDPRO, 46 arm×h5, AR,BIC-relative parity) → G3(all 5 targets + axis_contribution Figs 1-2/Table 2) → G4(Table 3 interactions + robustness subsamples). All four gates are actionable against the current API.

Scale ≈ 520k fits (1,150 cells × 456 origins; retune only 19×/cell). Survival path = result_store incremental + n_jobs parallel + staged arm submission — all native: `ResultStore` class (`pipeline/result_store.py:126`), `pipeline_spec(result_store=…, n_jobs=…)`, `result_store_summary`, `purge_result_store`. Incremental cell-identity digest (`result_cell_identity` + `SearchSpec.to_dict`) is implemented and its tests pass in this worktree (`tests/pipeline/test_result_store.py` validation_splitter/digest_tracks: 2 passed per codex_progress_a2). B4 depends on result_store for G2+ resumption; the "separate result_store bug" is NOT visible as an open defect in this worktree's dev-notes — re-run `test_result_store.py` green before scale-out, but it is not a demonstrated blocker here.

Transferable asset confirmed: `docs/replication/gcls_2021_replication.{md,py}` + `data/` — clone the layout for `gcls_2022_*` (no 2022 stub exists yet).

### FIRST BLOCKING DEPENDENCY
Extract + validate the panel from the archive and lock the ARDI/(p_y,p_f,n_f) arm construction. Concretely, before any fit: (1) unzip `MainAnalysis/2018-01.csv` from the integrity-stamped `glss-files.zip`, apply McCracken-Ng tcodes with the CPI I(1) override (footnote 19), reconcile retained-series count vs the paper's variable table (128 vs nominal 134); (2) settle the ARDI arm mapping since `far` carries a single `n_lag` but GCLS needs separated (p_y, p_f, n_f) — decide feature-pipeline route vs minor `far` extension. Everything downstream (46-arm registry, G1 smoke) is gated on these two. Second-order (non-blocking for G1, needed by G2/G3): reconstruct continuous λ/σ/C/ε̄ grids (borrow GCLS-2021 R code) and confirm USREC offline for the NBER split.
