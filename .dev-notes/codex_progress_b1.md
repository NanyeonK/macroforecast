# B1 Medeiros (2021) replication — progress log

Worktree: `~/project/mf-b1-medeiros` on `repro/medeiros-2021` @ 4bbae29e (from main).
Rules: reproduction scripts only under `scripts/replication/`, no package-code edits,
no push/gh. Stop before G3. Oracles cloned into gitignored `qa/` (read-only reference).

## Status: STOPPED at S1 (pre-G1) — 3 hard gates hit, human input needed.

### Resolved / unblocked
- **Worktree** created and isolated.
- **Data acquisition SOLVED without network.** stlouisfed S3 is blocked (HTTP 403), but the
  FRED-MD 2016-01 vintage is already cached at
  `~/project/macroforecast_replication_cache/fred_md/historical/historical-vintages-of-fred-md-2015-01-to-2024-12.zip`
  -> member `2016-01.csv` (589 KB, 685x135 = 134 series + sasdate, 684 months).
- **Oracles obtained (GitHub raw IS reachable).** Cloned into `qa/`:
  - `qa/HDeconometrics` (author Vasconcelos R package): csr, boosting, bagging, lbvar (BVAR),
    ic.glmnet (LASSO/adaLASSO/ElNet + BIC lambda), fitLambda.
  - `qa/ForecastingInflation` (author's actual paper replication scripts):
    `01_get_fred_data.R`, `03_call_model.R`, `functions/functions.R`, `functions/rolling_window.R`.
- **Tuning [GAP]s resolved from author code** (design risk #2/#4):
  - RF: `randomForest(Xin,yin,importance=TRUE)` = package DEFAULTS (ntree=500, mtry=p/3). No custom ntree.
  - RF/OLS hybrid: `randomForest(..., maxnodes=25, ntree=500)`.
  - adaLASSO: penalty `(|beta|+1/sqrt(n))^(-1)`, lambda via `ic.glmnet` BIC.
  - CSR: `csr(Xin,yin,fixed.controls=c(f.seq,ncol))` -> HDeconometrics defaults **K=20, k=4**.
  - bagging: `bagging(Xin,yin,R=100,l=5,pre.testing="group-joint")`.
  - target = **CPIAUCSL**, direct h=1..12, rolling `nwindows=180` (2001-15 block).

### GATE 1 [BLOCKER] — reference paper identity/citation is WRONG
- On-disk `~/archive/macroforecast_bak_stale_20260407/references/medeiros_etal_2021_jbes.pdf`
  is actually **Hauzenberger, Huber, Koop & Onorante (2022), "Fast and Flexible Bayesian
  Inference in Time-varying Parameter Regression Models," JBES 40(4):1904-1918** — a
  DIFFERENT paper (confirmed from PDF page-1 title/authors).
- The correct paper is **Medeiros, Vasconcelos, Veiga & Zilberman (2021), International
  Journal of Forecasting 37(2):419-436, DOI 10.1016/j.ijforecast.2021.01.001** (per
  `~/second_brain/00_wiki/sources/inflation-forecasting-machine-learning-medeiros-2019.md`).
  The design doc's venue "JBES 39(1) 2021" is incorrect. The correct paper is NOT on disk.

### GATE 2 [BLOCKER, S1] — UCSV gamma 0.12 is CONTAMINATED
- Design doc §0/§0.5-D2: "UCSV(Vtau=Vh=0.12)". The mislabeled Hauzenberger PDF's simulation
  DGP is `y_t=beta_t+eps_t, eps_t~N(0,0.12)` and `beta_t~N(m_t,0.12)` — two variances both
  0.12. This is an exact match to "Vtau=Vh=0.12": the design author almost certainly
  transcribed Hauzenberger's DGP variances as the Medeiros UCSV parameter.
- Medeiros's UCSV is the **Stock-Watson (2007) UCSV benchmark**, whose standard calibration is
  **gamma = 0.2 = the VARIANCE of both log-volatility RW innovations**. That is exactly the
  macroforecast package default (`models/bayesian.py:40`, documented as variance).
- The author's ForecastingInflation repo contains **no UCSV code** (RW/AR are its only
  univariate benchmarks), so the numeric value cannot be confirmed from author code; it must
  be read from the correct IJF paper text.
- **Decision / recommendation:** use `ucsv(gamma=0.2)` (variance, package default; do NOT
  square, do NOT use 0.12), pending confirmation against the IJF 2021 UCSV spec. Running UCSV
  with 0.12 would make all UCSV numbers wrong (the exact risk the handoff flagged).

### GATE 3 [BLOCKER] — 122-series screen gives 112, not 122
- Frozen 2016-01 raw completeness over 1960-01..2015-12 (672 months) -> **112** fully-observed
  series (of 134). Paper reports 122.
- Root cause (from author `01_get_fred_data.R`): the author does NOT freeze the 2016-01
  vintage. They pull `.../monthly/current.csv` via `fbi::fredmd`, apply tcode transforms
  FIRST, intersect with `fredmd_description`, DROP `group=="Prices" & tcode==6` and re-add
  those as `100*diff(log)`, then `select_if(~ !any(is.na(.)))`. Current-vintage backfill +
  transform-then-screen changes which series are balanced -> ~122.
- So the design's "freeze 2016-01 vintage" will NOT reproduce the paper's 122. Need a decision:
  (a) match the author exactly (current.csv + fbi transforms) for parity, or (b) keep the
  frozen 2016-01 vintage and accept ~112 + a vintage-sensitivity note. The paper's online
  appendix series list is required to lock either way.

### Downstream consequence — G2 parity cannot be scored yet
- G2 requires the paper's numeric tables (T5 RF/RW RMSE ratios, T8 relative-RMSE columns) as
  the parity oracle. Those live in the correct IJF paper (not on disk). The wiki note gives
  only qualitative results ("~30% MSE reduction", "UCSV slightly > AR"), insufficient for
  MATCH(+-0.01)/CLOSE(+-0.05) scoring.

### Also flag
- Design D7 "CSR ñ=25": author code uses HDeconometrics csr default **K=20**. Verify vs paper.

## Human gate requested
Provide the correct **Medeiros, Vasconcelos, Veiga & Zilberman (2021), IJF 37(2):419-436**
PDF + online appendix (series list / UCSV spec / exact CSR ñ / parity tables). The author's
replication CODE is already obtained; the missing piece is the paper PROSE + TABLES (for
parity target and UCSV gamma) and the exact 122-series manifest.

## 2026-07-08 -- Direction change: G2 runner moved onto pipeline, P4 blocked

- Read the required official pipeline docs first:
  `docs/guide/concepts/running.md`, `docs/reference/pipeline.md`,
  `docs/guide/getting_started.md`, and `docs/guide/model_overview.md`.
- Rewrote `scripts/replication/medeiros_2021_pipeline/run_block.py` to use
  `pipeline_spec()`/`run_pipeline()` instead of the single-arm model x horizon loop.
  The runner now:
  - uses arms `rw`, `ar`, `ucsv`, `rf`;
  - passes RF params explicitly as `n_estimators=500`, `max_features=1.0/3.0`,
    `random_state=42`;
  - omits RF `n_jobs`;
  - uses `n_jobs="auto"`, `seed=42`, `result_store="qa/result_cells"`,
    `preprocessing_cache_dir="qa/prep_cache"`, `save_models=False`, and
    `EvalSpec(benchmark="rw")`;
  - preflights RW/AR before RF/UCSV and stops before slow arms if P4 fails.
- Added runner-local `MedeirosRollingWindow`, a pickleable `WindowSpec` subclass, to preserve
  the paper's horizon-specific rolling sizes while staying inside pipeline execution:
  s1 R = 354/352/349/343 and s2 R = 486/484/481/475 for h = 1/3/6/12.
- Updated `scripts/replication/medeiros_2021_pipeline/registry.py` to remove hardcoded
  RF `n_jobs=48` and to give the custom Nov-2008 dummy an `__mf_digest__` for result-store reuse.
- P4 result: blocked. Current `run_pipeline` RW/AR ratios are
  `0.911304 / 0.789546 / 0.792246 / 0.764836`, with RW RMSE
  `0.2889 / 0.3847 / 0.3852 / 0.3990`. The archived target
  `qa/g2_rw_ar_v2.out` expects `0.914 / 0.761 / 0.759 / 0.763`.
- Diagnostic: the old low-level h=3 loop rerun under the current package also returns
  `0.7895463001107798`, so the pipeline rewrite matches current low-level semantics; the
  conflict is with the archived v2 parity artifact.
- RF/UCSV were not run, per gate discipline. No CSR/JMA/LASSO/bagging/hybrids/G3 were run.
- Wrote findings to `.dev-notes/replication_findings_medeiros.md` and updated
  `docs/replication/medeiros_2021.md` with the pipeline recipe, parity table, and gaps.

## 2026-07-08 -- Oracle corrected; full G2 four-arm pipeline run completed

- Read the workplan's "ORACLE CORRECTION (2026-07-08)" section and re-baselined P4 to the
  IJF Table 5 paper oracle, not stale `qa/g2_rw_ar_v2.out`.
- Edited only `scripts/replication/medeiros_2021_pipeline/run_block.py` in scripts:
  removed the pre-RF RW/AR stop gate and replaced the stale archived-AR assertion with a
  paper Table 5 tolerance summary.
- Ran the requested full `run_pipeline` arm set (`rw`, `ar`, `ucsv`, `rf`) with
  `n_jobs="auto"`, `seed=42`, `result_store="qa/result_cells"`,
  `preprocessing_cache_dir="qa/prep_cache"`, and `save_models=False`.
- P4 AR parity is stable against paper Table 5:
  `0.911304 / 0.789546 / 0.792246 / 0.764836` vs
  `0.902 / 0.790 / 0.791 / 0.753`.
- Full Table 5 scores:
  - AR: `MATCH / MATCH / MATCH / CLOSE`.
  - UCSV: `CLOSE / DIVERGENT / DIVERGENT / DIVERGENT`.
  - RF: `DIVERGENT / CLOSE / CLOSE / DIVERGENT`.
- Efficiency: successful cached/resumed scoring command reported
  `pipeline total = 2719.5s` vs sequential `15854.9s = 4.40h`, a `5.83x` validated speedup.
  Conservative cold/resume completed-cell lower bound is `4221.5s`, still `3.76x`.
- Gap found: first full command completed cold `s1` (`1504.7s`) but idled in
  `ProcessPoolExecutor.map` during `s2`; interrupt + result-store resume completed and
  produced the parity table.
- Updated `.dev-notes/replication_findings_medeiros.md` and
  `docs/replication/medeiros_2021.md` with the corrected oracle, parity table, arm map,
  runtime, and gaps. No package code edited. No git/gh used. No
  CSR/JMA/LASSO/bagging/hybrids/G3 run.

## 2026-07-09 -- B1 G2e RF/UCSV divergence resolution pass

- Loaded the replication objectives from the sibling `.dev-notes/REPLICATION_OBJECTIVES.md`
  because this worktree did not contain that file, then rechecked docs/reference,
  docs/guide, and `docs/replication/medeiros_2021.md`.
- Verified the author RF call: Table-5 `RF` uses `runrf()` and plain
  `randomForest::randomForest(Xin, yin, importance=TRUE)`. `maxnodes=25` is in
  `runrfols()`, so it belongs to RF/OLS, not the plain RF row.
- Updated only replication runner config, not package code: RF now passes
  `n_estimators=500`, `max_features=1/3`, `min_samples_leaf=5`, `bootstrap=True`,
  `max_samples=None`, `max_leaf_nodes=None`, `random_state=42`, and
  `model_selection={"random_forest": None}` in both `run_block.py` and the arm registry.
- Found that `model_selection=None` silently enabled the package default RF search and
  overrode fixed author params. An intermediate RF rerun with search still enabled took
  `1611.3s` and reproduced the old RF ratios; decoded cells showed varying
  `n_estimators`, `max_depth`, and `min_samples_leaf`.
- Final fixed-author RF rerun took `312.2s` and produced RF ratios
  `0.907273 / 0.740947 / 0.730065 / 0.737382`. RF h=3/h=6 are CLOSE; h=1/h=12 remain
  DIVERGENT but improved.
- UCSV audit found no lookahead in the direct-policy target filtering or `ucsv()` final-trend
  forecast path. The paper gives `Vtau=Vh=0.12` initial-prior variances and MCMC wording, but
  the author repos contain no UCSV implementation and do not publish draw/burn/gamma values.
- Ran the cached all-G2 scorer with `run_pipeline(..., n_jobs="auto")`; result store reuse
  completed in `6.8s` and left AR/RW unchanged. No G3 or other arms were run. No git/gh used.

## 2026-07-09 -- B1 G2f RF residual-divergence diagnosis

- `.dev-notes/REPLICATION_OBJECTIVES.md` was still absent in this worktree, so used the
  task note `.dev-notes/codex_task_g2f_rf.md` plus the existing replication objectives
  already captured in local findings/docs.
- Reconstructed author RF matrix from `qa/ForecastingInflation`: `runrf()` calls
  `dataprep()`, which fits `princomp(scale(df))` on the full in-window panel including
  `CPIAUCSL`, binds four factors to `df`, applies `embed(..., 4)`, removes the final
  `h` rows from `Xin`, aligns `yin=tail(y,nrow(Xin))`, and appends the Nov-2008 dummy.
- Diagnosed the residual RF gap as PROTOCOL, not backend: previous runner PCA omitted the
  target, included factors only at lag 0, and did not emit R `embed` lag-block order.
- Patched only `scripts/replication/medeiros_2021_pipeline/registry.py`: added a digestible
  runner-local custom feature step that reconstructs R-style `princomp(scale(df))` and
  `embed(cbind(df, factors), 4)` from `qa/medeiros_panel.parquet`.
- Ran `python3 scripts/replication/medeiros_2021_pipeline/run_block.py --arms rf`; RW was
  included automatically for scoring and reused from `qa/result_cells`; changed RF cells
  recomputed in `645.0s`.
- New RF ratios are `0.859709 / 0.732985 / 0.749831 / 0.710726`, all `CLOSE` against IJF
  Table 5. No package code patched. No UCSV, AR, G3, CSR/JMA/LASSO, bagging, hybrids,
  tolerance loosening, subsampling, reduced trees/draws, or coarser grids.
