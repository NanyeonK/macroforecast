# CODEX WORKPLAN — B1 Medeiros (2021) G2 four-model panel

**Author of this plan:** Fable (Claude), design/verification owner. You (codex) are the
executor. Do exactly what this plan says; escalate anything ambiguous into the progress log
rather than improvising on modeling choices.

**Mission (this run only):** produce the **G2 four-model parity panel** — RW / AR / RF / UCSV,
target **CPIAUCSL**, horizons **h in {1,3,6,12}** — RMSE ratios vs RW compared against the
paper's Table 5. Score MATCH/CLOSE/DIVERGENT, write the parity table into
`docs/replication/medeiros_2021.md`, log to `.dev-notes/codex_progress_b1.md`, then **STOP**.

## 0. HARD CONSTRAINTS (do not violate)
1. Work **only** inside `~/project/mf-b1-medeiros` (branch `repro/medeiros-2021`).
2. Edit **only** reproduction scripts under `scripts/replication/medeiros_2021_pipeline/`
   and docs/progress files. **Never edit package code** under `src/` / `macroforecast/`.
   If a package bug blocks you, work around it in the repro script and log it — do NOT patch
   the package.
3. **No git commit, no push, no `gh`.** Chan handles all git/GitHub.
4. **STOP after the 4-model G2 panel is scored.** Do NOT run CSR, JMA, LASSO/ridge/elastic-net,
   bagging, boosting, hybrids, the other subperiod tables, or any G3 sweep. Those are gated on
   Chan's confirmation.
5. If you hit a genuine blocker (data missing, model errors you cannot work around, runtime
   explosion), STOP and write the blocker to `.dev-notes/codex_progress_b1.md`.

## 1. LOCKED FINDINGS (verified by Fable — treat as ground truth, do not re-litigate)
- **Canonical paper:** Medeiros, Vasconcelos, Veiga & Zilberman (2021), *Int. J. Forecasting*
  37(2):419-436. Full text extracted at `qa/medeiros_correct_fulltext.txt`. (The old
  `medeiros_etal_2021_jbes.pdf` is a DIFFERENT paper — ignore it.)
- **UCSV:** Stock-Watson (2007) benchmark. `ucsv(gamma=0.2)` where **gamma = 0.2 is the
  VARIANCE** of both log-vol RW innovations (package default). Do NOT use 0.12. Do NOT square.
- **RF:** `randomForest` **defaults**: `ntree=500`, `mtry=p/3`, importance on. No custom ntree.
- **AR:** univariate, **BIC** order selection, direct multistep. (Already wired in `run_block.py`
  via `UNIVAR_AR = dict(predictors=None, target_lags=(0,1,2,3))`.)
- **Author-code-wins rule:** where the design handoff and author R code (`qa/ForecastingInflation`,
  `qa/HDeconometrics`) disagree, author code is authoritative.
- (Out of G2 scope, recorded for later: **CSR K=20 k=4**; bagging R=100 l=5 group-joint;
  RF/OLS maxnodes=25 ntree=500; adaLASSO ic.glmnet BIC. Do NOT run these now.)

## 2. WORKING ASSETS (already built and validated — reuse, do not rebuild)
- **Prepared panel:** `qa/medeiros_panel.parquet` (author transform-then-screen order; 112 series
  vs paper's 122 = vintage-timing GAP, already documented — acceptable for G2).
- **Arm registry:** `scripts/replication/medeiros_2021_pipeline/registry.py` (params locked to
  author code).
- **G2 runner:** `scripts/replication/medeiros_2021_pipeline/run_block.py`. Two-regime fixed
  rolling concatenation:
    - s1 1990-01..2000-12: R = 360 - h - p - 1
    - s2 2001-01..2015-12: R = 492 - h - p - 1   (p = 4 embedding lags)
  **RW and AR are already VALIDATED** against Table 5 (latest good run `qa/g2_rw_ar_v2.out`):
    AR ratios 0.914/0.761/0.759/0.763 vs paper 0.902/0.790/0.791/0.753 (d = +0.012/-0.029/
    -0.032/+0.010). This tracks the paper — do not touch the RW/AR path.
  The runner already accepts `--ucsv` (adds `ucsv(gamma=0.2)`). It does **not yet** run RF.

> NOTE: the "STOPPED at S1" banner at the top of `.dev-notes/codex_progress_b1.md` is STALE.
> G1 passed and G2 RW/AR is validated. Append a fresh dated section; do not delete history.

## 3. TASK — add RF + UCSV to the panel and score all four
1. Extend `run_block.py` so `models` can include **RF** and **UCSV** alongside RW/AR:
   - RF arm: `model="random_forest"`, `features=base_features()` from `registry.py`
     (all series + 4 PCA factors, lags 0..3, + Nov-2008 dummy). RF **defaults** (do not pass
     ntree/mtry — the package default is ntree=500, mtry=p/3; confirm and note the actual
     defaults the package uses in the log).
   - UCSV arm: `model="ucsv"`, `features=None`, `params={"gamma": 0.2}`.
   - RW/AR paths stay exactly as they are.
2. Keep `save_models=False` (memory). RF over ~460 features x rolling refits x 2 regimes x 4
   horizons is the slow part — expect this run to take a while. If a single (model,horizon)
   exceeds ~45 min wall, log it and continue with what completed; partial is fine, note gaps.
3. Compute RMSE ratio vs RW for each model at h in {1,3,6,12}, aligned to RW's OOS set per horizon.
4. Also emit RW absolute RMSE and OOS n per horizon (for provenance).

## 4. PARITY ORACLE — paper Table 5, Panel (a) RMSE ratios vs RW (full sample 1990-2015)
Source lines `qa/medeiros_correct_fulltext.txt:876-878`.

| model | h=1  | h=3  | h=6  | h=12 |
|-------|------|------|------|------|
| AR    | 0.902| 0.790| 0.791| 0.753|
| UCSV  | 0.954| 0.797| 0.777| 0.781|
| RF    | 0.844| 0.706| 0.715| 0.685|

(For reference only, not required this run — MAE panel (b) `:882-884`, MAD panel (c) `:888-890`.)

## 5. GRADING RULE
- **RW / AR:** deterministic benchmarks. Target MATCH (|d| <= 0.01). AR is already at |d| up to
  ~0.03 due to BIC order-selection differences vs the author's `runar`; treat |d| <= 0.03 as
  **CLOSE-ACCEPT** and note the BIC-order cause. |d| > 0.05 = DIVERGENT -> investigate.
- **RF:** stochastic (bootstrap + sklearn-vs-ranger). CLOSE (|d| <= 0.05). Fix `seed=42`.
  |d| > 0.05 = DIVERGENT -> log candidate causes (impl diff, mtry, feature set) but do NOT
  start tuning; that is a Chan decision.
- **UCSV:** MCMC benchmark. CLOSE (|d| <= 0.05). If DIVERGENT, first re-confirm gamma=0.2
  variance semantics in the package, log finding, do not change the value without escalating.

## 6. DELIVERABLES (all inside the worktree)
1. Updated `run_block.py` that runs the 4-model panel (invocation documented at top of file).
2. Console/log capture of the 4-model panel to `qa/g2_four_model.out`.
3. A **parity table** appended to `docs/replication/medeiros_2021.md` under a new
   "## G2 four-model parity (CPI, h in {1,3,6,12})" section: columns = model, h, our ratio,
   paper ratio, d, verdict (MATCH/CLOSE/DIVERGENT), plus RW abs RMSE + OOS n rows.
4. A fresh dated section in `.dev-notes/codex_progress_b1.md`: what ran, the scored table,
   any GAP/blocker, runtime notes, and a one-paragraph "G3 scope estimate" (how many
   model x horizon x subperiod cells remain for the full replication).
5. Then **STOP** and print a final summary (this becomes your last message).

## 7. FINAL SUMMARY FORMAT (your last message — keep it tight)
- Panel table: model x h with our ratio / paper ratio / d / verdict.
- Count of MATCH / CLOSE / DIVERGENT.
- Any DIVERGENT cell with your one-line suspected cause (no fixes applied).
- GAPs / blockers, if any.
- Confirmation that you did NOT run anything beyond the 4-model G2 panel.

---
## CORRECTION (2026-07-08, supersedes the RF guidance in §1/§3) — RF must be AUTHOR-CONFIGURED

**Principle (Chan directive):** replication = configure the macroforecast package to the AUTHOR
spec. Package defaults are irrelevant. Every arm's hyperparameters come from author code
(`qa/ForecastingInflation`) and must be set EXPLICITLY as package arm params.

**RF spec deviation caught:** the first run used the package `random_forest` DEFAULT
(`n_estimators=200`, `max_features=all`). Author RF (`functions/functions.R:120`,
`randomForest(Xin,yin,importance=TRUE)`) = R defaults = **ntree=500, mtry=floor(p/3)**.
That run was KILLED (do not wait on it). Record RF@200/all as a one-line footnote only.

**Package exposure CONFIRMED (no gap):** `random_forest(...)` (tree.py:62-100) exposes
`n_estimators` as a named param and forwards `**kwargs` straight into
`RandomForestRegressor(**estimator_params)`. So `max_features` and `n_jobs` pass through.
No package edit needed.

**RF arm — set these params EXPLICITLY:**
```python
Arm("rf", model="random_forest", features=base_features(),
    params={"n_estimators": 500,        # author ntree=500
            "max_features": 1.0/3.0,     # sklearn float fraction => int(1/3 * p) per split = R mtry floor(p/3)
            "random_state": 42,
            "n_jobs": 48})               # 48-core box; seed fixes results, n_jobs only affects speed
```
- sklearn `max_features` float semantics: per-split features = `max(1, int(max_features*n_features))`;
  `1/3` reproduces R `mtry=floor(p/3)` regardless of the actual p. Use the FLOAT, not a hardcoded int.
- `n_jobs=48` is a pure speedup — RandomForestRegressor with `random_state` set is deterministic
  across n_jobs. Document in the log that results are n_jobs-invariant.

**Action:** re-run the FULL 4-model panel fresh (RW/AR/UCSV recompute is cheap; RF now at author
config + 48 cores). Score vs IJF Table 5 (RF 0.844/0.706/0.715/0.685; AR 0.902/0.790/0.791/0.753;
UCSV 0.954/0.797/0.777/0.781). Then STOP + report.

**Forward principle for ALL later arms (LASSO/adaLASSO/ElNet/CSR/JMA/bagging/hybrids):** pull config
from author code, express as explicit package params, never rely on package defaults.

---
## GOVERNING FRAME — 4 co-equal objectives (per .dev-notes/REPLICATION_OBJECTIVES.md, rules B1-B5)
Parity numbers are the VEHICLE, not the whole mission. Every STOP/report summarizes ALL FOUR.

**P1 — Trust docs.** `docs/replication/medeiros_2021.md` must be docs-site quality: paper+venue,
exhibits replicated, **arm -> author-param map expressed as explicit package params (never defaults)**,
parity table w/ tolerances, runnable recipe pointer, honest gaps/caveats. Public trust artifact.

**P2 — Hunt package bugs/gaps.** Record EVERY defect/gap/silent-wrong risk/missing param/confusing
API in `.dev-notes/replication_findings_medeiros.md` (BUGS/GAPS section: file:line + repro + severity).
Do NOT patch package from this worktree — record + report to Fable. Zero findings = under-looked.
  Seeded findings to write up:
  - [SILENT-WRONG RISK, med] `models/tree.py:66` `random_forest` default n_estimators=200 &
    max_features unset (sklearn=all). R `randomForest` convention (what econ users expect) = 500,
    mtry=p/3. Silently wrong for anyone not overriding. repro: run_block RF@default vs @author.
  - [GAPS] design deltas D3 (MCS single-statistic -> two runs), D4 (JMA gap-LOO absent),
    D7 (CSR ñ vs K naming) — see phaseB_design_b1_medeiros_FINAL.md §0.5; verify + log.

**P3 — Efficiency (real workload).** Record slow/redundant/wasteful compute in the EFFICIENCY section
(what/why/file:line/est cost). Cross-ref `.dev-notes/pipeline_efficiency_review.md`.
  Seeded finding to write up:
  - [EFFICIENCY, high] default `meta.resolve_n_jobs()` under-utilized a 48-core box (~2 cores) on the
    RF grid -> ~20x wall-clock. RF@200 ran ~1957s/horizon at ~2 cores. Confirm resolve_n_jobs default
    + recommend core-aware default or docs note. file:line = macroforecast/meta resolve_n_jobs.

**P4 — Speedups must be STATISTICALLY IDENTICAL.** Forbidden: fewer draws/trees, looser tol,
subsample, coarser grid. Allowed: n_jobs (seeded), cache/reuse, vectorization, exact algebra.
Every speedup VALIDATED by identical-numbers before/after + recorded.
  REQUIRED validation for this lane's n_jobs=48 speedup:
  - Fit RandomForestRegressor on one fixed (X,y) slice with random_state=42 at n_jobs=1 vs n_jobs=48;
    assert predictions bitwise-identical. Record the check + result in the EFFICIENCY section.
    (Establishes the RF@48-core parity numbers are valid, not an artifact of parallelism.)

At STOP: report (1) parity status, (2) docs page state, (3) new bugs/gaps, (4) efficiency findings +
validated speedups.

---
## MEASURED DIAGNOSIS (2026-07-08) — CORRECTS the P3 seed above

**n_jobs propagation WORKS — retract the "n_jobs dropped/overridden" hypothesis.**
Evidence (isolated bench, same worktree env, /tmp/njobs_diag.py + /tmp/njobs_mf.py):
- `get_model("random_forest", params={...,"n_jobs":48})` returns spec params INCLUDING n_jobs=48.
- `mf.random_forest(X,y, n_estimators=500, max_features=1/3, random_state=42, n_jobs=48)`:
  2.79s (n_jobs=1) -> 0.37s (n_jobs=48) = 7.5x on ONE fit (350x150). sklearn direct: 9.5x.
- So the RF fit DOES parallelize. There is no global thread cap (isolated 48-core test succeeded).

**P4 speedup validation (n_jobs) — PASS.** RF predictions n_jobs=1 vs n_jobs=48 differ by max
5.55e-17 (last-bit float noise from parallel tree-sum reduction order) = statistically identical.
Record: n_jobs=48 is a SAFE speedup. Also: RF@500+n_jobs=48 (1637s/horizon) < RF@200+2core (1957s).

**REAL P3 finding — per-origin feature-pipeline recomputation dominates (NOT the fit).**
One RF horizon = ~1637s, but RF fitting is only ~312 origins x ~0.37s ~= 115s (~7%). The other
~93% (~1520s) is the feature pipeline (PCA + embed lags + predictor screen + Nov2008 dummy)
recomputed FROM SCRATCH every rolling origin (and again per horizon). This matches the known
prepared-stage cache gap (cache key = origin_pos+horizon; transform recomputes though fit is shared)
and the cross-horizon transform-cache gap. THIS is where the ~5-10x wall-clock win is, and it must be
statistically identical (P4): feature-fit reuse / prepared-stage cache, NOT fewer origins.
  Action for findings file: EFFICIENCY entry with these measured numbers; cross-ref
  .dev-notes/pipeline_efficiency_review.md + the transform-cache memory notes. Propose: cache the
  fitted feature transforms per (regime, horizon) and reuse across origins where the fit window is
  shared; validate identical forecasts before/after.

**P2 seed status:** KEEP the tree.py:66 RF-default silent-wrong-risk finding (200/all vs author
500/p3) — still valid, independent of n_jobs. DROP any "n_jobs override" claim.

---
## MEASURED UCSV COST (2026-07-08) — P3 EFFICIENCY seed
- UCSV rolling probe (/tmp/ucsv_probe.py): 11 origins / 76.9s = **6.99 s/origin**.
- Full G2 UCSV extrapolation: ~312 origins x 4 horizons x 6.99s = **~145 min (2.4h) SEQUENTIAL**.
- `get_model("ucsv", params={"gamma":0.2,"n_jobs":48})` preserves n_jobs, BUT UCSV is a Gibbs
  sampler (draws are a sequential Markov chain) — model-internal n_jobs is meaningless. Parallelism
  must be at the ORIGIN/HORIZON level in the harness (each origin fit is seed-fixed, independent,
  deterministic => statistically identical when parallelized). draws=5000 must NOT be reduced (P4).
- Full G2 sequential cost estimate: RW/AR ~0.3h + RF ~1.8h + UCSV ~2.4h ~= 4.5h.
- [GAP to verify] does the rolling harness / pipeline expose origin-level parallelism for UCSV? If
  only horizon-level (pipeline spec.n_jobs over cells) => 4x. If neither reaches UCSV => package GAP,
  record + report. ResultStore (persist G2 cells) needed so G3 reuses, not recomputes.

---
## DIRECTION CHANGE (2026-07-08) — REWRITE run_block AS A SINGLE run_pipeline CALL (box-maximizing)
Chan: the single-arm `mf.run(model,horizon)` double-loop BYPASSES the pipeline and leaves 47/48
cores idle. `run_pipeline` dispatches all (arm x horizon) cells across a worker pool -> wall-clock
sum -> max, statistically identical (P4). This is the STANDARD path for ALL papers B1-B5.

**STEP 1 — read the official docs FIRST (authoritative):**
  docs/guide/concepts/running.md (esp. §Parallel + Key Callables ex., lines 80-131; §Seeds/parallel
  148-152; §Incremental horse races 160-207), docs/reference/pipeline.md (pipeline_spec / EvalSpec /
  auto_parallelism / result_store), docs/guide/getting_started.md, docs/guide/model_overview.md.

**STEP 2 — rewrite run_block.py as ONE run_pipeline call.** Discard the hand loop.
```python
from macroforecast.pipeline import pipeline_spec, run_pipeline, Arm, EvalSpec, TargetSpec
RF_PARAMS = {"n_estimators":500, "max_features":1.0/3.0, "random_state":42}  # DROP n_jobs — pipeline auto-budgets model_threads
arms = [
  Arm("rw",   model="naive", features=None, is_benchmark=True),
  Arm("ar",   model="ar",    features=<univariate feature_spec>),   # BIC order, own lags only
  Arm("ucsv", model="ucsv",  features=None, params={"gamma":0.2}),
  Arm("rf",   model="random_forest", features=base_features(), params=RF_PARAMS),
]
# two-regime: run_pipeline PER REGIME (s1 1990-2000, s2 2001-2015) with its rolling window,
# concatenate forecasts for the full-sample table. VERIFY per-horizon R semantics vs docs
# (paper R = base - h - p - 1); the pipeline may handle the horizon offset internally.
spec = pipeline_spec(data=PANEL, targets=[TargetSpec("CPIAUCSL")], horizons=[1,3,6,12],
                     window=<regime window>, arms=arms, evaluation=EvalSpec(benchmark="rw"),
                     n_jobs="auto", seed=42,
                     result_store="qa/result_cells", preprocessing_cache_dir="qa/prep_cache",
                     save_models=False)
report = run_pipeline(spec)
```

**STEP 3 — ACCEPTANCE GATE = P4 identical-numbers proof.** The run_pipeline RW/AR ratios MUST
reproduce the validated single-arm numbers (AR d = +0.012/-0.029/-0.032/+0.010 vs paper T5,
qa/g2_sequential_baseline_timings.txt). If they drift, the window/policy config differs -> fix
before trusting RF/UCSV. Then score RF (0.844/0.706/0.715/0.685) + UCSV (0.954/0.797/0.777/0.781).

**STEP 4 — EFFICIENCY log (P3/P4), in .dev-notes/replication_findings_medeiros.md:**
  BEFORE (sequential single-arm, measured, qa/g2_sequential_baseline_timings.txt):
    RW ~77s, AR ~95s, RF ~1620s PER CELL, run serially -> RF alone ~1.8h; UCSV ~6.99s/origin
    (~145min for 4h). Total sequential ~4.5h. Root cause: run_block bypassed run_pipeline.
  AFTER (run_pipeline n_jobs=auto on 48 cores): record wall-clock. Expect sum->max collapse.
  Confirm parity numbers IDENTICAL before/after (P4). Record both + the identical-numbers assertion.

**STEP 5 — result_store ON** (qa/result_cells) so G3 reuses G2 cells (incremental horse race).

**STEP 6 — P1 docs + P2 findings** as before: arm->author-param map (explicit params), parity table,
recipe, gaps in docs/replication/medeiros_2021.md; BUGS/GAPS (tree.py:66 RF default silent-wrong;
single-arm-loop-bypasses-pipeline friction; any n_jobs/window surprises) in the findings file.

**STEP 7 — STOP + report ALL FOUR dimensions** (parity / docs / bugs+gaps / efficiency+validated
speedup). Do NOT run CSR/JMA/LASSO/bagging/G3.

---
## ORACLE CORRECTION (2026-07-08) — acceptance gate = PAPER T5, NOT g2_rw_ar_v2.out
The g2c run correctly STOPPED at the P4 gate, but the gate target I gave (g2_rw_ar_v2.out:
0.914/0.761/0.759/0.763) was a STALE earlier baseline. The authoritative oracle is IJF Table 5.
Compare AR RMSE ratios:
  Paper T5      : 0.902 / 0.790 / 0.791 / 0.753   <- TRUE oracle
  v2.out (stale): 0.914 / 0.761 / 0.759 / 0.763   (h=3,6 off by ~0.03)
  run_pipeline  : 0.911 / 0.790 / 0.792 / 0.765   -> d vs paper = +0.009/-0.000/+0.001/+0.012
The pipeline numbers track the PAPER essentially exactly at h=3/h=6 and are within 0.012 elsewhere
= PASS. codex diagnostic confirmed pipeline == current low-level loop (not drift); v2.out predates
a config change. **The run_pipeline rewrite is FAITHFUL. Proceed.**

**PROCEED instructions:**
1. Acceptance gate = PAPER T5 (§4), tolerance RW/AR |d|<=0.03 (BIC-order), RF/UCSV |d|<=0.05. Current
   RW/AR PASSES. Remove the pre-RF stop gate.
2. Run the FULL 4-arm run_pipeline (rw/ar/ucsv/rf), n_jobs='auto', result_store='qa/result_cells'.
   Score RF (0.844/0.706/0.715/0.685) + UCSV (0.954/0.797/0.777/0.781) vs paper.
3. EFFICIENCY: record pipeline parallel wall-clock (after) vs sequential 4.4h (before); confirm
   parity numbers stable = P4. RW/AR already showed 6.29x + result_store cache hits.
4. Finalize docs/replication/medeiros_2021.md (parity table w/ paper T5) + findings; STOP + 4-dim report.
