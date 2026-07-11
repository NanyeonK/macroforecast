# B1 (Medeiros 2021) — RF residual-divergence diagnostic: PROTOCOL (fixable) vs BACKEND (caveat)

Replication worktree `/home/nanyeon99/project/mf-b1-medeiros` (branch `repro/medeiros-2021`).
Read `.dev-notes/REPLICATION_OBJECTIVES.md` first (four purposes; Purpose 1 = faithful
replication is primary). The RF Arm now uses author-faithful settings
(ntree=500, mtry=p/3, nodesize=5=min_samples_leaf, maxnodes=NULL, replace=TRUE, seed 42,
model_selection={"random_forest":None}) yet h=1 (0.9073 vs 0.844) and h=12 (0.7374 vs 0.685)
remain DIVERGENT while h=3/h=6 are CLOSE. Your ONLY job: determine WHY, and if it is fixable,
fix it toward parity.

## The one decisive question
Is the residual RF gap a PROTOCOL difference (our predictor matrix / preprocessing differs from
the author's, => FIXABLE in run_block.py) or an algorithmic BACKEND difference (sklearn
RandomForestRegressor vs R randomForest, => an honest documented caveat, NOT a package bug)?

## Method
1. From the author code (`qa/ForecastingInflation`, `qa/HDeconometrics`) reconstruct EXACTLY the
   design matrix fed to `randomForest` for target CPIAUCSL, direct horizon h: which columns
   (all 122 FRED-MD predictors? factors/PCA? which?), how many lags of each, how many AR lags of
   the target, any standardization/scaling, and the exact rows (estimation window). Write this as
   an explicit spec.
2. Reconstruct EXACTLY what `scripts/replication/medeiros_2021_pipeline/run_block.py` builds for
   the RF arm (its feature_spec / base_features / embed order / lag count / factor inclusion).
3. DIFF the two. Enumerate every difference (columns present/absent, lag depth, factor vs raw,
   scaling, embed order, row/window alignment).
4. If a difference exists AND macroforecast can express the author's choice, FIX run_block.py's
   RF feature spec to match the author's design matrix (author-faithful only — no stats hacks),
   then rerun ONLY the RF cells via `run_pipeline(n_jobs="auto")` reusing `qa/result_cells`,
   and rescore vs IJF Table 5. Report the new RF row.
5. If the feature matrices already MATCH (no expressible protocol difference), then the residual
   is the sklearn-vs-R randomForest backend difference. Quantify it: it is a KNOWN, expected,
   non-bitwise-reproducible difference between the two RF implementations; macroforecast
   legitimately uses sklearn. Document it as a caveat in the docs page (state that h=3/h=6 land
   CLOSE and h=1/h=12 sit just outside tolerance due to backend, with the measured gaps). This
   is NOT a macroforecast bug — do not log it as one.

## Hard constraints
- NEVER edit `macroforecast/**`. Only author-faithful config/feature changes to run_block.py.
- FORBIDDEN (Purpose 4): reducing trees/draws, loosening tolerances, subsampling, coarser grids,
  or any change whose purpose is to force a match rather than match the author.
- AR/RW/UCSV cached numbers must stay unchanged — reuse `qa/result_cells`; recompute RF only.
- Do NOT touch UCSV. Do NOT run G3, CSR/JMA/LASSO/bagging/hybrids. No git commit/push/gh.

## Deliverables (update in place; no commit)
- `docs/replication/medeiros_2021.md`: updated RF row if fixed; a precise "RF feature design"
  subsection (author matrix vs runner matrix); PROTOCOL-fix note or BACKEND-caveat, whichever applies.
- `.dev-notes/replication_findings_medeiros.md`: record the diagnosis (protocol vs backend) with
  evidence; if any macroforecast feature-API gap blocked matching the author matrix, log it as a
  fix-lane input (file:line + severity). Do NOT patch the package.
- `.dev-notes/codex_progress_b1.md`: append.

## STOP report -> write to `qa/codex_last_msg_g2f.txt` AND print. Summarize all four purposes:
- P1: verdict PROTOCOL vs BACKEND; the post-fix RF row (if fixed) or the quantified backend caveat.
- P2: any feature-API gap found (or "none"); confirm no package code patched.
- P3/P4: reruns reused result store; RF-only recompute; no stats-changing shortcut; AR/RW/UCSV unchanged.
Then STOP.
