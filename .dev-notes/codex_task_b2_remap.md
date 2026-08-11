# B2 (Hounyo-Li) — rebase onto the new methods, re-map arms, rerun G1 smoke

Worktree `/home/nanyeon99/project/mf-b2-hounyoli` (branch `repro/hounyo-li-2026`). The 6 general
methods/options are now on `main` (HEAD `3cebafc9`): `pcr`, `ar_bic`, `pls score_projection`,
`standardize_scope`, `nan_policy=zero_after_standardize`, `score_aggregation=mean_fold`,
`preselect_stage`. Read `.dev-notes/b2_method_gap_spec.md` §4 (the arm re-mapping) and
`.dev-notes/REPLICATION_OBJECTIVES.md` (4 purposes). This run TESTS whether the corrected mappings
close the Table 2 parity gap.

## Steps
1. **Rebase** `repro/hounyo-li-2026` onto `main` (`git rebase main` or merge). Expect no conflict
   (B2 changes are under scripts/replication + docs/replication; main touched macroforecast/**).
2. **Re-map the arms** in `scripts/replication/hounyo_li_2026_pipeline/` per gap-spec §4, verifying
   each against the actual current API on main:
   - `AR_BIC` → `model="ar_bic"` with author options: `min_lag=1,max_lag=12,criterion="bic",
     ic_parameter_count="lag_square", estimator="matlab_ar", forecast_mode="coefficient_power",
     include_constant=True`. Target preprocessing (diff + movmean(12) + LEAK-FREE in-window
     standardization) stays CALLER-SIDE; do NOT reproduce the author target-y leak.
   - `PCA` → `model="pcr"` with `control_columns=(target_lag_col,), include_constant=True,
     drop_control_columns=True, standardize=True, nan_policy="zero_after_standardize"`, K grid.
   - `sPCA` → `scaled_pca` (`scale=False`, target-lag control, constant, K grid).
   - `SPCA` → `supervised_pca` (`scale=False`, `preselect="none"`, target-lag control, K/qN grid).
   - `SsPCA` → `supervised_scaled_pca` (same controls/grid).
   - `PLS` → `pls` with `score_projection="x_weights_raw"`.
   - Apply the leak-free predictor standardization scope (`standardize_scope=
     "origin_available_predictors"` or the exact param name on main) + `score_aggregation="mean_fold"`
     for the fold CV.
3. **Rerun G1 smoke** (inflation, h=1, full sample, threshold none, the 5 methods + AR_BIC),
   `run_pipeline(n_jobs="auto")`, reuse `runs/hl2026_store`, and set a GENEROUS
   `--parallel-cell-timeout` (e.g. 21600s / 6h) so SPCA/SsPCA are not killed. If the supervised
   cells are still too slow to finish in this run, that is acceptable — REPORT the parity for every
   method that completes (the cheap completed methods PCA/sPCA/PLS/AR_BIC are the key signal for
   whether the corrected mappings closed the gap).
4. **Score vs Table 2** (inflation/h=1): per method ratio / paper / Δ / verdict (|Δ|≤0.03 pass).
   The one irreducible divergence is the author's TARGET-y standardization leak (do NOT reproduce);
   if a residual gap remains after correct mappings + leak-free config, attribute/quantify it and
   state whether it is consistent with the target-y leak magnitude.

## Constraints
- Author-faithful config; NEVER package defaults. Do NOT patch `macroforecast/**` (the additions
  are on main now; if you find a NEW gap, record it, don't patch here). No stats-changing shortcuts.
  Do NOT reproduce the target-y leak. Reuse result_store. No G2/G3/finance. No push.

## Deliverables
- Re-mapped `scripts/replication/hounyo_li_2026_pipeline/` runner.
- `docs/replication/hounyo_li_2026.md` — updated arm→method map (now using pcr/ar_bic/pls-raw + config)
  + the G1 parity table.
- `.dev-notes/replication_findings_hounyo_li.md` — did the corrected mappings close the gap? residual
  attribution (target-y leak); any new gap.

## STOP report → write `qa/codex_last_msg_b2remap.txt` AND print (4 purposes):
- P1: G1 parity table (each completed method: ratio/paper/Δ/verdict); did PCA→pcr, PLS→raw-score,
  AR_BIC→ar_bic move toward the paper vs the old far/sklearn mapping? Which methods now MATCH/CLOSE?
- P2: confirm rebase clean; any NEW package gap; confirm no package patch; confirm target-y leak NOT reproduced.
- P3/P4: result_store reuse; timeout used; which cells completed vs timed out; no fidelity shortcut.
Then STOP.
