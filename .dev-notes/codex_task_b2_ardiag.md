# B2 (Hounyo-Li) — diagnose why the COMPLETED factor methods diverge from Table 2 (cheap, cached)

Worktree `/home/nanyeon99/project/mf-b2-hounyoli` (branch `repro/hounyo-li-2026`). The G1 smoke
FAILED on the completed methods: PCA 1.031 (paper 0.970), sPCA 0.901 (paper 0.768), PLS 1.112
(paper 0.861) — all worse-than-paper (ratios too high). SPCA/SsPCA timed out (handled separately;
DO NOT run them here). Your job: find WHY the completed methods diverge, cheaply, using the CACHED
cells (`runs/hl2026_store` already has AR_BIC, PCA, PLS, sPCA at 361 rows each) + the author MATLAB
oracle. Do NOT rerun the expensive supervised cells. Do NOT run G2/G3.

## Top hypothesis: the AR_BIC benchmark differs from the author's MATLAB AR
All paper ratios are model/AR. If our AR is MORE accurate than the author's AR, every ratio inflates
(worse-looking) uniformly-ish. Test this FIRST.

## Method
1. Read the author's AR benchmark script in `qa/hounyo_li_matlab/` (the AR/benchmark .m used for the
   inflation/h=1 full-sample column of Table 2). Extract its EXACT spec: lag selection (BIC, max 12?),
   direct vs iterated, whether the target is standardized in-window, intercept, estimation window.
2. Get our AR_BIC absolute RMSE for inflation, h=1, full sample from the cached cell (the AR_BIC
   forecasts are the ratio denominator; recover the 361-row forecast/error series from the result
   store or the g1_smoke raw outputs). Compute our AR absolute RMSFE.
3. Reproduce the author AR forecast in a small standalone Python script (from the .m spec) on the
   SAME built data (`qa/hounyo_li_*panel*.csv`) for inflation/h=1 full sample, and compute the
   author-AR absolute RMSFE. Compare to our AR. Quantify the ratio our_AR_RMSE / author_AR_RMSE.
4. Decompose the divergence: for PCA/sPCA/PLS, is the paper-vs-ours gap EXPLAINED by the AR
   denominator difference? Recompute each completed method's ratio using the AUTHOR-AR denominator
   (method_RMSE / author_AR_RMSE) and see if it moves toward the paper. If yes → the AR benchmark is
   the (main) cause. If a method-specific gap remains → identify it (candidates: data build /
   in-window standardization convention / factor extraction / (K,qN) tuning selection / preselection
   order). Check the most likely: compare our built panel + target to the author workbooks (row count,
   dates, transforms, standardization) and confirm the factor arms use author-faithful params.
5. Also sanity-check the ABSOLUTE RMSFE levels against any author-provided absolute numbers in the
   MATLAB results folder (the oracle may store raw RMSE), not just ratios.

## Deliverable = the fix, expressed as config
- If the AR benchmark is the cause and macroforecast can express the author's exact AR (lag/BIC/
  standardization) → specify the exact `Arm("AR_BIC", model="ar", params={...})` change for the
  runner and, if cheap, re-score the CACHED factor cells against the corrected AR to show the
  completed methods now match the paper (this needs only re-scoring, not recomputing factors — the
  factor forecasts are cached; only the denominator changes). Report the corrected ratios.
- If the cause is data/standardization/tuning → specify the exact runner fix and evidence.
- If a package capability is missing → record in `.dev-notes/replication_findings_hounyo_li.md` as a
  fix-lane input (file:line + severity). Do NOT patch `macroforecast/**`.

## Constraints
- Use CACHED cells; do NOT recompute PCA/sPCA/PLS/AR unless a config change (e.g., AR spec) requires
  recomputing ONLY the AR cell (cheap). Do NOT run SPCA/SsPCA/G2/G3/finance. Author-faithful only,
  no stats-changing shortcuts. No `macroforecast/**` patch. No push.

## STOP report → write `qa/codex_last_msg_b2ardiag.txt` AND print:
- The AR comparison: our AR RMSFE vs author AR RMSFE (inflation, h=1, full), ratio, and which is more
  accurate + why.
- Corrected completed-method ratios (method/author-AR) vs paper — does the AR fix explain the gap?
- Any residual method-specific divergence + its diagnosed cause.
- The exact runner config fix to apply (AR params, and/or data/standardization), and whether the
  completed methods then match Table 2 (inflation/h=1) within |Δ|≤0.03.
Then STOP.
