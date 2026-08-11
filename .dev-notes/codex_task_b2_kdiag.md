# B2 (Hounyo-Li) — WHY does the package fold-CV select different K than the author? (fixable config vs A2 difference)

Worktree `/home/nanyeon99/project/mf-b2-hounyoli`. The PCA decomposition showed macroforecast `pcr`
algebra is bit-identical to the author, and the residual Table-2 gap is mostly the **K (n_components)
SELECTION differing**: author K counts {1:30,2:1,3:30,4:13,5:1,6:67,7:22,8:107,9:24,10:66} vs package
K counts {1:48,2:1,3:41,4:3,6:15,7:47,8:67,9:78,10:61}. The difference is SYSTEMATIC (package leans
K=9, author leans K=8; package NEVER selects K=5). Your job: find WHY, cheaply, and classify it as a
FIXABLE B2-runner config-mapping issue OR a genuine package A2 (`explicit_folds` / `within_fold` /
`score_aggregation="mean_fold"`) difference vs the author `PCA_tune.m`.

Reuse `qa/hounyo_li_b2_pca_decomp.py` (the faithful author-PCA-path port that already reproduces the
author K distribution) as the AUTHOR oracle for K-selection.

## Method (cheap — sample origins, do NOT rerun all 361)
1. Read `qa/hounyo_li_matlab/.../PCA_tune.m` EXACTLY and state the author K-selection algorithm:
   fold geometry ({80|81-130},{130|131-190},{190|191-240}); within-fold per-observation expanding
   refit cadence; the K (and qN) grid; how validation MSE is computed and AGGREGATED across folds;
   the selection rule (min avg MSE) and tie-breaking.
2. Read how the B2 runner + package select K: the `_arms`/pcr arm's `model_selection` /
   `explicit_folds([80,130,190,240], within_fold="expanding")` config, `score_aggregation`, the K grid,
   and the package model-selection code path that consumes the A2 splitter
   (`macroforecast/model_selection/` — splitters.py, runner.py, search.py). State the package's
   effective K-selection algorithm.
3. Pick ~6-10 sample origins where the author K is known (from the port). For EACH, compute the
   per-fold validation MSE curve over K=1..10 BOTH ways (author `PCA_tune.m` logic via the port, and
   the package's fold-CV path) and compare: the per-fold MSEs, the aggregated score, and the argmin K.
   Find the FIRST concrete divergence (which fold, which K, which quantity differs).
4. Classify the systematic difference. Candidate causes to check explicitly:
   - fold BOUNDARIES or count (does the package use exactly {80,130,190,240} with 3 folds?);
   - within-fold expanding refit CADENCE (per-observation vs per-block; does the package refit every
     observation inside the fold like the author, or once per fold?);
   - MSE AGGREGATION (`mean_fold` = mean of per-fold MSEs vs mean over all validation points; unequal
     fold sizes make these differ — the author averages fold MSEs);
   - the validation prediction ALIGNMENT / standardization inside the fold;
   - tie-breaking / first-min vs last-min in argmin K;
   - the K grid actually searched.

## Deliverable = classification with evidence
- If the B2 RUNNER config is wrong (e.g., not exactly {80,130,190,240}, wrong aggregation flag, wrong
  grid) → FIXABLE CONFIG: state the exact runner fix; if cheap, apply it and re-check that sample-origin
  K now matches the author, and report the new PCA ratio.
- If the package A2 (`explicit_folds`/`within_fold`/`mean_fold`) genuinely differs from `PCA_tune.m`
  (e.g., refit cadence or aggregation semantics) → PACKAGE A2 GAP: record precisely in
  `.dev-notes/replication_findings_hounyo_li.md` (file:line + the exact semantic difference + a general,
  leak-free option that would close it) as a fix-lane input. Do NOT patch `macroforecast/**` here.

## Constraints
- PCA/pcr only, inflation, h=1. Sample origins (not all 361). Reuse the port + result_store. No
  supervised, no G2/G3, no other horizons. No `macroforecast/**` patch (record if A2). No push.

## STOP report → write `qa/codex_last_msg_b2kdiag.txt` AND print:
- The author vs package K-selection algorithms, side by side.
- The first concrete divergence on the sample origins (fold/K/quantity), with numbers.
- Root cause of the systematic K difference, classified: FIXABLE B2-RUNNER CONFIG (with the fix +
  re-checked K/ratio) OR PACKAGE A2 DIFFERENCE (with the precise semantic gap + fix-lane note).
Then STOP.
