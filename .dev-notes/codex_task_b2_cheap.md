# B2 (Hounyo-Li) — CHEAP-ONLY parity (exclude supervised) + document the supervised compute wall

Worktree `/home/nanyeon99/project/mf-b2-hounyoli` (branch `repro/hounyo-li-2026`, rebased on main
`3cebafc9`). The runner `scripts/replication/hounyo_li_2026_pipeline/run_g1_smoke.py` is ALREADY
re-mapped (PCA→`pcr`, AR_BIC→`ar_bic`, PLS→`pls score_projection="x_weights_raw"`, sPCA→`scaled_pca`,
+ leak-free `standardize_scope`/`nan`/`fold` config). A prior full-smoke run was KILLED because the
supervised methods (SsPCA/SPCA) are a compute wall (>2h for ONE cell, 1 target/1 horizon). Decision
(Chan): get the CHEAP-method parity now, document supervised as a wall.

## Task
1. Run G1 smoke with the CHEAP arms ONLY — RW, AR_BIC(`ar_bic`), PCA(`pcr`), sPCA(`scaled_pca`),
   PLS(`pls` raw-weight) — and EXCLUDE SsPCA(`supervised_scaled_pca`) and SPCA(`supervised_pca`).
   Add an `--arms`/`--exclude` filter to run_g1_smoke.py (or a small arm-subset switch) so the
   supervised arms are skipped. REUSE `runs/hl2026_store` (the cheap cells from the killed run are
   cached — pcr/ar_bic/sPCA/PLS/RW). This should score in minutes.
2. Score vs IJF Table 2 (inflation, h=1, full sample, threshold none): per method
   ratio / paper / Δ / verdict (|Δ|≤0.03). Compare to the OLD (pre-remap) numbers to show whether the
   corrected mappings moved each method TOWARD the paper:
   - OLD (far/sklearn): PCA 1.031, sPCA 0.901, PLS 1.112 (all FAIL vs paper 0.970/0.768/0.861).
   - NEW: report PCA(pcr), sPCA, PLS(raw), AR_BIC — did they improve? Which now MATCH/CLOSE?
3. If a residual gap remains after the correct mappings + leak-free config, attribute it: is it
   consistent with the author's TARGET-y standardization leak (which we deliberately do NOT
   reproduce)? Quantify if possible.

## Document the supervised compute wall (P1 + P3)
In `docs/replication/hounyo_li_2026.md` and `.dev-notes/replication_findings_hounyo_li.md`, record:
supervised SsPCA/SPCA at the exact author fold-internal expanding-refit × (K,qN) grid = >2h per cell
(1 target/1 horizon) measured; table2 = 12 supervised cells, full = weeks — computationally
infeasible, which is exactly the paper's own footnote ("numerical results were not reproduced, owing
to the substantial computational cost"). The package CAN express the exact supervised config (SsPCA
= `supervised_scaled_pca` leak-free MATCH), but the exact-author full run is a documented compute
wall, not a package limitation. Note that a smaller-scale supervised feasibility run (reduced
grid/origins) would NOT be the exact author result and is left as an optional labeled exercise.

## Constraints
- Author-faithful cheap config; NEVER package defaults. Do NOT patch `macroforecast/**` (record any
  new gap). Do NOT reproduce the target-y leak. No stats-changing shortcuts on the cheap methods
  (excluding supervised is a SCOPE reduction for this run, clearly labeled — not a fidelity change to
  the methods that DO run). Reuse result_store. No G2/G3/finance. No push.

## STOP report → write `qa/codex_last_msg_b2cheap.txt` AND print (4 purposes):
- P1: cheap-method parity table (PCA/sPCA/PLS/AR_BIC: NEW ratio / paper / Δ / verdict), the OLD→NEW
  comparison, and whether the corrected mappings closed the gap. Supervised = documented wall.
- P2: confirm no package patch; any new gap; target-y leak NOT reproduced.
- P3/P4: result_store reuse (cached cheap cells); no fidelity shortcut on the cheap methods.
Then STOP.
