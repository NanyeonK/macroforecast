# B2 (Hounyo-Li) — VERIFY K-prefix identity, then run the FULL Table 2 (author-oracle reproduction)

Worktree `/home/nanyeon99/project/mf-b2-hounyoli` (branch `repro/hounyo-li-2026`). The K-prefix
grouped evaluator (opt-in exact speedup for supervised-PCA) is now on `main` (HEAD `ab0536d2`). Goal:
reproduce the FULL Table 2 (macro/inflation + IP-growth + unemployment × horizons 1/6/12/24 × PCA/
sPCA/SPCA/SsPCA/PLS vs AR_BIC) on the author-methodology surface, using the K-prefix speedup + 48-core
parallelism. Read `.dev-notes/b2_method_gap_spec.md` §4 and `.dev-notes/REPLICATION_OBJECTIVES.md`.

## STEP 1 — rebase + VERIFY K-prefix identity (P4 GATE — do this before the full run)
1. Rebase `repro/hounyo-li-2026` onto `main` (brings in the K-prefix grouped evaluator). Resolve
   trivially (replication deliverables don't touch `macroforecast/**`).
2. VERIFY the K-prefix grouped evaluator is BITWISE-IDENTICAL to the per-candidate path: run a SMALL
   supervised selection (SsPCA, ~5-10 origins, full K/qN grid) with the grouped evaluator ENABLED vs
   DISABLED (the opt-in flag), and assert identical selected K AND identical forecasts (max abs diff
   0, or ≤1e-12). Report the proof. **If NOT identical → STOP immediately, do NOT run the full Table 2,
   report the mismatch.** This gate protects the entire run's validity.

## STEP 2 — full Table 2 (author-oracle surface) — only if STEP 1 identity passes
Configure the runner for the FULL Table 2 grid on the AUTHOR-ORACLE surface (`--surface author_oracle`,
the labeled leaky-standardization path that reproduces Table 2):
- targets: inflation (CPIAUCSL), IP growth, unemployment (the paper's 3 macro targets);
- horizons: 1, 6, 12, 24;
- methods: PCA(`pcr`), sPCA(`scaled_pca`), SPCA(`supervised_pca`), SsPCA(`supervised_scaled_pca`),
  PLS(`pls` raw-weight), vs AR_BIC(`ar_bic`) denominator; threshold=none (Table 2 base).
- Enable the K-prefix grouped evaluator. `n_jobs="auto"` (use all 48 logical cores via cell-level
  parallelism), `result_store` enabled (resumable), `parallel_cell_timeout=none` (or well above the
  supervised cell time so no false timeouts). Author K/qN grids, author fold geometry, author config
  (NEVER package defaults).
- P4 ABSOLUTE: no reduced grid/folds/origins/draws. The K-prefix speedup is the ONLY speed change and
  it is verified identical in STEP 1.
Score every cell vs the paper Table 2 (RMSE ratio vs RW... note: Table 2 ratios are vs the AR/RW
benchmark per the paper — use the paper's exact denominator). Produce the full Table 2 parity grid
(3 targets × 4 horizons × 5 methods): reproduced ratio / paper / Δ / verdict (|Δ|≤0.03).

## STEP 3 — report; STOP before D-tables
Do NOT run the D-tables (subsamples × thresholds ~4.5 days) — that is a separate go-ahead. Update the
trust page `docs/replication/hounyo_li_2026.md` with the full Table 2 author-oracle reproduction grid.

## Constraints
- Author-faithful; never package defaults. Do NOT patch `macroforecast/**` (K-prefix is on main; if a
  NEW gap appears, record it). No stats-changing shortcut (K-prefix verified identical is the only
  speed change). Reuse result_store. No D-tables. No finance. No push.

## STOP report → write `qa/codex_last_msg_b2table2.txt` AND print:
- STEP 1: K-prefix identity proof (grouped vs per-candidate: identical? max diff).
- STEP 2: the FULL Table 2 parity grid (3×4×5), reproduced-vs-paper, how many cells within |Δ|≤0.03,
  and the total runtime + core utilization. Does the full Table 2 reproduce?
- P2/P4: confirm no package patch; confirm the only speedup is the verified-identical K-prefix.
Then STOP (do NOT start D-tables).
