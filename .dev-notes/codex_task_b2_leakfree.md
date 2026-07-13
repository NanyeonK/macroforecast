# B2 (Hounyo-Li) — FULL LEAK-FREE Table 2 with our package (load_fred_md), + document WHY it differs

Worktree `/home/nanyeon99/project/mf-b2-hounyoli` (branch `repro/hounyo-li-2026`, rebased on main with
K-prefix). Chan's decision (B): produce macroforecast's HONEST leak-free Table 2 END-TO-END with OUR
package — OUR data via `load_fred_md`, OUR leak-free methods — and document clearly WHY the honest
numbers differ from the paper's Table 2. This is the package-trust deliverable (Purpose 1) + the leak
finding (Purpose 2) + the verified speedup (Purpose 3/4). Do NOT use the `author_oracle` leaky surface
for the PRIMARY numbers; do NOT run the author's MATLAB to compute forecasts (it is only a documented
oracle for the why-different proof). Do NOT patch `macroforecast/**`.

## STEP 1 — data via OUR pipeline (load_fred_md)
- Build the predictor panel + the 3 targets (inflation = CPIAUCSL; IP growth = INDPRO per the paper's
  transform; unemployment = UNRATE per the paper's transform) using `mf.data.load_fred_md` (our native
  FRED-MD loader), matching the paper's sample window (~1971:4–2023:3) and the paper's target
  transforms (read `qa/hounyo_li_fulltext.txt` for the exact target definitions/tcodes). Document the
  FRED-MD vintage/provenance. This is OUR data pipeline — NOT the author workbooks.

## STEP 2 — full LEAK-FREE Table 2
- Run all 3 targets × 4 horizons (1,6,12,24) × 5 methods {PCA=`pcr`, sPCA=`scaled_pca`,
  SPCA=`supervised_pca`, SsPCA=`supervised_scaled_pca`, PLS=`pls` raw-weight} vs AR_BIC(`ar_bic`), on the
  LEAK-FREE surface (origin-available standardization, NO target-y leak; NOT author_oracle). Author-
  faithful METHOD config (K=1..10, qN grid, author fold geometry) but leak-free preprocessing.
- K-prefix grouped evaluator ENABLED (verified bitwise-identical). `n_jobs="auto"` (48 cores),
  `result_store`, `parallel_cell_timeout=none`. P4: no reduced grids/folds/origins/draws.

## STEP 3 — score + the WHY-DIFFERENT analysis
- Produce the honest leak-free Table 2 grid (3×4×5): our ratio / paper / Δ. These WILL differ from the
  paper; that is expected and is the point.
- Document WHY, rigorously, using the ALREADY-ESTABLISHED evidence (do not recompute — cite the proofs
  in `qa/codex_last_msg_b2decomp.txt`, `b2kdiag.txt`, `b2finaldocs.txt`, `runs/hl2026_table2_author_oracle/`):
  1. PRIMARY = the author's look-ahead target-y standardization (leak): on MATCHED data (author series)
     leak-free `pcr` = 1.081 vs paper 0.970; the gap = the leak, directly (~19%) and via K-selection
     tuned on the leaky surface (~81%); and macroforecast on the author surface reproduces the paper
     EXACTLY (inflation 20/20, 55/60 overall). So the methodological difference IS the paper's leak.
  2. SECONDARY = data pipeline: our `load_fred_md` vintage/transforms vs the paper's exact dataset —
     quantify where our data differs (sample rows, any series/transform mismatch).

## STEP 4 — finalize `docs/replication/hounyo_li_2026.md` (docs-site quality trust page)
Include: paper+venue+journal-not-reproduced note; the OUR-PACKAGE leak-free Table 2 grid (honest
output); the paper's Table 2 alongside; the **"Why our numbers differ from the paper"** section (leak
primary + data secondary, with the decomposition + author-surface-reproduces-exactly proof); the
package additions (6 methods + K-prefix speedup, all verified) and bugs found; the 4-purpose framing;
a clear verdict — macroforecast is verified correct and TRANSPARENTLY explains why its honest leak-free
output differs from a published table that depends on a look-ahead. Also write a 1-page
`.dev-notes/b2_replication_summary.md`.

## Constraints
- Leak-free surface for the primary numbers; `author_oracle` only cited in the why-different proof.
- OUR `load_fred_md` data. K-prefix (verified identical) is the only speedup. No `macroforecast/**`
  patch. No stats-changing shortcut. Reuse result_store. No D-tables/finance. No push.

## STOP report → write `qa/codex_last_msg_b2leakfree.txt` AND print:
- The full leak-free Table 2 grid (3×4×5): our ratio / paper / Δ.
- The why-different summary (leak primary with numbers; data secondary).
- Confirm: leak-free surface, our load_fred_md data, K-prefix only speedup, no package patch, docs written.
Then STOP (no D-tables).
