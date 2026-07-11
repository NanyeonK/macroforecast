# B2 (Hounyo-Li) — FINAL trust docs: honest leak-free + labeled author-methodology reproduction

Worktree `/home/nanyeon99/project/mf-b2-hounyoli`. The diagnosis is COMPLETE and definitive:
macroforecast `pcr` is bit-identical to the author algebra; the A2 splitter is not defective; the
entire Table-2 gap is the author's LOOK-AHEAD standardization surface (240-row predictor + 241-row
target block incl. `y_{T+h}`, standardized once before CV — driving BOTH the forecast and the K
selection). Proven: `macroforecast.pcr` on that author surface reproduces the author K and the paper
PCA ratio 0.9706 ≈ 0.970. Chan's decision: document BOTH the honest leak-free numbers AND a LABELED
author-methodology reproduction. Reuse `qa/hounyo_li_b2_pca_decomp.py`,
`qa/hounyo_li_b2_author_pca_port.csv`, `qa/hounyo_li_b2_kdiag_sample_curves.json`.

## Build the labeled author-methodology reproduction (CHEAP methods: PCA, sPCA, PLS + AR_BIC)
This is a LABELED DIAGNOSTIC replication path in the runner (e.g. `--surface author_oracle`), clearly
marked as reproducing the author's look-ahead standardization. It is NOT a macroforecast package
feature — do NOT patch `macroforecast/**`. For each cheap method, on the author surface (per-origin
240-row predictor block + 241-row target block standardized once incl. `y_{T+h}`, `pcr`/`scaled_pca`/
`pls` fed pre-standardized inputs with `standardize=False`, fold-id-preserving 3-fold splitter, author
K/qN grid), reproduce the inflation/h=1/full-sample Table 2 ratio vs the author AR_BIC denominator:
- PCA (`pcr`): target 0.970 (already proven 0.9706).
- sPCA (`scaled_pca`): target 0.768.
- PLS (`pls` raw-weight): target 0.861.
Report each reproduced ratio vs Table 2 (|Δ|≤0.03 = reproduced). Use the same package models — this
proves macroforecast's models reproduce the paper EXACTLY when given the author's (leaky) surface.

## Write the trust page `docs/replication/hounyo_li_2026.md` (docs-site quality)
Include:
1. Paper + venue (IJF 42:414-433) + the journal's "not reproduced, owing to computational cost" note.
2. Arm→method map (the leak-free config: `pcr`/`ar_bic`/`pls` raw-weight/`scaled_pca`/`supervised_*`
   + `standardize_scope`/`nan`/`fold`).
3. **Two result columns** for the cheap methods (inflation/h=1/full):
   (a) macroforecast LEAK-FREE (honest): PCA 1.081, sPCA 0.965, PLS 1.037 (vs AR_BIC=1.0);
   (b) macroforecast on the AUTHOR-METHODOLOGY surface (labeled leaky reproduction): ≈ Table 2
       (0.970/0.768/0.861).
   Paper column alongside. Verdict per column.
4. **The leak, explained plainly**: the author standardizes the 241-row target block INCLUDING the
   realized `y_{T+h}` before CV/forecast (`inflation_linear.m:196-197`), a look-ahead. This drives both
   the forecast and the K selection. macroforecast is leak-free by design, so its honest numbers differ;
   fed the author surface, it reproduces Table 2 exactly (proving the package is correct and the gap is
   entirely this leak).
5. **Supervised (SsPCA/SPCA) = documented compute wall**: exact-author fold-internal expanding-refit ×
   (K,qN) grid measured at >2h per cell (1 target/1 horizon) → table2 = 12 cells, full = weeks; matches
   the paper's own footnote. macroforecast CAN express the exact supervised config (leak-free MATCH);
   the full-author supervised run is a compute wall, not a package limitation.
6. **Verdict**: macroforecast is verified correct (`pcr` bit-identical to the author; A2 not defective);
   Hounyo-Li Table 2 depends entirely on the author's look-ahead standardization surface; the package
   reproduces it exactly via the labeled author-methodology path; the leak-free numbers are the honest
   result. B2 also permanently enriched the package with 6 general methods (pcr/ar_bic/pls-raw + config).

## Constraints
- Cheap methods only (PCA/sPCA/PLS + AR_BIC). Supervised = documented wall (do NOT run at exact config).
- The leaky author-methodology path is a LABELED runner diagnostic; do NOT patch `macroforecast/**`.
  Reuse the decomp/kdiag artifacts. No push.

## STOP report → write `qa/codex_last_msg_b2finaldocs.txt` AND print:
- The author-methodology reproduction table (PCA/sPCA/PLS: reproduced ratio vs Table 2, |Δ|≤0.03?).
- Confirm the two-column trust page is written to docs-site quality.
- Confirm no macroforecast/** patch; supervised documented as wall.
Then STOP.
