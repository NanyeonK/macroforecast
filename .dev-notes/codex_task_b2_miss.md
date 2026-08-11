# B2 (Hounyo-Li) — close the 5 Table-2 misses via the author's IP-growth / unemployment source

Worktree `/home/nanyeon99/project/mf-b2-hounyoli` (branch `repro/hounyo-li-2026`). The full Table 2
author-oracle run reproduced 55/60 (|Δ|≤0.03). Inflation is 20/20 exact (we have the author MATLAB
`Inflation_results`). The 5 misses are ALL IP-growth / unemployment, attributed to a local-source
limitation: only `Inflation_results` was extracted from the 754MB reproducibility ZIP; IP-growth and
unemployment used a converted panel + documented geometry, NOT the author's exact per-target source.
Goal: get the author's exact IP-growth + unemployment source and re-run to close the misses toward 60/60.

Author ZIP: `~/second_brain/00_wiki/raw/paper_support/downloads_related_20260530/code/Reproducibility package_Hounyo and Li (2024).zip`.

## Steps
1. INSPECT the ZIP structure under `Empirical/Macro/` (or wherever): does it contain per-target result
   dirs for IP-growth and unemployment (e.g. `IPgrowth_results`, `Unemployment_results`, or the
   per-target `inflation_linear*.m`-analog scripts + the exact target series / transform codes)?
   `unzip -l` first; extract the relevant IP-growth + unemployment scripts + data to
   `qa/hounyo_li_matlab/` (selectively, not the whole 754MB).
2. From those, establish the author's EXACT per-target setup for IP-growth and unemployment: the target
   series + transform (tcode), the predictor panel, the fold/K/qN geometry, and any target-specific
   config that differs from inflation. Compare to what the full-run used (the converted panel +
   documented geometry) and identify the difference causing the misses.
3. FIX the runner's IP-growth + unemployment target/data configuration to the author's exact source,
   then RE-RUN the IP-growth + unemployment columns (2 targets × 4 horizons × 5 methods = 40 cells) on
   the author-oracle surface, K-prefix enabled, n_jobs="auto" (48 cores), result_store,
   parallel_cell_timeout=none. (Inflation's 20 cells are correct and cached — do not recompute.)
4. Re-score vs Table 2. Report the updated IP-growth + unemployment parity and whether the 5 misses
   closed → total now X/60.

## If the author source is NOT in the ZIP
If the ZIP does NOT contain the IP-growth / unemployment per-target source (only Inflation), say so
plainly: the 5 misses are then an irreducible local-source limitation (the author did not ship those
target scripts), and Table 2 reproduces 55/60 with inflation exact. Document that honestly; do NOT
fabricate or curve-fit the missing-target config.

## Constraints
- Author-faithful; author-oracle surface; K-prefix (verified identical) enabled. No reduced
  grids/folds/origins. No package patch (`macroforecast/**` untouched; record any new gap). Reuse
  result_store (inflation cached). No D-tables, no finance. No push.

## STOP report → write `qa/codex_last_msg_b2miss.txt` AND print:
- Whether the author IP-growth/unemployment source was found in the ZIP; what differed from the
  converted-panel run.
- The updated IP-growth + unemployment Table 2 parity (did the 5 misses close?); new total X/60.
- If irreducible (source absent), state so honestly.
- Confirm no package patch; K-prefix only speedup; no scope reduction.
Then STOP (do NOT start D-tables).
