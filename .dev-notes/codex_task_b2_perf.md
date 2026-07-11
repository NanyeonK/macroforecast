# B2 (Hounyo-Li) — profile the supervised methods + find EXACT (statistically-identical) speedups

Worktree `/home/nanyeon99/project/mf-b2-hounyoli`. Goal: make the full B2 replication (all targets ×
horizons × methods incl. supervised SsPCA/SPCA, author-oracle surface) COMPUTATIONALLY TRACTABLE by
finding EXACT speedups (P3) that produce BITWISE-IDENTICAL results (P4 — no reduced grid/draws/folds).
The supervised cell is currently >2h (1 target/1 horizon). Do NOT reduce fidelity. This is analysis +
a proposal; implement only the safe exact optimizations you can verify identical, and clearly report
which are macroforecast package changes (main fix-lane) vs runner.

## Profile
1. Run ONE supervised cell (SsPCA = `supervised_scaled_pca`, inflation, h=1, full sample, author config,
   author-oracle surface) under a profiler (cProfile / py-spy) with a SMALL bounded slice if needed to
   get the hotspot breakdown WITHOUT waiting 2h (e.g., profile ~20-30 origins, extrapolate). Report
   where the time goes: the fold-internal per-observation expanding refit, the SVD, the K grid, the qN
   grid, the residual-correlation screening, standardization, etc.

## Find EXACT speedups (statistically identical — verify)
Investigate each; for each, state whether it changes numbers (must be NO) and the expected speedup:
1. **K-grid recursion reuse.** SsPCA/SPCA extract K components by a GREEDY residual-recursion: computing
   K=10 already produces the K=1..9 intermediate fits. Does the model-selection / model code recompute
   the recursion independently for each K in the grid? If so, computing K_max ONCE and reading off all
   K is an EXACT ~grid-size speedup. Check `macroforecast/models/linear.py` (SsPCA/SPCA helper) and the
   model-selection K enumeration. This is likely the biggest win.
2. **qN-grid handling.** Similar: is the screening/recursion recomputed per qN independently when it
   could be shared, exactly?
3. **Fold-internal expanding refit.** Each validation observation refits on 1:(train_end+o-1) from
   scratch. Can the SVD / covariance / screening be updated incrementally (rank-1 / streaming) as the
   window grows by one row, giving bitwise-identical results? Or at least avoid recomputing shared
   sub-results.
4. **Redundant standardization / feature recomputation** across cells/folds (reuse where identical).
5. **Parallelism headroom.** Current n_jobs and where the parallelism is (cell-level vs within-cell).
   For the full run, the cell grid (targets × horizons × methods × K/qN) is embarrassingly parallel —
   confirm the pipeline can use ALL cores (report server1 core count) and that result_store makes it
   resumable. Note if a within-cell parallelization (over origins / K) would help.

## Deliverable
- Profile hotspot table (where the >2h goes).
- For each exact speedup: numbers-identical? (verify on a small slice — same output before/after),
  expected speedup factor, and classification (macroforecast PACKAGE change [file:line, main fix-lane]
  vs runner change). Implement ONLY the ones you can verify bitwise-identical on a small slice, and
  report the measured before/after time + identical-output proof. Do NOT implement anything you cannot
  verify identical; record those as proposals.
- A full-run compute ESTIMATE after the safe speedups + max parallelism: hours for Table 2 supervised,
  and for the D-tables.

## Constraints
- P4 ABSOLUTE: no reduced K/qN grid, no fewer folds/origins, no coarser anything. Only exact algebraic
  reuse / incremental updates / parallelism / caching. Every implemented change must be proven identical.
- If a speedup is a PACKAGE change, record it precisely for a main fix-lane (do NOT patch
  `macroforecast/**` from this replication worktree — record file:line + the exact change). Runner-level
  changes may be applied here. No push.

## STOP report → write `qa/codex_last_msg_b2perf.txt` AND print:
- The profile hotspots.
- The exact speedups found: each with numbers-identical proof, speedup factor, package-vs-runner class.
- The revised full-run compute estimate (Table 2 supervised hours; D-tables) after safe speedups + max
  parallelism (state server1 core count).
- Which speedups need a main package fix-lane (with the precise spec) vs which are runner-side.
Then STOP.
