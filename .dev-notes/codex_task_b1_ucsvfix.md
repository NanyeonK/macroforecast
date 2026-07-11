# B1 (Medeiros 2021) — fix the UCSV extraction so it is the paper's flat one-sided benchmark

Worktree `/home/nanyeon99/project/mf-b1-medeiros` (branch `repro/medeiros-2021`). Read
`.dev-notes/replication_findings_medeiros.md` (the UCSV BUG entry from the correctness diagnostic).

## The confirmed bug (do not re-diagnose — fix it)
Low-level `macroforecast.models.bayesian.ucsv` is CORRECT: it returns the one-sided filtered
final-trend posterior mean `tau_{T|T}` and `predict()` is flat across horizons. But the B1 UCSV arm
uses `Arm("ucsv", model="ucsv", features=None, params=...)` under `TargetSpec(policy="direct")`,
which makes the pipeline fit ucsv to HORIZON-SHIFTED direct targets `y_{t+h}` — a different UCSV
per horizon — so the B1 UCSV forecast is NOT flat across h (observed h1=2.97, h3=6.53, h6=6.54,
h12=4.94). The paper's UCSV benchmark forecasts the filtered trend `tau_{T|T}` fit ONCE on the
UNSHIFTED inflation target through origin T, and uses that SAME value for every horizon
(flat across h) — exactly like the RW/naive benchmark forecasts `y_T` flat.

## Goal
Make the UCSV arm reproduce the paper's UCSV benchmark: fit `ucsv` once on the unshifted target
through origin T; forecast = `tau_{T|T}`, identical for h=1/3/6/12 (flat). Then rerun the UCSV
cells and rescore vs IJF Table 5. This should move UCSV toward the paper; it may finally close it.

## Method
1. Determine the correct macroforecast configuration to express a "fit-once, flat one-sided
   level-trend forecast across all horizons" for `ucsv`, mirroring how the RW benchmark
   (`Arm("rw", model="naive", is_benchmark=True)`) stays flat. Investigate:
   - How does the pipeline treat `naive`/`is_benchmark=True` so it is flat under direct policy?
     (read the policy/target handling — likely the naive model ignores the horizon shift.)
   - Can UCSV be configured to fit the UNSHIFTED target and emit a flat `tau_{T|T}` across
     horizons? Candidate levers: use a non-shifting policy for this arm (recursive/level rather
     than horizon-shifted direct); or a per-arm target override so ucsv sees the unshifted series
     and its forecast is broadcast flat across h.
   - IMPORTANT: UCSV must remain a COMPETITOR, not the benchmark. RW (naive) stays the SOLE
     `is_benchmark=True` denominator — do NOT set `is_benchmark=True` on UCSV (that would change
     the evaluation denominator and is wrong). Flatness must come from the policy/target config,
     not from benchmark status.
2. If macroforecast CAN express it (config only), apply it in
   `scripts/replication/medeiros_2021_pipeline/run_block.py` (+ registry.py if used). Confirm the
   corrected UCSV forecasts are FLAT across h (h=1/3/6/12 identical from each origin) via a quick
   printed check. Then rerun ONLY the UCSV cells via `run_pipeline(n_jobs="auto")` reusing
   `qa/result_cells`; rescore vs Table 5. AR/RF/RW cached numbers must stay byte-identical.
3. If macroforecast CANNOT express a flat one-sided level-trend benchmark for `ucsv` under the
   direct horse-race (i.e., there is no config to fit the unshifted target + broadcast `tau_{T|T}`
   flat), that is a PACKAGE GAP: record it precisely in
   `.dev-notes/replication_findings_medeiros.md` as a fix-lane input (what config is missing +
   file:line + severity). Do NOT patch `macroforecast/**` from this worktree. Do NOT fake flatness
   by post-processing outside the package's forecast path in a way that hides the gap — if you must
   compute the flat forecast in the runner as a documented workaround, clearly label it a WORKAROUND
   pending the package fix.

## Constraints
- This is a CORRECTNESS fix to match the paper's UCSV definition — NOT curve-fitting. Do NOT sweep
  gamma/draws/priors to move the number. The only change is making UCSV the flat one-sided
  `tau_{T|T}` forecast the paper specifies.
- AR/RF/RW cached numbers unchanged (reuse `qa/result_cells`; recompute UCSV only). No
  stats-changing shortcuts. No G3/other arms. No `macroforecast/**` patch. No push.

## Deliverables
- `scripts/replication/medeiros_2021_pipeline/run_block.py` (+ registry.py) — corrected UCSV arm.
- `docs/replication/medeiros_2021.md` — updated UCSV row + arm→paper map noting the flat one-sided
  `tau_{T|T}` benchmark; final parity verdict for all 4 arms; if fully faithful now, say so.
- `.dev-notes/replication_findings_medeiros.md` — resolution note (config fix or package gap).

## STOP report → write `qa/codex_last_msg_b1ucsvfix.txt` AND print (4 purposes):
- P1: corrected UCSV row vs Table 5 (per horizon MATCH/CLOSE/DIVERGENT); confirm forecasts now FLAT
  across h; the full 4-arm parity table and whether B1 now fully replicates.
- P2: was it a config fix or a package gap? evidence. Confirm no package patch.
- P3/P4: reused result_store; recomputed UCSV only; AR/RF/RW byte-identical; no stats-changing shortcut.
Then STOP.
