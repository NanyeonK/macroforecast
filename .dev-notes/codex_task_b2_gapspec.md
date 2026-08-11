# B2 (Hounyo-Li) — precise per-method GAP enumeration + exact specs for PACKAGE ADDITIONS

Worktree `/home/nanyeon99/project/mf-b2-hounyoli`. This is a READ-ONLY analysis + spec-writing task.
Do NOT patch `macroforecast/**`, do NOT rerun models, do NOT run SPCA/SsPCA/G2/G3. Output is a
specification document that will feed a separate MAIN package fix-lane.

## Framing (important)
The G1 smoke diverged from Table 2 because the package is MISSING or MIS-MAPPING some of the
author's methods — NOT because of a paper-specific wrapper. The author's methods (e.g. residualized
PCR) are GENERAL forecasting methods from the Stock-Watson diffusion-index / supervised-PCA
literature that belong in the package as reusable atomic units. Your job: pin down exactly what is
missing/different, and spec each addition as a GENERAL method (never "Hounyo-Li-specific").

## Task 1 — per-method gap table (all 5 + benchmark)
For each of {AR_BIC, PCA, sPCA, SPCA, SsPCA, PLS}, read the author's EXACT algorithm from the MATLAB
oracle under `qa/hounyo_li_matlab/` (e.g. `PCA_emp002.m`, `inflation_linear*.m`, `SsPCA_tune.m`,
`inflation_arbic.m`, `lag_bic.m`, and the sPCA/SPCA/PLS scripts). Compare to the CURRENT package
model it was mapped to (`far`, `scaled_pca`, `supervised_pca`, `supervised_scaled_pca`, `pls`,
`ar`). For each, output: author algorithm (concise math/steps), current package model behavior,
verdict = MATCH / DIFFERENT(how, precisely) / MISSING. Confirmed so far: author "PCA" = residualized
PCR (residualize future target on [target lag; constant], SVD standardized X_insample, regress
residual on top-K scores, add control forecast) which is NOT `far`. Verify the others rather than
trusting the design doc's "native" claims.

## Task 2 — exact spec for each method to ADD or CORRECT (as a general atomic unit)
For every DIFFERENT/MISSING method, write an implementation-ready spec expressed as a GENERAL method
(inputs, fit steps, forecast formula, params), suitable to add to `macroforecast/models/`. E.g. for
residualized PCR: name candidate `pcr` (principal component regression with optional control-column
residualization), params (n_components, control_columns, include_constant, standardize), fit/predict
math. Cite the author .m lines as the correctness oracle but describe the method generally. Note
which are genuinely new vs a param/option addition to an existing model.

## Task 3 — the standardization convention: leak or leak-free? (decide with code evidence)
The author standardizes X over a 240-row in-window block and the target over a 241-row block. Read
the EXACT .m lines and determine precisely which observations feed the mean/std:
- Does the predictor standardization use ONLY data available at origin T (rows <= T => leak-FREE
  in-window standardization), or does it use x at t>T / the OOS row's own future (=> look-ahead)?
- Does the target standardization use y_{T+h} (the realized forecast target => LEAK), or only
  training targets <= T?
Give a definitive verdict with line evidence. If it is a leak-FREE in-window rolling standardization
that macroforecast's target-availability guard currently over-restricts, spec it as a CONFIG OPTION
to add (general, leak-free). If it genuinely uses the realized future target (LEAK), say so plainly —
that part is NOT to be reproduced; it becomes a documented divergence, not a package addition.

## Output
Write `.dev-notes/b2_method_gap_spec.md` containing: the per-method gap table (Task 1), the
add/correct specs (Task 2), the standardization verdict + config-vs-leak decision (Task 3), and a
final "B2 arm re-mapping" (which package model — existing or to-be-added — each of the 5 arms should
use). This doc is the input to the main package fix-lane.

## Constraints
- READ-ONLY: no `macroforecast/**` patch, no model reruns, no SPCA/SsPCA/G2/G3. You may write only
  the spec doc (+ small throwaway probe scripts under qa/ if needed to inspect the .m logic).
- Describe every method GENERALLY (atomic-unit philosophy; no paper-specific naming in the method
  itself). No push.

## STOP report → write `qa/codex_last_msg_b2gapspec.txt` AND print: the per-method verdict table
(MATCH/DIFFERENT/MISSING), the list of methods/params to ADD to the package, the standardization
leak-vs-config verdict, and the B2 arm re-mapping. Then STOP.
