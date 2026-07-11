# B1 (Medeiros et al. 2021, IJF 37(2):419-436) — G2 divergence resolution toward FAITHFUL parity

You work in the REPLICATION worktree `/home/nanyeon99/project/mf-b1-medeiros` (branch
`repro/medeiros-2021`). FIRST read `.dev-notes/REPLICATION_OBJECTIVES.md`: the four co-equal
purposes govern this task and your STOP report MUST summarize all four. Also skim
`docs/reference` and `docs/guide` for the pipeline / window / model APIs, and
`docs/replication/medeiros_2021.md` for current state.

## GOAL — Purpose 1 is primary
The AR arm already replicates IJF Table 5 faithfully (MATCH x3 + CLOSE x1). Five cells do NOT,
and your job is to make macroforecast reproduce them FAITHFULLY by configuring the package to
the AUTHOR'S EXACT specification, so the docs page shows trustworthy 1:1 replication. Do NOT
settle for "documented divergence" if faithful parity is achievable through correct
author-spec configuration. Classification (config-diff vs package-bug) is the MEANS to parity,
not the end.

Divergent cells (RMSE ratio vs RW / paper / d):
- RF   h=1  0.9156 / 0.844 / +0.072   (our RF is WORSE-relative-to-RW than paper at the extreme)
- RF   h=12 0.7507 / 0.685 / +0.066
- UCSV h=3  0.7264 / 0.797 / -0.071   (our UCSV BEATS RW MORE than paper -- systematically "too good")
- UCSV h=6  0.7204 / 0.777 / -0.057
- UCSV h=12 0.6951 / 0.781 / -0.086

## ORACLE
The author's actual code is cloned at `qa/ForecastingInflation` and `qa/HDeconometrics` —
ground truth for every tuning constant. Correct paper fulltext: `qa/medeiros_correct_fulltext.txt`.

## METHOD — per divergent arm, diagnose THEN remediate toward parity

### (1) RF
- From the author R code, extract the EXACT `randomForest` call: `ntree`, `mtry`, and the
  tree-size controls (`maxnodes`, `nodesize`, `sampsize`, `replace`). Prior notes indicate
  `maxnodes=25` — VERIFY against the R source, do not assume.
- Compare to (a) the RF `Arm` in `scripts/replication/medeiros_2021_pipeline/run_block.py`
  and (b) the macroforecast RF wrapper (`macroforecast/models/tree.py`). List every author
  setting the runner currently does NOT set. Hypothesis to test first: the runner omits a
  tree-depth cap, so our sklearn RF grows full-depth trees and overfits (worse OOS at h=1/h=12),
  while the author caps at `maxnodes=25` (=> sklearn `max_leaf_nodes=25`).
- If the package CAN express the missing author setting (`max_leaf_nodes` for `maxnodes`,
  `min_samples_leaf` for `nodesize`, etc.), ADD it to the RF Arm params in `run_block.py`
  (legitimate author-spec config, not a stats hack). Then rerun ONLY the RF cells via
  `run_pipeline` (reuse `qa/result_cells` for RW/AR/UCSV; `n_jobs="auto"`) and rescore vs Table 5.
- If the package CANNOT express an author setting (the knob is not wired through the wrapper),
  that is a macroforecast GAP -> record in `.dev-notes/replication_findings_medeiros.md` as a
  fix-lane input (file:line + what's missing + severity). Do NOT patch macroforecast package
  code from this worktree. Do NOT fake the setting.

### (2) UCSV
- Establish the paper's UCSV benchmark (Stock-Watson 2007 UCSV) and its exact spec: gamma
  (signal-to-noise / innovation variance), number of Gibbs draws, burn-in, and whether the
  forecast at time t uses ONLY data through t (no lookahead).
- Compare to macroforecast `ucsv` (`macroforecast/models/bayesian.py`) and the runner's UCSV
  Arm (`params={"gamma":0.2}`).
- Our UCSV being systematically "too good" is a red flag for either (a) a wrong gamma / draw
  count that over-smooths, or (b) a LOOKAHEAD / centering bug where the forecast peeks past the
  estimation cutoff. SCRUTINIZE the forecast-construction path for ANY use of information beyond
  the cutoff — that would be a Purpose-2 package bug and is the priority hypothesis.
- If it is an author-faithful config difference expressible in the package (gamma, draws), fix
  in `run_block.py` and rerun UCSV cells. If it is a package bug (lookahead) or a knob the
  package cannot express, record precisely in findings as a fix-lane input; do NOT patch the
  package here.

## HARD CONSTRAINTS
- NEVER edit `macroforecast/**` from this replication worktree. Record package bugs/gaps in
  `.dev-notes/replication_findings_medeiros.md` only.
- Only AUTHOR-FAITHFUL config changes to `run_block.py`. FORBIDDEN (Purpose 4): reducing
  trees/draws, loosening tolerances, subsampling, coarser grids, or any change whose purpose is
  to force a match rather than match the author.
- AR and RW numbers must stay bitwise-identical (cached in `qa/result_cells` — do not recompute
  or perturb). Reuse the result store; recompute only the arm/cells you remediate.
- Configure to the AUTHOR spec, never macroforecast defaults.
- Use `run_pipeline` with `n_jobs="auto"` for every rerun (maximize the box).
- Do NOT `git commit`, `git push`, or use `gh`. Do NOT run G3 (full sweep) or
  CSR/JMA/LASSO/bagging/hybrid arms.

## DELIVERABLES (update in place; no git commit)
1. `docs/replication/medeiros_2021.md` — update the Table 5 parity table with post-fix numbers;
   for every cell that reaches MATCH/CLOSE, note the author-spec setting that fixed it; for any
   cell still DIVERGENT, state the diagnosed cause (package bug/gap -> fix-lane input, or genuine
   unresolvable protocol gap -> caveat). Keep the arm->author-param map current.
2. `.dev-notes/replication_findings_medeiros.md` — BUGS/GAPS: any package bug/gap found
   (file:line + repro + severity). EFFICIENCY: new findings; confirm reruns reused the store.
3. `.dev-notes/codex_progress_b1.md` — append what you did.

## STOP REPORT — write to `qa/codex_last_msg_g2e.txt` AND print it. Summarize ALL FOUR purposes:
- P1 (faithful replication/docs): the POST-FIX parity table (per-cell MATCH/CLOSE/DIVERGENT),
  and for each previously-divergent cell whether it now replicates and via which author setting.
- P2 (bugs/gaps): every package bug/gap found this run (or "none new"), classified
  config-diff-vs-package-bug per cell.
- P3 (efficiency): efficiency findings; confirm reruns reused the result store; timings.
- P4 (statistically identical): confirm no stats-changing shortcut was used; confirm AR/RW
  numbers unchanged.
Then STOP.
