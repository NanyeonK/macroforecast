# B1 (Medeiros 2021) — rebase onto fixed main + close UCSV parity with the new package knobs

Worktree `/home/nanyeon99/project/mf-b1-medeiros` (branch `repro/medeiros-2021`). Read
`.dev-notes/REPLICATION_OBJECTIVES.md` first (four purposes; P1 faithful replication primary).

## Situation
B1 parity is AR faithful (MATCH x3 + CLOSE), RF faithful (CLOSE x4 after the feature-matrix fix),
and **UCSV is the only remaining divergent arm** (h=3/h=6/h=12 DIVERGENT). UCSV was blocked
because the package `ucsv` only exposed `gamma`. The package fixes #1/#3/#4 just landed on `main`
(HEAD `c3cd17e5`). #3 added UCSV initial-prior-variance knobs
(`initial_obs_log_vol_variance`, `initial_level_log_vol_variance`, defaults 10.0) plus the
`random_state` seed. The paper specifies **Vτ = Vh = 0.12** initial-prior variances.

## Steps
1. **Rebase** `repro/medeiros-2021` onto `main` (`git rebase main` or merge main). Expect NO
   conflict — B1's changes are under `scripts/replication/` + `docs/replication/`; main touched
   `macroforecast/**` core + tests + `docs/reference`. If a conflict appears, resolve minimally
   keeping both sides' intent.
2. **#4 window** — replace the module-level `MedeirosRollingWindow` subclass in `run_block.py`
   with the new package horizon-dependent window (the `R(base, horizon)` rule or `size_by_horizon`
   from #4; read `macroforecast/window/core.py` for the exact API). Then rerun a CHEAP check that
   RW/AR/RF numbers are **byte-identical** to the pre-swap cached cells (they must not move). If
   they move, STOP and report — the window swap is not equivalent.
3. **#1 workaround** — the `model_selection={"random_forest": None}` in the RF arm can stay
   (harmless) OR be removed now that explicit `Arm.params` are honored. If you remove it, confirm
   RF numbers unchanged. (Low priority — leave it if unsure.)
4. **UCSV parity (the point)** — read the paper's UCSV specification in
   `qa/medeiros_correct_fulltext.txt` (and the author code if it exists) to MAP the paper's
   parameterization to the package knobs correctly: confirm whether Vτ=Vh=0.12 are the
   initial-prior variances (=> `initial_obs_log_vol_variance=0.12`,
   `initial_level_log_vol_variance=0.12`) and what the log-vol innovation variance should be
   (currently `gamma=0.2`). Set the UCSV arm to the paper's exact spec via the new knobs + a fixed
   `random_state`. Rerun **ONLY the UCSV cells** via `run_pipeline(n_jobs="auto")` reusing
   `qa/result_cells`; rescore vs IJF Table 5.
5. **Assess** — does UCSV now reach MATCH/CLOSE at h=1/3/6/12? If YES → B1 fully replicates
   (AR✓ RF✓ UCSV✓) — say so. If still divergent, diagnose: try the paper's exact
   parameterization (the knob-to-paper mapping is the likely lever); if a genuine residual gap
   remains, document it honestly as a caveat. Do NOT force a match with stats-changing shortcuts
   (no reduced draws/tolerances).

## Constraints
- Author-faithful config only. AR/RW/RF cached numbers must stay unchanged (reuse `qa/result_cells`,
  recompute UCSV only). No stats-changing shortcuts (Purpose 4).
- Do NOT patch `macroforecast/**` — all package fixes are already on main. This is replication
  config only. Do NOT run G3 (full sweep) or other arms. No git push/gh.
- Replication artifacts (docs/replication, scripts/replication, .dev-notes) are updated in place;
  the rebase itself may be committed on the branch, but do not push.

## Deliverables (update in place)
- `docs/replication/medeiros_2021.md` — final parity table for ALL FOUR arms (AR/RF/UCSV/RW),
  the UCSV arm→paper-param map via the NEW knobs, and a clear statement of whether B1 now fully
  replicates or what residual UCSV caveat remains.
- `.dev-notes/replication_findings_medeiros.md` — UCSV resolution note; confirm #3/#4 removed the
  prior workarounds; any new finding.

## STOP report → write `qa/codex_last_msg_b1ucsv.txt` AND print. 4 purposes:
- P1: post-rebase parity table (all 4 arms); UCSV verdict per horizon and the knob values that
  closed it (or the documented residual).
- P2: confirm rebase clean; #4 window swap equivalent (RW/AR/RF unchanged); any new bug/gap.
- P3/P4: reused result_store; recomputed UCSV only; RW/AR/RF byte-identical; no stats-changing shortcut.
Then STOP.
