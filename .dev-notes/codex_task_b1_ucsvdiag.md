# B1 (Medeiros 2021) — UCSV forecast-extraction CORRECTNESS diagnostic (NOT a match-hunt)

Worktree `/home/nanyeon99/project/mf-b1-medeiros` (branch `repro/medeiros-2021`, rebased on main
c3cd17e5). Read `.dev-notes/REPLICATION_OBJECTIVES.md`. This is a CORRECTNESS check, not a
parity-closing attempt. FORBIDDEN: sweeping gamma / draws / priors to move UCSV toward the paper
number. You are verifying ONE thing: does macroforecast's `ucsv` produce the exact forecast the
paper specifies?

## The question
The paper (Medeiros et al. 2021, Appendix B.1) uses the Stock-Watson UCSV benchmark with
**one-sided `tau_t|t` h-step forecasts**. In SW-UCSV, `y_t = tau_t + eps_t` with
`tau_t = tau_{t-1} + eta_t` (trend is a random walk with stochastic volatility). Because the trend
is a driftless random walk and `eps` is mean-zero, the correct h-step point forecast of `y_{T+h}`
given data through `T` is the **filtered (one-sided) current trend** `tau_{T|T}` — the SAME value
for every horizon h (flat across h), and it must use data through T ONLY (filtered, NOT smoothed).

## Verify (read-only diagnosis + a small controlled numeric check)
1. Read `macroforecast/models/bayesian.py` `ucsv` (and any forecast helper it calls). Determine
   EXACTLY what quantity it returns as the h-step forecast: (a) is it the filtered `tau_{T|T}`
   (one-sided, data through T), or the smoothed `tau_{T|n}` (uses full sample = lookahead)?
   (b) is it flat across h (RW trend) or does it do something horizon-specific? (c) is it the
   posterior mean over retained MCMC draws of `tau_{T|T}`?
2. Small controlled check (no curve-fitting): construct a short toy series with a known trend,
   run `ucsv` at a fixed seed, and confirm (i) the h=1/3/6/12 forecasts from a given origin are
   IDENTICAL (flat, as an RW trend forecast must be), and (ii) the forecast equals the one-sided
   filtered trend posterior mean, NOT the smoothed trend. Compare filtered vs smoothed explicitly
   if the code exposes both.
3. Also re-confirm (g2e already suggested this): the forecast at origin T uses only data <= T.

## Verdict
- **CORRECT**: if `ucsv` returns the one-sided filtered `tau_{T|T}` posterior-mean, flat across h,
  no lookahead — then the package's UCSV forecast IS the paper's specified forecast, and the B1
  UCSV divergence is CONFIRMED to be the paper's unpublished internals (draw count, burn-in,
  inverse-gamma hyperparameters, log-vol innovation variance / gamma), NOT a package defect.
  Upgrade the docs caveat from "assumed" to "VERIFIED: package forecast is the correct one-sided
  SW forecast; residual gap is paper under-specification."
- **BUG**: if `ucsv` returns the smoothed trend (lookahead), a non-flat h-step, a wrong posterior
  quantity, or otherwise deviates from the one-sided `tau_{T|T}` forecast — describe the exact
  defect (file:line + the wrong-vs-right quantity + a minimal repro) in
  `.dev-notes/replication_findings_medeiros.md` as a fix-lane input. Do NOT patch `macroforecast/**`
  from this worktree. Note whether fixing it would plausibly move UCSV toward the paper.

## Constraints
- Correctness only. NO gamma/draws/prior sweeping to match the paper. No stats-changing shortcuts.
- Do NOT patch `macroforecast/**`. Do NOT rerun G3 or other arms. The AR/RF/RW/UCSV parity numbers
  from the last pass stand; you may reuse `qa/result_cells`. No push.

## Deliverables
- `docs/replication/medeiros_2021.md` — UCSV caveat upgraded to VERIFIED (with the evidence) OR
  flagged as a package bug if found.
- `.dev-notes/replication_findings_medeiros.md` — the diagnosis (CORRECT+evidence, or BUG+repro).

## STOP report → write `qa/codex_last_msg_b1ucsvdiag.txt` AND print:
- Verdict CORRECT or BUG, the exact quantity `ucsv` returns (filtered vs smoothed, flat vs
  horizon-specific, posterior mean of what), the toy-check result, and the resulting docs caveat
  wording (or the bug repro). Then STOP.
