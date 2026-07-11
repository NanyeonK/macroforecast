# B1 (Medeiros 2021) — finalize the docs trust page with the paper-grounded UCSV conclusion

Worktree `/home/nanyeon99/project/mf-b1-medeiros`. This is a DOCS-ONLY task (Purpose 1 trust
artifact). Do NOT rerun models, do NOT patch code, do NOT change parity numbers. Only edit
`docs/replication/medeiros_2021.md` (and `.dev-notes/replication_findings_medeiros.md` if needed)
to record the final, paper-grounded UCSV conclusion at docs-site quality.

## What to write (all grounded in the paper fulltext `qa/medeiros_correct_fulltext.txt`)

Update the UCSV section of the docs page so a skeptical reader sees exactly why AR and RF replicate
faithfully while UCSV has a documented residual that is the PAPER's under-specification, not a
package defect. Include:

1. **Benchmark structure (verbatim anchor).** The paper's ratios are "with respect to the random
   walk model" (Table 5 caption). RW is the sole benchmark/denominator; RW, AR, UCSV are the three
   "usual univariate benchmarks", and AR/UCSV/RF are reported as RMSE ratios vs RW. Our replication
   matches this exactly.

2. **Rolling-window anchor (verbatim).** The paper: "R_h = 360 − h − p − 1" for 1990–2000 and
   "R_h = 492 − h − p − 1" for 2001–2015 — rolling windows, which our runner uses. (Not expanding.)

3. **The paper's UCSV specification (quote it).** Reconstructed from the paper:
   > π_t = τ_t + e^{h_t/2}·ε_t;  τ_t = τ_{t−1} + u_t;  h_t = h_{t−1} + v_t
   > ε_t ~ N(0,1); u_t, v_t normal, zero mean, variances given by inverse-gamma priors;
   > τ_1 ~ N(0,V), h_1 ~ N(0,V_h), with V = V_h = 0.12; estimated by MCMC.
   State clearly: the package UCSV arm sets V = V_h = 0.12 (initial_obs/level_log_vol_variance) and
   uses the standard SW gamma=0.2 log-vol innovation calibration, fit on the unshifted target through
   origin T with a FLAT one-sided τ_{T|T} h-step forecast (the SW forecast structure).

4. **What the paper does NOT publish** (the residual's true source): the inverse-gamma
   hyperparameters for the u_t/v_t innovation variances, the MCMC draw/burn counts, and the seed.
   The author replication repo contains no UCSV implementation. These unpublished choices govern the
   trend smoothing and therefore the UCSV accuracy.

5. **The over-performance anchor (verbatim).** The paper states (result 7): "both AR and UCSV
   outperform the RW alternative ... the UCSV model is slightly superior to the AR specification."
   In the paper UCSV (0.954/0.797/0.777/0.781) ≈ AR (0.902/0.790/0.791/0.753). Our UCSV
   (0.9148/0.7297/0.7252/0.6976) is materially BETTER than AR — i.e. our valid SW-UCSV implementation,
   under a reasonable but different (published-params-only) calibration, smooths the trend less and
   is "too accurate" relative to the paper's unpublished calibration. This is the honest diagnosis:
   the package faithfully implements a valid SW-UCSV; the paper's specific UCSV cannot be exactly
   reproduced from published information, and matching it by sweeping the unpublished
   inverse-gamma/draws would be curve-fitting, which we deliberately do NOT do.

6. **The replication also surfaced + fixed a real extraction bug** (record as a Purpose-2 win): the
   UCSV arm under `policy="direct"` was fitting horizon-shifted targets (non-flat forecasts); it was
   corrected to a flat one-sided τ_{T|T} forecast via a per-arm recursive policy override. Verified
   flat (max_abs_range=0 across 300 origins). AR/RF/RW numbers byte-identical.

## Final B1 verdict to state on the page
RW benchmark, AR, and RF replicate the paper faithfully (AR MATCH×3+CLOSE; RF CLOSE×4). UCSV
reproduces the paper's full published specification (model equations, V=V_h=0.12, rolling windows,
flat one-sided τ_{T|T}); its residual RMSE-ratio gap at h=3/6/12 is attributable to the paper's
unpublished UCSV internals (inverse-gamma hyperparameters, MCMC draws/burn), documented as an
honest caveat rather than curve-fit. This demonstrates package trust: every author-specified knob
is expressible and set; the only gap is information the paper itself does not provide.

Keep the existing parity table and arm→author-param map. Write to docs-site quality. Do NOT alter
numbers or rerun anything.

## STOP report → write `qa/codex_last_msg_b1docs.txt` AND print: which sections you updated, and
confirm no numbers/code changed. Then STOP.
