# B2 (Hounyo-Li) — decompose the residual PCA gap: LEAK alone vs residual config/impl difference

Worktree `/home/nanyeon99/project/mf-b2-hounyoli`. Our leak-free `pcr` gives PCA ratio **1.081** vs
paper **0.970** (h=1, inflation, full sample). Question (Chan chose to dig): is the entire 0.111 gap
the author's TARGET-y standardization leak (which we deliberately do NOT put in the package), or is
there a residual config/implementation difference on top? Isolate it rigorously and cheaply. Focus on
ONE method: **PCA/pcr, inflation, h=1, full sample** (cheap; do NOT run supervised or other horizons).

## This is DIAGNOSTIC only
Any "leaky" emulation here is a LABELED DIAGNOSTIC in the replication runner / a throwaway script —
NOT a macroforecast package feature. Do NOT patch `macroforecast/**`. The goal is attribution, not a
new capability.

## Method — three measurements for PCA (inflation, h=1, full sample)
1. **Author-code oracle (gold standard).** Check if Octave is on server1 (`which octave`). If yes, run
   the author's PCA path (`qa/hounyo_li_matlab/.../PCA_emp002.m` + `inflation_linear.m` + the AR_BIC
   benchmark scripts) for CPIAUCSL, h=1, full sample, with the author's exact fold-tuned K — get the
   author PCA RMSFE ratio vs author AR. Compare to Table 2 (0.970).
   - If author-MATLAB ≈ 0.970 → the author code reproduces Table 2 (our data/setup is faithful) — proceed.
   - If author-MATLAB ≠ 0.970 → a data/Table-2/setup mismatch exists; investigate THAT before blaming
     the leak (compare the built panel/target to what the .m scripts load).
   - If Octave is NOT available, say so and rely on measurements 2–3 (a faithful Python port of the
     author standardization is acceptable as the oracle, clearly labeled).
2. **Leak-free pcr** = 1.081 (already have; reuse the cached cell).
3. **Leaky-emulated pcr (diagnostic).** Run OUR `pcr` on the SAME data/window/K but with the author's
   leaky target standardization emulated caller-side: standardize the target block INCLUDING the
   realized `y_{T+h}` per `inflation_linear.m:196-197` (`ytplush(:,1+h:T+h)` uses the future target in
   the mean/std). Everything else identical to measurement 2. Get the leaky-emulated ratio.

## Conclusion to reach (quantified)
- If leaky-emulated ≈ author-MATLAB ≈ 0.970 → **the entire gap is the target-y leak**; our leak-free
  package is correct and the paper's factor advantage IS the leak. Report the exact leak contribution
  (Δ from 1.081 → leaky value).
- If leaky-emulated ≈ author-MATLAB but ≠ 0.970 → data/Table-2 mismatch (identify it).
- If leaky-emulated ≠ author-MATLAB → there is a RESIDUAL implementation difference beyond the leak;
  identify it by stepping our `pcr` fit against `PCA_emp002.m` (residualization order, SVD orientation,
  score construction, K, embed/lag handling) and report the specific divergence.
Decompose: gap 0.111 = [leak contribution] + [residual impl/config contribution], with numbers.

## Constraints
- PCA/pcr only, inflation, h=1, full sample. Reuse `runs/hl2026_store` for the leak-free cell. No
  supervised, no other horizons, no G2/G3. Do NOT patch `macroforecast/**` (leaky emulation is a
  labeled diagnostic outside the package). No stats-changing shortcut on the non-leak parts. No push.

## STOP report → write `qa/codex_last_msg_b2decomp.txt` AND print:
- The three PCA measurements (leak-free 1.081 / leaky-emulated X / author-MATLAB Y or "octave N/A") vs paper 0.970.
- The decomposition: how much of the 0.111 gap is the leak vs residual impl/config, with numbers.
- If a residual impl difference exists, its exact cause (file:line vs PCA_emp002.m).
- Verdict: is macroforecast's leak-free pcr CORRECT (gap = leak only), or is there a residual gap to fix?
Then STOP.
