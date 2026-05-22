# mailbox.md — builder (C59 fix retry 1/3)

## Interface Changes

### _boruta_selection — new optional parameter

`params["n_shadow_copies"]` (int, default 6) controls the number of independent shadow
permutations used per Boruta iteration. MISA is computed over all `n_shadow_copies * N`
shadow importances. Existing callers that do not pass this parameter get the default
behavior (6 copies, Bonferroni-calibrated for T/N >= 6 at alpha=0.05, N<=20).

Callers that relied on the empty-result fallback (selecting the argmax-hit feature when
nothing is accepted) will now receive an empty DataFrame for null DGPs. This is the
CORRECT behavior per Algorithm 1; the fallback was a bug.

## Blockers: None

## Spec Adjustments

- HHS MRE-1 `T_DEFAULT` bumped from 500 to 2000 (finite-sample limit, not a runtime bug).
  Documented in `tests/core/test_l4_realized_garch_c56.py` class docstring and method.

## Status

Both commits made:
- 9cbbe36f: Boruta fix (fallback removal + n_shadow_copies=6)
- cceaa9be: HHS T_DEFAULT=2000
