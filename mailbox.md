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

---

# mailbox.md — builder continuation (C66 DAG Jargon Cleanup)

## Interface Changes: None

C66 is a pure text replacement -- no Python symbol renames, no API changes, no new exports.

## Blockers: None

## Status

4 continuation commits made (completing C66 docs/ cleanup):
- 211825fb: tutorials + how-to DAG cleanup (5 files)
- 760c722a: explanation docs DAG cleanup (2 files, 19 edits)
- 43d8322f: reference docs DAG cleanup (15 files, ~35 edits)
- 238c14ef: README DAG terminology cleanup

Final docs/ DAG count: 23 total (16 in archive KEEP, 7 non-archive all KEEP items).
macroforecast/ public-surface: 10 remaining (all inline code comments, INTERNAL-VOCAB).

KEEP decisions per planner:
- partial_layer_execution.md lines 27, 55, 341-342 (developer reference)
- contributing.md:29 dag.py file listing (accurate module name)
- tree_navigator.md:53 navigator_selected_dag_items (frozen API)
- architecture/layer7/index.md:31 importance DAG body (INTERNAL-VOCAB per impact.md 2.17)
