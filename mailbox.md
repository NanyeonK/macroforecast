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

---

# mailbox.md -- builder (C67 Tutorial Standalone-First Rewrite)

## Interface Changes: None

C67 is documentation-only (workflow 3). No Python symbols were added, removed, or renamed.
No API changes, no new exports.

## Blockers: None

## Notes for Tester

`tests/docs/test_tutorial_smoke.py` will likely need updates after C67. The new tutorial
code blocks use:
- `LinearAR(p=2)` and `LinearAR(p=4)` -- NOT `n_lags`
- `FactorAugmentedAR(p=2, n_factors=3)` -- NOT `n_lags`
- `model.fit(X_train, y_train)` with an empty DataFrame for X when LinearAR is used alone
- `model.predict(X_test)` -- NOT `model.predict(n_periods=...)`

If the smoke test runs code blocks sequentially within a shared namespace, variable `y`
(generated in the data section) must be in scope when the model fitting block runs.
Tutorial 02 also requires `X` in scope for PCR and FAAR blocks.

## Status

4 commits made:
- ac4e5ac3: index.md + two_entry_points.md reorder/refresh
- e30cfb66: 01_first_forecast.md standalone-first rewrite
- 65959124: 02_full_study.md 3-model comparison
- 34b95655: 03_custom_model.md BaseEstimator subclass pattern

---

# mailbox.md — builder PR-F (L7 schema + ops body move)

## STATUS: DONE

## Interface Changes

- `macroforecast.layers.l7_interpretation.ops` is now the **canonical** location for all L7 ops body and constants. The old path `macroforecast.core.ops.l7_ops` is permanently deleted.
- `macroforecast.core.layers.l7` is permanently deleted. Use `macroforecast.layers.l7_interpretation.schema` for all L7 schema symbols.
- `macroforecast.core.runtime` now imports `macroforecast.layers.l7_interpretation.ops` for side-effect registration (line added alongside l5/l6/l8 pattern).

## Test files requiring update (tester pipeline)

15 test files import from the deleted legacy paths. Canonical replacements:
- `macroforecast.core.ops.l7_ops` → `macroforecast.layers.l7_interpretation.ops`
- `macroforecast.core.layers.l7` → `macroforecast.layers.l7_interpretation.schema`

See spec.md table for the full list of affected test files.

Note: `tests/layers/test_l7_interpretation_phase3e.py::test_backward_compat_core_ops_l7_ops` explicitly tests the deleted path — this test must be updated or removed by tester.
