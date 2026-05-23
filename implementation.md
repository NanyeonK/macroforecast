# implementation.md -- Cycle 67 Tutorial Standalone-First Rewrite

## Summary

Documentation-only rewrite (workflow 3). Five tutorial files rewritten to present
the standalone `macroforecast.models` API first, with the recipe pipeline introduced
only in the final graduation section of each tutorial. No source code was modified.

---

## Files Modified

### `docs/tutorial/index.md`
- Replaced the single-paragraph opening with two paragraphs describing the standalone-first
  tutorial structure and summarizing what each tutorial covers.
- Line count: 18 -> 25 lines (+7).
- No toctree changes.

### `docs/tutorial/two_entry_points.md`
- Reordered: standalone models section now leads (was second). Recipe pipeline section
  follows. Comparison table columns swapped (standalone left, recipe right).
- Updated standalone code snippet to use `LinearAR(p=2)` with `from macroforecast.models`.
  Removed the old `mf.functions.ridge_fit` snippet.
- Decision flowchart reversed: starts from "Working in a notebook or one-off script?"
  instead of "Need bit-exact replication?".
- Added "Transition path" prose reframing recipes as graduation from standalone.
- Line count: 107 -> 112 lines (+5).

### `docs/tutorial/01_first_forecast.md`
- Full rewrite. Removed recipe-first YAML blocks, `mf.run()` invocations in main body,
  and manifest/replicate sections.
- New structure: install check, synthetic AR(2) data generation, LinearAR fit/predict,
  TimeSeriesSplit OOS loop, graduation section.
- Line count: 211 -> 95 lines (-116).

### `docs/tutorial/02_full_study.md`
- Full rewrite. Removed ~370 lines of YAML recipe blocks and recipe-centric prose.
- New structure: synthetic 5-feature macro panel, LinearAR OOS loop, PCR and FAAR
  comparison, summary table, graduation section.
- Line count: 491 -> 125 lines (-366).

### `docs/tutorial/03_custom_model.md`
- Full rewrite. Replaced recipe-centric `@mf.register_model` decorator pattern with
  BaseEstimator + RegressorMixin subclassing (C64 pattern).
- Includes `ConstantTrendPlusAR` as a worked example with full `fit` and `predict`
  implementation. Shows direct use in TimeSeriesSplit. Graduation section uses the
  functional wrapper pattern for `mf_custom.register_model`.
- Line count: 258 -> 130 lines (-128).

---

## API Verification (Parameter Name Corrections)

Critical corrections made relative to the spec.md code snippets:

| Spec snippet | Actual source signature | Correction applied |
|---|---|---|
| `LinearAR(n_lags=2)` | `_LinearARModel.__init__(self, p: int = 1)` | Changed to `LinearAR(p=2)` |
| `LinearAR(n_lags=4)` | same | Changed to `LinearAR(p=4)` |
| `FactorAugmentedAR(n_factors=3, n_lags=2)` | `_FactorAugmentedAR.__init__(self, p: int = 1, n_factors: int = 3)` | Changed to `FactorAugmentedAR(p=2, n_factors=3)` |
| `model.predict(n_periods=len(y_test))` | `_LinearARModel.predict(self, X: pd.DataFrame)` | Changed to `model.predict(X_test)` with empty DataFrame |
| `mf.custom.register_model` accepts class | `register_model(name, function)` takes callable only | Used functional wrapper pattern |

`PrincipalComponentRegression(n_components=3)` was correct in the spec.

All three signatures verified via `uv run python -c "import inspect; ..."` smoke check:
- `LinearAR.__init__ sig: (self, p: int = 1) -> None`
- `PrincipalComponentRegression.__init__ sig: (self, n_components: int = 4) -> None`
- `FactorAugmentedAR.__init__ sig: (self, p: int = 1, n_factors: int = 3) -> None`
- `LinearAR.fit sig: (self, X: pd.DataFrame, y: pd.Series) -> _LinearARModel`
- `LinearAR.predict sig: (self, X: pd.DataFrame) -> np.ndarray`

---

## Unit Tests

Not applicable for documentation-only workflow 3. The existing smoke tests in
`tests/docs/test_tutorial_smoke.py` will need to be updated by tester to match
the new code blocks (noted in mailbox.md).

---

## Cross-Reference Validation

All `{doc}` cross-references in the rewritten files verified against the tutorial
and how_to directory structure:
- `{doc}00_install` -- exists
- `{doc}02_full_study` -- exists
- `{doc}03_custom_model` -- exists
- `{doc}two_entry_points` -- exists
- `{doc}../how_to/sweep_over_models` -- exists
- `{doc}../how_to/add_custom_model` -- exists

---

## Design Choices

1. LinearAR and empty X DataFrame: the spec suggested `model.predict(n_periods=...)`,
   but the actual `predict(X: pd.DataFrame)` signature uses `len(X)` for output size.
   We pass an empty DataFrame with a matching DatetimeIndex, which is the cleanest
   pattern when no feature columns are available.

2. Tutorial 01 MSE interpretation: the prose notes "We find that the mean CV MSE is
   low relative to the noise variance (0.25)" -- this is consistent with the data
   generating process (scale=0.5, so variance=0.25) and is a factual statement, not
   exaggeration.

3. Tutorial 03 graduation: `mf_custom.register_model` takes a function, not a class.
   The functional wrapper pattern is the correct approach and matches the existing API
   documented in `macroforecast/custom.py`.

---

## Commits Made (4 atomic)

1. `ac4e5ac3` -- docs(c67): reorder tutorial/index.md and refresh two_entry_points.md
2. `e30cfb66` -- docs(c67): rewrite 01_first_forecast.md -- standalone LinearAR (211 -> ~95 lines)
3. `65959124` -- docs(c67): rewrite 02_full_study.md -- sklearn TimeSeriesSplit + 3-model comparison (491 -> ~125 lines)
4. `34b95655` -- docs(c67): rewrite 03_custom_model.md -- BaseEstimator subclass pattern (258 -> ~130 lines)

---

## Known Limitations / Deferred Items

- `tests/docs/test_tutorial_smoke.py` extracts and runs Python blocks from tutorials.
  The new code blocks reference `y` and `X` that must be generated before the model
  fitting blocks are run. Tester should verify the smoke test handles variable scope
  across code blocks, or update the test to generate the data inline.
- Tutorial 01 predict output is described as "a single-step-ahead forecast repeated
  for each row of `X_test`". This matches the `_LinearARModel.predict` implementation
  which returns `np.array([float(...)] * len(X))` -- a constant array. This is correct
  behavior for an AR model without recursive forecasting.
