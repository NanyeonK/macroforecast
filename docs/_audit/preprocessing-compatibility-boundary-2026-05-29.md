# Preprocessing Compatibility Boundary

Date: 2026-05-29

This audit records why the old preprocessing contract modules remain in the
tree while the public preprocessing API moves to direct pandas callables.

## Current public path

New user code should use:

- `macroforecast.preprocessing.reprocess(...)`
- `macroforecast.preprocessing.plan(...)`
- `macroforecast.preprocessing.report(...)`
- step helpers such as `apply_transform_codes`, `handle_outliers`, and
  `impute_missing`

These functions accept canonical pandas panels, `DataBundle`, `DataSpec`, or a
`(panel, metadata)` tuple. They return `PreprocessedData` or plain pandas
objects.

## Compatibility-only path

The following are still importable, but they are not the current design target:

- `macroforecast.preprocessing.schema`
- `macroforecast.preprocessing.build`
- `macroforecast.preprocessing.types`
- `build_preprocess_contract`
- `PreprocessContract`
- `preprocess_summary`
- `preprocess_to_dict`

## Why not delete them in this pass

Direct imports still exist in:

- `macroforecast.core.runtime`
- `macroforecast.core.layers._bootstrap`
- `macroforecast.core.layers.l2`
- `macroforecast.layers.l2_preprocessing.schema`
- `tools/docgen/*`
- legacy tests under `tests/layers`, `tests/core`, `tests/tools/docgen`, and
  `tests/test_preprocess_contract.py`

Deleting these files now would turn this small preprocessing cleanup into a
runtime/docgen/YAML rewrite. That rewrite should happen after the current direct
callable API is stable.

## Deletion gate

These modules can be deleted after:

1. Runtime materialization no longer imports `macroforecast.preprocessing.schema`.
2. Doc generation no longer introspects `L2_LAYER_SPEC`.
3. Legacy `PreprocessContract` tests are removed or rewritten against
   `reprocess()`, `plan()`, and `report()`.
4. Compatibility shims under `macroforecast.layers.*` and
   `macroforecast.core.layers.*` are removed or replaced by the new wrapper.

## Next rewrite sequence

1. Replace `macroforecast.core.runtime` imports of
   `macroforecast.preprocessing.schema` with direct calls to
   `macroforecast.preprocessing.reprocess`, plus a small wrapper that maps old
   recipe/YAML keys to the direct-call arguments.
2. Replace `tools/docgen` use of `L2_LAYER_SPEC` with function-reference pages
   generated from semantic modules, or remove generated option docs entirely
   for preprocessing.
3. Rewrite `tests/test_preprocess_contract.py` into public API tests covering
   `reprocess`, `plan`, `report`, and the step helpers.
4. Delete `macroforecast.preprocessing.schema`, `build`, and `types` only after
   the runtime and docgen imports above are gone.

## Frequency alignment note

`macroforecast.data.combine(...)` remains the correct place to combine
FRED-MD/FRED-QD with FRED-SD. `preprocessing.handle_mixed_frequency(...)` is a
direct helper for already-formed panels. Its quarterly-to-monthly compatibility
alias `step_backward` now accepts the same in-quarter repeat semantics as
`data.combine(..., quarterly_to_monthly="repeat_within_quarter")`; the clearer
new spelling for direct calls is `repeat_within_quarter`.
