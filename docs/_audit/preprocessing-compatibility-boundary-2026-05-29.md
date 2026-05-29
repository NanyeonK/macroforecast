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
