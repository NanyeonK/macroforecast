# Preprocessing Legacy Removal

Date: 2026-05-29

This pass removes the old preprocessing contract layer from the package. The
current public surface is the direct pandas API:

- `macroforecast.preprocessing.reprocess(...)`
- `macroforecast.preprocessing.plan(...)`
- `macroforecast.preprocessing.report(...)`
- step helpers such as `apply_transform_codes`, `handle_mixed_frequency`,
  `handle_outliers`, and `impute_missing`

## Removed

- `macroforecast.preprocessing.schema`
- `macroforecast.preprocessing.build`
- `macroforecast.preprocessing.types`
- `macroforecast.preprocessing.target_scale`
- `macroforecast.preprocessing.option_docs`
- `macroforecast.preprocessing.errors`
- compatibility shims under `macroforecast.layers.l2_preprocessing`
- registry-facing shim `macroforecast.core.layers.l2`
- legacy `PreprocessContract` tests
- generated L2 option-reference pages

## Runtime Replacement

`macroforecast.core.runtime.materialize_preprocessing` no longer imports
`macroforecast.preprocessing.schema`. It resolves the old `preprocessing`
recipe block through a small local direct-default map so existing recipe
execution can continue while the future YAML wrapper is designed around
`reprocess(...)`.

## Docs Replacement

The old generated L2 option docs are removed from docgen introspection. The
maintained user-facing page is `docs/reference/preprocessing.md`, which
documents direct callable inputs, outputs, defaults, and choices.
