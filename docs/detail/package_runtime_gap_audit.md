# Package Runtime Gap Audit

Date: 2026-04-26

This audit keeps package/runtime support ahead of additional Navigator UI work.
The UI may expose many choices, but a choice should become selectable only when
registry status, compiler pruning, runtime dispatch, docs, and generated UI data
all tell the same story.

## Status Rule

Use these statuses consistently:

| Status | Runtime meaning |
|---|---|
| `operational` | The value executes in the default supported composition or has a direct runtime contract. |
| `operational_narrow` | The value executes only with named compatible sub-axes or required `leaf_config`; unsupported cells must be pruned by the compiler/runtime. |
| `registry_only` | The value is vocabulary, compatibility debt, or an extension hook without a built-in runtime path. |
| `not_supported_yet` | The value is reserved for a named future contract. |

An `operational_narrow` value is allowed in broad sweeps, but invalid variants
must surface as skipped/non-executable cells with reasons. It must not silently
fall back to another representation.

## Current Finding

`feature_block_set` had drifted. Runtime and compiler code already used it to
route Layer 2 feature families, but the registry still marked every value
`registry_only`, and explicit recipe selections could be overwritten by the
legacy `feature_builder` bridge in the manifest.

The patched contract is:

- explicit `feature_block_set` values survive bridge derivation in
  `layer2_representation_spec.feature_blocks.feature_block_set`;
- executable family selectors are marked `operational` or
  `operational_narrow`;
- narrow values require compatible sub-block axes and are rejected with a
  concrete reason when the required sub-block is absent;
- legacy `feature_builder` remains accepted as compatibility input and source
  provenance.

## Layer 2 Representation-Family Surface

| `feature_block_set` value | Status | Required companion choices |
|---|---|---|
| `target_lags_only` | `operational` | Target-lag runtime; bridge defaults can derive the fixed target-lag block. |
| `transformed_x` | `operational` | Raw-panel feature runtime. |
| `high_dimensional_x` | `operational` | Raw-panel feature runtime. |
| `factor_blocks_only` | `operational` | Executable factor block, currently static PCA/PLS or equivalent bridge. |
| `factors_plus_target_lags` | `operational` | Executable factor block plus target-lag block. |
| `transformed_x_lags` | `operational_narrow` | `x_lag_feature_block=fixed_x_lags` or legacy fixed X-lag creation. |
| `selected_sparse_x` | `operational_narrow` | Operational `feature_selection_policy`. |
| `level_augmented_x` | `operational_narrow` | Non-none executable `level_feature_block`. |
| `rotation_augmented_x` | `operational_narrow` | Non-none executable `rotation_feature_block`. |
| `mixed_blocks` | `operational_narrow` | At least two active block sources. |
| `custom_blocks` | `operational_narrow` | Registered custom block or custom combiner contract. |
| `legacy_feature_builder_bridge` | `registry_only` | Compatibility provenance only. |

## Next Runtime Queue

1. Audit remaining `operational_narrow` axes where direct registry status
   depends on `leaf_config` or companion axes.
2. Define/version the `prediction_row_schema_v1` projection if payload
   families continue expanding.
3. Make the Layer 1 official frame handoff explicit before deeper
   vintage/release-lag or mixed-source work.
4. Keep generated Navigator UI data checked after each registry/status change.
5. Defer deeper browser affordances until the package/runtime surface has no
   known registry-versus-runtime drift.
