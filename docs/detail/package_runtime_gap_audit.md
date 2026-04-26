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

The Layer 1 official-frame handoff is now explicit. Runtime writes
`layer1_official_frame.json` for every run, records
`layer1_official_frame_contract=layer1_official_frame_v1` in `manifest.json`,
and registers the file in `artifact_manifest.json` as a Layer 1 artifact. This
closes the previous implicit boundary between raw/source handling and Layer 2
representation builders.

## Previous Finding

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

## Remaining Operational-Narrow Audit

The remaining source-level `operational_narrow` surfaces are now explicit and
exported through `navigator_ui_data_v1.operational_narrow_contracts`, so the
static Navigator, docs, and compiler audit use the same contract list.

| Axis / contract | Narrow values | Required companions | Enforcement surface |
|---|---|---|---|
| `feature_block_set` / `feature_block_set_public_axis_v1` | `transformed_x_lags`, `selected_sparse_x`, `level_augmented_x`, `rotation_augmented_x`, `mixed_blocks`, `custom_blocks` | Matching Layer 2 sub-blocks: fixed X lags, non-none selection/level/rotation block, at least two active block sources, or a registered custom block/combiner. | Compiler blocked reasons, sweep skipped-cell manifests, Navigator status catalog. |
| `exogenous_x_path_policy` / `exogenous_x_path_contract_v1` | `hold_last_observed`, `observed_future_x`, `scheduled_known_future_x`, `recursive_x_model` | `forecast_type=iterated`, raw-panel feature runtime, `target_lag_block=fixed_target_lags`; scheduled paths require `scheduled_known_future_x_columns`; recursive paths require `recursive_x_model_family=ar1`. | Layer 3 capability matrix, compiler blocked reasons, Navigator virtual-axis status and compatibility reasons. |
| `recursive_x_model_family` / `exogenous_x_path_contract_v1` | `ar1` | `exogenous_x_path_policy=recursive_x_model` and the raw-panel iterated point-forecast slice. | Compiler blocked reasons and Navigator compatibility reasons. |

Ledger-only narrow contracts such as factor-score rotation, sequence handoff,
interval/density wrappers, and raw-panel iterated payloads are documented in
`docs/detail/layer_contract_ledger.md`; they do not create additional registry
values that the Navigator can freely enable without companion choices.

## Next Runtime Queue

1. Keep generated Navigator UI data checked after each registry/status change.
2. Defer deeper browser affordances until the package/runtime surface has no
   known registry-versus-runtime drift.
3. Use `layer1_official_frame_v1` as the extension point before deeper
   vintage/release-lag or mixed-source runtime widening.
