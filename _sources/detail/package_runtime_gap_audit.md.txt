# Package Runtime Gap Audit

Date: 2026-04-27

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
| `external_plugin` | The value is executable only after the user registers or supplies an external callable/plugin. |
| `registry_only` | The value is vocabulary, compatibility debt, or an extension hook without a built-in runtime path. |
| `future` | The value is named for a planned design but should stay disabled for execution. |
| `not_supported_yet` | The value is reserved for a named future contract. |

An `operational_narrow` value is allowed in broad sweeps, but invalid variants
must surface as skipped/non-executable cells with reasons. It must not silently
fall back to another representation.

## Current Finding After PR #92

The package/runtime surface is internally consistent at the top-level contract
level. The remaining work is no longer "make the named choices real in bulk";
it is now a layer-by-layer audit of which named choices should be:

- primary Navigator choices;
- hidden/lowered compatibility axes;
- executable narrow slices;
- custom extension templates;
- future vocabulary that must stay disabled.

The live registry currently has 147 axes across Layers 0-7. The companion
[Layer Axis Census](layer_axis_census.md) lists every axis and every value by
current status. That census is the starting point for the next manual audit.

Current important deltas since the previous package/runtime audit:

- FRED-SD source handling is stable at Layer 1: current/vintage loading,
  source availability, frequency report/policy, state/variable selectors,
  grouping recipes, and t-code policy choices are executable and documented.
- FRED-SD advanced mixed-frequency representation is `operational_narrow` at
  Layer 2 through `fred_sd_native_frequency_block_payload_v1` and
  `fred_sd_mixed_frequency_model_adapter_v1`.
- Built-in Layer 3 FRED-SD MIDAS routes are `operational_narrow`:
  `model_family=midas_almon`, `model_family=midasr`, and compatibility alias
  `model_family=midasr_nealmon`.
- `midasr_weight_family` now has five operational-narrow values:
  `nealmon`, `almonp`, `nbeta`, `genexp`, and `harstep`.
- The custom Layer 3 model surface is now documented and test-covered through
  `custom_model_contract_metadata()`, including FRED-SD payload conditions.

The main open package gap after PR #92 is extension ergonomics, not a missing
FRED-SD runtime path: Layer 3 has a custom FRED-SD mixed-frequency template,
but Layer 2 custom feature-block / combiner templates and examples are still
thin relative to the runtime contracts.

## Previous Finding: Layer 1 Handoff

The Layer 1 official-frame handoff is explicit. Runtime writes
`layer1_official_frame.json` for every run, records
`layer1_official_frame_contract=layer1_official_frame_v1` in `manifest.json`,
and registers the file in `artifact_manifest.json` as a Layer 1 artifact. This
closes the previous implicit boundary between raw/source handling and Layer 2
representation builders.

## Previous Finding: Feature-Block Set

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
| `fred_sd_mixed_frequency_representation` / `fred_sd_native_frequency_block_payload_v1` | `native_frequency_block_payload`, `mixed_frequency_model_adapter` | Dataset includes FRED-SD, `feature_builder=raw_feature_panel`, `forecast_type=direct`, and a registered custom model or built-in MIDAS model. The adapter route also emits `fred_sd_mixed_frequency_model_adapter_v1`. | Navigator compatibility, compiler blocked reasons, runtime route guards, artifact sidecars. |
| `exogenous_x_path_policy` / `exogenous_x_path_contract_v1` | `hold_last_observed`, `observed_future_x`, `scheduled_known_future_x`, `recursive_x_model` | `forecast_type=iterated`, raw-panel feature runtime, `target_lag_block=fixed_target_lags`; scheduled paths require `scheduled_known_future_x_columns`; recursive paths require `recursive_x_model_family=ar1`. | Layer 3 capability matrix, compiler blocked reasons, Navigator virtual-axis status and compatibility reasons. |
| `recursive_x_model_family` / `exogenous_x_path_contract_v1` | `ar1` | `exogenous_x_path_policy=recursive_x_model` and the raw-panel iterated point-forecast slice. | Compiler blocked reasons and Navigator compatibility reasons. |

Ledger-only narrow contracts such as factor-score rotation, sequence handoff,
interval/density wrappers, and raw-panel iterated payloads are documented in
`docs/detail/layer_contract_ledger.md`; they do not create additional registry
values that the Navigator can freely enable without companion choices.

## Next Runtime Queue

1. Run the layer-by-layer audit from [Layer Axis Census](layer_axis_census.md).
   Treat registry axes outside the primary Navigator tree as the review queue,
   not as automatic UI bugs.
2. Start with Layer 2 because it has the largest open surface:
   custom feature-block templates, custom combiner templates, target
   transformer examples, and bridge-only axes need explicit keep/hide/promote
   decisions.
3. Then audit Layer 3 training axes outside the primary Navigator tree and the
   virtual future-X route axes. Keep full sequence/tensor and richer
   recursive-X families gated unless a narrow runtime/test contract is opened.
4. Keep generated Navigator UI data checked after each registry/status change.
5. Defer heavier estimator families, such as regularized/group MIDAS or
   state-space nowcasting, until the extension-template surface is clear
   enough for researchers to compare their own methods against built-ins.
