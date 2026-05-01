# Contract Source of Truth

This page defines the canonical ownership rules for macrocast layer contracts.
When code, docs, UI, or legacy compiler paths disagree, use this page to decide
which side must change.

## Decision

macrocast has two canonical contract layers:

1. Public layer shape and artifact boundaries are canonical in the current
   L0-L8 layer contract:
   - YAML keys: `0_meta`, `1_data`, `2_preprocessing`,
     `3_feature_engineering`, `4_forecasting_model`, `5_evaluation`,
     `6_statistical_tests`, `7_interpretation`, `8_output`;
   - diagnostics: `1_5_data_summary`, `2_5_pre_post_preprocessing`,
     `3_5_feature_diagnostics`, `4_5_generator_diagnostics`;
   - DAG/list/diagnostic shape rules and sink names from
     [Recipe Layers](recipe_layers.md) and
     [Layer Boundary Contract](layer_boundary_contract.md).
2. Axis vocabulary is canonical in the machine-readable registry under
   `macrocast/registry/*/AXIS_DEFINITION`:
   - axis IDs;
   - allowed values;
   - support status (`operational`, `registry_only`, `future`, etc.);
   - default policy;
   - compatibility and incompatibility metadata.

The current docs and Navigator must describe and emit only those canonical
public keys and registry axis IDs. Runtime validators should accept the same
contract. If a runtime layer rejects a registry-backed public axis, the runtime
validator is stale unless the axis is explicitly marked schema-only or future
in the registry.

## Compatibility Status

The older compiler/navigation vocabulary is compatibility input, not the public
source of truth. Examples include:

- legacy layer names such as `1_data_task`, `3_training`,
  `4_evaluation`, `5_output_provenance`, `6_stat_tests`, and `7_importance`;
- legacy axis aliases preserved by `macrocast/registry/naming.py`;
- bridge axes kept for old recipes while current L0-L8 contracts migrate.

Compatibility code may read these names and canonicalize them, but generated
docs, generated UI, exported YAML, and new recipes should prefer current L0-L8
keys and registry axis IDs.

## Precedence

When sources disagree, apply this order:

1. `macrocast/registry/*/AXIS_DEFINITION` for axis name, value, support status,
   and default policy.
2. `macrocast/core/layers/registry.py`, [Recipe Layers](recipe_layers.md), and
   [Layer Boundary Contract](layer_boundary_contract.md) for layer topology,
   YAML key, DAG/list/diagnostic shape, sink names, and artifact boundaries.
3. Runtime layer parser/validator modules for executable validation behavior.
   They must be updated when they reject canonical registry-backed public
   contracts.
4. Docs and Navigator UI as generated or manually authored views of the above.
   They must not invent independent option lists.
5. Legacy compiler paths and aliases as compatibility shims only.

## Implementation Rules

- Add or remove an axis by changing the relevant `AXIS_DEFINITION` first.
- Add or remove an artifact boundary by changing the layer registry/boundary
  contract first.
- UI option lists must be generated from, or tested against, the registry.
- Docs value catalogs must be generated from, or tested against, the registry.
- Runtime validators must report whether an axis is executable, schema-only, or
  future; they should not silently reject a public registry axis as unknown.
- Conditional visibility belongs in contract metadata or an exported contract
  manifest, not in one-off UI code.
- YAML export must be parseable and must omit inactive conditional axes.
- Default profiles are versioned contract objects. Changing a default is a
  profile-version change, not an undocumented UI change.

## Current Audit Findings

The current codebase still has migration gaps:

- Layer 1 runtime validation now accepts the registry-backed public axes used
  by docs/UI, including `information_set_type`, `release_lag_rule`,
  `contemporaneous_x_rule`, `raw_missing_policy`,
  `official_transform_policy`, and `missing_availability`.
- Layer 1 uses registry value `multi_target` in docs/UI and public examples.
  Older design notes may say `multi_series_target`; runtime validation accepts
  it as a compatibility alias and normalizes to `multi_target`.
- Layer 2 docs/UI expose canonical representation axes such as
  `horizon_target_construction`, `target_transform`, `x_missing_policy`,
  `feature_block_combination`, and `feature_builder`, while
  `macrocast/core/layers/l2.py` still validates an older cleaning-only axis
  set.
- Layer 5 docs/UI can emit context-inactive axes unless generation prunes them
  against recipe context.

These are implementation gaps. The registry plus current L0-L8 public layer
contract remains the source of truth.
