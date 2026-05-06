# Layer 2: Representation / Research Preprocessing

- Parent: [4. Detail (code): Full](../index.md)
- Previous: [4.1 Layer 1: Data Source, Target y, Predictor x](../layer1/index.md)
- Current: Layer 2
- Next: [Layer 3: Feature Engineering](../layer3/index.md)

Layer 2 owns representation construction after Layer 1 produces the source frame. It supports research preprocessing choices such as t-code handling, target construction, missing/outlier handling after the source frame, scaling, feature blocks, factor blocks, lag blocks, rotations, feature selection, and custom representation hooks.

## Decision order

| Group | Axes |
|---|---|
| FRED-SD mixed frequency | `fred_sd_mixed_frequency_representation` |
| Target construction | `horizon_target_construction`, `target_transform`, `target_normalization` |
| Transform and cleaning | `tcode_policy`, `x_missing_policy`, `x_outlier_policy`, `scaling_policy` |
| Feature blocks | `target_lag_block`, `x_lag_feature_block`, `factor_feature_block`, `level_feature_block`, `temporal_feature_block`, `rotation_feature_block` |
| Composition and selection | `feature_block_combination`, `feature_selection_policy`, `feature_selection_semantics` |
| Handoff | `evaluation_scale`, `feature_builder` compatibility bridge |

## Canonical names

Layer 2 uses researcher-facing names for target construction, predictor construction, block composition, selection, and handoff. Generated recipes and Navigator paths emit the canonical IDs listed in the registry. Retired bridge values are not documented as supported choices.

## Layer contract

Input:
- Layer 1 source frame, target y, and candidate predictor x contract.

Output:
- `layer2_representation_v1`;
- feature matrices and representation metadata consumed by Layer 3;
- auxiliary payloads for narrow advanced routes.

## Related reference

## See encyclopedia

For the full per-axis × per-option catalogue (every value with its OptionDoc summary, when-to-use / when-NOT, references), see [`encyclopedia/l2/`](../../encyclopedia/l2/index.md).
