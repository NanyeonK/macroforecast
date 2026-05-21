# Layer 2: Preprocessing

- Parent: [Architecture](../index.md)
- Previous: [Layer 1: Data Source, Target y, Predictor x](../layer1/index.md)
- Current: Layer 2
- Next: [Layer 3: Feature Engineering](../layer3/index.md)

Layer 2 owns the canonical McCracken-Ng preprocessing pipeline plus
mixed-frequency alignment for combined panels. Five sub-layers run in
order: mixed-frequency alignment → transform → outlier handling →
imputation → frame edge. Feature-engineering choices (lags / factors /
rotations / selection) live in **Layer 3**, not Layer 2.

## Decision order (5 sub-layers, 15 axes)

| Sub-layer | Axes |
|---|---|
| L2.A — Mixed frequency alignment | `mixed_frequency_representation`, `sd_series_frequency_filter`, `quarterly_to_monthly_rule`, `monthly_to_quarterly_rule` |
| L2.B — Transform | `transform_policy`, `transform_scope`, `sd_tcode_policy` |
| L2.C — Outlier handling | `outlier_policy`, `outlier_action`, `outlier_scope` |
| L2.D — Imputation | `imputation_policy`, `imputation_temporal_rule`, `imputation_scope` |
| L2.E — Frame edge | `frame_edge_policy`, `frame_edge_scope` |

`mixed_frequency_representation` (v0.8.5+) is general — applies to any
mixed-frequency panel, not just FRED-SD. `sd_tcode_policy` (v0.8.5+) is
orthogonal to `transform_policy`: the official `transform_policy`
controls FRED's published t-codes, while `sd_tcode_policy` selects the
SD-specific inferred / empirical policy when working with FRED-SD.

For per-axis options + when-to-use guidance see the
[L2 encyclopedia page](../../encyclopedia/l2/index.md) and per-axis
pages such as
[`mixed_frequency_representation`](../../encyclopedia/l2/axes/mixed_frequency_representation.md)
and
[`sd_tcode_policy`](../../encyclopedia/l2/axes/sd_tcode_policy.md).

## Custom hook

L2 supports a user-supplied callable in two positions:

- `leaf_config.custom_preprocessor` — runs **before** the
  canonical `transform → outlier → impute → frame_edge` pipeline.
- `leaf_config.custom_postprocessor` — runs **after** the
  canonical pipeline; output replaces the L2 clean panel.

The simple-API method is `Experiment.use_preprocessor("name",
applied_at="l2"|"l3")`. See
[`docs/for_recipe_authors/custom_hooks.md`](../../for_recipe_authors/custom_hooks.md)
for the callable signature + I/O contract.

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
