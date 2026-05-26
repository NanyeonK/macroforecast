# Layer 2: Preprocessing

- Parent: [Architecture](index.md)
- Previous: [Layer 1: Data Source, Target y, Predictor x](layer1.md)
- Current: Layer 2
- Next: [Layer 3: Feature Engineering](layer3.md)

Layer 2 owns the canonical McCracken-Ng preprocessing pipeline plus
mixed-frequency alignment for combined panels. Five sub-layers run in
order: mixed-frequency alignment → transform → outlier handling →
imputation → frame edge. Feature-engineering choices (lags / factors /
rotations / selection) live in **Layer 3**, not Layer 2.

## Decision order (5 sub-layers, 11 axes)

| Sub-layer | Axes |
|---|---|
| L2.A — Mixed frequency alignment | `mixed_frequency_representation`, `sd_series_frequency_filter`, `quarterly_to_monthly_policy`, `monthly_to_quarterly_policy` |
| L2.B — Transform | `transform_policy`, `sd_tcode_policy` |
| L2.C — Outlier handling | `outlier_policy`, `outlier_action` |
| L2.D — Imputation | `imputation_policy`, `imputation_temporal_rule` |
| L2.E — Frame edge | `frame_edge_policy` |

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
[`sd_tcode_policy`](../../reference/encyclopedia/l2/axes/sd_tcode_policy.md).

## Custom extension point

L2 supports a user-supplied callable in two positions:

- `leaf_config.custom_preprocessor` — runs **before** the
  canonical `transform → outlier → impute → frame_edge` pipeline.
- `leaf_config.custom_postprocessor` — runs **after** the
  canonical pipeline; output replaces the L2 clean panel.

The simple-API method is `Experiment.use_preprocessor("name",
applied_at="l2"|"l3")`. See
[`docs/how_to/use_extension_points.md`](../../how_to/use_extension_points.md)
for the callable signature + I/O contract.

## Layer contract

Input:
- Layer 1 source frame, target y, and candidate predictor x contract.

Output:
- `l2_clean_panel_v1`;
- feature matrices and representation metadata consumed by Layer 3;
- auxiliary payloads for narrow advanced routes.

## Related reference

## See encyclopedia

For the full per-axis × per-option catalogue (every value with its OptionDoc summary, when-to-use / when-NOT, references), see [`encyclopedia/l2/`](../../reference/encyclopedia/l2/index.md).

## Cycle 50 update (2026-05-22)

Two new operational axis options:
- `quarterly_to_monthly_policy: chow_lin` -- Chow & Lin (1971) regression-based temporal disaggregation using a monthly indicator series (`leaf_config.chow_lin_indicator`).
- `outlier_action: keep_with_indicator` -- preserves the original outlier value and appends a `{col}__outlier_flag` binary column (1=flagged, 0=clean).
