# Standalone functions — L2 preprocessing

L2 defines the cleaning contract for a raw panel before feature engineering.
In the recipe DSL these are fixed-axis choices on `2_preprocessing`. In the
standalone paradigm they are exposed as individual callables.

> **Cycle 22 note** — L2 standalone callables are planned for a future cycle.
> This page documents the 13 primary L2 axes so you can understand the
> cleaning surface. The encyclopedia links below point to the full per-axis
> reference.

## L2 axes

| Axis | One-liner | Encyclopedia |
|---|---|---|
| `transform_policy` | How to apply official FRED tcodes (none / official / custom) | [transform_policy](../encyclopedia/l2/axes/transform_policy.md) |
| `outlier_policy` | Detection method: IQR / winsorize / z-score / none | [outlier_policy](../encyclopedia/l2/axes/outlier_policy.md) |
| `imputation_policy` | Fill-in strategy: EM-factor / multivariate / mean / ffill / interpolate / none | [imputation_policy](../encyclopedia/l2/axes/imputation_policy.md) |
| `frame_edge_policy` | How to handle ragged / unbalanced leading edges | [frame_edge_policy](../encyclopedia/l2/axes/frame_edge_policy.md) |
| `transform_scope` | Which series to transform: both / predictors-only / target-only / N/A | [transform_scope](../encyclopedia/l2/axes/transform_scope.md) |
| `outlier_scope` | Which series to outlier-screen | [outlier_scope](../encyclopedia/l2/axes/outlier_scope.md) |
| `outlier_action` | What to do with detected outliers: flag / replace-median / cap | [outlier_action](../encyclopedia/l2/axes/outlier_action.md) |
| `imputation_scope` | Which series to impute | [imputation_scope](../encyclopedia/l2/axes/imputation_scope.md) |
| `imputation_temporal_rule` | Re-estimate imputation per-origin or block-recompute | [imputation_temporal_rule](../encyclopedia/l2/axes/imputation_temporal_rule.md) |
| `mixed_frequency_representation` | How to represent mixed-frequency data in a common frame | [mixed_frequency_representation](../encyclopedia/l2/axes/mixed_frequency_representation.md) |
| `monthly_to_quarterly_rule` | Aggregation rule when downsampling monthly to quarterly | [monthly_to_quarterly_rule](../encyclopedia/l2/axes/monthly_to_quarterly_rule.md) |
| `quarterly_to_monthly_rule` | Interpolation rule when upsampling quarterly to monthly | [quarterly_to_monthly_rule](../encyclopedia/l2/axes/quarterly_to_monthly_rule.md) |
| `sd_tcode_policy` | FRED-SD tcode application: none / inferred / empirical | [sd_tcode_policy](../encyclopedia/l2/axes/sd_tcode_policy.md) |

## In recipe context

```yaml
2_preprocessing:
  fixed_axes:
    transform_policy: apply_official_tcode
    outlier_policy: mccracken_ng_iqr
    imputation_policy: em_factor
    frame_edge_policy: keep_unbalanced
```

## Related

- [L3 feature engineering](l3_transforms.md) — the cleaned panel becomes
  the source panel for L3 DAG nodes.
- [Encyclopedia L2 index](../encyclopedia/l2/index.md) — full axis × option
  reference.
