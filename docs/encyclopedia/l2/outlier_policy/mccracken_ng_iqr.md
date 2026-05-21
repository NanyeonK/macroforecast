# `mccracken_ng_iqr` -- McCracken-Ng's published IQR-multiple outlier rule.

[Back to `outlier_policy` axis](../axes/outlier_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `outlier_policy`, sub-layer `l2_c`, layer `l2`.
> Standalone callable: `mf.functions.iqr_outlier_clean`.

## Function signature

```python
mf.functions.iqr_outlier_clean(
    panel: pd.DataFrame,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Flags any observation more than ``leaf_config.outlier_iqr_threshold`` (default 10) IQRs from the per-series median. The 10×IQR threshold is the published McCracken-Ng default and matches their replication scripts.

Pairs with an L2.C ``outlier_action`` to specify what happens to flagged observations (replace with NaN by default, then L2.D imputation fills them).

**When to use**

Default for FRED-based studies. Canonical replication path.

## In recipe context

Set ``params.outlier_policy = "mccracken_ng_iqr"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  outlier_policy: mccracken_ng_iqr
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4). (doi:10.1080/07350015.2015.1086655)

## Related ops

See also: `winsorize`, `zscore_threshold`, `none`, `outlier_action` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
