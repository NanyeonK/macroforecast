# `apply_official_tcode` -- Apply McCracken-Ng's series-by-series stationarity transforms.

[Back to `transform_policy` axis](../axes/transform_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `transform_policy`, sub-layer `l2_b`, layer `l2`.
> Standalone callable: `mf.functions.apply_tcode_transform`.

## Function signature

```python
mf.functions.apply_tcode_transform(
    panel: pd.DataFrame,
    tcode_map: dict[str, int],
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `tcode_map` | `dict[str, int]` | — | — | Mapping from column name to McCracken-Ng t-code integer 1..7. Columns not in tcode_map are passed through unchanged. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Each FRED-MD/QD series ships with a transformation code (1-7) mapping to a stationarity transform. ``apply_official_tcode`` runs the canonical mapping per series:

* 1 = level
* 2 = first difference
* 3 = second difference
* 4 = log
* 5 = first difference of log (≈ growth rate)
* 6 = second difference of log
* 7 = log diff of (1 + growth rate)

Applied per-origin within walk-forward to avoid look-ahead.

**When to use**

Default for FRED-based studies. Canonical replication path.

## In recipe context

Set ``params.transform_policy = "apply_official_tcode"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  transform_policy: apply_official_tcode
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'
* McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4). (doi:10.1080/07350015.2015.1086655)

## Related ops

See also: `no_transform`, `custom_tcode`, `transform_scope` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
