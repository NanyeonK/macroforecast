# `mean` -- Replace missing cells with the per-series rolling mean.

[Back to `imputation_policy` axis](../axes/imputation_policy.md) | [Back to L2](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `imputation_policy`, sub-layer `l2_d`, layer `l2`.
> Standalone callable: `mf.functions.mean_impute_clean`.

## Function signature

```python
mf.functions.mean_impute_clean(
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

Simple, fast, deterministic. No iteration. Useful when the missing pattern is sparse.

**When to use**

Sparse missingness; quick smoke tests.

## In recipe context

Set ``params.imputation_policy = "mean"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L2 recipe fragment
params:
  imputation_policy: mean
```

## References

* macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'

## Related ops

See also: `em_factor`, `forward_fill` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
