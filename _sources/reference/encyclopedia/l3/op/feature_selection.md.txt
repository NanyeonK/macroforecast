# `feature_selection` -- Filter columns by variance / correlation / lasso pre-screen.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.feature_selection_transform`.

## Function signature

```python
mf.functions.feature_selection_transform(
    panel: pd.DataFrame,
    target: pd.Series | None,
    n_features: int | float,
    method: str,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series | None` | `None` | — | Optional supervisory signal. Required when method is 'correlation' or 'lasso'; ignored for method='variance'. |
| `n_features` | `int | float` | `0.5` | int >= 1 or float in (0, 1] | Number of features to keep. If a float in (0, 1], treated as a fraction of total columns. If an integer, used as a direct count clamped to [1, K]. |
| `method` | `str` | `'"variance"'` | "variance" | "correlation" | "lasso" | Selection criterion. 'variance' keeps highest-variance columns (no target needed). 'correlation' keeps columns most correlated with target. 'lasso' fits LassoCV and keeps largest-coefficient columns. 'correlation' and 'lasso' require target. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Drops columns failing one of three criteria configured via ``params.method``:

* ``variance`` -- drop columns with variance below ``params.threshold``.
* ``correlation`` -- drop columns with pairwise correlation above ``params.threshold`` (keeps the first).
* ``lasso`` -- fit a Lasso pre-screen and keep columns with non-zero coefficients.

**When to use**

Trimming the panel before expensive downstream estimators (NN, SVM, kernel) when high-dim noise dominates.

**When NOT to use**

Tree models -- they handle irrelevant features natively.

## In recipe context

Set ``params.op = "feature_selection"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: feature_selection
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'

## Related ops

See also: `scale`, `pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
