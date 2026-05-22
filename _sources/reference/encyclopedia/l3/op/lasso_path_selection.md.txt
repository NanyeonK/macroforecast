# `lasso_path_selection` -- Feature selection along the Lasso regularisation path (Efron et al. 2004).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.lasso_path_selection`.

## Function signature

```python
mf.functions.lasso_path_selection(
    panel: pd.DataFrame,
    target: pd.Series,
    n_features_to_select: int | float,
    normalize_features: bool,
    random_state: int,
    temporal_rule: str,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_features_to_select` | `int | float` | `0.5` | int >= 1 or float in (0, 1] | Target number of features. A float in (0, 1] is treated as a fraction of total columns; an integer is used directly. The path is traced until this count is first reached. |
| `normalize_features` | `bool` | `True` | — | If True, standardise each column to zero mean and unit variance before computing the Lasso path so coefficients are comparable. |
| `random_state` | `int` | `0` | — | Random seed for any stochastic components of the path solver. |
| `temporal_rule` | `str` | `'"expanding_window_per_origin"'` | "expanding_window_per_origin" | "rolling_window_per_origin" | Controls when the Lasso path is recomputed per forecast origin. ``full_sample_once`` is hard-rejected. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Traces the full Lasso regularisation path from lambda_max down to a lambda that retains approximately ``n_features_to_select`` columns, then returns the sub-panel of surviving predictors. Algorithm:

1. Optionally standardise each column to unit variance (``normalize_features=True``, default).
2. Compute the full Lasso path via LARS or coordinate descent.
3. Identify the regularisation value where the number of non-zero coefficients first reaches ``n_features_to_select``.
4. Return the columns with non-zero coefficients at that lambda.

Unlike ``feature_selection`` with ``method='lasso'``, this op traverses the entire path so the selection threshold adapts to the data geometry rather than a fixed penalty. ``temporal_rule`` controls refitting per forecast origin; ``full_sample_once`` is rejected.

**When to use**

Compact, theory-grounded feature selection for linear macro forecasting models where a path-based threshold is preferable to a manually tuned penalty.

**When NOT to use**

When the number of desired features is unknown and cross-validation is required -- use recursive_feature_elimination with use_cv=True instead.

## In recipe context

Set ``params.op = "lasso_path_selection"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: lasso_path_selection
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. (2004) 'Least Angle Regression', Annals of Statistics 32(2): 407-499. <https://doi.org/10.1214/009053604000000067>

## Related ops

See also: `feature_selection`, `recursive_feature_elimination`, `stability_selection` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
