# `recursive_feature_elimination` -- Backward stepwise feature pruning via estimator importance (Guyon et al. 2002).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.recursive_feature_elimination`.

## Function signature

```python
mf.functions.recursive_feature_elimination(
    panel: pd.DataFrame,
    target: pd.Series,
    n_features_to_select: int | float,
    step: int | float,
    estimator: str,
    use_cv: bool,
    cv_folds: int,
    random_state: int,
    temporal_rule: str,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_features_to_select` | `int | float` | `0.5` | int >= 1 or float in (0, 1] | Number of features to retain. A float in (0, 1] is treated as a fraction of total columns; an integer is used directly. |
| `step` | `int | float` | `1` | int >= 1 or float in (0, 1) | Features removed per iteration. An integer removes that many; a float removes that fraction of the remaining features. |
| `estimator` | `str` | `'"ridge"'` | "ridge" | "lasso" | "svr_linear" | Base estimator whose coefficient magnitudes rank feature importance. ``svr_linear`` uses the SVM weight vector. |
| `use_cv` | `bool` | `False` | — | If True, wrap RFE in cross-validated RFECV; ``n_features_to_select`` is ignored and the optimal count is determined by CV. |
| `cv_folds` | `int` | `5` | >= 2 | Number of time-series cross-validation folds when use_cv=True. |
| `random_state` | `int` | `0` | — | Random seed for estimators that require it (e.g. SVR with RBF). |
| `temporal_rule` | `str` | `'"expanding_window_per_origin"'` | "expanding_window_per_origin" | "rolling_window_per_origin" | Controls when the base estimator is refitted per forecast origin. ``full_sample_once`` is hard-rejected. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Recursively eliminates the weakest features according to coefficients or feature importances of a base estimator until the target count is reached. Algorithm:

1. Fit the base estimator (``ridge``, ``lasso``, or ``svr_linear``) on the full feature set aligned to the target.
2. Rank features by absolute coefficient magnitude (linear models) or impurity reduction (trees).
3. Remove the bottom ``step`` features (an int) or fraction (a float).
4. Repeat until exactly ``n_features_to_select`` remain.
5. If ``use_cv=True``, wrap in cross-validated RFECV with ``cv_folds`` time-series folds to auto-select the optimal count.

``temporal_rule`` governs refitting per forecast origin; ``full_sample_once`` is rejected by a hard rule.

**When to use**

Trimming macro panels to a compact predictor set before linear or penalised forecasters; especially useful when coefficient-based ranking aligns with the forecasting objective.

**When NOT to use**

When the estimator's coefficient ranking is a poor proxy for marginal predictive value (e.g. highly correlated groups); prefer stability_selection or boruta_selection.

## In recipe context

Set ``params.op = "recursive_feature_elimination"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: recursive_feature_elimination
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Guyon, I., Weston, J., Barnhill, S. & Vapnik, V. (2002) 'Gene Selection for Cancer Classification using Support Vector Machines', Machine Learning 46(1-3): 389-422. <https://doi.org/10.1023/A:1012487302797>

## Related ops

See also: `feature_selection`, `boruta_selection`, `lasso_path_selection` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
