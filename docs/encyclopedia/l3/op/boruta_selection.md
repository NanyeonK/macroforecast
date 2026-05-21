# `boruta_selection` -- All-relevant feature selection via shadow-feature random forest (Kursa-Rudnicki 2010).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.boruta_selection`.

## Function signature

```python
mf.functions.boruta_selection(
    panel: pd.DataFrame,
    target: pd.Series,
    n_estimators_rf: int,
    max_iter: int,
    alpha: float,
    include_tentative: bool,
    random_state: int,
    temporal_rule: str,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_estimators_rf` | `int` | `100` | >= 1 | Number of trees in each random forest fit. Larger values stabilise importance rankings at the cost of compute. |
| `max_iter` | `int` | `100` | >= 1 | Maximum number of Boruta iterations. Each iteration removes at least one rejected feature; convergence typically occurs well before the limit. |
| `alpha` | `float` | `0.05` | in (0, 1) | Two-sided binomial test significance level for classifying features as confirmed or rejected. Lower values are more conservative. |
| `include_tentative` | `bool` | `False` | — | If True, tentative features (not yet confirmed or rejected within max_iter rounds) are retained in the output panel. |
| `random_state` | `int` | `0` | — | Random seed for the random forest and shadow-feature shuffles. |
| `temporal_rule` | `str` | `'"expanding_window_per_origin"'` | "expanding_window_per_origin" | "rolling_window_per_origin" | Controls when the random forest is refitted relative to each forecast origin. ``full_sample_once`` is hard-rejected. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Boruta identifies all features that carry statistically relevant predictive information by comparing each feature's importance to the maximum importance achieved by random (shadow) copies. Algorithm:

1. Append shuffled shadow copies of every column to the panel.
2. Fit a random forest (``n_estimators_rf`` trees) and record mean impurity-reduction importance for each real and shadow feature.
3. Use a two-sided binomial test (threshold ``alpha``) to classify each real feature as confirmed, rejected, or tentative.
4. Remove rejected features and their shadows; repeat up to ``max_iter`` rounds or until no tentative features remain.
5. Return the sub-panel of confirmed (and optionally tentative, ``include_tentative=True``) features.

Because the null is the importance of random noise, Boruta is an all-relevant selector: it keeps every feature that beats chance, not just the minimal predictive set. ``temporal_rule`` controls whether the forest is fit once per origin (``expanding_window_per_origin``). ``full_sample_once`` is rejected by a hard rule.

**When to use**

Macro panels where many predictors may matter but standard Lasso-type selectors over-shrink; best before tree or neural forecasters where high-dim panels are acceptable.

**When NOT to use**

Very wide panels (K >> 500) -- shadow copies double memory; prefer lasso_path_selection or stability_selection for computational cost.

## In recipe context

Set ``params.op = "boruta_selection"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: boruta_selection
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Kursa, M.B. & Rudnicki, W.R. (2010) 'Feature Selection with the Boruta Package', Journal of Statistical Software 36(11): 1-13. <https://doi.org/10.18637/jss.v036.i11>

## Related ops

See also: `feature_selection`, `stability_selection`, `recursive_feature_elimination` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
