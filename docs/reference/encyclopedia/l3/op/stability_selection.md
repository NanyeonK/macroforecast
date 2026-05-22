# `stability_selection` -- Feature selection by subsampling stability -- selection probability threshold (Meinshausen-Bühlmann 2010).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.stability_selection`.

## Function signature

```python
mf.functions.stability_selection(
    panel: pd.DataFrame,
    target: pd.Series,
    n_subsamples: int,
    subsample_fraction: float,
    pi_thr: float,
    base_estimator: str,
    alpha: float,
    random_state: int,
    temporal_rule: str,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_subsamples` | `int` | `100` | >= 10 | Number of subsampling rounds. More rounds give stable frequency estimates at higher compute cost. |
| `subsample_fraction` | `float` | `0.5` | in (0, 1) | Fraction of observations in each subsample. Meinshausen-Bühlmann recommend 0.5 for the theoretical FWER bound to apply. |
| `pi_thr` | `float` | `0.6` | in (0.5, 1] | Selection-probability threshold. Features with empirical frequency above this value are retained. Values near 0.9 are very conservative. |
| `base_estimator` | `str` | `'"lasso"'` | "lasso" | "elastic_net" | Sparse base estimator applied to each subsample. ``elastic_net`` can improve stability when predictors are highly correlated. |
| `alpha` | `float` | `0.01` | > 0 | Regularisation strength passed to the base estimator on each subsample. Controls sparsity per subsample draw. |
| `random_state` | `int` | `0` | — | Random seed for reproducible subsampling. |
| `temporal_rule` | `str` | `'"expanding_window_per_origin"'` | "expanding_window_per_origin" | "rolling_window_per_origin" | Controls when subsampling is rerun per forecast origin. ``full_sample_once`` is hard-rejected. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Estimates the probability that each feature would be selected by a sparse estimator on a random subsample, then retains features whose selection probability exceeds ``pi_thr``. Algorithm:

1. Draw ``n_subsamples`` subsamples of size ``subsample_fraction * T`` from the aligned (panel, target) observations.
2. On each subsample, fit the base estimator (``lasso`` or ``elastic_net``) with regularisation ``alpha`` and record which features receive non-zero coefficients.
3. Compute the empirical selection frequency for each feature across all subsamples.
4. Return the sub-panel of features with frequency >= ``pi_thr``.

Stability selection provides FWER control under mild assumptions on the regularisation path (Meinshausen-Bühlmann Theorem 1). ``temporal_rule`` governs refitting per origin; ``full_sample_once`` is rejected.

**When to use**

Macro panels where robustness of selection across data perturbations is more important than computational speed; pairs well with L4 ridge or elastic_net forecasters.

**When NOT to use**

Short time series (T < 100) where subsampling leaves too few observations per draw; prefer lasso_path_selection for small T.

## In recipe context

Set ``params.op = "stability_selection"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: stability_selection
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Meinshausen, N. & Bühlmann, P. (2010) 'Stability selection', Journal of the Royal Statistical Society Series B 72(4): 417-473. <https://doi.org/10.1111/j.1467-9868.2010.00740.x>

## Related ops

See also: `feature_selection`, `boruta_selection`, `lasso_path_selection` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
