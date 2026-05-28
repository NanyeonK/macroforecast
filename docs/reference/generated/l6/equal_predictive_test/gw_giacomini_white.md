# `gw_giacomini_white` -- Giacomini-White (2006) conditional equal-predictive-ability test.

[Back to `equal_predictive_test` axis](../axes/equal_predictive_test.md) | [Back to L6](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `equal_predictive_test`, sub-layer `L6_A_equal_predictive`, layer `l6`.
> Standalone callable: `mf.functions.gw_test`.

## Function signature

```python
mf.functions.gw_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
) -> GWTestResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `loss_a` | `np.ndarray` | — | — | Per-period losses for model A (e.g. squared errors). |
| `loss_b` | `np.ndarray` | — | — | Per-period losses for model B. |

## Returns

`GWTestResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stat` | `float or None` | GW test statistic |
| `pvalue` | `float or None` | Two-sided p-value |
| `decision` | `bool` | Reject H0 at 5% |
| `n_obs` | `int` | Observations used |
| `hln_correction` | `bool` | HLN correction applied |

## Behavior

Generalises DM to test conditional predictive ability given a vector of predictors. Robust to non-stationary performance differentials and works with rolling / expanding-window forecasts.

**When to use**

Conditional / regime-dependent forecast comparisons.

## In recipe context

Set ``params.equal_predictive_test = "gw_giacomini_white"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L6 recipe fragment
params:
  equal_predictive_test: gw_giacomini_white
```

## References

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Giacomini & White (2006) 'Tests of Conditional Predictive Ability', Econometrica 74(6): 1545-1578.

## Related ops

See also: `dm_diebold_mariano`, `multi` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
