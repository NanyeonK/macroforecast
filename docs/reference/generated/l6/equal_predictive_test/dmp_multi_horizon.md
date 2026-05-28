# `dmp_multi_horizon` -- Diebold-Mariano-Pesaran joint multi-horizon test.

[Back to `equal_predictive_test` axis](../axes/equal_predictive_test.md) | [Back to L6](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `equal_predictive_test`, sub-layer `L6_A_equal_predictive`, layer `l6`.
> Standalone callable: `mf.functions.dmp_test`.

## Function signature

```python
mf.functions.dmp_test(
    loss_differentials: list[np.ndarray] or np.ndarray,
) -> DMPTestResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `loss_differentials` | `list[np.ndarray] or np.ndarray` | — | — | Per-period loss differentials, one array per horizon or pre-stacked. |

## Returns

`DMPTestResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stat` | `float or None` | DMP test statistic |
| `pvalue` | `float or None` | Two-sided p-value |
| `decision` | `bool` | Reject H0 at 5% |
| `n_obs_stacked` | `int` | Stacked observations |

## Behavior

HAC-adjusted stacked DM test that evaluates equality of predictive ability across all forecast horizons simultaneously. v0.3 implementation following Pesaran-Timmermann.

**When to use**

Joint significance across multiple horizons (avoids per-horizon p-value adjustment).

## In recipe context

Set ``params.equal_predictive_test = "dmp_multi_horizon"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L6 recipe fragment
params:
  equal_predictive_test: dmp_multi_horizon
```

## References

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Pesaran & Timmermann (2007) 'Selection of estimation window in the presence of breaks', JoE 137(1): 134-161.

## Related ops

See also: `dm_diebold_mariano` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
