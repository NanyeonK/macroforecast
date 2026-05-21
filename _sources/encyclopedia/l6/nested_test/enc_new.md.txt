# `enc_new` -- Enc-New forecast encompassing test (Clark-McCracken 2001).

[Back to `nested_test` axis](../axes/nested_test.md) | [Back to L6](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `nested_test`, sub-layer `L6_B_nested`, layer `l6`.
> Standalone callable: `mf.functions.enc_new_test`.

## Function signature

```python
mf.functions.enc_new_test(
    loss_small: np.ndarray,
    loss_large: np.ndarray,
) -> EncNewTestResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `loss_small` | `np.ndarray` | — | — | Squared losses for the small model. |
| `loss_large` | `np.ndarray` | — | — | Squared losses for the large model. |

## Returns

`EncNewTestResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stat` | `float or None` | Enc-New statistic |
| `pvalue` | `float or None` | One-sided p-value |
| `decision` | `bool` | Reject H0 at 5% |
| `n_obs` | `int` | Observations used |

## Behavior

Tests whether the large model's forecast contains information beyond the small (nested) model. Uses raw loss improvement ``f_t = loss_small - loss_large`` without CW adjustment, then applies one-sided DM inference. Complementary to the Clark-West test when the user does not want the CW penalty.

**When to use**

Testing forecast encompassing in nested model settings without the CW adjustment term.

**When NOT to use**

When the CW adjustment for bias is desired -- use clark_west instead.

## In recipe context

Set ``params.nested_test = "enc_new"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L6 recipe fragment
params:
  nested_test: enc_new
```

## References

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Clark & McCracken (2001) 'Tests of Equal Forecast Accuracy and Encompassing for Nested Models', JoE 105(2): 1-28.

## Related ops

See also: `clark_west`, `enc_t`, `multi` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
