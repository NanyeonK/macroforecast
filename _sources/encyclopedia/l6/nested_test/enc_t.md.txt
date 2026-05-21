# `enc_t` -- Enc-T forecast encompassing test (Ericsson 1992 t-form).

[Back to `nested_test` axis](../axes/nested_test.md) | [Back to L6](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `nested_test`, sub-layer `L6_B_nested`, layer `l6`.
> Standalone callable: `mf.functions.enc_t_test`.

## Function signature

```python
mf.functions.enc_t_test(
    loss_small: np.ndarray,
    loss_large: np.ndarray,
) -> EncTTestResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `loss_small` | `np.ndarray` | — | — | Squared losses for the small model. |
| `loss_large` | `np.ndarray` | — | — | Squared losses for the large model. |

## Returns

`EncTTestResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `stat` | `float or None` | Enc-T statistic |
| `pvalue` | `float or None` | One-sided p-value |
| `decision` | `bool` | Reject H0 at 5% |
| `n_obs` | `int` | Observations used |

## Behavior

Ericsson (1992) t-form of the encompassing test. Identical computation to enc_new in the current implementation (raw loss improvement, one-sided DM inference, no CW adjustment). The distinction is the conceptual labelling: enc_t is cast as a t-statistic on the mean loss improvement. Both enc_new and enc_t share the same runtime dispatch branch.

**When to use**

Encompassing tests in contexts where the Ericsson t-form labelling is preferred.

**When NOT to use**

When CW adjustment is needed -- use clark_west instead.

## In recipe context

Set ``params.nested_test = "enc_t"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L6 recipe fragment
params:
  nested_test: enc_t
```

## References

* macroforecast design Part 3, L6: 'tests must report (statistic, p-value, kernel, lag) and respect HAC dependence-correction.'
* Ericsson (1992) 'Parameter Constancy, Mean Square Forecast Errors, and Measuring Forecast Performance', JoE 52(1-2): 113-153.

## Related ops

See also: `clark_west`, `enc_new`, `multi` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
