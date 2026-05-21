# `sliced_inverse_regression` -- sSUFF / Sliced inverse regression (scaled) -- supervised dimension reduction (Huang-Jiang-Li-Tong-Zhou 2022).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.sliced_inverse_regression_transform`.

## Function signature

```python
mf.functions.sliced_inverse_regression_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    n_components: int,
    n_slices: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_components` | `int` | `3` | >= 1 | Number of SIR directions (effective rank of the between-slice covariance matrix). |
| `n_slices` | `int` | `10` | >= 2 | Number of contiguous slices of the target distribution. Clamped internally to the number of aligned clean rows. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Supervised dimension reduction extending ``scaled_pca`` to non-linear y → X dependence. Pipeline: (1) standardise X; (2) optional column-wise predictive scaling (``scaling_method`` = ``scaled_pca`` reuses the Huang-Zhou OLS-slope; ``marginal_R2`` uses sign(β_j)·√R²_j; ``none`` skips); (3) sort rows by y and partition into ``n_slices`` H contiguous slices; (4) compute weighted between-slice covariance ``Σ_S = Σ_h (n_h/n) · m̄_h · m̄_h^⊤``; (5) take the top-``n_components`` eigenvectors as factor loadings; (6) project the full panel onto these directions. The sSUFF augmentation (Huang-Zhou-Tong 2022) recovers latent factors with higher correlation than plain SIR in the macro-panel regime where signals are sparse over predictors.

Defaults: ``n_components = 2``, ``n_slices = 10``, ``scaling_method = 'scaled_pca'``. Requires a ``target_signal`` input port; ``temporal_rule`` is required and rejects ``full_sample_once``.

**When to use**

Supervised factor extraction from macro panels with non-linear y → X structure; alternative to ``scaled_pca`` when the predictive direction is non-monotone.

**When NOT to use**

Very small T (need ≥ 5·n_slices observations after dropping NaN); strictly linear y → X relationship (``scaled_pca`` is sufficient).

## In recipe context

Set ``params.op = "sliced_inverse_regression"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: sliced_inverse_regression
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Huang, Jiang, Li, Tong & Zhou (2022) 'Scaled PCA: A New Approach to Dimension Reduction', Management Science 68(3): 1678-1695.
* Fan, Xue & Yao (2017) 'Sufficient forecasting using factor models', Journal of Econometrics 201(2): 292-306.
* Li (1991) 'Sliced Inverse Regression for Dimension Reduction', JASA 86(414): 316-327.

## Related ops

See also: `scaled_pca`, `supervised_pca`, `partial_least_squares` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
