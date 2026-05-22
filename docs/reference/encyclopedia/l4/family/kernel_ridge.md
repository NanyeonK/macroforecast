# `kernel_ridge` -- Kernel Ridge Regression -- closed-form non-linear ridge in the dual.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.kernel_ridge_fit`.

## Function signature

```python
mf.functions.kernel_ridge_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> KernelRidgeFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`KernelRidgeFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.alpha` | `float` | Ridge regularisation strength. |
| `.kernel` | `str` | Kernel type (e.g. rbf, linear). |
| `.n_features_in_` | `int` | Number of input features. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: alpha, kernel, feature count. |

## Behavior

Ridge regression with a non-linear kernel: ``ŷ(x) = Σ_i α_i K(x, x_i) + b`` where the dual coefficients ``α = (K + λ I)⁻¹ y`` are recovered in closed form. Operational v0.9.1 dev-stage v0.9.0F (audit-fix). Surfaces as a first-class L4 family because Coulombe / Surprenant / Leroux / Stevanovic (2022 JAE) 'How is Machine Learning Useful for Macroeconomic Forecasting?' Eq. 16 / §3.1.1 uses KRR as the headline non-linearity feature in the macro horse race.

**Tunable params**: ``alpha`` (= ridge penalty λ; default 1.0); ``kernel`` ('rbf' default / 'linear' / 'poly' / 'sigmoid' / 'laplacian' / 'chi2' -- any sklearn-supported kernel); ``gamma`` (RBF bandwidth, default sklearn auto = 1/n_features); ``degree`` (poly kernel only, default 3); ``coef0`` (poly / sigmoid, default 1.0).

Distinct from ``svr_rbf`` (ε-insensitive loss, sparsity in support vectors) and from ``ridge`` (linear). The dual representation also pairs with the L7 ``dual_decomposition`` op for kernel-weighted training-target attribution.

**When to use**

Non-linear macro forecasting baselines; KRR vs SVR-RBF / RF ablations; replicating Coulombe et al. (2022) Feature 1 nonlinearity test.

## In recipe context

Set ``params.family = "kernel_ridge"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: kernel_ridge
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Saunders, Gammerman & Vovk (1998) 'Ridge Regression Learning Algorithm in Dual Variables', ICML.
* Coulombe, Leroux, Stevanovic & Surprenant (2022) 'How is Machine Learning Useful for Macroeconomic Forecasting?', Journal of Applied Econometrics 37(5): 920-964 -- Eq. 16 + §3.1.1.

## Related ops

See also: `ridge`, `svr_rbf`, `dual_decomposition` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
