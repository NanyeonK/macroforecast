# `generalized_irf` -- Pesaran-Shin (1998) order-invariant generalized impulse-response function.

[Back to `op` axis](../axes/op.md) | [Back to L7](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L7_A_importance_dag_body`, layer `l7`.
> Standalone callable: `mf.functions.generalized_irf`.

## Function signature

```python
mf.functions.generalized_irf(...)
```

## Behavior

Order-invariant IRF where each GIRF is computed as the projection of all K residuals onto the j-th shock direction, scaled by the j-th diagonal entry of the residual covariance:

``GIRF_h(j) = sigma_jj^{-1/2} * A_h * Sigma * e_j``

where ``A_h = irf_obj.irfs[h]`` are the **raw reduced-form MA coefficients** (NOT Cholesky-orthogonalised), obtained via ``fitted_results.irf(n_periods, var_decomp=np.eye(K))``. Passing ``var_decomp=I`` bypasses the Cholesky factorisation inside statsmodels ``IRAnalysis``, returning ``A_h @ I = A_h`` directly. ``Sigma = fitted_results.sigma_u`` is the residual covariance matrix. ``e_j`` is the j-th standard basis vector.

**Importance metric**: ``importance[j] = sum_{h=0}^{H} |GIRF_h(j)[target_index]|`` -- the L1 norm of the target variable's response to shock j across all horizons 0..H.

**Order invariance**: permuting the VAR column order produces identical importance values for any given (shock, response) pair. Verified empirically to atol=1e-8 (actual difference ~1e-19 in the C49 test suite). This is the defining property that distinguishes the Pesaran-Shin GIRF from the Cholesky ``orthogonalised_irf``.

**Non-VAR fallback**: when the fitted model is not a VAR family (e.g., ridge), falls back to ``_tree_importance_frame`` with ``status='fallback_non_var'``.

**Numerical guard**: when ``sigma_jj <= 0`` (degenerate diagonal), scale defaults to 1.0. When ``irf()`` fails (singular or non-PD data), the fallback is triggered automatically.

Distinct from ``orthogonalised_irf`` (Cholesky lower-triangular rotation; order-dependent; operational since v0.2). Use ``generalized_irf`` when the variable ordering in the VAR has no theoretical motivation. Use ``orthogonalised_irf`` when a recursive identification is theoretically motivated.

**When to use**

VAR analysis where the variable ordering has no theoretical motivation and order-invariance is required; replication of papers using Pesaran-Shin (1998) GIRFs; comparing shock propagation across different variable orderings.

**When NOT to use**

When a recursive identification IS theoretically motivated -- use ``orthogonalised_irf`` instead (e.g., monetary policy ordered last in a structural VAR). Non-VAR models (triggers fallback to tree importance; no meaningful GIRF interpretation).

## In recipe context

Set ``params.op = "generalized_irf"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L7 recipe fragment
params:
  op: generalized_irf
```

## References

* macroforecast design Part 3, L7: 'every importance op produces (table, figure) pairs; the L7.B sub-layer governs export shape.'
* Pesaran & Shin (1998) 'Generalized impulse response analysis in linear multivariate models', Economics Letters 58(1): 17-29.

## Related ops

See also: `fevd`, `historical_decomposition`, `orthogonalised_irf` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
