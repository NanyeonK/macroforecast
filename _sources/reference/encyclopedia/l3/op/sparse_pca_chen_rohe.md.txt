# `sparse_pca_chen_rohe` -- Chen-Rohe (2023) Sparse Component Analysis -- non-diagonal D variant.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.sparse_pca_chen_rohe_transform`.

## Function signature

```python
mf.functions.sparse_pca_chen_rohe_transform(
    panel: pd.DataFrame,
    n_components: int,
    zeta: float,
    max_iter: int,
    var_innovations: bool,
    random_state: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_components` | `int` | `4` | >= 1 | Number of sparse components (= J in the SCA objective). Clamped internally to min(T_clean, K). |
| `zeta` | `float` | `0.0` | >= 0 | L1 budget for loadings Theta. 0.0 routes to zeta = n_components (paper CV-optimal boundary). |
| `max_iter` | `int` | `200` | >= 1 | Maximum alternating-maximisation iterations. |
| `var_innovations` | `bool` | `'False'` | — | If True, fit VAR(1) on SCA scores and return residuals as sparse macro-finance factors (Rapach-Zhou 2025 step 2). |
| `random_state` | `int` | `0` | — | Seed for NumPy RNG used in Z/Theta initialisation. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Sparse component analysis solving ``min_{Z,D,Θ} ‖X − Z D Θ'‖_F`` s.t. ``Z ∈ S(T,J)``, ``Θ ∈ S(M,J)``, ``‖Θ‖_1 ≤ ζ`` (Chen-Rohe 2023; Rapach & Zhou 2025 eq. 3). Differs from ``sparse_pca`` (sklearn / Zou-Hastie-Tibshirani 2006) in two ways: (1) the central matrix D is *not* restricted to be diagonal, which lets SCA explain more total variation for a given sparsity budget; (2) the single hyperparameter ``ζ ∈ [J, J√M]`` enters as an ℓ_1 budget *constraint* rather than a Lagrangian penalty.

Implementation: alternating maximisation of the equivalent bilinear convex-hull form ``max_{Z,Θ} ‖Z' X Θ‖_F`` over ``H(T,J) × H(M,J)`` (Zhou-Rapach 2025 eq. 4), iterating SVD-projection of Z and L1-budget projection of Θ. Used as the macro-side stage in Rapach & Zhou (2025) Sparse Macro-Finance Factors. Operational v0.9.1 dev-stage v0.9.0C-3.

Hyperparams: ``n_components`` (= J; default 4), ``zeta`` (= L1 budget; ``0.0`` defaults to J = most-binding boundary the paper finds optimal in CV), ``max_iter`` (default 200), ``random_state``.

**When to use**

Sparse macro-finance factor extraction with non-diagonal D; the Rapach-Zhou (2025) macro-side procedure.

**When NOT to use**

When sklearn-style L1-penalised loadings are sufficient -- prefer the cheaper ``sparse_pca``.

## In recipe context

Set ``params.op = "sparse_pca_chen_rohe"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: sparse_pca_chen_rohe
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a pipeline of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Chen & Rohe (2023) 'A New Basis for Sparse Principal Component Analysis', Journal of Computational and Graphical Statistics. arXiv:2007.00596.
* Rapach & Zhou (2025) 'Sparse Macro-Finance Factors' working paper -- §2.1 eqs. (3)-(4).

## Related ops

See also: `sparse_pca`, `supervised_pca`, `scaled_pca`, `pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
