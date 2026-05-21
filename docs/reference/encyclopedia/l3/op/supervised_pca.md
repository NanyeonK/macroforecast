# `supervised_pca` -- Supervised PCA (Giglio-Xiu-Zhang 2025) -- screen-then-PCA on a target panel.

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.supervised_pca_transform`.

## Function signature

```python
mf.functions.supervised_pca_transform(
    panel: pd.DataFrame,
    target: pd.Series,
    n_components: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `target` | `pd.Series` | — | — | Supervisory signal aligned to the panel index. Must share at least one index value with panel; raises ValueError if the intersection is empty. |
| `n_components` | `int` | `3` | >= 1 | Number of supervised principal components (P). Clamped internally to the number of columns kept after correlation screening. |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Two-stage supervised reduction:
  1. For each target column ``g``, rank panel columns by univariate correlation with ``g`` and keep the top ``⌊q · M⌋`` (q ∈ (0, 1] hyperparameter; default 0.5);
  2. Run PCA on the screened sub-panel, returning P supervised components.

Refinement of Giglio-Xiu (2021) three-pass: screening makes the construction robust to weak factors and omitted-variable bias. Used as the asset-side stage of Rapach & Zhou (2025) Sparse Macro-Finance Factors for risk-premium estimation. Distinct from ``partial_least_squares`` (PLS uses covariance-maximising NIPALS over all columns; SPCA uses correlation-screened PCA on a sub-panel) and from ``scaled_pca`` (Huang-Jiang-Tu-Zhou 2022 weights every column; SPCA hard-screens).

Operational v0.9.1 dev-stage v0.9.0C-4. Hyperparams: ``n_components`` (= P; default 4), ``q`` (screening rate; default 0.5).

**When to use**

Cross-sectional asset-pricing factor extraction; weak-factor-robust supervised reduction; Rapach-Zhou (2025) replication.

**When NOT to use**

When the supervisory signal is dense (every panel column matters) -- prefer ``scaled_pca`` or ``partial_least_squares``.

## In recipe context

Set ``params.op = "supervised_pca"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: supervised_pca
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Giglio, Xiu & Zhang (2025) 'Test Assets and Weak Factors', Journal of Finance, forthcoming.
* Giglio & Xiu (2021) 'Asset Pricing with Omitted Factors', Journal of Political Economy 129(7): 1947-1990.
* Rapach & Zhou (2025) 'Sparse Macro-Finance Factors' working paper -- §2.2 eqs. (5)-(8).

## Related ops

See also: `partial_least_squares`, `scaled_pca`, `sparse_pca_chen_rohe`, `pca` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
