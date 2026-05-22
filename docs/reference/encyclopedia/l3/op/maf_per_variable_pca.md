# `maf_per_variable_pca` -- Per-variable MAF via PCA on lag-panels -- Coulombe et al. (2021 IJF) Eq. (7).

[Back to `op` axis](../axes/op.md) | [Back to L3](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `op`, sub-layer `L3_A_step_op`, layer `l3`.
> Standalone callable: `mf.functions.maf_per_variable_pca_transform`.

## Function signature

```python
mf.functions.maf_per_variable_pca_transform(
    panel: pd.DataFrame,
    n_lags: int,
    n_components_per_var: int,
) -> pd.DataFrame
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `panel` | `pd.DataFrame` | — | — | Input panel. Each column is a variable; rows are time periods. Series is promoted to a single-column DataFrame internally. |
| `n_lags` | `int` | `12` | >= 1 | Number of lags in the per-variable lag-panel. Paper default: 12 (monthly data). |
| `n_components_per_var` | `int` | `2` | >= 1 | Number of PCA components per variable. Paper default: 2 (footnote 11). |

## Returns

`pd.DataFrame` — scalar result.

## Behavior

Implements the paper-exact Moving Average Factor (MAF) construction from Coulombe, Leroux, Stevanovic & Surprenant (2021 IJF) §2.2 Eq. (7). For each variable ``k = 1..K`` in the input panel:

1. Build the ``T × (n_lags + 1)`` lag-panel ``[X_{t,k}, L X_{t,k}, ..., L^{n_lags} X_{t,k}]``.
2. Run PCA retaining ``n_components_per_var`` components (paper default: 2).
3. Append the resulting factor columns to the output.

Output shape: ``(T, K · n_components_per_var)``. With defaults ``n_lags=12``, ``n_components_per_var=2`` the output is ``(T, 2K)`` -- paper footnote 11: 'We keep two MAFs for each series and they are obtained by PCA.'

**Distinction from existing ``ma_increasing_order → pca(4)`` path**: the existing stacked-PCA MAF cell runs a single PCA over all MA columns at once (stacked, 4 global components). This op runs separate PCA per variable, yielding ``2K`` locally-structured factors rather than 4 global ones. Use this op when paper-Eq.7-exact replication is required.

First ``n_lags`` rows per variable are NaN (lag-panel boundary). ``temporal_rule`` is required; ``full_sample_once`` is rejected to enforce walk-forward boundaries.

Operational from v0.9.0 (phase-f16).

**When to use**

Paper-exact replication of Coulombe et al. (2021 IJF) MAF construction; when per-variable PCA factors are preferred over global stacked-PCA (4-component) path.

**When NOT to use**

When the 16-cell horse-race stacked-PCA MAF cell is sufficient (use ``ma_increasing_order`` → ``pca`` instead); or when K is large and 2K output columns would exceed T.

## In recipe context

Set ``params.op = "maf_per_variable_pca"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L3 recipe fragment
params:
  op: maf_per_variable_pca
```

## References

* macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'
* Coulombe, Leroux, Stevanovic & Surprenant (2021) 'Macroeconomic Data Transformations Matter', International Journal of Forecasting 37(4): 1338-1354. <https://doi.org/10.1016/j.ijforecast.2021.05.005>

## Related ops

See also: `maf`, `ma_increasing_order`, `ma_window` (on the same axis).

_Last reviewed 2026-05-05 by macroforecast author._
