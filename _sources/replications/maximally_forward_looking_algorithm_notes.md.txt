# Maximally Forward-Looking Core Inflation — algorithm notes

**Paper**: Goulet Coulombe, Klieber, Barrette, Göbel (2024) "Maximally
Forward-Looking Core Inflation". Draft 2024-03-15. R package
`assemblage` (CRAN).

**PDF**: `wiki/raw/papers/Maximally Forward-Looking Core Inflation.pdf`

## Decomposition into atomic primitives

The paper introduces two indicators built from a generalised
non-negative ridge ("Assemblage Regression"):

* **Albacore_comps** (component space) -- supervised weighting of
  inflation components.
* **Albacore_ranks** (rank space) -- supervised weighting of
  *order statistics* of components per period; "asymmetric trimming"
  emerges from the learned weights on each rank position.

Decomposition into the macroforecast vocabulary:

| Paper construct | macroforecast primitive |
|---|---|
| Assemblage Regression (nonneg ridge, no intercept) | `ridge(coefficient_constraint=nonneg)` ✓ v0.8.9 (B-1) |
| `||Dw||₂²` fused-ridge penalty over rank weights | future sub-axis on `ridge` (`penalty=fused_difference`); v0.9.x |
| Albacore_comps target | recipe pattern: nonneg ridge on components Π_t against forward inflation |
| Albacore_ranks target | recipe pattern: `asymmetric_trim` (L2 op) → nonneg ridge on order statistics |
| `asymmetric_trim` rank-space transformation | new L2 op (this doc) |

## asymmetric_trim algorithm (B-6 spec)

**Input**: panel `Π` of shape `(n_periods, K)`, where each row is the
contemporaneous component growth rates at time t.

**Output**: panel `O` of shape `(n_periods, K)`, where `O[t, r]` =
the r-th order statistic of `Π[t, :]` (ascending sort).

**Pseudocode**:

```python
def asymmetric_trim(panel: pd.DataFrame) -> pd.DataFrame:
    # panel: (T x K), columns = component labels.
    # output: (T x K), columns = ['rank_1', ..., 'rank_K'] (1-indexed).
    arr = panel.to_numpy(dtype=float)
    sorted_arr = np.sort(arr, axis=1)  # ascending per-row
    cols = [f"rank_{r + 1}" for r in range(arr.shape[1])]
    return pd.DataFrame(sorted_arr, index=panel.index, columns=cols)
```

**Optional 3-month MA smoothing** (paper §3 implementation note):
"Order statistics time series are smoothed using the 3-month moving
average." This is delegated to a downstream `ma_window` op when the
user wants the smoothed variant; the bare `asymmetric_trim` op does
not bake it in (decomposition discipline).

## Equation reference (paper §2)

Albacore_ranks regression:

    ŵ_r = argmin_w  Σ_{t=1}^{T-h} (π_{t+1:t+h} - w' O_t)²  +  λ ||Dw||₂²
                  s.t.  w >= 0,  π̄_{t+1:t+h} = π̄*_ranks,t

where `D` is the difference operator and `O_t = sort(Π_t)`.

## Verification (B-6 quality bar)

* `asymmetric_trim` is idempotent on already-sorted input: applying it
  twice yields the same matrix as one application.
* For a uniformly-distributed component panel, the rank-r series
  approximates the r/K quantile of the marginal distribution.
* Per-row monotone: `O[t, 0] <= O[t, 1] <= ... <= O[t, K-1]`.

## Status

Implemented in v0.8.9 (B-6) as an L2 op. The downstream
`ridge(coefficient_constraint=nonneg)` is also in v0.8.9 (B-1), so the
Albacore_ranks recipe runs end-to-end after sorting. The fused-ridge
penalty (`||Dw||₂²`) on the weight differences across ranks is *not*
yet in the package -- standard ridge `α||w||₂²` is a less informative
prior on the rank weights but still produces a valid (just less
smooth) solution. Fused-ridge is a v0.9.x sub-axis target.
