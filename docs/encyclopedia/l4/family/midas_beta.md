# `midas_beta` -- MIDAS with Beta distribution kernel lag weights (Ghysels-Sinko-Valkanov 2007).

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.midas_beta`.

## Function signature

```python
mf.functions.midas_beta(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    freq_ratio: int = 1,
    n_lags_high: int = 12,
    sum_to_one: bool = True,
    max_iter: int = 200,
    n_starts: int = 5,
    random_state: int = 0,
) -> _MidasBetaModel
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `pd.DataFrame` | — | — | Feature matrix as a DataFrame. When ``freq_ratio = 1``: LF-aligned lag columns. When ``freq_ratio > 1``: raw HF DataFrame internally lag-stacked. Shape (n_samples, n_features). |
| `y` | `pd.Series` | — | — | Low-frequency target series aligned to the LF index. |
| `freq_ratio` | `int` | `1` | >=1 | High-frequency periods per low-frequency period. 1 = X already LF-aligned. |
| `n_lags_high` | `int` | `12` | >=1 | Number of high-frequency lags K. |
| `sum_to_one` | `bool` | `True` | — | Normalize weights to sum to 1 (always True by construction of Beta kernel; parameter retained for API symmetry with midas_almon). |
| `max_iter` | `int` | `200` | — | Maximum Nelder-Mead iterations per start. |
| `n_starts` | `int` | `5` | >=1 | Number of NLS restarts. Start 0 uses [1,1]; remaining starts draw a, b from Gamma(2,1). |
| `random_state` | `int` | `0` | — | RNG seed for perturbed NLS starts. Propagated from L0 via per-origin RNG contract (#279). |

## Returns

`_MidasBetaModel` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `._w_hat` | `np.ndarray` | Estimated Beta kernel lag weights, shape (n_lags_high,). Zero-padded when K_eff < n_lags_high. |
| `._theta_hat` | `np.ndarray` | Estimated Beta shape parameters [a, b], shape (2,). |
| `._intercept` | `float` | Fitted intercept mu. |
| `._slope` | `float` | Fitted overall slope beta. |
| `._converged` | `bool` | NLS convergence flag. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_rows,). |

## Behavior

Estimates mixed-frequency regressions via non-linear least squares (Nelder-Mead) on Beta distribution kernel weights. The weight function ``b(k) ∝ x_k^{a-1} · (1-x_k)^{b-1}`` (Ghysels, Sinko & Valkanov 2007, §2) maps normalized lag positions ``x_k = (k+1)/(K+1) ∈ (0,1)`` to scalar weights. Only 2 shape parameters ``(a, b)`` control the entire lag profile, making this the most parsimonious MIDAS variant.

Frequency contract and multi-start NLS are identical to ``midas_almon``. Start 0: ``[1, 1]`` (uniform Beta = equal weights). Perturbed starts: ``a, b ~ Gamma(2, 1)`` to keep both parameters positive naturally. Post-optimization clamp: ``a, b >= 1e-3``.

The two-attribute design (``_w_hat`` zero-padded to ``n_lags_high``; ``_w_hat_effective`` of length ``K_eff``) is identical to ``midas_almon``.

**When to use**

Parsimonious mixed-frequency forecasting -- only 2 shape parameters regardless of K. Suitable when T is small relative to K, making the Almon polynomial over-parameterized.

**When NOT to use**

When the lag weight profile is known to be non-Beta-shaped (e.g., step-function or flat). Use ``midas_step`` or ``midas_almon`` instead.

## In recipe context

Set ``params.family = "midas_beta"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  family: midas_beta
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Ghysels, Sinko & Valkanov (2007) 'MIDAS Regressions: Further Results and New Directions', Econometric Reviews 26(1).

## Related ops

See also: `midas_almon`, `midas_step`, `dfm_unrestricted_midas` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
