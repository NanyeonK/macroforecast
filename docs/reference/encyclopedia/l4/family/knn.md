<!-- TODO(restructure-phase-1-followup): folder will be renamed from l4/family/ to l4/model/ in a separate doc-pass -->
# `knn` -- k-nearest-neighbours regression.

[Back to `family` axis](../axes/family.md) | [Back to L4](../index.md) | [Browse all options](../../browse_by_option.md)

> Operational op under axis `family`, sub-layer `L4_A_model_selection`, layer `l4`.
> Standalone callable: `mf.functions.knn_fit`.

## Function signature

```python
mf.functions.knn_fit(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> KNNFitResult
```

## Parameters

| name | type | default | constraint | description |
|---|---|---|---|---|
| `X` | `np.ndarray | pd.DataFrame` | — | — | Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames. |
| `y` | `np.ndarray | pd.Series` | — | — | Target vector. Shape (n_samples,). Accepts numpy arrays or Series. |

## Returns

`KNNFitResult` — frozen dataclass with fit results.

| Attribute | Type | Description |
|-----------|------|-------------|
| `.n_neighbors` | `int` | Neighbour count k requested. |
| `.n_neighbors_used` | `int` | Actual k used (clipped to training set size). |
| `.n_features_in_` | `int` | Number of input features. |
| `.predict(X)` | `np.ndarray` | Predictions for new data X, shape (n_samples,). |
| `.summary()` | `str` | Table: neighbour counts and feature count. |

## Behavior

Memorises training data; predicts via nearest-neighbour averaging. Cheap, non-parametric.

**When to use**

Non-parametric baselines; sensitivity studies.

## In recipe context

Set ``params.family = "knn"`` in the relevant layer to activate this op within a recipe:

```yaml
# Layer L4 recipe fragment
params:
  model: knn
```

## References

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Cover & Hart (1967) 'Nearest neighbor pattern classification', IEEE Trans. on Information Theory 13(1).

## Related ops

See also: `random_forest`, `svr_rbf` (on the same axis).

_Last reviewed 2026-05-04 by macroforecast author._
