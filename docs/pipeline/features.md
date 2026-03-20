# Features

`FeatureBuilder` assembles the predictor matrix Z_t that is passed to each forecasting model. It lives in `macrocast.pipeline.features`.

---

## Overview

Two construction modes are supported:

1. **Factors mode** (`use_factors=True`, used with `Regularization.FACTORS` / ARDI): Z_t = [PCA factors f_1 ... f_{p_f}, AR lags y_{t-1} ... y_{t-p_y}]
2. **AR-only mode** (`use_factors=False`): Z_t = [AR lags y_{t-1} ... y_{t-p_y}]. Used for data-poor linear baselines and the AR benchmark.

The number of factors `n_factors` and the number of lags `n_lags` are tuning parameters. In practice they are selected by the `CVScheme` that is active for the current model spec.

---

## Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_factors` | `8` | Number of PCA factors retained. Ignored when `use_factors=False`. Tuned by the CV loop. |
| `n_lags` | `4` | Number of AR lags appended. Applies in both modes. Tuned by the CV loop. |
| `use_factors` | `True` | Enable PCA diffusion index. Set to `False` for AR-only and linear baseline models. |
| `standardize_X` | `True` | Standardize predictor panel columns to zero mean and unit variance before PCA. Recommended for all panel sizes. |
| `standardize_Z` | `False` | Standardize the output Z matrix. Useful for distance-based models (KRR, SVR) and neural networks. |

---

## Methods

**`fit(X_panel, y)`**

Fit PCA loadings on the training window. `X_panel` has shape (T_train, N) with a DatetimeIndex. `y` is the target series aligned to the same index. Returns `self`. PCA is applied only to X_panel; y is never used in the decomposition.

**`transform(X_panel, y, is_train=False)`**

Apply the fitted PCA loadings to produce Z. `X_panel` can be a full training window or a single test row (shape (1, N)). AR lags are constructed from the last `n_lags` entries of y. Returns an array of shape (T - n_lags, n_features). The `is_train` flag controls whether a NaN-check for lag availability is enforced.

**`fit_transform(X_panel, y)`**

Convenience method: calls `fit` then `transform` on the same window. Equivalent to `builder.fit(X, y).transform(X, y, is_train=True)`.

---

## Pseudo-OOS Discipline

Correctness in a pseudo-OOS setting requires strict no-look-ahead:

- `fit` is called once per outer window, on the training observations only.
- `transform` re-uses the same fitted loadings for the test row. PCA loadings are never re-estimated after the training window is fixed.
- AR lags of y in the test row are drawn from the end of the training series, not from any future observations.

`ForecastExperiment` enforces this discipline automatically. Direct use of `FeatureBuilder` requires the caller to respect it.

---

## Properties

| Property | Description |
|----------|-------------|
| `n_features` | Total number of columns in Z: `n_factors + n_lags` in factors mode; `n_lags` in AR-only mode. |
| `is_fitted` | `True` after `fit` or `fit_transform` has been called. |

---

## Example

```python
from macrocast.pipeline import FeatureBuilder

builder = FeatureBuilder(n_factors=8, n_lags=4, use_factors=True)

# Fit on training window
Z_train = builder.fit_transform(X_train, y_train)
# Z_train.shape == (T_train - 4, 12)

# Transform a single test row (no refitting)
Z_test = builder.transform(X_test_row, y_train[-4:])
# Z_test.shape == (1, 12)

print(builder.n_features)   # 8 factors + 4 lags = 12
print(builder.is_fitted)    # True
```

---

## AR-only Mode

For the AR benchmark and linear baselines that do not use a factor structure, set `use_factors=False`:

```python
ar_builder = FeatureBuilder(n_lags=4, use_factors=False)
Z_ar = ar_builder.fit_transform(X_panel, y_train)
# Z_ar.shape == (T_train - 4, 4)  — lags only
print(ar_builder.n_features)   # 4
```

The AR benchmark selects `n_lags` by BIC (`CVScheme.BIC`) rather than by grid search.
