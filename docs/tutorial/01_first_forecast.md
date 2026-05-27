# Your first forecast

Fit and evaluate a linear autoregressive model on a synthetic monthly series using
`macroforecast.models.LinearAR`, with no YAML required. This takes about five minutes.

---

## Before you start

Verify that macroforecast is installed:

```python
import macroforecast
print(macroforecast.__version__)   # 0.9.2b1 or later
```

If the import fails, see {doc}`00_install`.

---

## Generate the synthetic series

We create a 100-observation monthly series following an AR(2) process. Using
`numpy.random.default_rng` with a fixed seed keeps the example reproducible.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
n = 100
dates = pd.date_range("2015-01-01", periods=n, freq="MS")

# AR(2) process: y_t = 0.6 * y_{t-1} - 0.3 * y_{t-2} + eps
y_vals = np.zeros(n)
eps = rng.normal(scale=0.5, size=n)
for t in range(2, n):
    y_vals[t] = 0.6 * y_vals[t - 1] - 0.3 * y_vals[t - 2] + eps[t]

y = pd.Series(y_vals, index=dates, name="gdp_growth")
```

---

## Fit LinearAR and predict

`LinearAR` estimates an AR(p) model by OLS on the lag matrix. The constructor
takes one required argument, `p`, which sets the lag order. The `fit` method
accepts both `X` and `y` to satisfy the sklearn convention. `X` is not used
internally by `LinearAR` but must be passed. Passing an empty DataFrame with
a matching index is the cleanest option when no predictors are available.

```python
from macroforecast.models import LinearAR
import pandas as pd

# Split: 80 observations for training, 20 for evaluation
train_end = 80
y_train, y_test = y.iloc[:train_end], y.iloc[train_end:]

# Empty feature frame -- LinearAR is a target-only model;
# X is accepted for API uniformity but is not used in estimation.
X_train = pd.DataFrame(index=y_train.index)
X_test  = pd.DataFrame(index=y_test.index)

# Fit on training data
model = LinearAR(p=2)
model.fit(X_train, y_train)

# Predict the remaining 20 periods -- predict uses len(X_test) for output size
preds = model.predict(X_test)
print(preds)   # np.ndarray of shape (20,)
```

The returned array contains a single-step-ahead forecast repeated for each
row of `X_test`. For a stationary AR(2) process the forecast converges quickly
toward the long-run mean.

---

## Evaluate out-of-sample accuracy

A single train/test split can be noisy. `TimeSeriesSplit` from scikit-learn
provides a principled expanding-window cross-validation that respects temporal
order. We compute mean squared error across five folds.

```python
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []

for train_idx, test_idx in tscv.split(y):
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    X_tr = pd.DataFrame(index=y_tr.index)
    X_te = pd.DataFrame(index=y_te.index)

    m = LinearAR(p=2)
    m.fit(X_tr, y_tr)
    preds = m.predict(X_te)
    mse_scores.append(np.mean((preds - y_te.values) ** 2))

print(f"Mean CV MSE: {np.mean(mse_scores):.4f}")
```

We find that the mean CV MSE is low relative to the noise variance (0.25),
because the AR(2) structure is directly estimable from the training data.

---

## When you need reproducibility -- graduate to recipes

The standalone approach above is ideal for notebooks and one-off scripts. When
you need bit-exact replication across machines, systematic parameter sweeps, or
the full L6 test battery, the recipe pipeline adds those capabilities in exchange
for a YAML recipe definition.

```python
import macroforecast.recipes as mf_recipes
# mf.run(...) is a valid alias for mf_recipes.run(...)

recipe = """
0_meta:
  fixed_axes:
    reproducibility_policy: seeded_reproducible
  leaf_config:
    random_seed: 42
1_data:
  fixed_axes:
    panel_composition: custom_panel_only
    frequency: monthly
    horizon_set: custom_list
  leaf_config:
    target: gdp_growth
    target_horizons: [1]
    custom_panel_inline: {date: [...], gdp_growth: [...]}
# ... (remaining layers defined in the recipe DSL reference)
"""

result = mf_recipes.run(recipe, output_directory="./out/tut01/")
```

For full recipe syntax, see the recipe DSL reference. For the conceptual
comparison of standalone and recipe modes, see {doc}`two_entry_points`. The
next tutorial, {doc}`02_full_study`, extends this setup to a three-model
comparison using `TimeSeriesSplit` and a five-feature macro panel.
