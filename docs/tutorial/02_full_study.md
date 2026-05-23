# Full out-of-sample study

Run a full out-of-sample evaluation comparing three model classes from
`macroforecast.models` using scikit-learn's `TimeSeriesSplit`, then graduate to
the recipe pipeline for systematic benchmarking. No YAML is required for the
main body of this tutorial.

---

## Before you start

This tutorial assumes you have completed {doc}`01_first_forecast` and are
comfortable with `LinearAR`. If not, start there.

---

## The synthetic macro panel

We construct a 300-observation monthly panel with five predictor series and a
factor-driven target. The seed is fixed for reproducibility.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=0)
n = 300
dates = pd.date_range("2000-01-01", periods=n, freq="MS")

# Five synthetic macro predictors
X = pd.DataFrame(
    rng.standard_normal((n, 5)),
    index=dates,
    columns=["ip_growth", "unemp_diff", "cpi_growth", "ffr_diff", "spread"],
)

# Target: factor-driven process with AR(1) residual
beta = np.array([0.4, -0.3, 0.2, -0.1, 0.15])
eps = rng.normal(scale=0.5, size=n)
y_vals = X.values @ beta + eps
for t in range(1, n):
    y_vals[t] += 0.3 * y_vals[t - 1]

y = pd.Series(y_vals, index=dates, name="gdp_growth")
```

The target combines a linear factor component with an AR(1) residual, so both
autoregressive and factor-based models should find signal.

---

## Out-of-sample loop with LinearAR

We use `TimeSeriesSplit` with five folds and a fixed test size of 20 observations
per fold. `LinearAR` is a target-only model; it accepts `X` for API uniformity
but does not use it in estimation. We pass the predictor frame anyway so that
the loop structure is uniform across all three models.

```python
from macroforecast.models import LinearAR
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

tscv = TimeSeriesSplit(n_splits=5, test_size=20)
results = {}
mse_ar = []

for train_idx, test_idx in tscv.split(y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    # p=4 means four AR lags; LinearAR ignores X internally
    m = LinearAR(p=4)
    m.fit(X_tr, y_tr)
    preds = m.predict(X_te)   # len(X_te) determines output size
    mse_ar.append(np.mean((preds - y_te.values) ** 2))

results["LinearAR"] = np.mean(mse_ar)
print(f"LinearAR mean MSE: {results['LinearAR']:.4f}")
```

---

## Adding PrincipalComponentRegression and FactorAugmentedAR

`PrincipalComponentRegression` extracts the top-k principal components from `X`
and regresses `y` on those components by OLS. `FactorAugmentedAR` augments an
AR(p) model with PCA factors from `X`, following the Stock-Watson two-step
procedure. Both models require `X` at fit and predict time.

The constructor signatures, verified from source:
- `PrincipalComponentRegression(n_components=3)` -- number of PC factors
- `FactorAugmentedAR(p=2, n_factors=3)` -- lag order `p` and number of factors

```python
from macroforecast.models import PrincipalComponentRegression, FactorAugmentedAR

for ModelClass, name, kwargs in [
    (PrincipalComponentRegression, "PCR",  {"n_components": 3}),
    (FactorAugmentedAR,            "FAAR", {"p": 2, "n_factors": 3}),
]:
    mse_list = []
    for train_idx, test_idx in tscv.split(y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        m = ModelClass(**kwargs)
        m.fit(X_tr, y_tr)
        preds = m.predict(X_te)
        mse_list.append(np.mean((preds - y_te.values) ** 2))

    results[name] = np.mean(mse_list)
    print(f"{name} mean MSE: {results[name]:.4f}")
```

---

## Comparing the three models

Collecting the results into a `pd.Series` lets us sort and inspect them at a glance.

```python
import pandas as pd

summary = pd.Series(results).rename("mean_cv_mse").sort_values()
print(summary.to_string())
```

We expect FAAR to outperform LinearAR when the factor structure in `X` explains
target variation, because the PCA step recovers latent predictors that AR lags
alone cannot. PCR provides a simpler factor extraction without an autoregressive
component; its advantage over LinearAR depends on the signal-to-noise ratio in
the predictor panel. On this synthetic dataset the factor loadings in `beta` are
moderate, so the ranking may vary across random seeds.

---

## When to graduate to recipes

The standalone loop above is sufficient for exploratory work. When the study
requires systematic sweeps over lag orders or factor counts, the Diebold-Mariano
test battery, or bit-exact replication with a verifiable manifest, the recipe
pipeline is the right tool.

```python
import macroforecast.recipes as mf_recipes
# mf.run(...) is a valid alias for mf_recipes.run(...)

# The recipe API runs the same model families as the standalone classes,
# adding bit-exact provenance, parameter sweeps, and the DM test battery.
result = mf_recipes.run("path/to/my_study.yaml", output_directory="./out/tut02/")
```

For full recipe syntax, see the recipe DSL reference. The next tutorial,
{doc}`03_custom_model`, shows how to define your own model class and use it
directly in the same `TimeSeriesSplit` loop. For systematic model sweeps in the
recipe pipeline, see {doc}`../how_to/sweep_over_models`. For the conceptual
comparison of standalone and recipe modes, see {doc}`two_entry_points`.
