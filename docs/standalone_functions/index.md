# Standalone functions — namespace overview

`macroforecast` ships two complementary usage paradigms:

1. **Recipe DSL** — a YAML-driven 12-layer DAG that runs an end-to-end
   reproducible study. Every sweep cell is bit-exact replicable via
   `mf.replicate(manifest_path)`.
2. **Standalone callables** — `mf.functions.<name>(...)` gives you a single
   operation as a plain Python function, with no YAML and no recipe
   overhead. Useful for scripting, Jupyter notebooks, and partial pipeline
   integration.

The two paradigms share the same underlying implementations — a standalone
call is a thin wrapper around the same adapter the recipe runtime uses.

## Namespace

```python
import macroforecast as mf

# POC surface (Cycle 22, v0.10 candidate):
result = mf.functions.ridge_fit(X, y, alpha=1.0)
u1     = mf.functions.theil_u1(y_true, y_pred)
u2     = mf.functions.theil_u2(y_true, naive_pred, y_pred)

# Subsequent cycles will expand to L3, L4, L5, L6, L7 op families.
```

## Layer summary

| Layer | Purpose | Ops (operational) | Typical use |
|---|---|---|---|
| [L2 preprocessing](l2_clean.md) | Transform, outlier, imputation, frame edge | 13 axes | Clean a panel before features |
| [L3 feature engineering](l3_transforms.md) | Lags, dimension reduction, filters, supervised transforms | 47 ops | Build predictor matrices |
| [L4 fit](l4_fit.md) | Forecasting model families + tuning | 43 ops | Fit and predict |
| [L5 metrics](l5_metrics.md) | Point, relative, density, direction, Theil | 15 options | Evaluate accuracy |
| [L6 tests](l6_tests.md) | Equal-predictive-ability + nested-model tests | 7 ops | Significance testing |
| [L7 importance](l7_importance.md) | Feature importance, SHAP, dependence, permutation | 8 grouped ops | Interpret models |

## End-to-end example

The following snippet uses two standalone callables — `ols_fit` (L4) and
`permutation_importance` (L7) — to fit a model and rank features, with no
YAML recipe needed.

```python
import macroforecast as mf
import numpy as np

rng = np.random.RandomState(42)
n, p = 120, 8
X = rng.randn(n, p)
beta = np.array([2.0, -1.5, 0.0, 3.0, 0.0, -0.5, 0.0, 1.0])
y = X @ beta + 0.5 * rng.randn(n)

# L4: fit OLS
fit_result = mf.functions.ols_fit(X, y)
print(fit_result.summary())
print("Coefficients:", fit_result.coef_)

# L7: permutation importance
imp = mf.functions.permutation_importance(fit_result, X, y, n_repeats=30, random_state=42)
print("Feature importances:", imp.importances_mean)
```

> **Note** — `ols_fit` and `permutation_importance` are planned for the
> L3/L4/L5/L6/L7 expansion in cycles following Cycle 22. The current POC
> surface is `mf.functions.ridge_fit`, `mf.functions.theil_u1`, and
> `mf.functions.theil_u2`. See [two_entry_points](../two_entry_points.md)
> for a decision guide on when to use the recipe DSL vs standalone callables.
