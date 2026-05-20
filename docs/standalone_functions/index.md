# Standalone functions - namespace overview

`macroforecast` ships two complementary usage paradigms:

1. **Recipe DSL** - a YAML-driven 12-layer DAG that runs an end-to-end
   reproducible study. Every sweep cell is bit-exact replicable via
   `mf.replicate(manifest_path)`.
2. **Standalone callables** - `mf.functions.<name>(...)` gives you a single
   operation as a plain Python function, with no YAML and no recipe
   overhead. Useful for scripting, Jupyter notebooks, and partial pipeline
   integration.

The two paradigms share the same underlying implementations - a standalone
call is a thin wrapper around the same adapter the recipe runtime uses.

## Namespace

```python
import macroforecast as mf
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 2.1, 2.9])
y_prev = np.array([0.9, 2.0, 2.8])

result = mf.functions.ridge_fit(X, y, alpha=1.0)
u1     = mf.functions.theil_u1(y_true, y_pred)
u2     = mf.functions.theil_u2(y_true, y_pred, y_prev)
```

## Layer summary

| Layer | Purpose | Callables | Typical use |
|---|---|---|---|
| [L2 clean](l2_clean.md) | Transform, outlier, imputation, frame edge, tcode | 14 | Clean a panel before features |
| [L3 transforms](l3_transforms.md) | Lags, dimension reduction, filters, supervised transforms | 36 | Build predictor matrices |
| [L4 fit](l4_fit.md) | Forecasting model families | 38 | Fit and predict |
| [L5 metrics](l5_metrics.md) | Point, relative, density, direction, Theil | 15 | Evaluate accuracy |
| [L6 tests](l6_tests.md) | Equal-predictive-ability + nested-model tests | 7 | Significance testing |
| [L7 importance](l7_importance.md) | Native, permutation, ALE, PDP, SHAP | 8 | Interpret models |

Total: 118 standalone callables.

## End-to-end example

The following snippet uses two standalone callables - `ols_fit` (L4) and
`permutation_importance` (L7) - to fit a model and rank features, with no
YAML recipe needed.

```python
import macroforecast as mf
import numpy as np

rng = np.random.default_rng(42)
n, p = 120, 8
X = rng.standard_normal((n, p))
beta = np.array([2.0, -1.5, 0.0, 3.0, 0.0, -0.5, 0.0, 1.0])
y = X @ beta + 0.5 * rng.standard_normal(n)

# L4: fit OLS
fit_result = mf.functions.ols_fit(X, y)
print(fit_result.summary())
print("Coefficients:", fit_result.coef_)

# L7: permutation importance (returns PermutationImportanceResult)
imp = mf.functions.permutation_importance(fit_result, X, y, n_repeats=10, random_state=42)
# .importances_mean_ is exposed by PermutationImportanceResult and
# CondPermutationImportanceResult only. Native / ALE / PDP / SHAP results use
# .importances_ or their type-specific attribute.
print("Feature importances:", imp.importances_mean_)
```

See [two_entry_points](../two_entry_points.md) for a decision guide on when
to use the recipe DSL vs standalone callables.
