# Standalone functions: L7 importance (8 ops)

L7 importance callables take a fitted L4 result object plus feature matrix `X` (and target `y` for permutation-based methods) and return a frozen result dataclass. Return attributes differ by callable - see each type-specific description below.

Result type summary:

- `NativeImportanceResult` exposes `.importances_`, `.feature_names_`, `.method`, `.summary()`.
- `PermutationImportanceResult` exposes `.importances_mean_`, `.importances_std_`, `.feature_names_`, `.n_repeats`, `.summary()`.
- `CondPermutationImportanceResult` exposes `.importances_mean_`, `.importances_std_`, `.feature_names_`, `.method`, `.n_repeats`, `.summary()`.
- `ALEImportanceResult` exposes `.importances_`, `.ale_values_`, `.feature_names_`, `.summary()`.
- `PDPImportanceResult` exposes `.importances_`, `.pdp_values_`, `.grid_values_`, `.feature_names_`, `.summary()`.
- `SHAPImportanceResult` exposes `.shap_values_`, `.expected_value_`, `.explainer_type`, `.feature_names_`, `.summary()`.

Only `PermutationImportanceResult` and `CondPermutationImportanceResult` expose `.importances_mean_` / `.importances_std_`. The other importance types use `.importances_`, `.ale_values_`, `.pdp_values_`, or `.shap_values_` instead.

## Model-native importance (2 ops)

#### `model_native_linear_coef_importance(result: Any, X: np.ndarray | pd.DataFrame) -> NativeImportanceResult`

Extract the fitted coefficient vector as signed importance scores (X required).

Returns `NativeImportanceResult`: `.feature_names_`, `.importances_`, `.method`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.ols_fit(X, y)
imp = mf.functions.model_native_linear_coef_importance(fit_result, X)
print(imp.importances_)
```

[Encyclopedia](../encyclopedia/l7/op/model_native_linear_coef.md)

#### `model_native_tree_importance(result: Any, X: np.ndarray | pd.DataFrame) -> NativeImportanceResult`

Tree impurity-based (gini/gain) feature importance (X required).

Returns `NativeImportanceResult`: `.feature_names_`, `.importances_`, `.method`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.random_forest_fit(X, y, n_estimators=20)
imp = mf.functions.model_native_tree_importance(fit_result, X)
print(imp.importances_)
```

[Encyclopedia](../encyclopedia/l7/op/model_native_tree_importance.md)

## Permutation-based importance (2 ops)

#### `permutation_importance(result: Any, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_repeats: int = 10, random_state: int | None = None) -> PermutationImportanceResult`

Breiman-Fisher permutation feature importance (model-agnostic).

Returns `PermutationImportanceResult`: `.feature_names_`, `.importances_mean_`, `.importances_std_`, `.n_repeats`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.ols_fit(X, y)
imp = mf.functions.permutation_importance(fit_result, X, y, n_repeats=10)
print(imp.importances_mean_)
```

[Encyclopedia](../encyclopedia/l7/op/permutation_importance.md)

#### `cond_permutation_importance(result: Any, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_repeats: int = 10, random_state: int | None = None) -> CondPermutationImportanceResult`

Strobl (2008) conditional permutation importance (bias-corrected for correlated features).

Returns `CondPermutationImportanceResult`: `.feature_names_`, `.importances_mean_`, `.importances_std_`, `.method`, `.n_repeats`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.ols_fit(X, y)
imp = mf.functions.cond_permutation_importance(fit_result, X, y, n_repeats=10)
print(imp.importances_mean_)
```

[Encyclopedia](../encyclopedia/l7/op/permutation_importance_strobl.md)

## Post-hoc importance (4 ops)

#### `ale_importance(result: Any, X: np.ndarray | pd.DataFrame, *, n_bins: int = 20) -> ALEImportanceResult`

Accumulated local effects (Apley-Zhu 2020) L1 importance.

Returns `ALEImportanceResult`: `.ale_values_`, `.feature_names_`, `.importances_`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.ols_fit(X, y)
imp = mf.functions.ale_importance(fit_result, X, n_bins=20)
print(imp.importances_)
```

[Encyclopedia](../encyclopedia/l7/op/accumulated_local_effect.md)

#### `partial_dependence_importance(result: Any, X: np.ndarray | pd.DataFrame, *, grid_resolution: int = 20) -> PDPImportanceResult`

Friedman (2001) partial dependence L1 importance.

Returns `PDPImportanceResult`: `.feature_names_`, `.grid_values_`, `.importances_`, `.pdp_values_`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.ols_fit(X, y)
imp = mf.functions.partial_dependence_importance(fit_result, X, grid_resolution=20)
print(imp.importances_)
```

[Encyclopedia](../encyclopedia/l7/op/partial_dependence.md)

#### `shap_linear_importance(result: Any, X: np.ndarray | pd.DataFrame) -> SHAPImportanceResult`

SHAP LinearExplainer importance (no extra deps for linear models).

Returns `SHAPImportanceResult`: `.expected_value_`, `.explainer_type`, `.feature_names_`, `.shap_values_`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.ols_fit(X, y)
imp = mf.functions.shap_linear_importance(fit_result, X)
print(imp.shap_values_.shape)
```

[Encyclopedia](../encyclopedia/l7/op/shap_linear.md)

#### `shap_tree_importance(result: Any, X: np.ndarray | pd.DataFrame) -> SHAPImportanceResult`

SHAP TreeExplainer importance (requires `pip install macroforecast[shap]`).

Returns `SHAPImportanceResult`: `.expected_value_`, `.explainer_type`, `.feature_names_`, `.shap_values_`, `.summary()`.

```python
rng = np.random.default_rng(7)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
fit_result = mf.functions.random_forest_fit(X, y, n_estimators=20)
imp = mf.functions.shap_tree_importance(fit_result, X)
# Requires: pip install macroforecast[shap]
```

[Encyclopedia](../encyclopedia/l7/op/shap_tree.md)

## Quick example

```python
import macroforecast as mf
import numpy as np

rng = np.random.default_rng(7)
X = rng.standard_normal((120, 8))
y = X[:, 1] * 3 + X[:, 4] - X[:, 6] * 0.5 + rng.standard_normal(120)

result = mf.functions.ridge_fit(X, y, alpha=1.0)
imp = mf.functions.permutation_importance(result, X, y)
print(imp.summary(top_n=5))
```
