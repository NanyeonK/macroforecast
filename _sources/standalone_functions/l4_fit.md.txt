# Standalone functions: L4 fit (38 ops)

L4 fit callables train a forecasting model on `(X, y)` and return a frozen result dataclass. Each result exposes `.predict(X)` and `.summary()`. Coefficients, importance scores, and training metadata are type-specific (see each Returns line below).

All signatures take `X: np.ndarray | pd.DataFrame` and `y: np.ndarray | pd.Series` as the first two positional arguments.

## Linear models (8 ops)

#### `ols_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> OLSFitResult`

Ordinary least squares regression - closed-form (X'X)^(-1) X'y.

Returns `OLSFitResult`: `.coef_`, `.intercept_`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.ols_fit(X, y)
print(result.summary())
y_pred = result.predict(X)
```

[Encyclopedia](../encyclopedia/l4/family/ols.md)

#### `ridge_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, alpha: float = 1.0, prior: "Literal[none, random_walk, shrink_to_target, fused_difference]" = none, coefficient_constraint: "Literal[none, nonneg]" = none, vol_model: "Literal[ewma, garch11] | None" = None, random_state: int | None = None) -> RidgeFitResult`

Ridge regression with optional prior, sign constraint, and volatility model.

Returns `RidgeFitResult`: `.alpha`, `.coef_`, `.intercept_`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.ridge_fit(X, y, alpha=1.0)
print(result.summary())
y_pred = result.predict(X)
```

[Encyclopedia](../encyclopedia/l4/family/ridge.md)

#### `lasso_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, alpha: float = 1.0, max_iter: int = 20000) -> LassoFitResult`

Lasso regression (L1-penalised) at a single alpha.

Returns `LassoFitResult`: `.alpha`, `.coef_`, `.intercept_`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.lasso_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/lasso.md)

#### `lasso_path_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, cv: int = 5, max_iter: int = 20000, random_state: int | None = None) -> LassoPathFitResult`

Lasso path with cross-validated alpha selection.

Returns `LassoPathFitResult`: `.alpha_selected`, `.coef_`, `.intercept_`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.lasso_path_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/lasso_path.md)

#### `elastic_net_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 20000) -> ElasticNetFitResult`

Elastic-net regression (L1 + L2 mix).

Returns `ElasticNetFitResult`: `.alpha`, `.coef_`, `.intercept_`, `.l1_ratio`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.elastic_net_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/elastic_net.md)

#### `bayesian_ridge_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> BayesianRidgeFitResult`

Bayesian ridge with ARD-style alpha/lambda estimation.

Returns `BayesianRidgeFitResult`: `.alpha_`, `.coef_`, `.intercept_`, `.lambda_`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.bayesian_ridge_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/bayesian_ridge.md)

#### `huber_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, epsilon: float = 1.35, max_iter: int = 1000) -> HuberFitResult`

Huber-robust regression (Huber loss with epsilon threshold).

Returns `HuberFitResult`: `.coef_`, `.epsilon`, `.intercept_`, `.predict()`, `.scale_`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.huber_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/huber.md)

#### `kernel_ridge_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, alpha: float = 1.0, kernel: str = rbf, gamma: float | None = None) -> KernelRidgeFitResult`

Kernel ridge regression (rbf default; gamma=None auto-scales).

Returns `KernelRidgeFitResult`: `.alpha`, `.kernel`, `.n_features_in_`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.kernel_ridge_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/kernel_ridge.md)

## Regularized linear and kernel (1 op)

#### `pcr_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_components: int = 3) -> PCRFitResult`

Principal-component regression (PCA + OLS).

Returns `PCRFitResult`: `.coef_`, `.n_components`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.pcr_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/principal_component_regression.md)

## Tree and ensemble models (7 ops)

#### `random_forest_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 0, n_jobs: int = 1) -> RandomForestFitResult`

Random-forest regression (Breiman 2001) with feature_importances_.

Returns `RandomForestFitResult`: `.feature_importances_`, `.n_estimators_used`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.random_forest_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/random_forest.md)

#### `extra_trees_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 0, n_jobs: int = 1) -> ExtraTreesFitResult`

Extremely randomised trees regressor (Geurts et al. 2006).

Returns `ExtraTreesFitResult`: `.feature_importances_`, `.n_estimators_used`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.extra_trees_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/extra_trees.md)

#### `gradient_boosting_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 3, random_state: int = 0) -> GradientBoostingFitResult`

Gradient boosting regression trees (Friedman 2001).

Returns `GradientBoostingFitResult`: `.feature_importances_`, `.n_estimators_used`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.gradient_boosting_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/gradient_boosting.md)

#### `glmboost_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_iter: int = 100, learning_rate: float = 0.1) -> GLMBoostFitResult`

Componentwise L2-boosting with linear base learners (kwarg is `n_iter`).

Returns `GLMBoostFitResult`: `.coef_`, `.intercept_`, `.learning_rate`, `.n_iter`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.glmboost_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/glmboost.md)

#### `xgboost_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_estimators: int = 300, learning_rate: float = 0.1, max_depth: int = 6, subsample: float = 1.0, random_state: int = 0) -> XGBoostFitResult`

XGBoost regressor (Chen and Guestrin 2016) - requires `macroforecast[xgboost]`.

Returns `XGBoostFitResult`: `.feature_importances_`, `.n_estimators_used`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.xgboost_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/xgboost.md)

#### `lightgbm_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_estimators: int = 300, learning_rate: float = 0.1, max_depth: int = -1, num_leaves: int = 31, random_state: int = 0) -> LightGBMFitResult`

LightGBM regressor (Ke et al. 2017) - requires `macroforecast[lightgbm]`.

Returns `LightGBMFitResult`: `.feature_importances_`, `.n_estimators_used`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.lightgbm_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/lightgbm.md)

#### `catboost_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_estimators: int = 300, learning_rate: float = 0.1, max_depth: int = 6, random_state: int = 0) -> CatBoostFitResult`

CatBoost regressor (Prokhorenkova et al. 2018) - requires `macroforecast[catboost]`.

Returns `CatBoostFitResult`: `.feature_importances_`, `.n_estimators_used`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.catboost_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/catboost.md)

## Neural networks (4 ops)

#### `mlp_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, hidden_layer_sizes: tuple = (32, 16), max_iter: int = 500, random_state: int = 0) -> MLPFitResult`

Multi-layer perceptron regressor (sklearn MLPRegressor).

Returns `MLPFitResult`: `.epochs_used`, `.final_loss`, `.hidden_layer_sizes`, `.n_features_in_`, `.n_params`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.mlp_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/mlp.md)

#### `lstm_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, hidden_size: int = 32, n_epochs: int = 50, random_state: int = 0) -> LSTMFitResult`

LSTM regressor (PyTorch nn.LSTM).

Returns `LSTMFitResult`: `.epochs_used`, `.final_loss`, `.hidden_size`, `.n_features_in_`, `.n_params`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.lstm_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/lstm.md)

#### `gru_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, hidden_size: int = 32, n_epochs: int = 50, random_state: int = 0) -> GRUFitResult`

GRU regressor (PyTorch nn.GRU).

Returns `GRUFitResult`: `.epochs_used`, `.final_loss`, `.hidden_size`, `.n_features_in_`, `.n_params`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.gru_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/gru.md)

#### `transformer_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, hidden_size: int = 32, n_epochs: int = 50, random_state: int = 0) -> TransformerFitResult`

Transformer encoder regressor (PyTorch nn.TransformerEncoder).

Returns `TransformerFitResult`: `.epochs_used`, `.final_loss`, `.hidden_size`, `.n_features_in_`, `.n_params`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.transformer_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/transformer.md)

## Time-series and factor models (12 ops)

#### `ar_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_lag: int = 1) -> ARFitResult`

Autoregressive model AR(n_lag).

Returns `ARFitResult`: `.coef_`, `.intercept_`, `.n_lag`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.ar_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/ar_p.md)

#### `var_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_lag: int = 1) -> VARFitResult`

Vector autoregression VAR(n_lag).

Returns `VARFitResult`: `.n_lag`, `.n_obs`, `.n_series`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.var_fit(X, y, n_lag=2)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/var.md)

#### `far_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_factors: int = 3, n_lag: int = 1) -> FARFitResult`

Factor-augmented autoregression (FAR).

Returns `FARFitResult`: `.coef_`, `.n_factors`, `.n_lag`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.far_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/factor_augmented_ar.md)

#### `favar_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_factors: int = 3, n_lag: int = 1) -> FAVARFitResult`

Factor-augmented VAR (FAVAR, Bernanke-Boivin-Eliasz 2005).

Returns `FAVARFitResult`: `.n_factors`, `.n_lag`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.favar_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/factor_augmented_var.md)

#### `bvar_minnesota_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_lag: int = 1, lambda1: float = 0.2) -> BVARMinnesotaFitResult`

Bayesian VAR with Minnesota prior (lambda1 controls shrinkage).

Returns `BVARMinnesotaFitResult`: `.lambda1`, `.n_lag`, `.n_obs`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.bvar_minnesota_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/bvar_minnesota.md)

#### `bvar_niw_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_lag: int = 1, lambda1: float = 0.2) -> BVARNIWFitResult`

Bayesian VAR with normal-inverse-Wishart prior.

Returns `BVARNIWFitResult`: `.lambda1`, `.n_lag`, `.n_obs`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.bvar_niw_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/bvar_normal_inverse_wishart.md)

#### `ets_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ETSFitResult`

ETS state-space exponential smoothing.

Returns `ETSFitResult`: `.error_trend_seasonal`, `.n_obs`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.ets_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/ets.md)

#### `holt_winters_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> HoltWintersFitResult`

Holt-Winters seasonal exponential smoothing.

Returns `HoltWintersFitResult`: `.n_obs`, `.predict()`, `.seasonal`, `.seasonal_periods`, `.summary()`, `.trend`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.holt_winters_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/holt_winters.md)

#### `theta_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> ThetaFitResult`

Theta forecasting method (Assimakopoulos and Nikolopoulos 2000).

Returns `ThetaFitResult`: `.alpha_`, `.n_obs`, `.predict()`, `.summary()`, `.theta`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.theta_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/theta_method.md)

#### `dfm_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_factors: int = 3) -> DFMFitResult`

Dynamic factor model (state-space EM).

Returns `DFMFitResult`: `.mode_`, `.n_factors`, `.n_obs`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.dfm_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/dfm_mixed_mariano_murasawa.md)

#### `garch11_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> GARCH11FitResult`

GARCH(1,1) conditional variance model - requires `macroforecast[arch]`.

Returns `GARCH11FitResult`: `.conditional_mu`, `.n_obs`, `.params_`, `.predict()`, `.predict_variance`, `.summary()`, `.variant`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.garch11_fit(X, y)
# Requires optional extra (see description above)
```

[Encyclopedia](../encyclopedia/l4/family/garch11.md)

#### `egarch_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> EGARCHFitResult`

EGARCH(1,1) asymmetric volatility model - requires `macroforecast[arch]`.

Returns `EGARCHFitResult`: `.conditional_mu`, `.n_obs`, `.params_`, `.predict()`, `.predict_variance`, `.summary()`, `.variant`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.egarch_fit(X, y)
# Requires optional extra (see description above)
```

[Encyclopedia](../encyclopedia/l4/family/egarch.md)

## Special models (6 ops)

#### `realized_garch_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, rv: np.ndarray | pd.Series) -> RealizedGARCHFitResult`

Realized GARCH (Hansen-Huang-Shek 2012); takes (X, y, rv) - requires `macroforecast[arch]`.

Returns `RealizedGARCHFitResult`: `.conditional_mu`, `.n_obs`, `.params_`, `.predict()`, `.predict_variance`, `.summary()`, `.variant`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
rv = np.abs(rng.standard_normal(80))
result = mf.functions.realized_garch_fit(X, y, rv)
# Requires: pip install macroforecast[arch]
```

[Encyclopedia](../encyclopedia/l4/family/realized_garch_with_rv_exog.md)

#### `mars_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> MARSFitResult`

Multivariate Adaptive Regression Splines (Friedman 1991) - requires `macroforecast[mars]`.

Returns `MARSFitResult`: `.n_features_in_`, `.n_terms`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.mars_fit(X, y)
# Requires optional extra (see description above)
```

[Encyclopedia](../encyclopedia/l4/family/mars.md)

#### `knn_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, n_neighbors: int = 5) -> KNNFitResult`

k-nearest-neighbours regression.

Returns `KNNFitResult`: `.n_features_in_`, `.n_neighbors`, `.n_neighbors_used`, `.predict()`, `.summary()`, `.weights`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.knn_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/knn.md)

#### `svr_linear_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, C: float = 1.0) -> SVRFitResult`

Support-vector regression with linear kernel.

Returns `SVRFitResult`: `.C`, `.degree`, `.gamma`, `.kernel`, `.n_support_vectors`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.svr_linear_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/svr_linear.md)

#### `svr_poly_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, C: float = 1.0, degree: int = 3) -> SVRFitResult`

Support-vector regression with polynomial kernel.

Returns `SVRFitResult`: `.C`, `.degree`, `.gamma`, `.kernel`, `.n_support_vectors`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.svr_poly_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/svr_poly.md)

#### `svr_rbf_fit(X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series, *, C: float = 1.0, gamma: str | float = scale) -> SVRFitResult`

Support-vector regression with RBF kernel (gamma='scale').

Returns `SVRFitResult`: `.C`, `.degree`, `.gamma`, `.kernel`, `.n_support_vectors`, `.predict()`, `.summary()`.

```python
rng = np.random.default_rng(0)
X = rng.standard_normal((80, 5))
y = rng.standard_normal(80)
result = mf.functions.svr_rbf_fit(X, y)
print(result.summary())
```

[Encyclopedia](../encyclopedia/l4/family/svr_rbf.md)

## Quick example

```python
import macroforecast as mf
import numpy as np

rng = np.random.default_rng(0)
X = rng.standard_normal((80, 10))
y = X[:, 0] + rng.standard_normal(80) * 0.3

result = mf.functions.ridge_fit(X, y, alpha=0.5)
print(result.summary())
y_pred = result.predict(X)
```
