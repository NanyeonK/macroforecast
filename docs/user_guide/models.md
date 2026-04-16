# Models

macrocast supports 24 model families organized by category. This guide helps you choose the right model for your forecasting task.

## Model selection guide

**Start here:**
- Few predictors (target lags only)? Use **AR** or **OLS**
- Many predictors, want regularization? Use **Ridge**, **Lasso**, or **ElasticNet**
- Want sparsity (variable selection)? Use **Lasso**, **AdaptiveLasso**, or **ComponentwiseBoosting**
- Want nonlinear relationships? Use **RandomForest**, **XGBoost**, or **LightGBM**
- Want robustness to outliers? Use **Huber** or **SVR**
- Want factor-based dimension reduction? Use **PCR**, **PLS**, or **FactorAugmentedLinear**
- Want a neural network baseline? Use **MLP**

## Benchmark / naive

### `ar` — Autoregressive model with BIC lag selection
- Fits AR(p) with p selected by BIC from 1 to max_lag
- Feature builder: `autoreg_lagged_target` only
- The standard naive benchmark in macro forecasting literature
- Reference: Stock & Watson (1999)

## Linear / regularized

### `ols` — Ordinary Least Squares
- No regularization. Use only when p << n (few predictors relative to observations)
- Feature builder: both `autoreg_lagged_target` and `raw_feature_panel`
- Key HP: none (closed-form solution)

### `ridge` — Ridge Regression (L2 penalty)
- Shrinks all coefficients toward zero. Never sets them exactly to zero.
- Good default for data-rich macro panels (FRED-MD has ~130 variables)
- Key HP: `alpha` (regularization strength, log-uniform [1e-4, 1e4])
- Importance: `linear_coefficients`, `linear_shap`

### `lasso` — Lasso Regression (L1 penalty)
- Produces sparse solutions: some coefficients are exactly zero (variable selection)
- Key HP: `alpha` (regularization strength, log-uniform [1e-6, 1e2])
- Importance: `linear_coefficients` (nonzero = selected variables)

### `elasticnet` — Elastic Net (L1 + L2 penalty)
- Combines Ridge and Lasso. Useful when you want some sparsity but also grouping.
- Key HP: `alpha`, `l1_ratio` (0 = Ridge, 1 = Lasso)

### `bayesianridge` — Bayesian Ridge Regression
- Automatic regularization via prior distributions on coefficients
- No HP tuning needed (learns regularization from data)
- Provides coefficient uncertainty estimates

### `huber` — Huber Regression
- Robust to outliers: uses Huber loss (L2 near zero, L1 for large errors)
- Key HP: `epsilon` (transition point between L2 and L1), `alpha`
- Use when FRED-MD target has extreme observations (e.g., COVID period)

### `adaptivelasso` — Adaptive Lasso
- Two-stage: (1) Ridge initial fit, (2) weighted Lasso with penalty w_j = 1/|beta_init_j|^gamma
- More consistent variable selection than standard Lasso (Zou 2006)
- Key HP: `gamma` (weight exponent), `init_estimator` (ridge or ols)

### `quantile_linear` — Quantile Regression
- Predicts conditional quantile (e.g., median) instead of conditional mean
- Requires `forecast_object: point_median` in recipe
- Key HP: `quantile`, `alpha`

## Kernel / margin

### `svr_linear` — Support Vector Regression (linear kernel)
- Linear SVR with epsilon-insensitive loss. Sparse solution in dual space.
- Has `.coef_` attribute: supports `linear_coefficients` importance
- Key HP: `C` (regularization), `epsilon` (tube width)

### `svr_rbf` — Support Vector Regression (RBF kernel)
- Nonlinear SVR with Gaussian RBF kernel. Can capture complex patterns.
- No `.coef_`: use `permutation_importance` or `kernel_shap`
- Key HP: `C`, `epsilon`, `gamma` (kernel width)
- Warning: O(n^2-n^3) complexity; may be slow for very large datasets

## Linear boosting

### `componentwise_boosting` — Componentwise L2 Boosting
- Each iteration selects ONE variable and fits a simple regression on it
- Implicit variable selection through early stopping
- Popular in macro forecasting: Bai & Ng (2009), Buchen & Wohlrabe (2011)
- Key HP: `n_iterations`, `learning_rate`
- Importance: selection frequency (how often each variable is chosen)

### `boosting_ridge` — Boosted Ridge
- Each iteration fits a full Ridge regression on residuals with shrinkage
- Accumulates coefficients across iterations
- Key HP: `n_iterations`, `learning_rate`, `ridge_alpha`

### `boosting_lasso` — Boosted Lasso
- Same as Boosted Ridge but with Lasso per iteration. Produces sparse updates.
- Key HP: `n_iterations`, `learning_rate`, `lasso_alpha`

## Factor-based linear

### `pcr` — Principal Component Regression
- PCA on X -> select k components -> OLS on components
- Dimension reduction built into the model. Classic diffusion index approach.
- Feature builder: `factors_plus_AR` or `factor_pca`
- Key HP: `n_components`

### `pls` — Partial Least Squares
- Supervised dimension reduction: finds components that maximize covariance with y
- Often outperforms PCR when only a few factors are relevant
- Key HP: `n_components`

### `factor_augmented_linear` — Factor-Augmented Linear Model
- Stock & Watson (2002) diffusion index forecasting
- PCA factors + target AR lags -> Ridge or OLS
- Feature builder: `factors_plus_AR` (includes AR lags alongside factors)
- Key HP: `n_components`

## Tree / ensemble

### `randomforest` — Random Forest
- Bagged ensemble of decision trees with random feature subsets
- Key HP: `n_estimators`, `max_depth`
- Importance: `tree_shap`, `RF_Gini_importance`, `permutation_importance`

### `extratrees` — Extremely Randomized Trees
- Like Random Forest but with random split thresholds. Lower variance.
- Key HP: `n_estimators`, `max_depth`

### `gbm` — Gradient Boosting Machine
- Sequential boosting of decision trees. sklearn GradientBoostingRegressor.
- Key HP: `n_estimators`, `learning_rate`, `max_depth`

### `xgboost` — XGBoost
- Optimized gradient boosting with regularization. Industry standard.
- Key HP: `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- Importance: `tree_shap` (exact, fast), `permutation_importance`
- Optional dependency: `pip install xgboost`

### `lightgbm` — LightGBM
- Gradient boosting with leaf-wise growth and histogram binning. Fast on large data.
- Key HP: `n_estimators`, `learning_rate`, `num_leaves`
- Optional dependency: `pip install lightgbm`

### `catboost` — CatBoost
- Gradient boosting with ordered boosting and native categorical support.
- Key HP: `iterations`, `learning_rate`, `depth`
- Optional dependency: `pip install catboost`

## Neural

### `mlp` — Multi-Layer Perceptron
- sklearn MLPRegressor. Simple feedforward neural network.
- Key HP: `hidden_layer_sizes`, `alpha` (L2 penalty), `learning_rate_init`
- Note: convergence warnings are common with small datasets. Not a problem for prediction quality.

## Feature builder compatibility

| Feature builder | Compatible models |
|----------------|------------------|
| `autoreg_lagged_target` | All 24 models (target lags only) |
| `raw_feature_panel` | All except `ar` (full X panel + imputation) |
| `raw_X_only` | All except `ar` (X panel without target lags) |
| `factors_plus_AR` | `pcr`, `pls`, `factor_augmented_linear` (PCA factors + AR lags) |
| `factor_pca` | `pcr`, `pls`, `factor_augmented_linear` (PCA factors only) |

## Importance method compatibility

| Model family | Supported importance methods |
|-------------|----------------------------|
| Linear (ridge, lasso, elasticnet, ...) | `linear_shap`, `linear_coefficients`, `permutation_importance`, `lime`, `pdp`, `ice`, `ale` |
| Tree (rf, xgboost, lightgbm, ...) | `tree_shap`, `permutation_importance`, `lime`, `pdp`, `ice`, `ale` |
| SVR (rbf) | `kernel_shap`, `permutation_importance`, `lime`, `pdp`, `ice`, `ale` |
| MLP | `kernel_shap`, `permutation_importance`, `lime`, `pdp`, `ice`, `ale` |

**See also:**
- [Mathematical Background: Evaluation Metrics](../math/evaluation_metrics.md) — metric formulas
- [User Guide: Tuning](tuning.md) — HP optimization for each model
- [Example: Basic Benchmark](../examples/basic_benchmark.md) — model comparison in action
- [API Reference: Execution](../api/execution.md) — executor function signatures
