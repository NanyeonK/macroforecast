# Standalone functions — L4 fit

L4 is the forecasting model layer. In the standalone paradigm each family is
accessible as:

```python
mf.functions.<name>_fit(X, y, **kwargs) -> <Name>FitResult
```

A `FitResult` is a frozen dataclass with:
- `.coef_` (or family-specific attributes, e.g. `.feature_importances_`)
- `.predict(X)` — generate point forecasts
- `.summary()` — human-readable parameter + fit summary

> **Cycle 22 POC** — `mf.functions.ridge_fit` is the only L4 standalone
> callable currently shipped. `RidgeFitResult` is the concrete return type.
> Other families below are planned for subsequent cycles.

## Linear family (8 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `ols_fit` | Ordinary least squares | [family axis](../encyclopedia/l4/axes/family.md#ols) |
| `ridge_fit` | L2-regularised OLS (Cycle 22 POC — **currently shipped**) | [ridge op page](../encyclopedia/l4/family/ridge.md) |
| `lasso_fit` | L1-regularised OLS (coordinate descent) | [family axis](../encyclopedia/l4/axes/family.md#lasso) |
| `elastic_net_fit` | Elastic net (L1 + L2) | [family axis](../encyclopedia/l4/axes/family.md#elastic-net) |
| `lasso_path_fit` | Lasso path over regularisation grid | [family axis](../encyclopedia/l4/axes/family.md#lasso-path) |
| `bayesian_ridge_fit` | Bayesian ridge (ARD) | [family axis](../encyclopedia/l4/axes/family.md#bayesian-ridge) |
| `huber_fit` | Huber regression (robust to outliers) | [family axis](../encyclopedia/l4/axes/family.md#huber) |
| `glmboost_fit` | GLM boosting via gradient descent on linear predictor | [family axis](../encyclopedia/l4/axes/family.md#glmboost) |

## Tree / boosting family (6 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `decision_tree_fit` | Single CART decision tree | [family axis](../encyclopedia/l4/axes/family.md#decision-tree) |
| `random_forest_fit` | Bootstrap-aggregated trees | [family axis](../encyclopedia/l4/axes/family.md#random-forest) |
| `extra_trees_fit` | Extremely randomised trees | [family axis](../encyclopedia/l4/axes/family.md#extra-trees) |
| `gradient_boosting_fit` | sklearn gradient boosting | [family axis](../encyclopedia/l4/axes/family.md#gradient-boosting) |
| `xgboost_fit` | XGBoost gradient boosting (optional extra) | [family axis](../encyclopedia/l4/axes/family.md#xgboost) |
| `lightgbm_fit` | LightGBM gradient boosting (optional extra) | [family axis](../encyclopedia/l4/axes/family.md#lightgbm) |

## Deep / neural family (4 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `mlp_fit` | Multi-layer perceptron (sklearn) | [family axis](../encyclopedia/l4/axes/family.md#mlp) |
| `lstm_fit` | LSTM (requires `macroforecast[deep]`) | [family axis](../encyclopedia/l4/axes/family.md#lstm) |
| `gru_fit` | GRU (requires `macroforecast[deep]`) | [family axis](../encyclopedia/l4/axes/family.md#gru) |
| `transformer_fit` | Transformer (requires `macroforecast[deep]`) | [family axis](../encyclopedia/l4/axes/family.md#transformer) |

## Time-series family (14 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `ar_p_fit` | Autoregressive model AR(p) | [family axis](../encyclopedia/l4/axes/family.md#ar-p) |
| `var_fit` | Vector autoregression VAR | [family axis](../encyclopedia/l4/axes/family.md#var) |
| `factor_augmented_ar_fit` | Factor-Augmented AR (Stock-Watson) | [family axis](../encyclopedia/l4/axes/family.md#factor-augmented-ar) |
| `bvar_minnesota_fit` | Bayesian VAR with Minnesota prior (Litterman 1986) | [family axis](../encyclopedia/l4/axes/family.md#bvar-minnesota) |
| `bvar_normal_inverse_wishart_fit` | Bayesian VAR with NIW prior | [family axis](../encyclopedia/l4/axes/family.md#bvar-normal-inverse-wishart) |
| `factor_augmented_var_fit` | FAVAR (Bernanke-Boivin-Eliasz 2005) | [family axis](../encyclopedia/l4/axes/family.md#factor-augmented-var) |
| `dfm_mixed_mariano_murasawa_fit` | Mixed-frequency DFM (Mariano-Murasawa MQ Kalman) | [family axis](../encyclopedia/l4/axes/family.md#dfm-mixed-mariano-murasawa) |
| `macroeconomic_random_forest_fit` | MRF GTVP (Coulombe 2024) | [family axis](../encyclopedia/l4/axes/family.md#macroeconomic-random-forest) |
| `ets_fit` | Exponential smoothing (ETS) | [family axis](../encyclopedia/l4/axes/family.md#ets) |
| `theta_method_fit` | Theta method (Assimakopoulos-Nikolopoulos 2000) | [family axis](../encyclopedia/l4/axes/family.md#theta-method) |
| `holt_winters_fit` | Holt-Winters triple exponential smoothing | [family axis](../encyclopedia/l4/axes/family.md#holt-winters) |
| `garch11_fit` | GARCH(1,1) volatility model | [family axis](../encyclopedia/l4/axes/family.md#garch11) |
| `egarch_fit` | EGARCH(1,1) asymmetric volatility | [family axis](../encyclopedia/l4/axes/family.md#egarch) |
| `realized_garch_with_rv_exog_fit` | Realized GARCH with RV exogenous variable | [family axis](../encyclopedia/l4/axes/family.md#realized-garch-with-rv-exog) |

## Misc family (6 ops)

| Op | One-liner | Encyclopedia |
|---|---|---|
| `svr_linear_fit` | Support vector regression (linear kernel) | [family axis](../encyclopedia/l4/axes/family.md#svr-linear) |
| `svr_rbf_fit` | Support vector regression (RBF kernel) | [family axis](../encyclopedia/l4/axes/family.md#svr-rbf) |
| `svr_poly_fit` | Support vector regression (polynomial kernel) | [family axis](../encyclopedia/l4/axes/family.md#svr-poly) |
| `knn_fit` | k-nearest neighbours regression | [family axis](../encyclopedia/l4/axes/family.md#knn) |
| `quantile_regression_forest_fit` | Quantile Regression Forest (Meinshausen 2006) | [family axis](../encyclopedia/l4/axes/family.md#quantile-regression-forest) |
| `bagging_fit` | Bootstrap-aggregated wrapper + quantile bands | [family axis](../encyclopedia/l4/axes/family.md#bagging) |

## Quick example (ridge — currently shipped)

```python
import macroforecast as mf
import numpy as np

rng = np.random.RandomState(42)
X = rng.randn(100, 5)
y = X @ np.array([1, 2, 3, 4, 5]) + 0.5 * rng.randn(100)

result = mf.functions.ridge_fit(X, y, alpha=1.0)
print(result.summary())
print(result.coef_)
preds = result.predict(X)
```

## Related

- [L5 metrics](l5_metrics.md) — evaluate the fit result.
- [L7 importance](l7_importance.md) — interpret the fit result.
- [Encyclopedia L4 family axis](../encyclopedia/l4/axes/family.md) — full
  per-family reference.
- [Encyclopedia ridge op page](../encyclopedia/l4/family/ridge.md) — detailed
  ridge documentation with sub-axis options (prior / coefficient_constraint /
  vol_model).
