# macroforecast.models

[Back to reference](index.md)

Direct callable model fits plus the model registry used by selection and pipeline runners.

Guide context: [../guide/model_overview.md](../guide/model_overview.md).

## Public Symbols

| Symbol | Kind | Summary |
| --- | --- | --- |
| `GARCHEstimator` | class | GARCH/EGARCH wrapper around the optional `arch` package. |
| `LGBAPlusRegressor` | class | Alternating LGB^A+ estimator. |
| `LGBPlusRegressor` | class | Competition-based LGB+ estimator. |
| `MODEL_SPECS` | data | dict() -> new empty dictionary |
| `MacroRandomForestRegressor` | class | Adapter for the vendored MacroRandomForest reference implementation. |
| `MARSRegressor` | class | Package-native multivariate adaptive regression splines estimator. |
| `ModelFit` | class | Fitted model wrapper returned by macroforecast model callables. |
| `ModelParameter` | class | One model-owned parameter description. |
| `ModelSpec` | class | Callable model plus model-owned defaults and hyperparameter spaces. |
| `QuantileRegressionForestRegressor` | class | Random-forest point forecasts plus empirical leaf quantiles. |
| `RealizedGARCHEstimator` | class | Compact Hansen-Huang-Shek-style realized GARCH joint MLE. |
| `SavedModel` | class | Paths and status returned by model fit persistence helpers. |
| `ScaledPCARegressor` | class | Huang et al. scaled PCA factor extraction with a linear forecast head. |
| `SupervisedAggregationRegressor` | class | Constrained supervised aggregation estimator. |
| `SupervisedPCARegressor` | class | Original-style SPCA with iterative screening, PCA, and projection. |
| `SupervisedScaledPCARegressor` | class | Hounyo-Li supervised scaled PCA: predictive-slope scaling plus SPCA. |
| `TVPRidgeRegressor` | class | Time-varying parameter ridge / 2SRR estimator. |
| `VolatilityFit` | class | Fitted volatility model wrapper. |
| `adaptive_elastic_net` | function | Fit adaptive elastic net using initial coefficient-based column weights. |
| `adaptive_lasso` | function | Fit adaptive lasso using initial coefficient-based penalty weights. |
| `albacore_components` | function | Fit the inflation-specific component-space Albacore wrapper. |
| `albacore_ranks` | function | Fit the inflation-specific rank-space Albacore wrapper. |
| `ar` | function | Fit a fixed-order AR(``n_lag``) by OLS. |
| `naive` | function | Random-walk (naive) forecaster: carry the last observed value forward. |
| `seasonal_naive` | function | Seasonal-naive forecaster: repeat the last full seasonal cycle. |
| `random_walk_drift` | function | Random-walk-with-drift forecaster. |
| `stlf` | function | STL + forecast: seasonally adjust, forecast, re-seasonalize (R forecast::stlf). |
| `assemblage_regression` | function | Fit the generic assemblage regression family. |
| `bayesian_ridge` | function | Fit empirical-Bayes Bayesian ridge regression. |
| `arima` | function | Fit a (seasonal) ARIMA model via statsmodels for target-only forecasts. |
| `auto_arima` | function | Automatic (seasonal) ARIMA order selection (forecast::auto.arima). |
| `bvar_minnesota` | function | Fit a FAVAR::BVAR-style Bayesian VAR with Minnesota prior variances. |
| `bvar_normal_inverse_wishart` | function | Fit a FAVAR::BVAR-style Bayesian VAR with normal-Wishart priors. |
| `catboost` | function | Fit a CatBoost regressor. Requires the `catboost` extra. |
| `custom_model` | function | Build a user-owned ``ModelSpec`` without registering a package model. |
| `component_aggregation` | function | Fit component-space supervised aggregation weights. |
| `dfm_mixed_mariano_murasawa` | function | Fit a monthly/quarterly DynamicFactorMQ model with MM aggregation. |
| `dfm_unrestricted_midas` | function | Fit DFM factors plus unrestricted MIDAS lag coefficients. |
| `decision_tree` | function | Fit a CART regression tree. |
| `describe_model` | function | Describe model-owned parameters and preset spaces. |
| `density_hnn` | function | Fit the paper-faithful Density Hemisphere neural-network forecaster. |
| `egarch` | function | Fit EGARCH. |
| `elastic_net` | function | Fit elastic net regression. |
| `ets` | function | Fit statsmodels ETSModel for target-only forecasts. |
| `extra_trees` | function | Fit an extremely randomized trees regressor. |
| `far` | function | Fit factor-augmented autoregression. |
| `favar` | function | Fit a FAVAR::FAVAR-aligned Bayesian factor-augmented VAR. |
| `fused_difference_ridge` | function | Fit ridge regression with adjacent-coefficient smoothness. |
| `garch11` | function | Fit GARCH(p, q), default GARCH(1, 1). |
| `gjr_garch` | function | Fit GJR-GARCH(p, o, q) (Glosten-Jagannathan-Runkle), default GJR-GARCH(1, 1, 1). |
| `tgarch` | function | Fit Threshold GARCH (TGARCH / Zakoian), default order (1, 1, 1). |
| `get_model` | function | Return a model spec by name, callable, or existing spec. |
| `glmboost` | function | Fit componentwise linear boosting. |
| `gradient_boosting` | function | Fit sklearn gradient-boosted regression trees. |
| `group_lasso` | function | Fit group lasso with one group label per predictor. |
| `gru` | function | Fit a torch-backed GRU regressor. Requires ``macroforecast[deep]``. |
| `hemisphere_nn` | function | Fit a torch-backed Hemisphere neural network mean/variance forecaster. |
| `huber` | function | Fit robust Huber regression. |
| `holt_winters` | function | Fit Holt-Winters exponential smoothing for target-only forecasts. |
| `kernel_ridge` | function | Fit sklearn kernel ridge regression. |
| `knn` | function | Fit sklearn k-nearest-neighbor regression. |
| `lasso` | function | Fit lasso regression with a user-supplied alpha. |
| `lightgbm` | function | Fit a LightGBM regressor. Requires the `lightgbm` extra. |
| `linear_svr` | function | Fit linear support-vector regression. |
| `lgba_plus` | function | Fit alternating LGB^A+ hybrid tree/linear boosting. |
| `lgb_plus` | function | Fit competition-based LGB+ hybrid tree/linear boosting. |
| `list_model_specs` | function | List registered model specs. |
| `load_fit` | function | Load a fitted model object saved by `save_fit()`. |
| `lstm` | function | Fit a torch-backed LSTM regressor. Requires ``macroforecast[deep]``. |
| `macro_random_forest` | function | Fit Macroeconomic Random Forest with the vendored reference backend. |
| `mars` | function | Fit package-native multivariate adaptive regression splines. |
| `midas_almon` | function | Fit linear MIDAS using Almon-polynomial lag weights. |
| `midas_beta` | function | Fit linear MIDAS using normalized beta lag weights. |
| `midas_step` | function | Fit linear MIDAS using equal step-function lag buckets. |
| `restricted_midas` | function | Fit a midasr::midas_r-style nonlinear restricted MIDAS regression. |
| `nn` | function | Fit a torch-backed feed-forward neural-network regressor. |
| `model_search_space` | function | Return a model-owned hyperparameter space. |
| `nonneg_ridge` | function | Fit ridge regression with non-negative coefficients. |
| `nu_svr` | function | Fit nu-support-vector regression. |
| `ols` | function | Fit ordinary least squares. |
| `pls` | function | Fit partial least squares regression. |
| `quantile_regression_forest` | function | Fit a quantile regression forest. |
| `rank_aggregation` | function | Fit rank-space supervised aggregation weights. |
| `random_walk_ridge` | function | Fit a random-walk coefficient ridge model and predict with the final coefficient. |
| `random_forest` | function | Fit a random forest regressor. |
| `realized_garch` | function | Fit realized GARCH with a realized-measurement equation. |
| `risk_forecast` | function | Value-at-Risk and Expected Shortfall forecast from a fitted volatility model. |
| `value_at_risk` | function | Lower-tail Value-at-Risk return quantile(s); see :func:`risk_forecast`. |
| `news_impact_curve` | function | Engle-Ng (1993) news impact curve for a fitted GARCH-family model. |
| `garch_roll` | function | Rolling 1-step volatility / Value-at-Risk backtest (rugarch::ugarchroll). |
| `expected_shortfall` | function | Expected Shortfall (mean return below VaR); see :func:`risk_forecast`. |
| `ridge` | function | Fit ridge regression. |
| `save_fit` | function | Persist a fitted model object and a JSON metadata sidecar. |
| `scaled_pca` | function | Fit Huang et al. scaled PCA with a linear forecast head. |
| `shrink_to_target_ridge` | function | Fit ridge regression with a coefficient prior target. |
| `solve_fused_difference_ridge` | function | Return nonnegative fused-difference weights for rank aggregation. |
| `solve_mean_aligned_ridge` | function | Return nonnegative weights constrained to match target mean. |
| `solve_nonnegative_ridge` | function | Return nonnegative ridge weights from the assemblage solver primitive. |
| `solve_simplex_ridge` | function | Return nonnegative sum-to-one ridge weights. |
| `solve_target_shrinkage_ridge` | function | Return weights for Albacore-style target-shrinkage ridge. |
| `sparse_group_lasso` | function | Fit sparse group lasso with group and feature-level sparsity. |
| `supervised_aggregation` | function | Fit a generic supervised component-to-aggregate weighting model. |
| `supervised_pca` | function | Fit original-style supervised PCA regression. |
| `supervised_scaled_pca` | function | Fit Hounyo-Li supervised scaled PCA regression. |
| `svr` | function | Fit support-vector regression. |
| `theta_method` | function | Fit the Theta forecasting method for target-only forecasts. |
| `transformer` | function | Fit a torch-backed Transformer encoder regressor. Requires ``macroforecast[deep]``. |
| `tvp_ridge` | function | Fit Goulet Coulombe TVP ridge / 2SRR as a macroforecast model. |
| `unrestricted_midas` | function | Fit unrestricted MIDAS over an explicit lag matrix. |
| `var` | function | Fit a vector autoregression on a multivariate panel. |
| `var_select_order` | function | Select the VAR lag order by information criteria (vars::VARselect). |
| `var_roots` | function | VAR stability: moduli of the companion-matrix eigenvalues (vars::roots). |
| `var_restrict` | function | Restricted VAR by sequential elimination of regressors (R vars::restrict). |
| `xgboost` | function | Fit an XGBoost regressor. Requires the `xgboost` extra. |

## Model Registry

These rows come from `macroforecast.models.MODEL_SPECS` / `list_model_specs()`.

| Model | Family | Input kind | Default preset | Requires extra | Requires scaling | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `adaptive_elastic_net` | `linear` | `supervised` | `standard` | none | no | Adaptive elastic net using initial coefficient-based column weights. |
| `adaptive_lasso` | `linear` | `supervised` | `standard` | none | no | Adaptive lasso using initial coefficient-based penalty weights. |
| `albacore_components` | `assemblage` | `supervised` | `standard` | none | no | Inflation-specific component-space Albacore wrapper. |
| `albacore_ranks` | `assemblage` | `supervised` | `standard` | none | no | Inflation-specific rank-space Albacore wrapper. |
| `ar` | `timeseries` | `supervised` | `standard` | none | no | Univariate autoregression. |
| `arima` | `timeseries` | `target` | `standard` | none | no | (Seasonal) ARIMA model. |
| `assemblage_regression` | `assemblage` | `supervised` | `standard` | none | no | Generic assemblage regression wrapper with component and rank variants. |
| `auto_arima` | `timeseries` | `target` | `standard` | none | no | Automatic (seasonal) ARIMA order selection (forecast::auto.arima). |
| `bayesian_ridge` | `linear` | `supervised` | `standard` | none | no | Empirical-Bayes Bayesian ridge. |
| `bvar_minnesota` | `timeseries` | `panel` | `standard` | none | no | FAVAR::BVAR / bvartools Minnesota-prior Bayesian VAR posterior sampler. |
| `bvar_normal_inverse_wishart` | `timeseries` | `panel` | `standard` | none | no | FAVAR::BVAR-aligned Bayesian VAR with normal/inverse-Wishart prior controls. |
| `catboost` | `tree` | `supervised` | `standard` | `catboost` | no | CatBoost regressor. |
| `component_aggregation` | `assemblage` | `supervised` | `standard` | none | no | Component-space supervised aggregation; generic Albacorecomps primitive. |
| `decision_tree` | `tree` | `supervised` | `standard` | none | no | CART regression tree. |
| `density_hnn` | `neural` | `supervised` | `standard` | `deep` | no | Paper-faithful Density Hemisphere neural network with prior-DNN OOB volatility emphasis and OOB volatility rescaling. |
| `dfm_mixed_mariano_murasawa` | `mixed_frequency` | `panel` | `standard` | none | no | Mixed-frequency dynamic factor model using Mariano-Murasawa quarterly aggregation. |
| `dfm_unrestricted_midas` | `mixed_frequency` | `panel` | `standard` | none | no | Composite DynamicFactorMQ factors plus unrestricted MIDAS forecast head. |
| `egarch` | `volatility` | `volatility` | `standard` | `arch` | no | EGARCH volatility model. |
| `elastic_net` | `linear` | `supervised` | `standard` | none | no | Elastic net regression. |
| `ets` | `timeseries` | `target` | `standard` | none | no | Statsmodels ETS target-only forecasting model. |
| `extra_trees` | `tree` | `supervised` | `standard` | none | no | Extremely randomized trees. |
| `far` | `factor` | `supervised` | `standard` | none | no | Factor-augmented autoregression. |
| `favar` | `factor` | `supervised` | `standard` | none | no | FAVAR::FAVAR-aligned Bayesian factor-augmented VAR sampler. |
| `fused_difference_ridge` | `linear` | `supervised` | `standard` | none | no | Ridge regression with a fused-difference coefficient prior. |
| `garch11` | `volatility` | `volatility` | `standard` | `arch` | no | GARCH volatility model. |
| `gjr_garch` | `volatility` | `volatility` | `standard` | `arch` | no | GJR-GARCH asymmetric volatility model. |
| `glmboost` | `linear` | `supervised` | `standard` | none | no | Componentwise linear boosting. |
| `gradient_boosting` | `tree` | `supervised` | `standard` | none | no | Gradient-boosted regression trees. |
| `group_lasso` | `linear` | `supervised` | `standard` | none | no | Package-native group lasso with group-level sparsity. |
| `gru` | `neural` | `supervised` | `standard` | `deep` | no | Torch-backed GRU regressor. |
| `hemisphere_nn` | `neural` | `supervised` | `standard` | `deep` | no | Bagged Hemisphere neural network with mean and variance heads. |
| `holt_winters` | `timeseries` | `target` | `standard` | none | no | Holt-Winters exponential smoothing target-only forecasting model. |
| `huber` | `linear` | `supervised` | `standard` | none | no | Robust Huber regression. |
| `kernel_ridge` | `nonparametric` | `supervised` | `standard` | none | yes | Kernel ridge regression. |
| `knn` | `nonparametric` | `supervised` | `standard` | none | yes | K-nearest-neighbor regression. |
| `lasso` | `linear` | `supervised` | `standard` | none | no | Lasso regression. |
| `lgb_plus` | `tree` | `supervised` | `standard` | `lightgbm` | no | LGB+ competition hybrid boosting with tree/linear channel diagnostics. |
| `lgba_plus` | `tree` | `supervised` | `standard` | `lightgbm` | no | LGB^A+ alternating tree-block and greedy linear boosting. |
| `lightgbm` | `tree` | `supervised` | `standard` | `lightgbm` | no | LightGBM regressor. |
| `linear_svr` | `support_vector` | `supervised` | `standard` | none | yes | Linear support-vector regression. |
| `lstm` | `neural` | `supervised` | `standard` | `deep` | no | Torch-backed LSTM regressor. |
| `macro_random_forest` | `tree` | `supervised` | `standard` | `macro_random_forest` | no | Adapter for the external MacroRandomForest package. |
| `mars` | `spline` | `supervised` | `standard` | none | no | Package-native MARS-style hinge-basis regression. |
| `midas_almon` | `mixed_frequency` | `supervised` | `standard` | none | no | Fixed-shape MIDAS over lag groups using midasr::nealmon-style normalized exponential Almon weights. |
| `midas_beta` | `mixed_frequency` | `supervised` | `standard` | none | no | Fixed-shape MIDAS over lag groups using midasr::nbetaMT-style beta weights. |
| `midas_step` | `mixed_frequency` | `supervised` | `standard` | none | no | Fixed-shape MIDAS over lag groups using normalized midasr::polystep-style step weights. |
| `naive` | `timeseries` | `target` | `standard` | none | no | Random-walk (naive) baseline: carry the last value forward (forecast::naive). |
| `nn` | `neural` | `supervised` | `standard` | `deep` | no | Torch-backed feed-forward multilayer perceptron regressor. |
| `nonneg_ridge` | `linear` | `supervised` | `standard` | none | no | Ridge regression with non-negative coefficients. |
| `nu_svr` | `support_vector` | `supervised` | `standard` | none | yes | Nu support-vector regression. |
| `ols` | `linear` | `supervised` | `standard` | none | no | Ordinary least squares with no model-owned tuning space. |
| `pls` | `composite` | `supervised` | `standard` | none | no | Partial least squares regression with optional Hounyo-Li-style control residualization. |
| `quantile_regression_forest` | `tree` | `supervised` | `standard` | none | no | Quantile regression forest. |
| `random_forest` | `tree` | `supervised` | `standard` | none | no | Random forest regression. |
| `random_walk_drift` | `timeseries` | `target` | `standard` | none | no | Random-walk-with-drift baseline (forecast::rwf(drift=TRUE)). |
| `random_walk_ridge` | `linear` | `supervised` | `standard` | none | no | Time-varying random-walk ridge fit, predicting with the final coefficient vector. |
| `rank_aggregation` | `assemblage` | `supervised` | `standard` | none | no | Rank-space supervised aggregation; generic Albacoreranks primitive. |
| `realized_garch` | `volatility` | `volatility` | `standard` | none | no | Compact realized GARCH volatility model. |
| `restricted_midas` | `mixed_frequency` | `supervised` | `standard` | none | no | midasr::midas_r-style nonlinear restricted MIDAS over explicit lag columns. |
| `ridge` | `linear` | `supervised` | `standard` | none | no | Ridge regression. |
| `scaled_pca` | `composite` | `supervised` | `standard` | none | no | Huang et al. scaled PCA: marginal predictive-slope scaling followed by PCA. |
| `seasonal_naive` | `timeseries` | `target` | `standard` | none | no | Seasonal-naive baseline: repeat the last seasonal cycle (forecast::snaive). |
| `shrink_to_target_ridge` | `linear` | `supervised` | `standard` | none | no | Ridge regression shrinking coefficients toward a target vector. |
| `sparse_group_lasso` | `linear` | `supervised` | `standard` | none | no | Package-native sparse group lasso with group and feature-level sparsity. |
| `stlf` | `timeseries` | `target` | `standard` | none | no | STL decomposition + forecast of the seasonally-adjusted series (forecast::stlf). |
| `supervised_aggregation` | `assemblage` | `supervised` | `standard` | none | no | Generic constrained supervised aggregation derived from Albacore/assemblage primitives. |
| `supervised_pca` | `composite` | `supervised` | `standard` | none | no | Original-style iterative supervised PCA with residual correlation screening and projection. |
| `supervised_scaled_pca` | `composite` | `supervised` | `standard` | none | no | Hounyo-Li supervised scaled PCA: marginal predictive-slope scaling followed by SPCA. |
| `svr` | `support_vector` | `supervised` | `standard` | none | yes | Kernel support-vector regression. |
| `tgarch` | `volatility` | `volatility` | `standard` | `arch` | no | Threshold GARCH (TGARCH/Zakoian) volatility model. |
| `theta_method` | `timeseries` | `target` | `standard` | none | no | Theta method target-only forecasting model. |
| `transformer` | `neural` | `supervised` | `standard` | `deep` | no | Torch-backed Transformer encoder regressor. |
| `tvp_ridge` | `linear` | `supervised` | `standard` | none | no | Goulet Coulombe TVP ridge / 2SRR estimator. |
| `unrestricted_midas` | `mixed_frequency` | `supervised` | `standard` | none | no | Unrestricted MIDAS over explicit lag columns. |
| `var` | `timeseries` | `panel` | `standard` | none | no | R vars::VAR-aligned vector autoregression point forecast. |
| `xgboost` | `tree` | `supervised` | `standard` | `xgboost` | no | XGBoost regressor. |

## Registered Model Details

### adaptive_elastic_net

Family: `linear`

#### Fit Signature

```python
macroforecast.models.adaptive_elastic_net(X: Any, y: Any | None = None, *, alpha: float = 1.0, l1_ratio: float = 0.5, gamma: float = 1.0, initial: str = "ridge", initial_alpha: float = 1.0, eps: float = 0.0001, normalize_weights: bool = True, max_iter: int = 20000, tol: float = 0.0001, random_state: int | None = None) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Adaptive elastic net using initial coefficient-based column weights.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Final adaptive elastic-net penalty strength. |
| `l1_ratio` | `0.5` | `float` | True | L1 share of the final elastic-net penalty. |
| `gamma` | `1.0` | `float` | True | Exponent applied to initial coefficient weights. |
| `initial` | `"ridge"` | `str` | False | Initial model: 'ridge' or 'ols'. |
| `initial_alpha` | `1.0` | `float` | False | Initial ridge penalty. |
| `eps` | `0.0001` | `float` | False | Small denominator floor for adaptive weights. |
| `normalize_weights` | `True` | `bool` | False | Rescale adaptive penalty weights to mean one, matching glmnet penalty.factor scaling. |
| `max_iter` | `20000` | `int` | False | Final solver iteration cap. |
| `tol` | `0.0001` | `float` | False | Final solver convergence tolerance. |
| `random_state` | `None` | `int | None` | False | Final solver random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)`, `gamma`: `(1.0,)`, `l1_ratio`: `(0.25, 0.5, 0.75)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)`, `gamma`: `(0.5, 1.0, 2.0)`, `l1_ratio`: `(0.1, 0.25, 0.5, 0.75, 0.9)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)`, `gamma`: `(0.5, 1.0, 1.5, 2.0)`, `l1_ratio`: `(0.05, 0.1, 0.25, 0.5, 0.75, 0.9)` |

### adaptive_lasso

Family: `linear`

#### Fit Signature

```python
macroforecast.models.adaptive_lasso(X: Any, y: Any | None = None, *, alpha: float = 1.0, gamma: float = 1.0, initial: str = "ridge", initial_alpha: float = 1.0, eps: float = 0.0001, normalize_weights: bool = True, max_iter: int = 20000, tol: float = 0.0001, random_state: int | None = None) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Adaptive lasso using initial coefficient-based penalty weights.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Final adaptive lasso penalty strength. |
| `gamma` | `1.0` | `float` | True | Exponent applied to initial coefficient weights. |
| `initial` | `"ridge"` | `str` | False | Initial model: 'ridge' or 'ols'. |
| `initial_alpha` | `1.0` | `float` | False | Initial ridge penalty. |
| `eps` | `0.0001` | `float` | False | Small denominator floor for adaptive weights. |
| `normalize_weights` | `True` | `bool` | False | Rescale adaptive penalty weights to mean one, matching glmnet penalty.factor scaling. |
| `max_iter` | `20000` | `int` | False | Final solver iteration cap. |
| `tol` | `0.0001` | `float` | False | Final solver convergence tolerance. |
| `random_state` | `None` | `int | None` | False | Final solver random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)`, `gamma`: `(1.0,)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)`, `gamma`: `(0.5, 1.0, 2.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)`, `gamma`: `(0.5, 1.0, 1.5, 2.0)` |

### albacore_components

Family: `assemblage`

#### Fit Signature

```python
macroforecast.models.albacore_components(X: Any, y: Any | None = None, *, reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Inflation-specific component-space Albacore wrapper.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `reference_weights` | `None` | `mapping | sequence | None` | False | Official or reference basket weights. |
| `alpha` | `1.0` | `float` | True | Target-shrinkage penalty strength. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### albacore_ranks

Family: `assemblage`

#### Fit Signature

```python
macroforecast.models.albacore_ranks(X: Any, y: Any | None = None, *, alpha: float = 1.0, difference_order: int = 1, max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Inflation-specific rank-space Albacore wrapper.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Fused-difference penalty strength. |
| `difference_order` | `1` | `int` | False | Finite-difference order for rank weights. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### ar

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.ar(X: Any, y: Any | None = None, *, n_lag: int = 1, direct: bool = False) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Univariate autoregression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_lag` | `1` | `int` | True | Autoregressive lag order. |
| `direct` | `False` | `bool` | False | Direct multi-step projection onto fresh lags (set by the forecast policy). |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_lag`: `(1, 2, 4)` |
| `standard` | `n_lag`: `(1, 2, 4, 6, 12)` |
| `wide` | `n_lag`: `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

### arima

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.arima(y: Any, *, order: tuple[int, int, int] = (1, 0, 0), seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0), trend: str | None = None) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

(Seasonal) ARIMA model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `order` | `(1, 0, 0)` | `tuple[int, int, int]` | True | ARIMA (p, d, q) order. |
| `seasonal_order` | `(0, 0, 0, 0)` | `tuple[int, int, int, int]` | False | Seasonal (P, D, Q, m) order. |
| `trend` | `None` | `str | None` | False | Deterministic trend ('n','c','t','ct'). |

### assemblage_regression

Family: `assemblage`

#### Fit Signature

```python
macroforecast.models.assemblage_regression(X: Any, y: Any | None = None, *, space: AggregationSpace = "component", alpha: float = 1.0, reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None, penalty: AggregationPenalty | None = None, max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Generic assemblage regression wrapper with component and rank variants.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `space` | `"component"` | `component | rank` | False | Use component-space or rank-space aggregation. |
| `alpha` | `1.0` | `float` | True | Penalty strength. |
| `reference_weights` | `None` | `mapping | sequence | None` | False | Optional reference weights for component space. |
| `penalty` | `None` | `str | None` | False | Optional penalty override. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### auto_arima

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.auto_arima(y: Any, *, max_p: int = 5, max_q: int = 5, max_d: int = 2, seasonal: bool = False, m: int = 1, max_P: int = 1, max_Q: int = 1, seasonal_D: int = 0, ic: str = "aicc", trend: str | None = None) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Automatic (seasonal) ARIMA order selection (forecast::auto.arima).

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `max_p` | `5` | `int` | False | Maximum non-seasonal AR order. |
| `max_q` | `5` | `int` | False | Maximum non-seasonal MA order. |
| `max_d` | `2` | `int` | False | Maximum differencing order. |
| `seasonal` | `False` | `bool` | False | Search seasonal orders. |
| `m` | `1` | `int` | False | Seasonal period. |
| `ic` | `"aicc"` | `str` | False | Selection criterion ('aicc','aic','bic'). |

### bayesian_ridge

Family: `linear`

#### Fit Signature

```python
macroforecast.models.bayesian_ridge(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Empirical-Bayes Bayesian ridge.

### bvar_minnesota

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.bvar_minnesota(panel: Any, *, target: str | None = None, n_lag: int = 1, kappa0: float = 2.0, kappa1: float = 0.5, nu0: float = 0.0, s0: float | Sequence[Sequence[float]] | None = None, iter: int = 300, burnin: int = 100, random_state: int = 0, own_lag_prior_mean: float = 0.0) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `panel` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

FAVAR::BVAR / bvartools Minnesota-prior Bayesian VAR posterior sampler.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `target` | `None` | `str | None` | False | Target column in the panel. |
| `n_lag` | `1` | `int` | True | VAR lag order. |
| `kappa0` | `2.0` | `float` | True | FAVAR/bvartools Minnesota own-lag prior scale. |
| `kappa1` | `0.5` | `float` | True | FAVAR/bvartools Minnesota lag-decay exponent. |
| `nu0` | `0.0` | `float` | False | Inverse-Wishart degrees-of-freedom prior parameter. |
| `s0` | `None` | `float | matrix | None` | False | Inverse-Wishart scale prior parameter. None (default): data-dependent diag(AR/OLS residual variance) scale. |
| `iter` | `300` | `int` | False | Total Gibbs iterations (deep/paper-faithful default is 10000; pass explicitly to restore it). |
| `burnin` | `100` | `int` | False | Burn-in iterations discarded from posterior summaries (deep/paper-faithful default is 5000; pass explicitly to restore it). |
| `random_state` | `0` | `int` | False | Random seed for posterior draws. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `kappa0`: `(1.0, 2.0)`, `kappa1`: `(0.5,)`, `n_lag`: `(1, 2)` |
| `standard` | `kappa0`: `(0.5, 1.0, 2.0)`, `kappa1`: `(0.5, 1.0)`, `n_lag`: `(1, 2)` |
| `wide` | `kappa0`: `(0.25, 0.5, 1.0, 2.0)`, `kappa1`: `(0.5, 1.0, 2.0)`, `n_lag`: `(1, 2, 4, 6, 12)` |

### bvar_normal_inverse_wishart

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.bvar_normal_inverse_wishart(panel: Any, *, target: str | None = None, n_lag: int = 1, b0: float = 0.0, vb0: float = 0.0, nu0: float = 0.0, s0: float | Sequence[Sequence[float]] | None = None, iter: int = 300, burnin: int = 100, random_state: int = 0) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `panel` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

FAVAR::BVAR-aligned Bayesian VAR with normal/inverse-Wishart prior controls.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `target` | `None` | `str | None` | False | Target column in the panel. |
| `n_lag` | `1` | `int` | True | VAR lag order. |
| `b0` | `0.0` | `float` | False | Normal prior mean for VAR coefficients. |
| `vb0` | `0.0` | `float` | False | Normal prior variance scale for VAR coefficients. |
| `nu0` | `0.0` | `float` | False | Inverse-Wishart degrees-of-freedom prior parameter. |
| `s0` | `None` | `float | matrix | None` | False | Inverse-Wishart scale prior parameter. None (default): data-dependent diag(AR/OLS residual variance) scale. |
| `iter` | `300` | `int` | False | Total Gibbs iterations (deep/paper-faithful default is 10000; pass explicitly to restore it). |
| `burnin` | `100` | `int` | False | Burn-in iterations discarded from posterior summaries (deep/paper-faithful default is 5000; pass explicitly to restore it). |
| `random_state` | `0` | `int` | False | Random seed for posterior draws. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_lag`: `(1, 2)` |
| `standard` | `n_lag`: `(1, 2)` |
| `wide` | `n_lag`: `(1, 2, 4, 6, 12)` |

### catboost

Family: `tree`

#### Fit Signature

```python
macroforecast.models.catboost(X: Any, y: Any | None = None, *, n_estimators: int = 300, learning_rate: float = 0.1, max_depth: int = 6, random_state: int = 0, verbose: bool = False, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `catboost` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

CatBoost regressor.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_estimators` | `300` | `int` | True | Number of boosting stages. |
| `learning_rate` | `0.1` | `float` | True | Shrinkage per stage. |
| `max_depth` | `6` | `int` | True | Tree depth. |
| `random_state` | `0` | `int` | False | Boosting random seed. |
| `verbose` | `False` | `bool` | False | CatBoost console output flag. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.05, 0.1)`, `max_depth`: `(2, 3)`, `n_estimators`: `(50, 100)` |
| `standard` | `learning_rate`: `(0.03, 0.05, 0.1)`, `max_depth`: `(2, 3, 5)`, `n_estimators`: `(100, 200, 500)` |
| `wide` | `learning_rate`: `(0.01, 0.03, 0.05, 0.1)`, `max_depth`: `(2, 3, 5, 8)`, `n_estimators`: `(100, 200, 500, 1000)` |

### component_aggregation

Family: `assemblage`

#### Fit Signature

```python
macroforecast.models.component_aggregation(X: Any, y: Any | None = None, *, alpha: float = 1.0, reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None, penalty: AggregationPenalty | None = None, simplex: bool = True, nonneg: bool = True, penalty_scale: "Literal['none', 'feature_std']" = "feature_std", max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Component-space supervised aggregation; generic Albacorecomps primitive.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Penalty strength. |
| `reference_weights` | `None` | `mapping | sequence | None` | False | Optional reference component weights. |
| `penalty` | `None` | `ridge | target_shrinkage | None` | False | Penalty; None selects target_shrinkage when weights are supplied. |
| `simplex` | `True` | `bool` | False | Constrain weights to sum to one. |
| `nonneg` | `True` | `bool` | False | Constrain weights to be non-negative. |
| `penalty_scale` | `"feature_std"` | `none | feature_std` | False | Scale penalties by component standard deviations. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### decision_tree

Family: `tree`

#### Fit Signature

```python
macroforecast.models.decision_tree(X: Any, y: Any | None = None, *, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 0, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

CART regression tree.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `max_depth` | `None` | `int | None` | True | Maximum tree depth. |
| `min_samples_leaf` | `1` | `int` | True | Minimum samples per terminal leaf. |
| `random_state` | `0` | `int` | False | Tree random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `max_depth`: `(3, 5, None)`, `min_samples_leaf`: `(1, 3)` |
| `standard` | `max_depth`: `(3, 5, 10, None)`, `min_samples_leaf`: `(1, 3, 5)` |
| `wide` | `max_depth`: `(2, 3, 5, 10, 20, None)`, `min_samples_leaf`: `(1, 2, 3, 5, 10)` |

### density_hnn

Family: `neural`

#### Fit Signature

```python
macroforecast.models.density_hnn(X: Any, y: Any | None = None, *, common_layers: int = 2, mean_layers: int = 2, volatility_layers: int = 2, prior_layers: int = 3, neurons: int = 400, dropout: float = 0.2, learning_rate: float = 0.001, max_epochs: int = 40, n_estimators: int = 20, prior_estimators: int = 10, subsample: float = 0.8, block_size: int = 8, volatility_emphasis: float | None = None, rescale_volatility: bool = True, patience: int = 8, random_state: int = 0, device: TorchDevice = "auto", quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95), volatility_clip: float = 0.05) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `deep` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("feature lags/trends are built before fitting; X and y are standardized inside each fit",)` |

Paper-faithful Density Hemisphere neural network with prior-DNN OOB volatility emphasis and OOB volatility rescaling.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `common_layers` | `2` | `int` | False | Shared common-core depth. |
| `mean_layers` | `2` | `int` | False | Conditional-mean hemisphere depth. |
| `volatility_layers` | `2` | `int` | False | Conditional-volatility hemisphere depth. |
| `prior_layers` | `3` | `int` | False | Plain prior-DNN depth. |
| `neurons` | `400` | `int` | True | Hidden width used by all dense blocks. |
| `dropout` | `0.2` | `float` | False | Dropout rate. |
| `learning_rate` | `0.001` | `float` | True | Adam learning rate. |
| `max_epochs` | `40` | `int` | False | Training epoch cap. |
| `n_estimators` | `20` | `int` | True | Density-HNN bootstrap ensemble size. |
| `prior_estimators` | `10` | `int` | True | Prior-DNN bootstrap ensemble size used to estimate volatility emphasis. |
| `subsample` | `0.8` | `float` | False | Blocked bootstrap sampling rate. |
| `block_size` | `8` | `int` | False | Time-series bootstrap block size. |
| `volatility_emphasis` | `None` | `float | None` | False | Override for Aionx volatility-emphasis parameter; None estimates it from prior-DNN OOB MSE. |
| `rescale_volatility` | `True` | `bool` | False | Apply Aionx blocked-OOB log residual-square volatility recalibration. |
| `patience` | `8` | `int` | False | Early-stopping patience. |
| `random_state` | `0` | `int` | False | Random seed. |
| `device` | `"auto"` | `str` | False | Torch device: auto, cpu, or cuda. |
| `quantile_levels` | `(0.05, 0.5, 0.95)` | `tuple[float, ...]` | False | Default normal-approximation density quantiles. |
| `volatility_clip` | `0.05` | `float` | False | Minimum volatility in Gaussian negative log likelihood. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.001,)`, `n_estimators`: `(3, 5)`, `neurons`: `(16, 32)`, `prior_estimators`: `(2, 3)` |
| `standard` | `learning_rate`: `(0.0005, 0.001)`, `n_estimators`: `(3, 5)`, `neurons`: `(16, 32)`, `prior_estimators`: `(2, 3)` |
| `wide` | `learning_rate`: `(0.0001, 0.0005, 0.001)`, `n_estimators`: `(5, 10, 25)`, `neurons`: `(32, 64, 128)`, `prior_estimators`: `(3, 5, 10)` |

### dfm_mixed_mariano_murasawa

Family: `mixed_frequency`

#### Fit Signature

```python
macroforecast.models.dfm_mixed_mariano_murasawa(panel: Any, *, target: str | None = None, metadata: Mapping[str, Any] | None = None, monthly_columns: Iterable[str] | None = None, quarterly_columns: Iterable[str] | None = None, unsupported: str = "raise", n_factors: int = 1, factor_order: int = 1, idiosyncratic_ar1: bool = True, standardize: bool = True, maxiter: int = 500, tolerance: float = 1e-06) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `panel` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("pass a native mixed monthly/quarterly panel from macroforecast.data.combine(..., frequency='native')", "keep quarterly flow variables on their observed quarterly dates; the model applies Mariano-Murasawa aggregation")` |

Mixed-frequency dynamic factor model using Mariano-Murasawa quarterly aggregation.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `target` | `None` | `str | None` | False | Target column; defaults to first quarterly column. |
| `metadata` | `None` | `Mapping[str, Any] | None` | False | Data metadata with native frequencies. |
| `monthly_columns` | `None` | `Iterable[str] | None` | False | Explicit monthly columns. |
| `quarterly_columns` | `None` | `Iterable[str] | None` | False | Explicit quarterly columns. |
| `unsupported` | `"raise"` | `str` | False | Unsupported-frequency policy: 'raise' or 'drop'. |
| `n_factors` | `1` | `int` | True | Number of dynamic factors. |
| `factor_order` | `1` | `int` | True | VAR order for factor dynamics. |
| `idiosyncratic_ar1` | `True` | `bool` | False | Whether idiosyncratic disturbances are AR(1). |
| `standardize` | `True` | `bool` | False | Whether statsmodels standardizes observed series. |
| `maxiter` | `500` | `int` | False | EM iteration cap. |
| `tolerance` | `1e-06` | `float` | False | EM convergence tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `factor_order`: `(1,)`, `n_factors`: `(1,)` |
| `standard` | `factor_order`: `(1, 2)`, `n_factors`: `(1, 2)` |
| `wide` | `factor_order`: `(1, 2, 3)`, `n_factors`: `(1, 2, 3)` |

### dfm_unrestricted_midas

Family: `mixed_frequency`

#### Fit Signature

```python
macroforecast.models.dfm_unrestricted_midas(panel: Any, *, target: str, metadata: Mapping[str, Any] | None = None, lag_columns: Iterable[str] | None = None, lags: Iterable[int] | int = (0, 1, 2), factor_lags: Iterable[int] | int = (0,), target_frequency: str | None = "quarterly", anchor_position: str = "period_end", n_factors: int = 1, factor_order: int = 1, idiosyncratic_ar1: bool = True, standardize: bool = True, maxiter: int = 500, tolerance: float = 1e-06, alpha: float = 0.0, fit_intercept: bool = True, drop_missing: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `panel` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("pass a native mixed monthly/quarterly panel with column-level frequency metadata", "use feature_engineering.mixed_frequency_lags directly when you need full manual control")` |

Composite DynamicFactorMQ factors plus unrestricted MIDAS forecast head.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `target` | `None` | `str` | False | Target column. |
| `metadata` | `None` | `Mapping[str, Any] | None` | False | Data metadata with native frequencies. |
| `lag_columns` | `None` | `Iterable[str] | None` | False | Observed columns to add as unrestricted MIDAS lags. |
| `lags` | `(0, 1, 2)` | `Iterable[int] | int` | True | Observed-column native-frequency lags. |
| `factor_lags` | `(0,)` | `Iterable[int] | int` | True | DFM factor monthly lags. |
| `target_frequency` | `"quarterly"` | `str | None` | False | Frequency used to position target anchors. |
| `anchor_position` | `"period_end"` | `str` | False | Anchor date positioning. |
| `n_factors` | `1` | `int` | True | Number of DFM factors. |
| `factor_order` | `1` | `int` | True | VAR order for DFM factor dynamics. |
| `idiosyncratic_ar1` | `True` | `bool` | False | Whether DFM idiosyncratic disturbances are AR(1). |
| `standardize` | `True` | `bool` | False | Whether DynamicFactorMQ standardizes observed variables. |
| `maxiter` | `500` | `int` | False | DFM EM iteration cap. |
| `tolerance` | `1e-06` | `float` | False | DFM EM convergence tolerance. |
| `alpha` | `0.0` | `float` | True | Optional ridge penalty on unrestricted MIDAS head. |
| `fit_intercept` | `True` | `bool` | False | Whether to fit an intercept. |
| `drop_missing` | `True` | `bool` | False | Drop incomplete composite-design rows. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.0, 0.1, 1.0)`, `factor_lags`: `((0,),)`, `factor_order`: `(1,)`, `lags`: `((0, 1, 2),)`, `n_factors`: `(1,)` |
| `standard` | `alpha`: `(0.0, 0.01, 0.1, 1.0)`, `factor_lags`: `((0,), (0, 1))`, `factor_order`: `(1, 2)`, `lags`: `((0, 1, 2), (0, 1, 2, 3, 4, 5))`, `n_factors`: `(1, 2)` |
| `wide` | `alpha`: `(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)`, `factor_lags`: `((0,), (0, 1), (0, 1, 2))`, `factor_order`: `(1, 2, 3)`, `lags`: `((0, 1, 2), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))`, `n_factors`: `(1, 2, 3)` |

### egarch

Family: `volatility`

#### Fit Signature

```python
macroforecast.models.egarch(y: Any, *, X: Any | None = None, p: int = 1, o: int = 1, q: int = 1, mean_model: str = "constant", dist: str = "normal", rescale: bool = False, **kwargs: Any) -> VolatilityFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `volatility` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | `arch` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

EGARCH volatility model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `p` | `1` | `int` | True | EGARCH innovation lag order. |
| `o` | `1` | `int` | True | Asymmetric innovation lag order (leverage term). |
| `q` | `1` | `int` | True | EGARCH variance lag order. |
| `mean_model` | `"constant"` | `str` | False | Conditional mean model. |
| `dist` | `"normal"` | `str` | True | Innovation distribution. |
| `rescale` | `False` | `bool` | False | arch package rescale option. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `dist`: `("normal", "t")`, `o`: `(0, 1)`, `p`: `(1,)`, `q`: `(1,)` |
| `standard` | `dist`: `("normal", "t")`, `o`: `(0, 1)`, `p`: `(1, 2)`, `q`: `(1, 2)` |
| `wide` | `dist`: `("normal", "t", "skewt")`, `o`: `(0, 1, 2)`, `p`: `(1, 2, 3)`, `q`: `(1, 2, 3)` |

### elastic_net

Family: `linear`

#### Fit Signature

```python
macroforecast.models.elastic_net(X: Any, y: Any | None = None, *, alpha: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 20000, standardize: bool = False, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Elastic net regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Overall penalty strength. |
| `l1_ratio` | `0.5` | `float` | True | L1 share of the elastic-net penalty. |
| `max_iter` | `20000` | `int` | False | Optimization iteration cap. |
| `standardize` | `False` | `bool` | False | Standardize predictors inside the fitted estimator. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)`, `l1_ratio`: `(0.25, 0.5, 0.75)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)`, `l1_ratio`: `(0.1, 0.25, 0.5, 0.75, 0.9)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)`, `l1_ratio`: `(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)` |

### ets

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.ets(y: Any, *, error: str = "add", trend: str | None = None, seasonal: str | None = None, seasonal_periods: int | None = None, damped_trend: bool = False, model: str | None = None) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Statsmodels ETS target-only forecasting model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `error` | `"add"` | `str` | False | ETS error form. |
| `trend` | `None` | `str | None` | False | ETS trend form. |
| `seasonal` | `None` | `str | None` | False | ETS seasonal form. |
| `seasonal_periods` | `None` | `int | None` | False | Seasonal period. |
| `damped_trend` | `False` | `bool` | False | Whether to damp the trend. |

### extra_trees

Family: `tree`

#### Fit Signature

```python
macroforecast.models.extra_trees(X: Any, y: Any | None = None, *, n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 0, n_jobs: int | None = None, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Extremely randomized trees.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_estimators` | `200` | `int` | True | Number of trees. |
| `max_depth` | `None` | `int | None` | True | Maximum depth per tree. |
| `min_samples_leaf` | `1` | `int` | True | Minimum samples per terminal leaf. |
| `random_state` | `0` | `int` | False | Forest random seed. |
| `n_jobs` | `None` | `int | None` | False | Parallel worker count (None resolves to meta.configure(n_jobs)). |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `max_depth`: `(3, 5, None)`, `min_samples_leaf`: `(1, 3)`, `n_estimators`: `(50, 100)` |
| `standard` | `max_depth`: `(3, 5, 10, None)`, `min_samples_leaf`: `(1, 3, 5)`, `n_estimators`: `(100, 200, 500)` |
| `wide` | `max_depth`: `(3, 5, 10, 20, None)`, `min_samples_leaf`: `(1, 2, 3, 5, 10)`, `n_estimators`: `(100, 200, 500, 1000)` |

### far

Family: `factor`

#### Fit Signature

```python
macroforecast.models.far(X: Any, y: Any | None = None, *, n_factors: int = 3, n_lag: int = 1, random_state: int = 0, direct: bool = False) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Factor-augmented autoregression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_factors` | `3` | `int` | True | Number of PCA factors. |
| `n_lag` | `1` | `int` | True | Autoregressive lag order. |
| `random_state` | `0` | `int` | False | PCA random seed. |
| `direct` | `False` | `bool` | False | Direct multi-step projection onto fresh lags (set by the forecast policy). |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_factors`: `(1, 2, 3)`, `n_lag`: `(1, 2, 4)` |
| `standard` | `n_factors`: `(1, 2, 3, 5, 8)`, `n_lag`: `(1, 2, 4, 6, 12)` |
| `wide` | `n_factors`: `(1, 2, 3, 5, 8, 10, 12)`, `n_lag`: `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

### favar

Family: `factor`

#### Fit Signature

```python
macroforecast.models.favar(X: Any, y: Any | None = None, *, n_factors: int = 2, n_lag: int = 2, fctmethod: str = "BGM", slowcode: Sequence[bool] | None = None, factorprior: Mapping[str, Any] | None = None, varprior: Mapping[str, Any] | None = None, nburn: int = 100, nrep: int = 200, standardize: bool = True, random_state: int = 0) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

FAVAR::FAVAR-aligned Bayesian factor-augmented VAR sampler.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_factors` | `2` | `int` | True | Number of latent factors. |
| `n_lag` | `2` | `int` | True | VAR lag order on target plus factors. |
| `fctmethod` | `"BGM"` | `str` | False | FAVAR factor identification method: BBE or BGM (default BGM; BBE requires slowcode). |
| `slowcode` | `None` | `Sequence[bool] | None` | False | Slow-variable mask required by BBE. |
| `factorprior` | `None` | `Mapping[str, Any] | None` | False | Factor loading prior controls. |
| `varprior` | `None` | `Mapping[str, Any] | None` | False | BVAR prior controls for the factor VAR block. |
| `nburn` | `100` | `int` | False | Burn-in iterations for loading/BVAR posterior draws (deep/paper-faithful default is 5000; pass explicitly to restore it). |
| `nrep` | `200` | `int` | False | Saved loading draws and post-burn BVAR draw count (deep/paper-faithful default is 15000; pass explicitly to restore it). |
| `standardize` | `True` | `bool` | False | Use R scale() semantics for X and y before factor extraction. |
| `random_state` | `0` | `int` | False | Random seed for posterior draws. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_factors`: `(1, 2)`, `n_lag`: `(1, 2)` |
| `standard` | `n_factors`: `(1, 2)`, `n_lag`: `(1, 2)` |
| `wide` | `n_factors`: `(1, 2, 3, 5)`, `n_lag`: `(1, 2, 4, 6)` |

### fused_difference_ridge

Family: `linear`

#### Fit Signature

```python
macroforecast.models.fused_difference_ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, difference_order: int = 1, mean_equality: bool = False, nonneg: bool = False, fit_intercept: bool = True, max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Ridge regression with a fused-difference coefficient prior.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Strength of the adjacent-coefficient smoothness penalty. |
| `difference_order` | `1` | `int` | False | Finite-difference order applied to coefficients. |
| `mean_equality` | `False` | `bool` | False | Constrain fitted and observed sums to match. |
| `nonneg` | `False` | `bool` | False | Constrain coefficients to be non-negative. |
| `fit_intercept` | `True` | `bool` | False | Fit an intercept unless mean_equality=True. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### garch11

Family: `volatility`

#### Fit Signature

```python
macroforecast.models.garch11(y: Any, *, X: Any | None = None, p: int = 1, q: int = 1, mean_model: str = "constant", dist: str = "normal", rescale: bool = False, **kwargs: Any) -> VolatilityFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `volatility` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | `arch` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

GARCH volatility model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `p` | `1` | `int` | True | GARCH innovation lag order. |
| `q` | `1` | `int` | True | GARCH variance lag order. |
| `mean_model` | `"constant"` | `str` | False | Conditional mean model. |
| `dist` | `"normal"` | `str` | True | Innovation distribution. |
| `rescale` | `False` | `bool` | False | arch package rescale option. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `dist`: `("normal", "t")`, `p`: `(1,)`, `q`: `(1,)` |
| `standard` | `dist`: `("normal", "t")`, `p`: `(1, 2)`, `q`: `(1, 2)` |
| `wide` | `dist`: `("normal", "t", "skewt")`, `p`: `(1, 2, 3)`, `q`: `(1, 2, 3)` |

### gjr_garch

Family: `volatility`

#### Fit Signature

```python
macroforecast.models.gjr_garch(y: Any, *, X: Any | None = None, p: int = 1, o: int = 1, q: int = 1, mean_model: str = "constant", dist: str = "normal", rescale: bool = False, **kwargs: Any) -> VolatilityFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `volatility` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | `arch` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

GJR-GARCH asymmetric volatility model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `p` | `1` | `int` | True | GARCH innovation lag order. |
| `o` | `1` | `int` | True | Asymmetric (leverage) lag order. |
| `q` | `1` | `int` | True | GARCH variance lag order. |
| `mean_model` | `"constant"` | `str` | False | Conditional mean model. |
| `dist` | `"normal"` | `str` | True | Innovation distribution. |
| `rescale` | `False` | `bool` | False | arch package rescale option. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `dist`: `("normal", "t")`, `o`: `(1,)`, `p`: `(1,)`, `q`: `(1,)` |
| `standard` | `dist`: `("normal", "t")`, `o`: `(1, 2)`, `p`: `(1, 2)`, `q`: `(1, 2)` |
| `wide` | `dist`: `("normal", "t", "skewt")`, `o`: `(1, 2)`, `p`: `(1, 2, 3)`, `q`: `(1, 2, 3)` |

### glmboost

Family: `linear`

#### Fit Signature

```python
macroforecast.models.glmboost(X: Any, y: Any | None = None, *, n_iter: int = 100, learning_rate: float = 0.1, center: bool = True, candidate_sampling: str = "all", candidate_count: int | None = None, candidate_fraction: float | None = None, candidate_cap: int | None = None, candidate_min: int = 1, candidate_rounding: str = "floor", random_state: int | None = None) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Componentwise linear boosting.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_iter` | `100` | `int` | True | Number of boosting iterations. |
| `learning_rate` | `0.1` | `float` | True | Shrinkage applied to each componentwise update. |
| `center` | `True` | `bool` | False | Center predictors before componentwise updates, matching mboost's default. |
| `candidate_sampling` | `"all"` | `str` | False | Candidate-subset policy per boosting step: all or random. |
| `candidate_count` | `None` | `int | None` | False | Fixed candidate count when candidate_sampling='random'. |
| `candidate_fraction` | `None` | `float | None` | False | Candidate fraction when candidate_sampling='random'. |
| `candidate_cap` | `None` | `int | None` | False | Maximum sampled candidate count after resolving count/fraction. |
| `candidate_min` | `1` | `int` | False | Minimum sampled candidate count. |
| `candidate_rounding` | `"floor"` | `str` | False | Rounding rule for candidate_fraction: floor, ceil, or round. |
| `random_state` | `None` | `int | None` | False | Seed for per-step candidate feature sampling. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.05, 0.1)`, `n_iter`: `(50, 100)` |
| `standard` | `learning_rate`: `(0.01, 0.05, 0.1)`, `n_iter`: `(50, 100, 200, 500)` |
| `wide` | `learning_rate`: `(0.005, 0.01, 0.05, 0.1, 0.2)`, `n_iter`: `(50, 100, 200, 500, 1000)` |

### gradient_boosting

Family: `tree`

#### Fit Signature

```python
macroforecast.models.gradient_boosting(X: Any, y: Any | None = None, *, n_estimators: int = 200, learning_rate: float = 0.1, max_depth: int = 3, random_state: int = 0, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Gradient-boosted regression trees.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_estimators` | `200` | `int` | True | Number of boosting stages. |
| `learning_rate` | `0.1` | `float` | True | Shrinkage per stage. |
| `max_depth` | `3` | `int` | True | Maximum tree depth. |
| `random_state` | `0` | `int` | False | Boosting random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.05, 0.1)`, `max_depth`: `(2, 3)`, `n_estimators`: `(50, 100)` |
| `standard` | `learning_rate`: `(0.03, 0.05, 0.1)`, `max_depth`: `(2, 3, 5)`, `n_estimators`: `(100, 200, 500)` |
| `wide` | `learning_rate`: `(0.01, 0.03, 0.05, 0.1)`, `max_depth`: `(2, 3, 5, 8)`, `n_estimators`: `(100, 200, 500, 1000)` |

### group_lasso

Family: `linear`

#### Fit Signature

```python
macroforecast.models.group_lasso(X: Any, y: Any | None = None, *, groups: Sequence[str | int] | None = None, alpha: float = 1.0, group_weights: dict[str, float] | None = None, max_iter: int = 5000, tol: float = 1e-05, scale: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Package-native group lasso with group-level sparsity.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `groups` | `None` | `sequence[str | int] | None` | False | One group label per predictor. |
| `alpha` | `1.0` | `float` | True | Group penalty strength. |
| `group_weights` | `None` | `dict[str, float] | None` | False | Optional group penalty weights. |
| `max_iter` | `5000` | `int` | False | Proximal-gradient iteration cap. |
| `tol` | `1e-05` | `float` | False | Proximal-gradient convergence tolerance. |
| `scale` | `True` | `bool` | False | Whether to standardize predictors inside the model. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)` |

### gru

Family: `neural`

#### Fit Signature

```python
macroforecast.models.gru(X: Any, y: Any | None = None, *, sequence_length: int = 4, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0, learning_rate: float = 0.001, max_epochs: int = 100, batch_size: int = 32, random_state: int = 0, device: TorchDevice = "auto") -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `deep` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("handled internally: X and y are standardized inside each fit",)` |

Torch-backed GRU regressor.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `sequence_length` | `4` | `int` | True | Trailing rows per recurrent sequence. |
| `hidden_size` | `32` | `int` | True | Recurrent hidden-state width. |
| `num_layers` | `1` | `int` | False | Number of recurrent layers. |
| `dropout` | `0.0` | `float` | False | Dropout between recurrent layers. |
| `learning_rate` | `0.001` | `float` | True | Adam learning rate. |
| `max_epochs` | `100` | `int` | False | Training epoch cap. |
| `batch_size` | `32` | `int` | False | Mini-batch size. |
| `random_state` | `0` | `int` | False | Random seed. |
| `device` | `"auto"` | `str` | False | Torch device: auto, cpu, or cuda. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `hidden_size`: `(16, 32)`, `learning_rate`: `(0.001,)`, `sequence_length`: `(2, 4)` |
| `standard` | `hidden_size`: `(16, 32, 64)`, `learning_rate`: `(0.0005, 0.001)`, `sequence_length`: `(2, 4, 8)` |
| `wide` | `hidden_size`: `(16, 32, 64, 128)`, `learning_rate`: `(0.0001, 0.0005, 0.001, 0.005)`, `sequence_length`: `(2, 4, 8, 12)` |

### hemisphere_nn

Family: `neural`

#### Fit Signature

```python
macroforecast.models.hemisphere_nn(X: Any, y: Any | None = None, *, lc: int = 2, lm: int = 2, lv: int = 2, neurons: int = 64, dropout: float = 0.2, learning_rate: float = 0.001, max_epochs: int = 40, n_estimators: int = 20, subsample: float = 0.8, nu: float | None = None, variance_penalty: float = 1.0, patience: int = 8, validation_fraction: float = 0.2, random_state: int = 0, device: TorchDevice = "auto", lr: float | None = None, n_epochs: int | None = None, B: int | None = None, sub_rate: float | None = None, lambda_emphasis: float | None = None, val_frac: float | None = None, quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `deep` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("handled internally: X is standardized inside each fit",)` |

Bagged Hemisphere neural network with mean and variance heads.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `lc` | `2` | `int` | False | Shared common-core depth. |
| `lm` | `2` | `int` | False | Mean-head depth after the common core. |
| `lv` | `2` | `int` | False | Variance-head depth after the common core. |
| `neurons` | `64` | `int` | True | Hidden width for all dense layers. |
| `dropout` | `0.2` | `float` | False | Dropout rate. |
| `learning_rate` | `0.001` | `float` | True | Adam learning rate. |
| `max_epochs` | `40` | `int` | False | Training epoch cap. |
| `n_estimators` | `20` | `int` | True | Number of blocked-subsample bags. |
| `subsample` | `0.8` | `float` | False | Blocked-subsample fraction. |
| `nu` | `None` | `float | None` | False | Variance-emphasis target ratio. |
| `variance_penalty` | `1.0` | `float` | False | Soft penalty on the variance-emphasis target. |
| `patience` | `8` | `int` | False | Early-stopping patience. |
| `validation_fraction` | `0.2` | `float` | False | Chronological validation fraction. |
| `random_state` | `0` | `int` | False | Random seed. |
| `device` | `"auto"` | `str` | False | Torch device: auto, cpu, or cuda. |
| `quantile_levels` | `(0.05, 0.5, 0.95)` | `tuple[float, ...]` | False | Default normal-approximation density quantiles. |
| `lr` | `None` | `float | None` | False | Legacy alias for learning_rate. |
| `n_epochs` | `None` | `int | None` | False | Legacy alias for max_epochs. |
| `B` | `None` | `int | None` | False | Legacy alias for n_estimators. |
| `sub_rate` | `None` | `float | None` | False | Legacy alias for subsample. |
| `lambda_emphasis` | `None` | `float | None` | False | Legacy alias for variance_penalty. |
| `val_frac` | `None` | `float | None` | False | Legacy alias for validation_fraction. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.001,)`, `n_estimators`: `(3, 5)`, `neurons`: `(16, 32)` |
| `standard` | `learning_rate`: `(0.0005, 0.001)`, `n_estimators`: `(3, 5)`, `neurons`: `(16, 32)` |
| `wide` | `learning_rate`: `(0.0001, 0.0005, 0.001)`, `n_estimators`: `(5, 10, 25)`, `neurons`: `(32, 64, 128)` |

### holt_winters

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.holt_winters(y: Any, *, trend: str | None = "add", seasonal: str | None = None, seasonal_periods: int | None = None, damped_trend: bool = False) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Holt-Winters exponential smoothing target-only forecasting model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `trend` | `"add"` | `str | None` | False | Trend component. |
| `seasonal` | `None` | `str | None` | False | Seasonal component. |
| `seasonal_periods` | `None` | `int | None` | False | Seasonal period. |
| `damped_trend` | `False` | `bool` | False | Whether to damp the trend. |

### huber

Family: `linear`

#### Fit Signature

```python
macroforecast.models.huber(X: Any, y: Any | None = None, *, epsilon: float = 1.35, max_iter: int = 1000, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Robust Huber regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `epsilon` | `1.35` | `float` | True | Huber loss transition threshold. |
| `max_iter` | `1000` | `int` | False | Optimization iteration cap. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `epsilon`: `(1.1, 1.35, 1.75)` |
| `standard` | `epsilon`: `(1.1, 1.35, 1.5, 1.75, 2.0)` |
| `wide` | `epsilon`: `(1.01, 1.1, 1.35, 1.5, 1.75, 2.0, 2.5)` |

### kernel_ridge

Family: `nonparametric`

#### Fit Signature

```python
macroforecast.models.kernel_ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, kernel: str = "linear", gamma: float | None = None, degree: int = 3, coef0: float = 1.0, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | yes |
| `recommended_preprocessing` | `("standardize predictors before nonlinear kernels",)` |

Kernel ridge regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Ridge penalty strength. |
| `kernel` | `"linear"` | `str` | True | Kernel name: linear, rbf, polynomial, sigmoid, etc. |
| `gamma` | `None` | `float | None` | False | Kernel coefficient. |
| `degree` | `3` | `int` | False | Polynomial kernel degree. |
| `coef0` | `1.0` | `float` | False | Independent term for polynomial/sigmoid kernels. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.1, 1.0, 10.0)`, `kernel`: `("linear", "rbf")` |
| `standard` | `alpha`: `(0.01, 0.1, 1.0, 10.0)`, `gamma`: `(None, 0.01, 0.1)`, `kernel`: `("linear", "rbf", "poly")` |
| `wide` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)`, `degree`: `(2, 3, 4)`, `gamma`: `(None, 0.001, 0.01, 0.1, 1.0)`, `kernel`: `("linear", "rbf", "poly", "sigmoid")` |

### knn

Family: `nonparametric`

#### Fit Signature

```python
macroforecast.models.knn(X: Any, y: Any | None = None, *, n_neighbors: int = 5, weights: str = "uniform", metric: str = "minkowski", p: int = 2, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | yes |
| `recommended_preprocessing` | `("standardize predictors before distance-based fitting",)` |

K-nearest-neighbor regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_neighbors` | `5` | `int` | True | Number of nearest neighbors. |
| `weights` | `"uniform"` | `str` | True | Neighbor weighting: uniform or distance. |
| `metric` | `"minkowski"` | `str` | False | Distance metric. |
| `p` | `2` | `int` | False | Minkowski distance order. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_neighbors`: `(3, 5, 10)`, `weights`: `("uniform", "distance")` |
| `standard` | `n_neighbors`: `(3, 5, 10, 20)`, `p`: `(1, 2)`, `weights`: `("uniform", "distance")` |
| `wide` | `n_neighbors`: `(1, 3, 5, 10, 20, 40)`, `p`: `(1, 2)`, `weights`: `("uniform", "distance")` |

### lasso

Family: `linear`

#### Fit Signature

```python
macroforecast.models.lasso(X: Any, y: Any | None = None, *, alpha: float = 1.0, max_iter: int = 20000, standardize: bool = False, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Lasso regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | L1 penalty strength. |
| `max_iter` | `20000` | `int` | False | Optimization iteration cap. |
| `standardize` | `False` | `bool` | False | Standardize predictors inside the fitted estimator. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### lgb_plus

Family: `tree`

#### Fit Signature

```python
macroforecast.models.lgb_plus(X: Any, y: Any | None = None, *, n_ensemble: int = 3, n_steps: int = 30, learning_rate: float = 0.05, subsample: float = 0.7, num_leaves: int = 5, min_data_in_leaf: int = 20, lambda_l2: float = 0.1, linear_candidate_fraction: float = 0.5, selection_method: "Literal['oob', 'validation', 'training']" = "oob", val_fraction: float = 0.2, early_stop_patience: int | None = 50, aggregation: "Literal['mean', 'median']" = "mean", random_state: int | None = 0, verbose: bool = False, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `lightgbm` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

LGB+ competition hybrid boosting with tree/linear channel diagnostics.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_ensemble` | `3` | `int` | True | Independent LGB+ ensemble members (deep/paper-faithful default is 10; pass explicitly to restore it). |
| `n_steps` | `30` | `int` | True | Maximum tree/linear competition steps per member (deep/paper-faithful default is 200; pass explicitly to restore it). |
| `learning_rate` | `0.05` | `float` | True | Shared shrinkage for tree and linear updates. |
| `subsample` | `0.7` | `float` | True | Row subsample share per competition step. |
| `num_leaves` | `5` | `int` | True | Maximum leaves for each one-step LightGBM tree. |
| `min_data_in_leaf` | `20` | `int` | True | Minimum rows in a LightGBM leaf. |
| `lambda_l2` | `0.1` | `float` | False | LightGBM tree L2 regularization. |
| `linear_candidate_fraction` | `0.5` | `float` | True | Fraction of features sampled before greedy linear residual selection. |
| `selection_method` | `"oob"` | `str` | False | Candidate judge: 'oob', 'validation', or 'training'. |
| `val_fraction` | `0.2` | `float` | False | Fixed validation share when selection_method='validation'. |
| `early_stop_patience` | `50` | `int | None` | False | Stop after no selection-loss improvement. |
| `aggregation` | `"mean"` | `str` | False | Ensemble aggregation: 'mean' or 'median'. |
| `random_state` | `0` | `int | None` | False | Base random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.03, 0.05)`, `linear_candidate_fraction`: `(0.33, 0.5)`, `min_data_in_leaf`: `(10, 20)`, `n_ensemble`: `(3, 5)`, `n_steps`: `(50, 100)`, `num_leaves`: `(5, 7)`, `subsample`: `(0.6, 0.7)` |
| `standard` | `learning_rate`: `(0.02, 0.05, 0.1)`, `linear_candidate_fraction`: `(0.33, 0.5, 1.0)`, `min_data_in_leaf`: `(10, 20, 30)`, `n_ensemble`: `(3, 5)`, `n_steps`: `(30, 50, 100)`, `num_leaves`: `(5, 7, 10)`, `subsample`: `(0.6, 0.7, 0.8)` |
| `wide` | `learning_rate`: `(0.01, 0.02, 0.05, 0.1)`, `linear_candidate_fraction`: `(0.25, 0.33, 0.5, 1.0)`, `min_data_in_leaf`: `(5, 10, 20, 30)`, `n_ensemble`: `(5, 10, 20)`, `n_steps`: `(100, 200, 400, 600)`, `num_leaves`: `(5, 7, 10, 15)`, `subsample`: `(0.5, 0.6, 0.7, 0.8)` |

### lgba_plus

Family: `tree`

#### Fit Signature

```python
macroforecast.models.lgba_plus(X: Any, y: Any | None = None, *, n_runs: int = 1, n_cycles: int = 25, trees_per_cycle: int = 10, lr_tree: float = 0.02, lr_linear: float = 0.1, num_leaves: int = 15, min_data_in_leaf: int = 20, subsample: float = 1.0, random_state: int | None = 0, verbose: bool = False, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `lightgbm` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

LGB^A+ alternating tree-block and greedy linear boosting.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_runs` | `1` | `int` | True | Independent alternating-run ensemble members. |
| `n_cycles` | `25` | `int` | True | Alternating tree-block plus linear-update cycles. |
| `trees_per_cycle` | `10` | `int` | True | LightGBM residual trees per cycle. |
| `lr_tree` | `0.02` | `float` | True | Shrinkage for tree-block updates. |
| `lr_linear` | `0.1` | `float` | True | Shrinkage for univariate linear updates. |
| `num_leaves` | `15` | `int` | True | Maximum leaves for each residual tree. |
| `min_data_in_leaf` | `20` | `int` | True | Minimum rows in a LightGBM leaf. |
| `subsample` | `1.0` | `float` | False | LightGBM bagging fraction for tree blocks. |
| `random_state` | `0` | `int | None` | False | Base random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `lr_linear`: `(0.05, 0.1)`, `lr_tree`: `(0.02, 0.05)`, `min_data_in_leaf`: `(10, 20)`, `n_cycles`: `(5, 10)`, `n_runs`: `(1, 3)`, `num_leaves`: `(5, 10)`, `trees_per_cycle`: `(3, 5)` |
| `standard` | `lr_linear`: `(0.05, 0.1, 0.2)`, `lr_tree`: `(0.01, 0.02, 0.05)`, `min_data_in_leaf`: `(10, 20, 30)`, `n_cycles`: `(10, 25)`, `n_runs`: `(1, 5)`, `num_leaves`: `(5, 10, 15)`, `subsample`: `(0.7, 1.0)`, `trees_per_cycle`: `(5, 10)` |
| `wide` | `lr_linear`: `(0.03, 0.05, 0.1, 0.2)`, `lr_tree`: `(0.01, 0.02, 0.05)`, `min_data_in_leaf`: `(5, 10, 20, 30)`, `n_cycles`: `(10, 25, 50)`, `n_runs`: `(1, 5, 10)`, `num_leaves`: `(5, 10, 15, 31)`, `subsample`: `(0.6, 0.7, 1.0)`, `trees_per_cycle`: `(5, 10, 20)` |

### lightgbm

Family: `tree`

#### Fit Signature

```python
macroforecast.models.lightgbm(X: Any, y: Any | None = None, *, n_estimators: int = 300, learning_rate: float = 0.1, max_depth: int = -1, num_leaves: int = 31, random_state: int = 0, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `lightgbm` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

LightGBM regressor.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_estimators` | `300` | `int` | True | Number of boosting stages. |
| `learning_rate` | `0.1` | `float` | True | Shrinkage per stage. |
| `max_depth` | `-1` | `int` | True | Maximum tree depth; -1 means no limit. |
| `num_leaves` | `31` | `int` | True | Maximum leaves per tree. |
| `random_state` | `0` | `int` | False | Boosting random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.05, 0.1)`, `max_depth`: `(-1, 3, 5)`, `n_estimators`: `(50, 100)`, `num_leaves`: `(15, 31)` |
| `standard` | `learning_rate`: `(0.03, 0.05, 0.1)`, `max_depth`: `(-1, 3, 5, 10)`, `n_estimators`: `(100, 200, 500)`, `num_leaves`: `(15, 31, 63)` |
| `wide` | `learning_rate`: `(0.01, 0.03, 0.05, 0.1)`, `max_depth`: `(-1, 3, 5, 10, 20)`, `n_estimators`: `(100, 200, 500, 1000)`, `num_leaves`: `(15, 31, 63, 127)` |

### linear_svr

Family: `support_vector`

#### Fit Signature

```python
macroforecast.models.linear_svr(X: Any, y: Any | None = None, *, C: float = 1.0, epsilon: float = 0.0, loss: str = "epsilon_insensitive", tol: float = 0.0001, max_iter: int = 10000, random_state: int | None = 0, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | yes |
| `recommended_preprocessing` | `("standardize predictors before fitting",)` |

Linear support-vector regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `C` | `1.0` | `float` | True | Regularization strength inverse. |
| `epsilon` | `0.0` | `float` | True | Epsilon-insensitive tube width. |
| `loss` | `"epsilon_insensitive"` | `str` | False | LinearSVR loss function. |
| `tol` | `0.0001` | `float` | False | Optimization tolerance. |
| `max_iter` | `10000` | `int` | False | Solver iteration cap. |
| `random_state` | `0` | `int | None` | False | Random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `C`: `(0.1, 1.0)`, `epsilon`: `(0.0, 0.1)` |
| `standard` | `C`: `(0.01, 0.1, 1.0, 10.0)`, `epsilon`: `(0.0, 0.01, 0.1)` |
| `wide` | `C`: `(0.001, 0.01, 0.1, 1.0, 10.0, 100.0)`, `epsilon`: `(0.0, 0.001, 0.01, 0.1, 0.2)` |

### lstm

Family: `neural`

#### Fit Signature

```python
macroforecast.models.lstm(X: Any, y: Any | None = None, *, sequence_length: int = 4, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0, learning_rate: float = 0.001, max_epochs: int = 100, batch_size: int = 32, random_state: int = 0, device: TorchDevice = "auto") -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `deep` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("handled internally: X and y are standardized inside each fit",)` |

Torch-backed LSTM regressor.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `sequence_length` | `4` | `int` | True | Trailing rows per recurrent sequence. |
| `hidden_size` | `32` | `int` | True | Recurrent hidden-state width. |
| `num_layers` | `1` | `int` | False | Number of recurrent layers. |
| `dropout` | `0.0` | `float` | False | Dropout between recurrent layers. |
| `learning_rate` | `0.001` | `float` | True | Adam learning rate. |
| `max_epochs` | `100` | `int` | False | Training epoch cap. |
| `batch_size` | `32` | `int` | False | Mini-batch size. |
| `random_state` | `0` | `int` | False | Random seed. |
| `device` | `"auto"` | `str` | False | Torch device: auto, cpu, or cuda. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `hidden_size`: `(16, 32)`, `learning_rate`: `(0.001,)`, `sequence_length`: `(2, 4)` |
| `standard` | `hidden_size`: `(16, 32, 64)`, `learning_rate`: `(0.0005, 0.001)`, `sequence_length`: `(2, 4, 8)` |
| `wide` | `hidden_size`: `(16, 32, 64, 128)`, `learning_rate`: `(0.0001, 0.0005, 0.001, 0.005)`, `sequence_length`: `(2, 4, 8, 12)` |

### macro_random_forest

Family: `tree`

#### Fit Signature

```python
macroforecast.models.macro_random_forest(X: Any, y: Any | None = None, *, x_columns: Sequence[str] | None = None, S_columns: Sequence[str] | None = None, x_pos: Sequence[int] | None = None, S_pos: Sequence[int] | None = None, y_pos: int = 0, B: int = 25, minsize: int = 10, mtry_frac: float = 0.3333333333333333, min_leaf_frac_of_x: float = 1.0, VI: bool = False, ERT: bool = False, quantile_rate: float | None = None, S_priority_vec: Sequence[float] | None = None, random_x: bool = False, trend_push: int = 1, howmany_random_x: int = 1, howmany_keep_best_VI: int = 20, cheap_look_at_GTVPs: bool = True, prior_var: Sequence[float] | None = None, prior_mean: Sequence[float] | None = None, subsampling_rate: float = 0.75, rw_regul: float = 0.75, keep_forest: bool = False, block_size: int = 12, fast_rw: bool = True, ridge_lambda: float = 0.1, HRW: int = 0, resampling_opt: int = 2, print_b: bool = False, parallelise: bool = False, n_cores: int = 1, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `macro_random_forest` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Adapter for the external MacroRandomForest package.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `x_columns` | `None` | `list[str] | None` | False | Predictors in the time-varying linear equation. |
| `S_columns` | `None` | `list[str] | None` | False | State variables entering the forest split function. |
| `x_pos` | `None` | `list[int] | None` | False | Reference-package predictor positions after the target column. |
| `S_pos` | `None` | `list[int] | None` | False | Reference-package state positions after the target column. |
| `y_pos` | `0` | `int` | False | Fixed target position for the X/y callable adapter; must remain 0. |
| `B` | `25` | `int` | True | Number of MRF trees (deep/paper-faithful default is 50; pass explicitly to restore it). |
| `minsize` | `10` | `int` | True | Minimum node size before split attempts. |
| `mtry_frac` | `0.3333333333333333` | `float` | True | Fraction of state variables considered at each split. |
| `min_leaf_frac_of_x` | `1.0` | `float` | True | Minimum leaf-size multiplier relative to local x dimension. |
| `VI` | `False` | `bool` | False | Enable variable-importance split search mode. |
| `ERT` | `False` | `bool` | False | Enable extremely randomized tree split mode. |
| `quantile_rate` | `None` | `float | None` | False | Optional quantile rate for quantile-oriented output. |
| `S_priority_vec` | `None` | `list[float] | None` | False | Optional priority weights over state variables. |
| `random_x` | `False` | `bool` | False | Use random subsets of local-linear predictors. |
| `trend_push` | `1` | `int` | False | Reference-package trend-push option. |
| `howmany_random_x` | `1` | `int` | False | Number of random local-linear predictor draws. |
| `howmany_keep_best_VI` | `20` | `int` | False | Number of best VI candidates retained. |
| `cheap_look_at_GTVPs` | `True` | `bool` | False | Use the reference package's cheaper GTVP inspection. |
| `prior_var` | `None` | `list[float] | None` | False | Optional prior variances for local coefficients. |
| `prior_mean` | `None` | `list[float] | None` | False | Optional prior means for local coefficients. |
| `subsampling_rate` | `0.75` | `float` | True | Subsample share used by each tree. |
| `rw_regul` | `0.75` | `float` | True | Random-walk shrinkage strength. |
| `keep_forest` | `False` | `bool` | False | Keep full reference forest object in memory. |
| `block_size` | `12` | `int` | False | Reference-package block size for time-series resampling. |
| `fast_rw` | `True` | `bool` | False | Use fast random-walk regularization path. |
| `ridge_lambda` | `0.1` | `float` | True | Ridge penalty for local linear fits. |
| `HRW` | `0` | `int` | False | Reference-package hierarchical random-walk option. |
| `resampling_opt` | `2` | `int` | True | Reference MRF resampling option. |
| `parallelise` | `False` | `bool` | False | Run the reference implementation in parallel. |
| `n_cores` | `1` | `int` | False | Reference implementation worker count. |
| `print_b` | `False` | `bool` | False | Print tree progress. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `B`: `(10, 25)`, `min_leaf_frac_of_x`: `(1.0,)`, `minsize`: `(5, 10)`, `mtry_frac`: `(0.5, 1.0)`, `resampling_opt`: `(2,)`, `ridge_lambda`: `(0.1, 0.5)`, `rw_regul`: `(0.5, 0.75)`, `subsampling_rate`: `(0.75,)` |
| `standard` | `B`: `(10, 25)`, `min_leaf_frac_of_x`: `(0.5, 1.0)`, `minsize`: `(5, 10, 20)`, `mtry_frac`: `(0.3333333333333333, 0.5, 1.0)`, `resampling_opt`: `(1, 2)`, `ridge_lambda`: `(0.1, 0.5, 1.0)`, `rw_regul`: `(0.5, 0.75, 0.9)`, `subsampling_rate`: `(0.5, 0.75)` |
| `wide` | `B`: `(50, 100, 250)`, `min_leaf_frac_of_x`: `(0.25, 0.5, 1.0, 2.0)`, `minsize`: `(5, 10, 20, 40)`, `mtry_frac`: `(0.25, 0.3333333333333333, 0.5, 0.75, 1.0)`, `resampling_opt`: `(1, 2)`, `ridge_lambda`: `(0.01, 0.1, 0.5, 1.0)`, `rw_regul`: `(0.25, 0.5, 0.75, 0.9)`, `subsampling_rate`: `(0.5, 0.63, 0.75, 0.9)` |

### mars

Family: `spline`

#### Fit Signature

```python
macroforecast.models.mars(X: Any, y: Any | None = None, *, max_terms: int = 20, max_degree: int = 1, n_knots: int = 10, min_improvement: float = 1e-06, penalty: float = 2.0, prune: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Package-native MARS-style hinge-basis regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `max_terms` | `20` | `int` | True | Maximum number of hinge basis terms including intercept. |
| `max_degree` | `1` | `int` | True | Maximum interaction degree among hinge factors. |
| `n_knots` | `10` | `int` | True | Candidate quantile knots per predictor. |
| `min_improvement` | `1e-06` | `float` | False | Forward-step relative RSS improvement floor. |
| `penalty` | `2.0` | `float` | False | GCV pruning complexity penalty. |
| `prune` | `True` | `bool` | False | Whether to prune terms by GCV. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `max_degree`: `(1,)`, `max_terms`: `(8, 12)`, `n_knots`: `(5, 10)` |
| `standard` | `max_degree`: `(1, 2)`, `max_terms`: `(10, 20)`, `n_knots`: `(5, 10)` |
| `wide` | `max_degree`: `(1, 2)`, `max_terms`: `(10, 20, 30, 50)`, `n_knots`: `(5, 10, 20)` |

### midas_almon

Family: `mixed_frequency`

#### Fit Signature

```python
macroforecast.models.midas_almon(X: Any, y: Any | None = None, *, polynomial_order: int = 2, theta: tuple[float, ...] | None = None, alpha: float = 0.0, fit_intercept: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Fixed-shape MIDAS over lag groups using midasr::nealmon-style normalized exponential Almon weights.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `polynomial_order` | `2` | `int` | False | Almon polynomial order. |
| `theta` | `None` | `tuple[float, ...] | None` | False | midasr::nealmon shape coefficients; length must equal polynomial_order. |
| `alpha` | `0.0` | `float` | True | Optional ridge penalty on aggregated MIDAS regressors. |
| `fit_intercept` | `True` | `bool` | False | Whether to fit an intercept. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.0, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.0, 0.01, 0.1, 1.0)`, `polynomial_order`: `(1, 2, 3)` |
| `wide` | `alpha`: `(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)`, `polynomial_order`: `(1, 2, 3, 4)` |

### midas_beta

Family: `mixed_frequency`

#### Fit Signature

```python
macroforecast.models.midas_beta(X: Any, y: Any | None = None, *, beta_params: tuple[float, float] = (1.0, 1.0), alpha: float = 0.0, fit_intercept: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Fixed-shape MIDAS over lag groups using midasr::nbetaMT-style beta weights.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `beta_params` | `(1.0, 1.0)` | `tuple[float, float]` | True | midasr::nbetaMT beta lag-weight shape parameters. |
| `alpha` | `0.0` | `float` | True | Optional ridge penalty on aggregated MIDAS regressors. |
| `fit_intercept` | `True` | `bool` | False | Whether to fit an intercept. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.0, 0.1, 1.0)`, `beta_params`: `((1.0, 1.0), (1.0, 2.0), (2.0, 1.0))` |
| `standard` | `alpha`: `(0.0, 0.01, 0.1, 1.0)`, `beta_params`: `((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0))` |
| `wide` | `alpha`: `(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)`, `beta_params`: `((0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 2.0), (3.0, 1.0))` |

### midas_step

Family: `mixed_frequency`

#### Fit Signature

```python
macroforecast.models.midas_step(X: Any, y: Any | None = None, *, n_steps: int = 3, step_bounds: tuple[int, ...] | None = None, step_weights: tuple[float, ...] | None = None, alpha: float = 0.0, fit_intercept: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Fixed-shape MIDAS over lag groups using normalized midasr::polystep-style step weights.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_steps` | `3` | `int` | True | Number of lag buckets when step_bounds is not supplied. |
| `step_bounds` | `None` | `tuple[int, ...] | None` | False | Optional midasr::polystep-style interior cut points. |
| `step_weights` | `None` | `tuple[float, ...] | None` | False | Optional raw height for each step bucket. |
| `alpha` | `0.0` | `float` | True | Optional ridge penalty on aggregated MIDAS regressors. |
| `fit_intercept` | `True` | `bool` | False | Whether to fit an intercept. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.0, 0.1, 1.0)`, `n_steps`: `(2, 3)` |
| `standard` | `alpha`: `(0.0, 0.01, 0.1, 1.0)`, `n_steps`: `(2, 3, 4)` |
| `wide` | `alpha`: `(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)`, `n_steps`: `(2, 3, 4, 6)` |

### naive

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.naive(y: Any, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Random-walk (naive) baseline: carry the last value forward (forecast::naive).

### nn

Family: `neural`

#### Fit Signature

```python
macroforecast.models.nn(X: Any, y: Any | None = None, *, hidden_layer_sizes: tuple[int, ...] = (100,), activation: str = "relu", dropout: float = 0.0, learning_rate: float = 0.001, max_epochs: int = 100, batch_size: int = 32, weight_decay: float = 0.0, optimizer: TorchOptimizer = "adam", loss: TorchLoss = "mse", random_state: int = 0, device: TorchDevice = "auto") -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `deep` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("handled internally: X and y are standardized inside each fit",)` |

Torch-backed feed-forward multilayer perceptron regressor.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `hidden_layer_sizes` | `(100,)` | `tuple[int, ...]` | True | Feed-forward hidden layer widths. |
| `activation` | `"relu"` | `str` | False | Activation: identity, logistic, sigmoid, tanh, relu, or gelu. |
| `dropout` | `0.0` | `float` | True | Dropout rate between hidden layers. |
| `learning_rate` | `0.001` | `float` | True | Optimizer learning rate. |
| `max_epochs` | `100` | `int` | False | Training epoch cap. |
| `batch_size` | `32` | `int` | False | Mini-batch size. |
| `weight_decay` | `0.0` | `float` | True | L2 weight decay. |
| `optimizer` | `"adam"` | `str` | False | Torch optimizer: adam, sgd, or rmsprop. |
| `loss` | `"mse"` | `str` | False | Torch loss: mse or huber. |
| `random_state` | `0` | `int` | False | Random seed. |
| `device` | `"auto"` | `str` | False | Torch device: auto, cpu, or cuda. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `dropout`: `(0.0,)`, `hidden_layer_sizes`: `((32,), (64,))`, `learning_rate`: `(0.001,)`, `weight_decay`: `(0.0, 0.0001)` |
| `standard` | `dropout`: `(0.0, 0.1)`, `hidden_layer_sizes`: `((64,), (100,), (64, 32))`, `learning_rate`: `(0.0005, 0.001)`, `weight_decay`: `(0.0, 0.0001, 0.001)` |
| `wide` | `dropout`: `(0.0, 0.1, 0.25)`, `hidden_layer_sizes`: `((32,), (64,), (100,), (128,), (100, 50), (128, 64))`, `learning_rate`: `(0.0001, 0.0005, 0.001, 0.005)`, `weight_decay`: `(0.0, 1e-05, 0.0001, 0.001, 0.01)` |

### nonneg_ridge

Family: `linear`

#### Fit Signature

```python
macroforecast.models.nonneg_ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, fit_intercept: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Ridge regression with non-negative coefficients.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | L2 penalty strength. |
| `fit_intercept` | `True` | `bool` | False | Fit an intercept outside the constrained coefficients. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### nu_svr

Family: `support_vector`

#### Fit Signature

```python
macroforecast.models.nu_svr(X: Any, y: Any | None = None, *, kernel: str = "rbf", C: float = 1.0, nu: float = 0.5, gamma: str | float = "scale", degree: int = 3, coef0: float = 0.0, shrinking: bool = True, tol: float = 0.001, cache_size: float = 200.0, max_iter: int = -1, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | yes |
| `recommended_preprocessing` | `("standardize predictors before fitting",)` |

Nu support-vector regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `kernel` | `"rbf"` | `str` | False | NuSVR kernel: linear, poly, rbf, or sigmoid. |
| `C` | `1.0` | `float` | True | Regularization strength inverse. |
| `nu` | `0.5` | `float` | True | Upper/lower training-error and support-vector fraction control. |
| `gamma` | `"scale"` | `str | float` | True | Kernel coefficient for rbf/poly/sigmoid. |
| `degree` | `3` | `int` | False | Polynomial kernel degree. |
| `coef0` | `0.0` | `float` | False | Independent term for poly/sigmoid kernels. |
| `shrinking` | `True` | `bool` | False | Whether to use the shrinking heuristic. |
| `tol` | `0.001` | `float` | False | Optimization tolerance. |
| `cache_size` | `200.0` | `float` | False | Kernel cache size in MB. |
| `max_iter` | `-1` | `int` | False | Solver iteration cap; -1 means no cap. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `C`: `(0.1, 1.0)`, `gamma`: `("scale",)`, `nu`: `(0.25, 0.5)` |
| `standard` | `C`: `(0.1, 1.0, 10.0)`, `gamma`: `("scale", "auto")`, `nu`: `(0.25, 0.5, 0.75)` |
| `wide` | `C`: `(0.01, 0.1, 1.0, 10.0, 100.0)`, `gamma`: `("scale", "auto")`, `nu`: `(0.1, 0.25, 0.5, 0.75, 0.9)` |

### ols

Family: `linear`

#### Fit Signature

```python
macroforecast.models.ols(X: Any, y: Any | None = None, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Ordinary least squares with no model-owned tuning space.

### pls

Family: `composite`

#### Fit Signature

```python
macroforecast.models.pls(X: Any, y: Any | None = None, *, n_components: int = 3, scale: bool = True, max_iter: int = 500, tol: float = 1e-06, control_columns: Sequence[str] | None = None, include_constant: bool = True, drop_control_columns: bool = True, quadratic_factors: bool = False, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Partial least squares regression with optional Hounyo-Li-style control residualization.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_components` | `3` | `int` | True | Number of latent PLS components. |
| `scale` | `True` | `bool` | False | Whether to standardize predictors before PLS. |
| `max_iter` | `500` | `int` | False | NIPALS iteration cap. |
| `tol` | `1e-06` | `float` | False | NIPALS convergence tolerance. |
| `control_columns` | `None` | `Sequence[str] | None` | False | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | `bool` | False | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | `bool` | False | Whether controls are excluded from the PLS block. |
| `quadratic_factors` | `False` | `bool` | False | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_components`: `(1, 2, 3)` |
| `standard` | `n_components`: `(1, 2, 3, 5, 8)` |
| `wide` | `n_components`: `(1, 2, 3, 5, 8, 10, 12, 20)` |

### quantile_regression_forest

Family: `tree`

#### Fit Signature

```python
macroforecast.models.quantile_regression_forest(X: Any, y: Any | None = None, *, n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 0, quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Quantile regression forest.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_estimators` | `200` | `int` | True | Number of trees. |
| `max_depth` | `None` | `int | None` | True | Maximum depth per tree. |
| `min_samples_leaf` | `1` | `int` | True | Minimum samples per terminal leaf. |
| `random_state` | `0` | `int` | False | Forest random seed. |
| `quantile_levels` | `(0.05, 0.5, 0.95)` | `tuple[float, ...]` | False | Default quantile levels returned by predict_quantiles(). |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `max_depth`: `(3, 5, None)`, `min_samples_leaf`: `(1, 3)`, `n_estimators`: `(50, 100)` |
| `standard` | `max_depth`: `(3, 5, 10, None)`, `min_samples_leaf`: `(1, 3, 5)`, `n_estimators`: `(100, 200, 500)` |
| `wide` | `max_depth`: `(3, 5, 10, 20, None)`, `min_samples_leaf`: `(1, 2, 3, 5, 10)`, `n_estimators`: `(100, 200, 500, 1000)` |

### random_forest

Family: `tree`

#### Fit Signature

```python
macroforecast.models.random_forest(X: Any, y: Any | None = None, *, n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 0, n_jobs: int | None = None, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Random forest regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_estimators` | `200` | `int` | True | Number of trees. |
| `max_depth` | `None` | `int | None` | True | Maximum depth per tree. |
| `min_samples_leaf` | `1` | `int` | True | Minimum samples per terminal leaf. |
| `random_state` | `0` | `int` | False | Forest random seed. |
| `n_jobs` | `None` | `int | None` | False | Parallel worker count (None resolves to meta.configure(n_jobs)). |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `max_depth`: `(3, 5, None)`, `min_samples_leaf`: `(1, 3)`, `n_estimators`: `(50, 100)` |
| `standard` | `max_depth`: `(3, 5, 10, None)`, `min_samples_leaf`: `(1, 3, 5)`, `n_estimators`: `(100, 200, 500)` |
| `wide` | `max_depth`: `(3, 5, 10, 20, None)`, `min_samples_leaf`: `(1, 2, 3, 5, 10)`, `n_estimators`: `(100, 200, 500, 1000)` |

### random_walk_drift

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.random_walk_drift(y: Any, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Random-walk-with-drift baseline (forecast::rwf(drift=TRUE)).

### random_walk_ridge

Family: `linear`

#### Fit Signature

```python
macroforecast.models.random_walk_ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, initial_alpha: float = 1.0, fit_intercept: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Time-varying random-walk ridge fit, predicting with the final coefficient vector.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Penalty on changes in adjacent coefficient vectors. |
| `initial_alpha` | `1.0` | `float` | False | Penalty on the first coefficient vector. |
| `fit_intercept` | `True` | `bool` | False | Fit an intercept outside the time-varying coefficients. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### rank_aggregation

Family: `assemblage`

#### Fit Signature

```python
macroforecast.models.rank_aggregation(X: Any, y: Any | None = None, *, alpha: float = 1.0, penalty: AggregationPenalty = "fused_difference", mean_match: bool = True, nonneg: bool = True, difference_order: int = 1, penalty_scale: "Literal['none', 'feature_std']" = "feature_std", max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Rank-space supervised aggregation; generic Albacoreranks primitive.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Penalty strength. |
| `penalty` | `"fused_difference"` | `ridge | fused_difference` | False | Rank-weight penalty family. |
| `mean_match` | `True` | `bool` | False | Constrain fitted aggregate mean to match target mean. |
| `nonneg` | `True` | `bool` | False | Constrain rank weights to be non-negative. |
| `difference_order` | `1` | `int` | False | Finite-difference order for rank weights. |
| `penalty_scale` | `"feature_std"` | `none | feature_std` | False | Scale penalties by rank standard deviations. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### realized_garch

Family: `volatility`

#### Fit Signature

```python
macroforecast.models.realized_garch(y: Any, *, X: Any | None = None, rv: Any | None = None, realized_variance: str | None = None, max_iter: int = 2000, n_starts: int = 5, random_state: int = 0) -> VolatilityFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `volatility` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Compact realized GARCH volatility model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `realized_variance` | `None` | `str | None` | False | Column name for realized variance; if omitted, an r_t^2 proxy is used. |
| `max_iter` | `2000` | `int` | False | Optimizer iteration cap. |
| `n_starts` | `5` | `int` | True | Number of optimizer starting points. |
| `random_state` | `0` | `int` | False | Optimizer random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_starts`: `(3, 5)` |
| `standard` | `n_starts`: `(3, 5, 10)` |
| `wide` | `n_starts`: `(3, 5, 10, 20)` |

### restricted_midas

Family: `mixed_frequency`

#### Fit Signature

```python
macroforecast.models.restricted_midas(X: Any, y: Any | None = None, *, weighting: str = "almon", polynomial_order: int = 2, start_params: Mapping[str, Sequence[float]] | Sequence[float] | None = None, n_steps: int = 3, step_bounds: tuple[int, ...] | None = None, fit_intercept: bool = True, maxiter: int = 200, tolerance: float = 1e-06) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

midasr::midas_r-style nonlinear restricted MIDAS over explicit lag columns.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `weighting` | `"almon"` | `str` | False | Restriction map: 'almon'/'nealmon', 'beta'/'nbetaMT', or 'step'/'polystep'. |
| `polynomial_order` | `2` | `int` | False | Almon polynomial order after the aggregate scale parameter. |
| `start_params` | `None` | `Mapping[str, Sequence[float]] | Sequence[float] | None` | False | midasr::midas_r-style starting values for each lag group. |
| `n_steps` | `3` | `int` | False | Number of step buckets when weighting='step' and step_bounds is not supplied. |
| `step_bounds` | `None` | `tuple[int, ...] | None` | False | Optional midasr::polystep-style interior cut points. |
| `fit_intercept` | `True` | `bool` | False | Whether to estimate an intercept outside the restricted lag maps. |
| `maxiter` | `200` | `int` | False | Maximum SciPy least_squares function evaluations (deep/paper-faithful default is 1000; pass explicitly to restore it). |
| `tolerance` | `1e-06` | `float` | False | least_squares xtol/ftol/gtol (deep/paper-faithful default is 1e-8; pass explicitly to restore it). |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `polynomial_order`: `(1, 2)`, `weighting`: `("almon",)` |
| `standard` | `polynomial_order`: `(1, 2, 3)`, `weighting`: `("almon", "beta")` |
| `wide` | `n_steps`: `(2, 3, 4)`, `polynomial_order`: `(1, 2, 3, 4)`, `weighting`: `("almon", "beta", "step")` |

### ridge

Family: `linear`

#### Fit Signature

```python
macroforecast.models.ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Ridge regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | L2 penalty strength. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### scaled_pca

Family: `composite`

#### Fit Signature

```python
macroforecast.models.scaled_pca(X: Any, y: Any | None = None, *, n_components: int = 3, scale: bool = True, control_columns: Sequence[str] | None = None, include_constant: bool = True, drop_control_columns: bool = True, winsorize_slopes: tuple[float, float] | None = None, quadratic_factors: bool = False) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Huang et al. scaled PCA: marginal predictive-slope scaling followed by PCA.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_components` | `3` | `int` | True | Number of Huang scaled-PCA factors. |
| `scale` | `True` | `bool` | False | Whether to standardize predictors inside the model. |
| `control_columns` | `None` | `Sequence[str] | None` | False | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | `bool` | False | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | `bool` | False | Whether controls are excluded from the PCA block. |
| `winsorize_slopes` | `None` | `tuple[float, float] | None` | False | Optional percentile winsorization for scaling slopes. |
| `quadratic_factors` | `False` | `bool` | False | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_components`: `(1, 2, 3)` |
| `standard` | `n_components`: `(1, 2, 3, 5, 8)` |
| `wide` | `n_components`: `(1, 2, 3, 5, 8, 10, 12, 20)` |

### seasonal_naive

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.seasonal_naive(y: Any, *, period: int | None = None, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Seasonal-naive baseline: repeat the last seasonal cycle (forecast::snaive).

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `period` | `None` | `int | None` | False | Seasonal period m; repeats the last m values. |

### shrink_to_target_ridge

Family: `linear`

#### Fit Signature

```python
macroforecast.models.shrink_to_target_ridge(X: Any, y: Any | None = None, *, alpha: float = 1.0, prior_target: float | Sequence[float] | dict[str, float] | None = None, simplex: bool = False, nonneg: bool = False, fit_intercept: bool = True, max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Ridge regression shrinking coefficients toward a target vector.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `1.0` | `float` | True | Strength of the coefficient-target shrinkage. |
| `prior_target` | `None` | `float | sequence | mapping | None` | False | Coefficient target; None means zero, or uniform when simplex=True. |
| `simplex` | `False` | `bool` | False | Constrain coefficients to sum to one. |
| `nonneg` | `False` | `bool` | False | Constrain coefficients to be non-negative. |
| `fit_intercept` | `True` | `bool` | False | Fit an intercept unless simplex=True. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### sparse_group_lasso

Family: `linear`

#### Fit Signature

```python
macroforecast.models.sparse_group_lasso(X: Any, y: Any | None = None, *, groups: Sequence[str | int] | None = None, alpha: float = 1.0, l1_ratio: float = 0.5, group_weights: dict[str, float] | None = None, max_iter: int = 5000, tol: float = 1e-05, scale: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Package-native sparse group lasso with group and feature-level sparsity.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `groups` | `None` | `sequence[str | int] | None` | False | One group label per predictor. |
| `alpha` | `1.0` | `float` | True | Total sparse-group penalty strength. |
| `l1_ratio` | `0.5` | `float` | True | Feature-level L1 share; remaining share is group penalty. |
| `group_weights` | `None` | `dict[str, float] | None` | False | Optional group penalty weights. |
| `max_iter` | `5000` | `int` | False | Proximal-gradient iteration cap. |
| `tol` | `1e-05` | `float` | False | Proximal-gradient convergence tolerance. |
| `scale` | `True` | `bool` | False | Whether to standardize predictors inside the model. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)`, `l1_ratio`: `(0.25, 0.5, 0.75)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)`, `l1_ratio`: `(0.1, 0.25, 0.5, 0.75, 0.9)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0)`, `l1_ratio`: `(0.05, 0.1, 0.25, 0.5, 0.75, 0.9)` |

### stlf

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.stlf(y: Any, *, period: int | None = None, sa_method: str = "ets", **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

STL decomposition + forecast of the seasonally-adjusted series (forecast::stlf).

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `period` | `None` | `int | None` | False | Seasonal period; inferred from the index if omitted. |
| `sa_method` | `"ets"` | `str` | False | Forecaster for the seasonally-adjusted series ('ets'). |

### supervised_aggregation

Family: `assemblage`

#### Fit Signature

```python
macroforecast.models.supervised_aggregation(X: Any, y: Any | None = None, *, space: AggregationSpace = "component", penalty: AggregationPenalty = "ridge", alpha: float = 1.0, reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None, nonneg: bool = True, simplex: bool = False, mean_match: bool = False, difference_order: int = 1, fit_intercept: bool = False, penalty_scale: "Literal['none', 'feature_std']" = "feature_std", max_iter: int = 1000, tol: float = 1e-09) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Generic constrained supervised aggregation derived from Albacore/assemblage primitives.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `space` | `"component"` | `component | rank` | False | Aggregation space. |
| `penalty` | `"ridge"` | `ridge | target_shrinkage | fused_difference` | False | Coefficient penalty family. |
| `alpha` | `1.0` | `float` | True | Penalty strength. |
| `reference_weights` | `None` | `mapping | sequence | None` | False | Reference basket weights for target shrinkage. |
| `nonneg` | `True` | `bool` | False | Constrain weights to be non-negative. |
| `simplex` | `False` | `bool` | False | Constrain weights to sum to one. |
| `mean_match` | `False` | `bool` | False | Constrain fitted aggregate mean to match target mean. |
| `difference_order` | `1` | `int` | False | Finite-difference order for fused rank weights. |
| `fit_intercept` | `False` | `bool` | False | Fit an intercept outside the aggregation weights. |
| `penalty_scale` | `"feature_std"` | `none | feature_std` | False | Scale penalties by component standard deviations. |
| `max_iter` | `1000` | `int` | False | SLSQP solver iteration cap. |
| `tol` | `1e-09` | `float` | False | SLSQP solver tolerance. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.01, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.001, 0.01, 0.1, 1.0, 10.0)` |
| `wide` | `alpha`: `(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)` |

### supervised_pca

Family: `composite`

#### Fit Signature

```python
macroforecast.models.supervised_pca(X: Any, y: Any | None = None, *, n_components: int = 3, n_selected: int | None = 50, min_abs_corr: float = 0.0, scale: bool = True, control_columns: Sequence[str] | None = None, include_constant: bool = True, drop_control_columns: bool = True, preselect: str = "none", t_threshold: float = 1.28, elastic_net_alpha: float = 0.0002, elastic_net_l1_ratio: float = 0.5, quadratic_factors: bool = False, random_state: int = 0) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Original-style iterative supervised PCA with residual correlation screening and projection.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_components` | `3` | `int` | True | Number of sequential supervised components. |
| `n_selected` | `50` | `int | None` | True | Predictors selected at each SPCA step. |
| `min_abs_corr` | `0.0` | `float` | True | Minimum absolute residual correlation retained before PCA. |
| `scale` | `True` | `bool` | False | Whether to standardize predictors and target inside the model. |
| `control_columns` | `None` | `Sequence[str] | None` | False | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | `bool` | False | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | `bool` | False | Whether controls are excluded from the PCA block. |
| `preselect` | `"none"` | `str` | False | Optional pre-selection: none, hard_tstat, or elastic_net. |
| `t_threshold` | `1.28` | `float` | False | Hard t-stat pre-selection threshold. |
| `elastic_net_alpha` | `0.0002` | `float` | False | Elastic-net pre-selection penalty. |
| `elastic_net_l1_ratio` | `0.5` | `float` | False | Elastic-net pre-selection L1 ratio. |
| `quadratic_factors` | `False` | `bool` | False | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |
| `random_state` | `0` | `int` | False | Elastic-net pre-selection random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `min_abs_corr`: `(0.0,)`, `n_components`: `(1, 2, 3)`, `n_selected`: `(10, 25, 50)` |
| `standard` | `min_abs_corr`: `(0.0, 0.05, 0.1)`, `n_components`: `(1, 2, 3, 5)`, `n_selected`: `(10, 25, 50, 100)` |
| `wide` | `min_abs_corr`: `(0.0, 0.03, 0.05, 0.1, 0.2)`, `n_components`: `(1, 2, 3, 5, 8)`, `n_selected`: `(10, 25, 50, 100, 200)` |

### supervised_scaled_pca

Family: `composite`

#### Fit Signature

```python
macroforecast.models.supervised_scaled_pca(X: Any, y: Any | None = None, *, n_components: int = 3, n_selected: int | None = 50, min_abs_corr: float = 0.0, scale: bool = True, control_columns: Sequence[str] | None = None, include_constant: bool = True, drop_control_columns: bool = True, preselect: str = "none", t_threshold: float = 1.28, elastic_net_alpha: float = 0.0002, elastic_net_l1_ratio: float = 0.5, quadratic_factors: bool = False, random_state: int = 0) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Hounyo-Li supervised scaled PCA: marginal predictive-slope scaling followed by SPCA.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_components` | `3` | `int` | True | Number of sequential SsPCA components. |
| `n_selected` | `50` | `int | None` | True | Predictors selected at each SPCA step after slope scaling. |
| `min_abs_corr` | `0.0` | `float` | True | Minimum absolute residual correlation retained before PCA. |
| `scale` | `True` | `bool` | False | Whether to standardize predictors and target inside the model. |
| `control_columns` | `None` | `Sequence[str] | None` | False | Optional X columns used as forecasting controls. |
| `include_constant` | `True` | `bool` | False | Whether to include a constant in the control block. |
| `drop_control_columns` | `True` | `bool` | False | Whether controls are excluded from the PCA block. |
| `preselect` | `"none"` | `str` | False | Optional pre-selection: none, hard_tstat, or elastic_net. |
| `t_threshold` | `1.28` | `float` | False | Hard t-stat pre-selection threshold. |
| `elastic_net_alpha` | `0.0002` | `float` | False | Elastic-net pre-selection penalty. |
| `elastic_net_l1_ratio` | `0.5` | `float` | False | Elastic-net pre-selection L1 ratio. |
| `quadratic_factors` | `False` | `bool` | False | Whether to add the Hounyo-Li PC2 squared-factor forecast head. |
| `random_state` | `0` | `int` | False | Elastic-net pre-selection random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `min_abs_corr`: `(0.0,)`, `n_components`: `(1, 2, 3)`, `n_selected`: `(10, 25, 50)` |
| `standard` | `min_abs_corr`: `(0.0, 0.05, 0.1)`, `n_components`: `(1, 2, 3, 5)`, `n_selected`: `(10, 25, 50, 100)` |
| `wide` | `min_abs_corr`: `(0.0, 0.03, 0.05, 0.1, 0.2)`, `n_components`: `(1, 2, 3, 5, 8)`, `n_selected`: `(10, 25, 50, 100, 200)` |

### svr

Family: `support_vector`

#### Fit Signature

```python
macroforecast.models.svr(X: Any, y: Any | None = None, *, kernel: str = "rbf", C: float = 1.0, epsilon: float = 0.1, gamma: str | float = "scale", degree: int = 3, coef0: float = 0.0, shrinking: bool = True, tol: float = 0.001, cache_size: float = 200.0, max_iter: int = -1, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | none |
| `requires_scaling` | yes |
| `recommended_preprocessing` | `("standardize predictors before fitting",)` |

Kernel support-vector regression.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `kernel` | `"rbf"` | `str` | False | SVR kernel: linear, poly, rbf, or sigmoid. |
| `C` | `1.0` | `float` | True | Regularization strength inverse. |
| `epsilon` | `0.1` | `float` | True | Epsilon-insensitive tube width. |
| `gamma` | `"scale"` | `str | float` | True | Kernel coefficient for rbf/poly/sigmoid. |
| `degree` | `3` | `int` | False | Polynomial kernel degree. |
| `coef0` | `0.0` | `float` | False | Independent term for poly/sigmoid kernels. |
| `shrinking` | `True` | `bool` | False | Whether to use the shrinking heuristic. |
| `tol` | `0.001` | `float` | False | Optimization tolerance. |
| `cache_size` | `200.0` | `float` | False | Kernel cache size in MB. |
| `max_iter` | `-1` | `int` | False | Solver iteration cap; -1 means no cap. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `C`: `(0.1, 1.0)`, `epsilon`: `(0.01, 0.1)`, `gamma`: `("scale",)` |
| `standard` | `C`: `(0.1, 1.0, 10.0)`, `epsilon`: `(0.01, 0.1, 0.2)`, `gamma`: `("scale", "auto")` |
| `wide` | `C`: `(0.01, 0.1, 1.0, 10.0, 100.0)`, `epsilon`: `(0.001, 0.01, 0.1, 0.2)`, `gamma`: `("scale", "auto")` |

### tgarch

Family: `volatility`

#### Fit Signature

```python
macroforecast.models.tgarch(y: Any, *, X: Any | None = None, p: int = 1, o: int = 1, q: int = 1, mean_model: str = "constant", dist: str = "normal", rescale: bool = False, **kwargs: Any) -> VolatilityFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `volatility` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | `arch` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Threshold GARCH (TGARCH/Zakoian) volatility model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `p` | `1` | `int` | True | GARCH innovation lag order. |
| `o` | `1` | `int` | True | Asymmetric (leverage) lag order. |
| `q` | `1` | `int` | True | GARCH variance lag order. |
| `mean_model` | `"constant"` | `str` | False | Conditional mean model. |
| `dist` | `"normal"` | `str` | True | Innovation distribution. |
| `rescale` | `False` | `bool` | False | arch package rescale option. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `dist`: `("normal", "t")`, `o`: `(1,)`, `p`: `(1,)`, `q`: `(1,)` |
| `standard` | `dist`: `("normal", "t")`, `o`: `(1, 2)`, `p`: `(1, 2)`, `q`: `(1, 2)` |
| `wide` | `dist`: `("normal", "t", "skewt")`, `o`: `(1, 2)`, `p`: `(1, 2, 3)`, `q`: `(1, 2, 3)` |

### theta_method

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.theta_method(y: Any, *, period: int | None = None, deseasonalize: bool = True, use_test: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `target` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Theta method target-only forecasting model.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `period` | `None` | `int | None` | False | Seasonal period. |
| `deseasonalize` | `True` | `bool` | False | Whether to deseasonalize. |
| `use_test` | `True` | `bool` | False | Statsmodels seasonality test flag. |

### transformer

Family: `neural`

#### Fit Signature

```python
macroforecast.models.transformer(X: Any, y: Any | None = None, *, sequence_length: int = 4, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.0, learning_rate: float = 0.001, max_epochs: int = 100, batch_size: int = 32, random_state: int = 0, device: TorchDevice = "auto") -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `deep` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `("handled internally: X and y are standardized inside each fit",)` |

Torch-backed Transformer encoder regressor.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `sequence_length` | `4` | `int` | True | Trailing rows per Transformer sequence. |
| `hidden_size` | `32` | `int` | True | Transformer feed-forward width. |
| `num_layers` | `1` | `int` | False | Number of encoder layers. |
| `dropout` | `0.0` | `float` | False | Transformer dropout rate. |
| `learning_rate` | `0.001` | `float` | True | Adam learning rate. |
| `max_epochs` | `100` | `int` | False | Training epoch cap. |
| `batch_size` | `32` | `int` | False | Mini-batch size. |
| `random_state` | `0` | `int` | False | Random seed. |
| `device` | `"auto"` | `str` | False | Torch device: auto, cpu, or cuda. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `hidden_size`: `(16, 32)`, `learning_rate`: `(0.001,)`, `sequence_length`: `(2, 4)` |
| `standard` | `hidden_size`: `(16, 32, 64)`, `learning_rate`: `(0.0005, 0.001)`, `sequence_length`: `(2, 4, 8)` |
| `wide` | `hidden_size`: `(16, 32, 64, 128)`, `learning_rate`: `(0.0001, 0.0005, 0.001, 0.005)`, `sequence_length`: `(2, 4, 8, 12)` |

### tvp_ridge

Family: `linear`

#### Fit Signature

```python
macroforecast.models.tvp_ridge(X: Any, y: Any | None = None, *, lambda_candidates: Any | None = None, oosX: Any | None = None, lambda2: float = 0.1, kfold: int = 5, cv_plot: bool = False, cv_2srr: bool = True, sig_u_param: float = 0.75, sig_eps_param: float = 0.75, ols_prior: bool = False, random_state: int = 1071, use_garch: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `cv_path` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Goulet Coulombe TVP ridge / 2SRR estimator.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `lambda_candidates` | `None` | `sequence[float] | None` | True | Candidate lambda values for the time-variation ridge penalty. |
| `lambda2` | `0.1` | `float` | False | Penalty on starting coefficient values beta_0. |
| `kfold` | `5` | `int` | False | Random k-fold count for lambda CV. |
| `cv_2srr` | `True` | `bool` | False | Run the second lambda CV after 2SRR variance reweighting. |
| `sig_u_param` | `0.75` | `float` | False | Shrinkage exponent for coefficient-innovation variance weights. |
| `sig_eps_param` | `0.75` | `float` | False | Shrinkage exponent for residual-volatility weights. |
| `ols_prior` | `False` | `bool` | False | Shrink starting coefficients toward OLS instead of zero. |
| `random_state` | `1071` | `int` | False | Random fold seed. |
| `use_garch` | `True` | `bool` | False | Use optional arch GARCH(1,1) residual volatility if installed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `lambda_candidates`: `((0.01, 0.1, 1.0, 10.0), (0.1, 1.0, 10.0, 100.0))` |
| `standard` | `lambda_candidates`: `((0.0024787521766663585, 0.015877422572448646, 0.10170139230422684, 0.6514390575310555, 4.172733883598097, 26.728068975964966, 171.20422512253052, 1096.6331584284585, 7024.384376636005, 44994.05794134296, 288205.36312940903, 1846073.3513931935, 11824855.657505034, 75743041.96271756, 485165195.4097903),)` |
| `wide` | `lambda_candidates`: `((0.00033546262790251185, 0.001661557273173934, 0.00822974704902003, 0.040762203978366246, 0.20189651799465547, 1.0, 4.953032424395122, 24.532530197109374, 121.51041751873497, 601.8450378720822, 2980.9579870417283, 14764.781565577294, 73130.44183341571, 362217.44961124816, 1794074.7726062182, 8886110.520507872, 44013193.53483411, 217998774.67921108, 1079754999.464535, 5348061522.750579, 26489122129.84347),)` |

### unrestricted_midas

Family: `mixed_frequency`

#### Fit Signature

```python
macroforecast.models.unrestricted_midas(X: Any, y: Any | None = None, *, alpha: float = 0.0, fit_intercept: bool = True) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

Unrestricted MIDAS over explicit lag columns.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `alpha` | `0.0` | `float` | True | Optional ridge penalty on unrestricted lag coefficients. |
| `fit_intercept` | `True` | `bool` | False | Whether to fit an intercept. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `alpha`: `(0.0, 0.1, 1.0)` |
| `standard` | `alpha`: `(0.0, 0.01, 0.1, 1.0)` |
| `wide` | `alpha`: `(0.0, 0.001, 0.01, 0.1, 1.0, 10.0)` |

### var

Family: `timeseries`

#### Fit Signature

```python
macroforecast.models.var(panel: Any, *, target: str | None = None, n_lag: int = 1, type: str = "const", season: int | None = None) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `panel` |
| `default_preset` | `standard` |
| `default_search_method` | `grid` |
| `requires_extra` | none |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

R vars::VAR-aligned vector autoregression point forecast.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `target` | `None` | `str | None` | False | Target column in the panel. |
| `n_lag` | `1` | `int` | True | VAR lag order. |
| `type` | `"const"` | `str` | False | R vars::VAR deterministic terms: const, trend, both, or none. |
| `season` | `None` | `int | None` | False | Optional centered seasonal dummies, matching vars::VAR(season=...). |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `n_lag`: `(1, 2, 4)` |
| `standard` | `n_lag`: `(1, 2, 4, 6, 12)` |
| `wide` | `n_lag`: `(1, 2, 3, 4, 6, 9, 12, 18, 24)` |

### xgboost

Family: `tree`

#### Fit Signature

```python
macroforecast.models.xgboost(X: Any, y: Any | None = None, *, n_estimators: int = 300, learning_rate: float = 0.1, max_depth: int = 6, subsample: float = 1.0, random_state: int = 0, **kwargs: Any) -> ModelFit
```

| Field | Value |
| --- | --- |
| `input_kind` | `supervised` |
| `default_preset` | `standard` |
| `default_search_method` | `random` |
| `requires_extra` | `xgboost` |
| `requires_scaling` | no |
| `recommended_preprocessing` | `()` |

XGBoost regressor.

#### Model Parameters

| Name | Default | Kind | Tunable | Description |
| --- | --- | --- | --- | --- |
| `n_estimators` | `300` | `int` | True | Number of boosting stages. |
| `learning_rate` | `0.1` | `float` | True | Shrinkage per stage. |
| `max_depth` | `6` | `int` | True | Maximum tree depth. |
| `subsample` | `1.0` | `float` | True | Row subsample share. |
| `random_state` | `0` | `int` | False | Boosting random seed. |

#### Search Spaces

| Preset | Parameters |
| --- | --- |
| `small` | `learning_rate`: `(0.05, 0.1)`, `max_depth`: `(2, 3)`, `n_estimators`: `(50, 100)`, `subsample`: `(0.6, 0.8, 1.0)` |
| `standard` | `learning_rate`: `(0.03, 0.05, 0.1)`, `max_depth`: `(2, 3, 5)`, `n_estimators`: `(100, 200, 500)`, `subsample`: `(0.6, 0.8, 1.0)` |
| `wide` | `learning_rate`: `(0.01, 0.03, 0.05, 0.1)`, `max_depth`: `(2, 3, 5, 8)`, `n_estimators`: `(100, 200, 500, 1000)`, `subsample`: `(0.6, 0.8, 1.0)` |


## Data And Module Values

### `MODEL_SPECS`

Kind: `data`

```python
MODEL_SPECS = dict(76 entries: adaptive_elastic_net, adaptive_lasso, albacore_components, albacore_ranks, ar, arima, assemblage_regression, auto_arima, bayesian_ridge, bvar_minnesota, bvar_normal_inverse_wishart, catboost, ...)
```

## Callable And Class Reference

### GARCHEstimator

Qualified name: `macroforecast.models.volatility.GARCHEstimator`

#### Signature

```python
macroforecast.models.GARCHEstimator(*, variant: str = "garch11", p: int = 1, o: int = 0, q: int = 1, mean_model: str = "constant", dist: str = "normal", rescale: bool = False, realized_variance: str | None = None, power: float = 2.0, **kwargs: Any) -> None
```

#### Description

GARCH/EGARCH wrapper around the optional `arch` package.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `variant` | keyword only | `str` | `"garch11"` |
| `p` | keyword only | `int` | `1` |
| `o` | keyword only | `int` | `0` |
| `q` | keyword only | `int` | `1` |
| `mean_model` | keyword only | `str` | `"constant"` |
| `dist` | keyword only | `str` | `"normal"` |
| `rescale` | keyword only | `bool` | `False` |
| `realized_variance` | keyword only | `str \| None` | `None` |
| `power` | keyword only | `float` | `2.0` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.GARCHEstimator(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'GARCHEstimator'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
| `predict_variance` | `predict_variance(self, horizon: int = 1) -> np.ndarray` | No public docstring is available. |
### LGBAPlusRegressor

Qualified name: `macroforecast.models.tree.LGBAPlusRegressor`

#### Signature

```python
macroforecast.models.LGBAPlusRegressor(*, n_runs: int = 1, n_cycles: int = 25, trees_per_cycle: int = 10, lr_tree: float = 0.02, lr_linear: float = 0.1, num_leaves: int = 15, min_data_in_leaf: int = 20, subsample: float = 1.0, random_state: int | None = None, verbose: bool = False, **lgb_params: Any) -> None
```

#### Description

Alternating LGB^A+ estimator.

Source alignment:
- Reference repository: https://github.com/philgoucou/lgbplus
- Python file: `python/lgb_plus_A.py`; R file: `R/lgb_plus_A.R`.
- Each cycle fits a LightGBM residual tree block, applies `lr_tree`, then
  selects the feature with the largest absolute residual correlation and
  applies one univariate OLS linear update with intercept and `lr_linear`.
- R also provides `lgb_plus_A_ensemble`; macroforecast exposes the same idea
  as `n_runs` on this estimator rather than creating a separate public
  helper, keeping model selection and forecasting runner integration simple.
- The linear slope is computed by centered dot products, exactly equivalent
  to R's `cov(x, resid) / var(x)` and mathematically cleaner than the Python
  file's mixed `np.cov(..., ddof=1) / x.var(ddof=0)` expression.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_runs` | keyword only | `int` | `1` |
| `n_cycles` | keyword only | `int` | `25` |
| `trees_per_cycle` | keyword only | `int` | `10` |
| `lr_tree` | keyword only | `float` | `0.02` |
| `lr_linear` | keyword only | `float` | `0.1` |
| `num_leaves` | keyword only | `int` | `15` |
| `min_data_in_leaf` | keyword only | `int` | `20` |
| `subsample` | keyword only | `float` | `1.0` |
| `random_state` | keyword only | `int \| None` | `None` |
| `verbose` | keyword only | `bool` | `False` |
| `lgb_params` | var keyword | `Any` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.LGBAPlusRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `channel_importance` | `channel_importance(self, X: Any \| None = None, y: Any \| None = None) -> pd.DataFrame` | No public docstring is available. |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series, feature_names: Sequence[str] \| None = None) -> "'LGBAPlusRegressor'"` | No public docstring is available. |
| `get_linear_feature_summary` | `get_linear_feature_summary(self) -> dict[str, int]` | No public docstring is available. |
| `get_total_trees` | `get_total_trees(self) -> int` | No public docstring is available. |
| `predict` | `predict(self, X: Any) -> np.ndarray` | No public docstring is available. |
| `predict_components` | `predict_components(self, X: Any) -> pd.DataFrame` | No public docstring is available. |
| `summary` | `summary(self) -> str` | No public docstring is available. |
### LGBPlusRegressor

Qualified name: `macroforecast.models.tree.LGBPlusRegressor`

#### Signature

```python
macroforecast.models.LGBPlusRegressor(*, n_ensemble: int = 10, n_steps: int = 200, learning_rate: float = 0.05, subsample: float = 0.7, num_leaves: int = 5, min_data_in_leaf: int = 20, lambda_l2: float = 0.1, linear_candidate_fraction: float = 0.5, selection_method: "Literal['oob', 'validation', 'training']" = "oob", val_fraction: float = 0.2, early_stop_patience: int | None = 50, aggregation: "Literal['mean', 'median']" = "mean", random_state: int | None = None, verbose: bool = False, **lgb_params: Any) -> None
```

#### Description

Competition-based LGB+ estimator.

Source alignment:
- Reference repository: https://github.com/philgoucou/lgbplus
- Python file: `python/lgb_plus.py`; R file: `R/lgb_plus.R`.
- Both versions fit one LightGBM residual tree and one greedy univariate
  linear residual update at every boosting step, then accept the lower-loss
  candidate under `oob`, `validation`, or `training` selection.
- The R implementation adds `linear_candidate_fraction`, randomly sampling
  feature candidates before choosing the best residual correlation. The
  Python implementation omits that public argument. macroforecast keeps the
  R candidate-subsampling logic and the Python in-class ensemble structure.
- The competition linear update intentionally has no intercept:
  `coef = sum(x_j * residual) / sum(x_j^2)`, matching both reference files.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_ensemble` | keyword only | `int` | `10` |
| `n_steps` | keyword only | `int` | `200` |
| `learning_rate` | keyword only | `float` | `0.05` |
| `subsample` | keyword only | `float` | `0.7` |
| `num_leaves` | keyword only | `int` | `5` |
| `min_data_in_leaf` | keyword only | `int` | `20` |
| `lambda_l2` | keyword only | `float` | `0.1` |
| `linear_candidate_fraction` | keyword only | `float` | `0.5` |
| `selection_method` | keyword only | `Literal['oob', 'validation', 'training']` | `"oob"` |
| `val_fraction` | keyword only | `float` | `0.2` |
| `early_stop_patience` | keyword only | `int \| None` | `50` |
| `aggregation` | keyword only | `Literal['mean', 'median']` | `"mean"` |
| `random_state` | keyword only | `int \| None` | `None` |
| `verbose` | keyword only | `bool` | `False` |
| `lgb_params` | var keyword | `Any` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.LGBPlusRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `channel_importance` | `channel_importance(self, X: Any \| None = None, y: Any \| None = None) -> pd.DataFrame` | No public docstring is available. |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series, feature_names: Sequence[str] \| None = None) -> "'LGBPlusRegressor'"` | No public docstring is available. |
| `get_linear_feature_summary` | `get_linear_feature_summary(self) -> dict[str, int]` | No public docstring is available. |
| `get_step_type_summary` | `get_step_type_summary(self) -> dict[str, Any]` | No public docstring is available. |
| `predict` | `predict(self, X: Any) -> np.ndarray` | No public docstring is available. |
| `predict_components` | `predict_components(self, X: Any) -> pd.DataFrame` | No public docstring is available. |
| `predict_individual` | `predict_individual(self, X: Any) -> np.ndarray` | No public docstring is available. |
| `summary` | `summary(self) -> str` | No public docstring is available. |
### MacroRandomForestRegressor

Qualified name: `macroforecast.models.tree.MacroRandomForestRegressor`

#### Signature

```python
macroforecast.models.MacroRandomForestRegressor(*, x_columns: Sequence[str] | None = None, S_columns: Sequence[str] | None = None, x_pos: Sequence[int] | None = None, S_pos: Sequence[int] | None = None, y_pos: int = 0, B: int = 25, minsize: int = 10, mtry_frac: float = 0.3333333333333333, min_leaf_frac_of_x: float = 1.0, VI: bool = False, ERT: bool = False, quantile_rate: float | None = None, S_priority_vec: Sequence[float] | None = None, random_x: bool = False, trend_push: int = 1, howmany_random_x: int = 1, howmany_keep_best_VI: int = 20, cheap_look_at_GTVPs: bool = True, prior_var: Sequence[float] | None = None, prior_mean: Sequence[float] | None = None, subsampling_rate: float = 0.75, rw_regul: float = 0.75, keep_forest: bool = False, block_size: int = 12, fast_rw: bool = True, ridge_lambda: float = 0.1, HRW: int = 0, resampling_opt: int = 2, print_b: bool = False, parallelise: bool = False, n_cores: int = 1, **kwargs: Any) -> None
```

#### Description

Adapter for the vendored MacroRandomForest reference implementation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `x_columns` | keyword only | `Sequence[str] \| None` | `None` |
| `S_columns` | keyword only | `Sequence[str] \| None` | `None` |
| `x_pos` | keyword only | `Sequence[int] \| None` | `None` |
| `S_pos` | keyword only | `Sequence[int] \| None` | `None` |
| `y_pos` | keyword only | `int` | `0` |
| `B` | keyword only | `int` | `25` |
| `minsize` | keyword only | `int` | `10` |
| `mtry_frac` | keyword only | `float` | `0.3333333333333333` |
| `min_leaf_frac_of_x` | keyword only | `float` | `1.0` |
| `VI` | keyword only | `bool` | `False` |
| `ERT` | keyword only | `bool` | `False` |
| `quantile_rate` | keyword only | `float \| None` | `None` |
| `S_priority_vec` | keyword only | `Sequence[float] \| None` | `None` |
| `random_x` | keyword only | `bool` | `False` |
| `trend_push` | keyword only | `int` | `1` |
| `howmany_random_x` | keyword only | `int` | `1` |
| `howmany_keep_best_VI` | keyword only | `int` | `20` |
| `cheap_look_at_GTVPs` | keyword only | `bool` | `True` |
| `prior_var` | keyword only | `Sequence[float] \| None` | `None` |
| `prior_mean` | keyword only | `Sequence[float] \| None` | `None` |
| `subsampling_rate` | keyword only | `float` | `0.75` |
| `rw_regul` | keyword only | `float` | `0.75` |
| `keep_forest` | keyword only | `bool` | `False` |
| `block_size` | keyword only | `int` | `12` |
| `fast_rw` | keyword only | `bool` | `True` |
| `ridge_lambda` | keyword only | `float` | `0.1` |
| `HRW` | keyword only | `int` | `0` |
| `resampling_opt` | keyword only | `int` | `2` |
| `print_b` | keyword only | `bool` | `False` |
| `parallelise` | keyword only | `bool` | `False` |
| `n_cores` | keyword only | `int` | `1` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.MacroRandomForestRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'MacroRandomForestRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### MARSRegressor

Qualified name: `macroforecast.models.spline.MARSRegressor`

#### Signature

```python
macroforecast.models.MARSRegressor(*, max_terms: int = 20, max_degree: int = 1, n_knots: int = 10, min_improvement: float = 1e-06, penalty: float = 2.0, prune: bool = True) -> None
```

#### Description

Package-native multivariate adaptive regression splines estimator.

This is an additive/low-order hinge-basis implementation for the package's
callable API. It uses forward pair insertion and optional backward pruning
by generalized cross-validation. It does not claim bit-level equivalence to
proprietary or unmaintained MARS backends.

Source comparison:
- Friedman (1991), "Multivariate Adaptive Regression Splines", represents
  the model as a linear expansion over products of one-sided hinge basis
  functions and uses a forward pass followed by pruning.
- R `earth` is the main open implementation surface to compare against:
  CRAN `earth` documents that it builds models using Friedman's "Fast
  MARS" and "Multivariate Adaptive Regression Splines"; `R/earth.R` routes
  to `earth.fit()`, whose default forward/pruning controls include
  `degree`, `nk`, `thresh`, `minspan`, `endspan`, `fast.k`, and
  `pmethod`.
- This estimator aligns with the core numeric regression skeleton:
  paired hinge basis functions, product interactions up to `max_degree`,
  least-squares refits after each candidate insertion, and GCV-style
  backward deletion.
- It intentionally does not reproduce the full `earth` implementation:
  no formula/factor expansion, case or response weights, GLM mode,
  minspan/endspan rules, Fast MARS parent ageing, exhaustive/seqrep/CV
  pruning, leverage calculations, variance models, or earth object
  compatibility. Candidate knots are a quantile grid controlled by
  `n_knots`, not every eligible observed cut used by `earth`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `max_terms` | keyword only | `int` | `20` |
| `max_degree` | keyword only | `int` | `1` |
| `n_knots` | keyword only | `int` | `10` |
| `min_improvement` | keyword only | `float` | `1e-06` |
| `penalty` | keyword only | `float` | `2.0` |
| `prune` | keyword only | `bool` | `True` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.MARSRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'MARSRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### ModelFit

Qualified name: `macroforecast.models.types.ModelFit`

#### Signature

```python
macroforecast.models.ModelFit(estimator: Any, model: str, feature_names: tuple[str, ...] = (), target_name: str | None = None, metadata: dict[str, Any] = <factory>, diagnostics: dict[str, Any] = <factory>) -> None
```

#### Description

Fitted model wrapper returned by macroforecast model callables.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `estimator` | positional or keyword | `Any` | `required` |
| `model` | positional or keyword | `str` | `required` |
| `feature_names` | positional or keyword | `tuple[str, ...]` | `()` |
| `target_name` | positional or keyword | `str \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `diagnostics` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.ModelFit(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `estimator` | `Any` | `required` |
| `model` | `str` | `required` |
| `feature_names` | `tuple[str, ...]` | `()` |
| `target_name` | `str \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `diagnostics` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `predict` | `predict(self, X: Any) -> pd.Series` | Return point predictions as a pandas Series. |
| `summary` | `summary(self) -> str` | Return a compact text summary of the fitted object. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return JSON-ready fit metadata without serializing the estimator. |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return a compact metadata block for downstream runners. |
### ModelParameter

Qualified name: `macroforecast.models.specs.ModelParameter`

#### Signature

```python
macroforecast.models.ModelParameter(name: str, default: Any, kind: str, description: str, tunable: bool = True) -> None
```

#### Description

One model-owned parameter description.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `default` | positional or keyword | `Any` | `required` |
| `kind` | positional or keyword | `str` | `required` |
| `description` | positional or keyword | `str` | `required` |
| `tunable` | positional or keyword | `bool` | `True` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.ModelParameter(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | `required` |
| `default` | `Any` | `required` |
| `kind` | `str` | `required` |
| `description` | `str` | `required` |
| `tunable` | `bool` | `True` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready parameter description. |
### ModelSpec

Qualified name: `macroforecast.models.specs.ModelSpec`

#### Signature

```python
macroforecast.models.ModelSpec(name: str, family: str, fit_func: Callable[..., Any], default_params: dict[str, Any] = <factory>, parameters: tuple[ModelParameter, ...] = (), search_spaces: SearchSpaces = <factory>, default_search_method: str = "grid", default_preset: str = "standard", input_kind: InputKind = "supervised", preset: str = "standard", params: dict[str, Any] = <factory>, backend: str = "internal", requires_extra: str | None = None, requires_scaling: bool = False, recommended_preprocessing: tuple[str, ...] = (), description: str = "", selection_method: str = "cv") -> None
```

#### Description

Callable model plus model-owned defaults and hyperparameter spaces.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `family` | positional or keyword | `str` | `required` |
| `fit_func` | positional or keyword | `Callable[..., Any]` | `required` |
| `default_params` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `parameters` | positional or keyword | `tuple[ModelParameter, ...]` | `()` |
| `search_spaces` | positional or keyword | `SearchSpaces` | `<factory>` |
| `default_search_method` | positional or keyword | `str` | `"grid"` |
| `default_preset` | positional or keyword | `str` | `"standard"` |
| `input_kind` | positional or keyword | `InputKind` | `"supervised"` |
| `preset` | positional or keyword | `str` | `"standard"` |
| `params` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `backend` | positional or keyword | `str` | `"internal"` |
| `requires_extra` | positional or keyword | `str \| None` | `None` |
| `requires_scaling` | positional or keyword | `bool` | `False` |
| `recommended_preprocessing` | positional or keyword | `tuple[str, ...]` | `()` |
| `description` | positional or keyword | `str` | `""` |
| `selection_method` | positional or keyword | `str` | `"cv"` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.ModelSpec(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | `required` |
| `family` | `str` | `required` |
| `fit_func` | `Callable[..., Any]` | `required` |
| `default_params` | `dict[str, Any]` | `default_factory` |
| `parameters` | `tuple[ModelParameter, ...]` | `()` |
| `search_spaces` | `SearchSpaces` | `default_factory` |
| `default_search_method` | `str` | `"grid"` |
| `default_preset` | `str` | `"standard"` |
| `input_kind` | `InputKind` | `"supervised"` |
| `preset` | `str` | `"standard"` |
| `params` | `dict[str, Any]` | `default_factory` |
| `backend` | `str` | `"internal"` |
| `requires_extra` | `str \| None` | `None` |
| `requires_scaling` | `bool` | `False` |
| `recommended_preprocessing` | `tuple[str, ...]` | `()` |
| `description` | `str` | `""` |
| `selection_method` | `str` | `"cv"` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `all_params` | `all_params(self, **params: Any) -> dict[str, Any]` | Merge default, fixed, and trial parameters. |
| `describe` | `describe(self) -> pd.DataFrame` | Return parameter documentation as a DataFrame. |
| `fit` | `fit(self, X: Any, y: Any \| None = None, **params: Any) -> Any` | Fit the model according to the model's input convention. |
| `search_space` | `search_space(self, preset: str \| None = None) -> dict[str, tuple[Any, ...]]` | Return the model-owned hyperparameter space for one preset. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return a JSON-ready model specification. |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return compact model metadata for selection and forecasting runners. |
| `with_params` | `with_params(self, **params: Any) -> "'ModelSpec'"` | Return the same model spec with fixed model parameters. |
| `with_preset` | `with_preset(self, preset: str) -> "'ModelSpec'"` | Return the same model spec with a different hyperparameter preset. |
### QuantileRegressionForestRegressor

Qualified name: `macroforecast.models.tree.QuantileRegressionForestRegressor`

#### Signature

```python
macroforecast.models.QuantileRegressionForestRegressor(*, n_estimators: int = 200, max_depth: int | None = None, min_samples_leaf: int = 1, random_state: int = 0, quantile_levels: tuple[float, ...] = (0.05, 0.5, 0.95)) -> None
```

#### Description

Random-forest point forecasts plus empirical leaf quantiles.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_estimators` | keyword only | `int` | `200` |
| `max_depth` | keyword only | `int \| None` | `None` |
| `min_samples_leaf` | keyword only | `int` | `1` |
| `random_state` | keyword only | `int` | `0` |
| `quantile_levels` | keyword only | `tuple[float, ...]` | `(0.05, 0.5, 0.95)` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.QuantileRegressionForestRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'QuantileRegressionForestRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
| `predict_quantiles` | `predict_quantiles(self, X: pd.DataFrame, levels: tuple[float, ...] \| None = None) -> dict[float, np.ndarray]` | No public docstring is available. |
### RealizedGARCHEstimator

Qualified name: `macroforecast.models.volatility.RealizedGARCHEstimator`

#### Signature

```python
macroforecast.models.RealizedGARCHEstimator(*, realized_variance: str | None = None, max_iter: int = 2000, n_starts: int = 5, random_state: int = 0) -> None
```

#### Description

Compact Hansen-Huang-Shek-style realized GARCH joint MLE.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `realized_variance` | keyword only | `str \| None` | `None` |
| `max_iter` | keyword only | `int` | `2000` |
| `n_starts` | keyword only | `int` | `5` |
| `random_state` | keyword only | `int` | `0` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.RealizedGARCHEstimator(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'RealizedGARCHEstimator'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
| `predict_variance` | `predict_variance(self, horizon: int = 1) -> np.ndarray` | No public docstring is available. |
### SavedModel

Qualified name: `macroforecast.models.persistence.SavedModel`

#### Signature

```python
macroforecast.models.SavedModel(model_path: str | None, metadata_path: str, save_error: str | None = None) -> None
```

#### Description

Paths and status returned by model fit persistence helpers.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model_path` | positional or keyword | `str \| None` | `required` |
| `metadata_path` | positional or keyword | `str` | `required` |
| `save_error` | positional or keyword | `str \| None` | `None` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.SavedModel(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `model_path` | `str \| None` | `required` |
| `metadata_path` | `str` | `required` |
| `save_error` | `str \| None` | `None` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | No public docstring is available. |
### ScaledPCARegressor

Qualified name: `macroforecast.models.linear.ScaledPCARegressor`

#### Signature

```python
macroforecast.models.ScaledPCARegressor(*, n_components: int = 3, scale: bool = True, control_columns: Sequence[str] | None = None, include_constant: bool = True, drop_control_columns: bool = True, winsorize_slopes: tuple[float, float] | None = None, quadratic_factors: bool = False) -> None
```

#### Description

Huang et al. scaled PCA factor extraction with a linear forecast head.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_components` | keyword only | `int` | `3` |
| `scale` | keyword only | `bool` | `True` |
| `control_columns` | keyword only | `Sequence[str] \| None` | `None` |
| `include_constant` | keyword only | `bool` | `True` |
| `drop_control_columns` | keyword only | `bool` | `True` |
| `winsorize_slopes` | keyword only | `tuple[float, float] \| None` | `None` |
| `quadratic_factors` | keyword only | `bool` | `False` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.ScaledPCARegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'ScaledPCARegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
| `transform` | `transform(self, X: pd.DataFrame) -> pd.DataFrame` | No public docstring is available. |
### SupervisedAggregationRegressor

Qualified name: `macroforecast.models.assemblage.SupervisedAggregationRegressor`

#### Signature

```python
macroforecast.models.SupervisedAggregationRegressor(*, space: AggregationSpace = "component", penalty: AggregationPenalty = "ridge", alpha: float = 1.0, reference_weights: Mapping[str, float] | Sequence[float] | pd.Series | None = None, nonneg: bool = True, simplex: bool = False, mean_match: bool = False, difference_order: int = 1, fit_intercept: bool = False, penalty_scale: "Literal['none', 'feature_std']" = "feature_std", max_iter: int = 1000, tol: float = 1e-09) -> None
```

#### Description

Constrained supervised aggregation estimator.

This is the generic, inflation-free primitive behind Albacore-style
assemblage regression. It learns nonnegative, optionally simplex or
mean-matched weights that map a component panel to a future aggregate
target. Source cue: R ``assemblage`` functions ``nonneg.ridge``,
``nonneg.ridge.sum1``, ``nonneg.ridge.mean``, and ``nonneg.ridge.meanD``.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `space` | keyword only | `AggregationSpace` | `"component"` |
| `penalty` | keyword only | `AggregationPenalty` | `"ridge"` |
| `alpha` | keyword only | `float` | `1.0` |
| `reference_weights` | keyword only | `Mapping[str, float] \| Sequence[float] \| pd.Series \| None` | `None` |
| `nonneg` | keyword only | `bool` | `True` |
| `simplex` | keyword only | `bool` | `False` |
| `mean_match` | keyword only | `bool` | `False` |
| `difference_order` | keyword only | `int` | `1` |
| `fit_intercept` | keyword only | `bool` | `False` |
| `penalty_scale` | keyword only | `Literal['none', 'feature_std']` | `"feature_std"` |
| `max_iter` | keyword only | `int` | `1000` |
| `tol` | keyword only | `float` | `1e-09` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.SupervisedAggregationRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'SupervisedAggregationRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### SupervisedPCARegressor

Qualified name: `macroforecast.models.linear.SupervisedPCARegressor`

#### Signature

```python
macroforecast.models.SupervisedPCARegressor(*, n_components: int = 3, n_selected: int | None = 50, min_abs_corr: float = 0.0, scale: bool = True, control_columns: Sequence[str] | None = None, include_constant: bool = True, drop_control_columns: bool = True, preselect: str = "none", t_threshold: float = 1.28, elastic_net_alpha: float = 0.0002, elastic_net_l1_ratio: float = 0.5, slope_scale: bool = False, quadratic_factors: bool = False, random_state: int = 0) -> None
```

#### Description

Original-style SPCA with iterative screening, PCA, and projection.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `n_components` | keyword only | `int` | `3` |
| `n_selected` | keyword only | `int \| None` | `50` |
| `min_abs_corr` | keyword only | `float` | `0.0` |
| `scale` | keyword only | `bool` | `True` |
| `control_columns` | keyword only | `Sequence[str] \| None` | `None` |
| `include_constant` | keyword only | `bool` | `True` |
| `drop_control_columns` | keyword only | `bool` | `True` |
| `preselect` | keyword only | `str` | `"none"` |
| `t_threshold` | keyword only | `float` | `1.28` |
| `elastic_net_alpha` | keyword only | `float` | `0.0002` |
| `elastic_net_l1_ratio` | keyword only | `float` | `0.5` |
| `slope_scale` | keyword only | `bool` | `False` |
| `quadratic_factors` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int` | `0` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.SupervisedPCARegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'SupervisedPCARegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### SupervisedScaledPCARegressor

Qualified name: `macroforecast.models.linear.SupervisedScaledPCARegressor`

#### Signature

```python
macroforecast.models.SupervisedScaledPCARegressor(**kwargs: Any) -> None
```

#### Description

Hounyo-Li supervised scaled PCA: predictive-slope scaling plus SPCA.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.SupervisedScaledPCARegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series) -> "'SupervisedPCARegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### TVPRidgeRegressor

Qualified name: `macroforecast.models.tvp.TVPRidgeRegressor`

#### Signature

```python
macroforecast.models.TVPRidgeRegressor(*, lambda_candidates: Any | None = None, oosX: Any | None = None, lambda2: float = 0.1, kfold: int = 5, cv_plot: bool = False, cv_2srr: bool = True, sig_u_param: float = 0.75, sig_eps_param: float = 0.75, ols_prior: bool = False, random_state: int = 1071, use_garch: bool = True) -> None
```

#### Description

Time-varying parameter ridge / 2SRR estimator.

Source alignment:
- R package: `TVPRidge`
- Source file: `R/MV2SRR_v210407.R`
- Main R callable: `tvp.ridge`

The implementation keeps the R decomposition:
`Zfun` basis expansion -> `dualGRR` generalized ridge ->
`CV.KF.MV` / `cv.univariate` lambda tuning -> 2SRR variance
reweighting. Indexing changes are explicit because R is 1-based and NumPy
is 0-based.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `lambda_candidates` | keyword only | `Any \| None` | `None` |
| `oosX` | keyword only | `Any \| None` | `None` |
| `lambda2` | keyword only | `float` | `0.1` |
| `kfold` | keyword only | `int` | `5` |
| `cv_plot` | keyword only | `bool` | `False` |
| `cv_2srr` | keyword only | `bool` | `True` |
| `sig_u_param` | keyword only | `float` | `0.75` |
| `sig_eps_param` | keyword only | `float` | `0.75` |
| `ols_prior` | keyword only | `bool` | `False` |
| `random_state` | keyword only | `int` | `1071` |
| `use_garch` | keyword only | `bool` | `True` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.TVPRidgeRegressor(...)
```

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `fit` | `fit(self, X: pd.DataFrame, y: pd.Series \| pd.DataFrame) -> "'TVPRidgeRegressor'"` | No public docstring is available. |
| `predict` | `predict(self, X: pd.DataFrame) -> np.ndarray` | No public docstring is available. |
### VolatilityFit

Qualified name: `macroforecast.models.types.VolatilityFit`

#### Signature

```python
macroforecast.models.VolatilityFit(estimator: Any, model: str, feature_names: tuple[str, ...] = (), target_name: str | None = None, metadata: dict[str, Any] = <factory>, diagnostics: dict[str, Any] = <factory>) -> None
```

#### Description

Fitted volatility model wrapper.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `estimator` | positional or keyword | `Any` | `required` |
| `model` | positional or keyword | `str` | `required` |
| `feature_names` | positional or keyword | `tuple[str, ...]` | `()` |
| `target_name` | positional or keyword | `str \| None` | `None` |
| `metadata` | positional or keyword | `dict[str, Any]` | `<factory>` |
| `diagnostics` | positional or keyword | `dict[str, Any]` | `<factory>` |

#### Returns

`None`

#### Minimal Use

```python
import macroforecast as mf
# Construct with the signature above:
# mf.models.VolatilityFit(...)
```

#### Dataclass Fields

| Field | Type | Default |
| --- | --- | --- |
| `estimator` | `Any` | `required` |
| `model` | `str` | `required` |
| `feature_names` | `tuple[str, ...]` | `()` |
| `target_name` | `str \| None` | `None` |
| `metadata` | `dict[str, Any]` | `default_factory` |
| `diagnostics` | `dict[str, Any]` | `default_factory` |

#### Public Methods

| Method | Signature | Summary |
| --- | --- | --- |
| `predict` | `predict(self, X: Any) -> pd.Series` | Return point predictions as a pandas Series. |
| `predict_variance` | `predict_variance(self, horizon: int = 1) -> pd.Series` | No public docstring is available. |
| `summary` | `summary(self) -> str` | Return a compact text summary of the fitted object. |
| `to_dict` | `to_dict(self) -> dict[str, Any]` | Return JSON-ready fit metadata without serializing the estimator. |
| `to_metadata` | `to_metadata(self) -> dict[str, Any]` | Return a compact metadata block for downstream runners. |
### custom_model

Qualified name: `macroforecast.models.specs.custom_model`

#### Signature

```python
macroforecast.models.custom_model(name: str, fit_func: Callable[..., Any], *, family: str = "custom", default_params: Mapping[str, Any] | None = None, parameters: tuple[ModelParameter, ...] = (), search_spaces: SearchSpaces | None = None, default_search_method: str = "grid", default_preset: str = "standard", input_kind: InputKind = "supervised", backend: str = "custom", requires_extra: str | None = None, requires_scaling: bool = False, recommended_preprocessing: tuple[str, ...] = (), description: str | None = None) -> ModelSpec
```

#### Description

Build a user-owned ``ModelSpec`` without registering a package model.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `name` | positional or keyword | `str` | `required` |
| `fit_func` | positional or keyword | `Callable[..., Any]` | `required` |
| `family` | keyword only | `str` | `"custom"` |
| `default_params` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `parameters` | keyword only | `tuple[ModelParameter, ...]` | `()` |
| `search_spaces` | keyword only | `SearchSpaces \| None` | `None` |
| `default_search_method` | keyword only | `str` | `"grid"` |
| `default_preset` | keyword only | `str` | `"standard"` |
| `input_kind` | keyword only | `InputKind` | `"supervised"` |
| `backend` | keyword only | `str` | `"custom"` |
| `requires_extra` | keyword only | `str \| None` | `None` |
| `requires_scaling` | keyword only | `bool` | `False` |
| `recommended_preprocessing` | keyword only | `tuple[str, ...]` | `()` |
| `description` | keyword only | `str \| None` | `None` |

#### Returns

`ModelSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.custom_model(...)
```
### describe_model

Qualified name: `macroforecast.models.specs.describe_model`

#### Signature

```python
macroforecast.models.describe_model(model: str | Callable[..., Any] | ModelSpec) -> pd.DataFrame
```

#### Description

Describe model-owned parameters and preset spaces.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.describe_model(...)
```
### get_model

Qualified name: `macroforecast.models.specs.get_model`

#### Signature

```python
macroforecast.models.get_model(model: str | Callable[..., Any] | ModelSpec, *, preset: str | None = None, params: Mapping[str, Any] | None = None) -> ModelSpec
```

#### Description

Return a model spec by name, callable, or existing spec.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `preset` | keyword only | `str \| None` | `None` |
| `params` | keyword only | `Mapping[str, Any] \| None` | `None` |

#### Returns

`ModelSpec`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.get_model(...)
```
### list_model_specs

Qualified name: `macroforecast.models.specs.list_model_specs`

#### Signature

```python
macroforecast.models.list_model_specs(*, family: str | None = None) -> pd.DataFrame
```

#### Description

List registered model specs.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `family` | keyword only | `str \| None` | `None` |

#### Returns

`pd.DataFrame`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.list_model_specs(...)
```
### load_fit

Qualified name: `macroforecast.models.persistence.load_fit`

#### Signature

```python
macroforecast.models.load_fit(model_path: str | Path) -> Any
```

#### Description

Load a fitted model object saved by `save_fit()`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model_path` | positional or keyword | `str \| Path` | `required` |

#### Returns

`Any`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.load_fit(...)
```
### model_search_space

Qualified name: `macroforecast.models.specs.model_search_space`

#### Signature

```python
macroforecast.models.model_search_space(model: str | Callable[..., Any] | ModelSpec, *, preset: str | None = None) -> dict[str, tuple[Any, ...]]
```

#### Description

Return a model-owned hyperparameter space.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `model` | positional or keyword | `str \| Callable[..., Any] \| ModelSpec` | `required` |
| `preset` | keyword only | `str \| None` | `None` |

#### Returns

`dict[str, tuple[Any, ...]]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.model_search_space(...)
```
### risk_forecast

Qualified name: `macroforecast.models.volatility.risk_forecast`

#### Signature

```python
macroforecast.models.risk_forecast(fit: VolatilityFit, *, alpha: float = 0.05, horizon: int = 1) -> dict[str, Any]
```

#### Description

Value-at-Risk and Expected Shortfall forecast from a fitted volatility model.

Returns, per horizon step, the lower-tail return quantile (``var``) and the
mean return conditional on falling below it (``es``) at level ``alpha``, using
the fitted conditional mean, the h-step conditional variance forecast, and the
fitted innovation distribution (Normal or standardized Student-t; other
distributions fall back to Normal). VaR/ES are in return space (losses are
negative), matching rugarch's quantile()/ES convention.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `fit` | positional or keyword | `VolatilityFit` | `required` |
| `alpha` | keyword only | `float` | `0.05` |
| `horizon` | keyword only | `int` | `1` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.risk_forecast(...)
```
### value_at_risk

Qualified name: `macroforecast.models.volatility.value_at_risk`

#### Signature

```python
macroforecast.models.value_at_risk(fit: VolatilityFit, *, alpha: float = 0.05, horizon: int = 1) -> np.ndarray
```

#### Description

Lower-tail Value-at-Risk return quantile(s); see :func:`risk_forecast`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `fit` | positional or keyword | `VolatilityFit` | `required` |
| `alpha` | keyword only | `float` | `0.05` |
| `horizon` | keyword only | `int` | `1` |

#### Returns

`np.ndarray`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.value_at_risk(...)
```
### news_impact_curve

Qualified name: `macroforecast.models.volatility.news_impact_curve`

#### Signature

```python
macroforecast.models.news_impact_curve(fit: VolatilityFit, *, shocks: Any | None = None, n_points: int = 101, span: float = 3.0, reference_variance: float | None = None) -> dict[str, Any]
```

#### Description

Engle-Ng (1993) news impact curve for a fitted GARCH-family model.

Traces the one-step conditional variance ``h_t`` as a function of the lagged
shock ``eps_{t-1}``, holding the lagged variance ``h_{t-1}`` fixed at a
reference level. A symmetric model gives a symmetric U-shaped curve centred at
zero; leverage (GJR/EGARCH/TGARCH ``gamma``) tilts it so negative shocks raise
variance more than positive ones.

``reference_variance`` is ``h_{t-1}`` (defaults to the last fitted conditional
variance). ``shocks`` is an explicit eps grid; if omitted, ``n_points`` points
spanning +/- ``span`` reference standard deviations are used. Returns the shock
grid, the implied conditional variance, the reference variance, and the model
variant. EGARCH uses the Gaussian ``E|z| = sqrt(2/pi)``; for non-normal
innovations its curve is approximate.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `fit` | positional or keyword | `VolatilityFit` | `required` |
| `shocks` | keyword only | `Any \| None` | `None` |
| `n_points` | keyword only | `int` | `101` |
| `span` | keyword only | `float` | `3.0` |
| `reference_variance` | keyword only | `float \| None` | `None` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.news_impact_curve(...)
```
### garch_roll

Qualified name: `macroforecast.models.volatility.garch_roll`

#### Signature

```python
macroforecast.models.garch_roll(returns: Any, *, variant: str = "garch11", forecast_length: int | None = None, refit_every: int = 25, window: str = "expanding", window_size: int | None = None, alpha: float = 0.05, dist: str = "normal", **kwargs: Any) -> dict[str, Any]
```

#### Description

Rolling 1-step volatility / Value-at-Risk backtest (rugarch::ugarchroll).

Walks forward over the last ``forecast_length`` observations, producing a
one-step-ahead conditional volatility and VaR forecast at each origin. The
model is re-estimated every ``refit_every`` origins and filtered forward in
between (the k-step variance forecast from the last fit, exactly as
``ugarchroll`` filters between refits). ``window`` is ``"expanding"`` or
``"rolling"`` (with ``window_size``). Returns the per-origin forecast sigma,
lower-tail VaR and Expected Shortfall at ``alpha``, the realised return, the
VaR-violation indicator, and a coverage summary (empirical vs nominal
violation rate) for backtesting.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `returns` | positional or keyword | `Any` | `required` |
| `variant` | keyword only | `str` | `"garch11"` |
| `forecast_length` | keyword only | `int \| None` | `None` |
| `refit_every` | keyword only | `int` | `25` |
| `window` | keyword only | `str` | `"expanding"` |
| `window_size` | keyword only | `int \| None` | `None` |
| `alpha` | keyword only | `float` | `0.05` |
| `dist` | keyword only | `str` | `"normal"` |
| `kwargs` | var keyword | `Any` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.garch_roll(...)
```
### expected_shortfall

Qualified name: `macroforecast.models.volatility.expected_shortfall`

#### Signature

```python
macroforecast.models.expected_shortfall(fit: VolatilityFit, *, alpha: float = 0.05, horizon: int = 1) -> np.ndarray
```

#### Description

Expected Shortfall (mean return below VaR); see :func:`risk_forecast`.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `fit` | positional or keyword | `VolatilityFit` | `required` |
| `alpha` | keyword only | `float` | `0.05` |
| `horizon` | keyword only | `int` | `1` |

#### Returns

`np.ndarray`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.expected_shortfall(...)
```
### save_fit

Qualified name: `macroforecast.models.persistence.save_fit`

#### Signature

```python
macroforecast.models.save_fit(fit: Any, model_path: str | Path, *, metadata_path: str | Path | None = None, metadata: Mapping[str, Any] | None = None, allow_pickle_error: bool = True) -> SavedModel
```

#### Description

Persist a fitted model object and a JSON metadata sidecar.

This helper owns only the storage format. Forecasting runners decide which
fit to save, where to save it, and which experiment metadata to attach.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `fit` | positional or keyword | `Any` | `required` |
| `model_path` | positional or keyword | `str \| Path` | `required` |
| `metadata_path` | keyword only | `str \| Path \| None` | `None` |
| `metadata` | keyword only | `Mapping[str, Any] \| None` | `None` |
| `allow_pickle_error` | keyword only | `bool` | `True` |

#### Returns

`SavedModel`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.save_fit(...)
```
### solve_fused_difference_ridge

Qualified name: `macroforecast.models.assemblage.solve_fused_difference_ridge`

#### Signature

```python
macroforecast.models.solve_fused_difference_ridge(X: Any, y: Any, *, alpha: float = 1.0, difference_order: int = 1, mean_match: bool = True, penalty_scale: "Literal['none', 'feature_std']" = "feature_std") -> pd.Series
```

#### Description

Return nonnegative fused-difference weights for rank aggregation.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `1.0` |
| `difference_order` | keyword only | `int` | `1` |
| `mean_match` | keyword only | `bool` | `True` |
| `penalty_scale` | keyword only | `Literal['none', 'feature_std']` | `"feature_std"` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.solve_fused_difference_ridge(...)
```
### solve_mean_aligned_ridge

Qualified name: `macroforecast.models.assemblage.solve_mean_aligned_ridge`

#### Signature

```python
macroforecast.models.solve_mean_aligned_ridge(X: Any, y: Any, *, alpha: float = 1.0, penalty_scale: "Literal['none', 'feature_std']" = "feature_std") -> pd.Series
```

#### Description

Return nonnegative weights constrained to match target mean.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `1.0` |
| `penalty_scale` | keyword only | `Literal['none', 'feature_std']` | `"feature_std"` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.solve_mean_aligned_ridge(...)
```
### solve_nonnegative_ridge

Qualified name: `macroforecast.models.assemblage.solve_nonnegative_ridge`

#### Signature

```python
macroforecast.models.solve_nonnegative_ridge(X: Any, y: Any, *, alpha: float = 1.0, penalty_scale: "Literal['none', 'feature_std']" = "feature_std") -> pd.Series
```

#### Description

Return nonnegative ridge weights from the assemblage solver primitive.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `1.0` |
| `penalty_scale` | keyword only | `Literal['none', 'feature_std']` | `"feature_std"` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.solve_nonnegative_ridge(...)
```
### solve_simplex_ridge

Qualified name: `macroforecast.models.assemblage.solve_simplex_ridge`

#### Signature

```python
macroforecast.models.solve_simplex_ridge(X: Any, y: Any, *, alpha: float = 1.0, penalty_scale: "Literal['none', 'feature_std']" = "feature_std") -> pd.Series
```

#### Description

Return nonnegative sum-to-one ridge weights.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `alpha` | keyword only | `float` | `1.0` |
| `penalty_scale` | keyword only | `Literal['none', 'feature_std']` | `"feature_std"` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.solve_simplex_ridge(...)
```
### solve_target_shrinkage_ridge

Qualified name: `macroforecast.models.assemblage.solve_target_shrinkage_ridge`

#### Signature

```python
macroforecast.models.solve_target_shrinkage_ridge(X: Any, y: Any, *, reference_weights: Mapping[str, float] | Sequence[float] | pd.Series, alpha: float = 1.0, simplex: bool = True, penalty_scale: "Literal['none', 'feature_std']" = "feature_std") -> pd.Series
```

#### Description

Return weights for Albacore-style target-shrinkage ridge.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `X` | positional or keyword | `Any` | `required` |
| `y` | positional or keyword | `Any` | `required` |
| `reference_weights` | keyword only | `Mapping[str, float] \| Sequence[float] \| pd.Series` | `required` |
| `alpha` | keyword only | `float` | `1.0` |
| `simplex` | keyword only | `bool` | `True` |
| `penalty_scale` | keyword only | `Literal['none', 'feature_std']` | `"feature_std"` |

#### Returns

`pd.Series`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.solve_target_shrinkage_ridge(...)
```
### var_select_order

Qualified name: `macroforecast.models.timeseries.var_select_order`

#### Signature

```python
macroforecast.models.var_select_order(panel: Any, *, maxlags: int | None = None, trend: str = "c") -> dict[str, Any]
```

#### Description

Select the VAR lag order by information criteria (vars::VARselect).

Fits VAR(p) for p up to ``maxlags`` and reports the lag order minimising each
of AIC, BIC (Schwarz), HQ (Hannan-Quinn) and FPE, via statsmodels
``VAR.select_order``. ``trend`` is the deterministic term ('n','c','ct','ctt').
Returns the selected order per criterion plus the criterion values per lag.

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `maxlags` | keyword only | `int \| None` | `None` |
| `trend` | keyword only | `str` | `"c"` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.var_select_order(...)
```
### var_roots

Qualified name: `macroforecast.models.timeseries.var_roots`

#### Signature

```python
macroforecast.models.var_roots(fit: ModelFit) -> dict[str, Any]
```

#### Description

VAR stability: moduli of the companion-matrix eigenvalues (vars::roots).

Builds the kp x kp companion matrix from the fitted lag coefficients and
returns the eigenvalue moduli (descending), the spectral radius, and
``is_stable`` (all moduli < 1, i.e. the VAR is covariance-stationary).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `fit` | positional or keyword | `ModelFit` | `required` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.var_roots(...)
```
### var_restrict

Qualified name: `macroforecast.models.timeseries.var_restrict`

#### Signature

```python
macroforecast.models.var_restrict(panel: Any, *, n_lag: int = 1, type: str = "const", season: int | None = None, threshold: float = 2.0) -> dict[str, Any]
```

#### Description

Restricted VAR by sequential elimination of regressors (R vars::restrict).

Fits a reduced-form VAR and, equation by equation, removes the regressor with
the smallest absolute ``t`` statistic whenever it falls below ``threshold``,
re-estimating after each removal (``method="ser"`` in ``vars::restrict``). This
yields a parsimonious VAR with zero restrictions on the insignificant
coefficients. Returns, per equation, the retained coefficients (and zeros for
the eliminated terms), their ``t`` statistics, the names of the eliminated
regressors, and a ``K x m`` restriction matrix (1 = retained, 0 = restricted).

#### Parameters

| Name | Kind | Type | Default |
| --- | --- | --- | --- |
| `panel` | positional or keyword | `Any` | `required` |
| `n_lag` | keyword only | `int` | `1` |
| `type` | keyword only | `str` | `"const"` |
| `season` | keyword only | `int \| None` | `None` |
| `threshold` | keyword only | `float` | `2.0` |

#### Returns

`dict[str, Any]`

#### Minimal Use

```python
import macroforecast as mf
# Call with the signature above:
# mf.models.var_restrict(...)
```
