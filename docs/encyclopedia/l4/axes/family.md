# `family`

[Back to L4](../index.md) | [Browse all axes](../../browse_by_axis.md) | [Browse all options](../../browse_by_option.md)

> Axis ``family`` on sub-layer ``L4_A_model_selection`` (layer ``l4``).

## Sub-layer

**L4_A_model_selection**

## Axis metadata

- Default: `'ridge'`
- Sweepable: True
- Status: operational

## Operational status summary

- Operational: 46 option(s)
- Future: 1 option(s)

## Options

### `ar_p`  --  operational

Autoregressive AR(p) on the target.

See [ar_p function page](../family/ar_p.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.ar_fit``.

### `ols`  --  operational

Ordinary least squares -- baseline linear regression.

See [ols function page](../family/ols.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.ols_fit``.

### `ridge`  --  operational

Ridge regression (L2-regularised OLS).

See [ridge function page](../family/ridge.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.ridge_fit``.

### `lasso`  --  operational

Lasso regression (L1-regularised OLS).

See [lasso function page](../family/lasso.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.lasso_fit``.

### `elastic_net`  --  operational

Elastic net (L1 + L2 hybrid).

See [elastic_net function page](../family/elastic_net.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.elastic_net_fit``.

### `lasso_path`  --  operational

Lasso with CV-selected alpha (LassoCV).

See [lasso_path function page](../family/lasso_path.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.lasso_path_fit``.

### `bayesian_ridge`  --  operational

Bayesian ridge with empirical-Bayes prior.

See [bayesian_ridge function page](../family/bayesian_ridge.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.bayesian_ridge_fit``.

### `glmboost`  --  operational

Componentwise L2-boosting with linear base learners.

See [glmboost function page](../family/glmboost.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.glmboost_fit``.

### `huber`  --  operational

Huber regression (robust to outliers).

See [huber function page](../family/huber.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.huber_fit``.

### `var`  --  operational

Vector autoregression VAR(p).

See [var function page](../family/var.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.var_fit``.

### `factor_augmented_ar`  --  operational

Factor-augmented AR (PCA factors + AR lags on target).

See [factor_augmented_ar function page](../family/factor_augmented_ar.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.far_fit``.

### `principal_component_regression`  --  operational

Principal component regression (PCA → OLS).

See [principal_component_regression function page](../family/principal_component_regression.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.pcr_fit``.

### `decision_tree`  --  operational

Single decision tree (sklearn) or Slow-Growing Tree (SGT, in-fit shrinkage).

Cheapest non-linear model. Useful as an ablation against random forests / boosting -- if a single tree matches RF performance, the ensemble isn't buying much.

**v0.9 sub-axis ``params.split_shrinkage`` = η** -- per-split soft-weight learning rate. ``0.0`` (default) keeps the standard greedy sklearn CART. Non-zero values activate **Slow-Growing Trees** (Goulet Coulombe 2024 — operational v0.9.1 dev-stage v0.9.0B-6): a *soft-weighted* tree where rows on the side that does not satisfy the split rule receive weight ``(1 − η)`` instead of 0. Implements Algorithm 1 from the paper exactly: leaf weights propagate multiplicatively through every split, the Herfindahl index ``H_l = Σ(ω²)/(Σω)²`` of the leaf weight vector controls stopping, and the leaf prediction is the weighted mean.

Limit cases (verified by tests):
  * η = 1.0 → recovers standard CART (hard splits).
  * η ≈ 0.1, H̄ ≈ 0.05 → SGT regime ('matches RF on Linear DGP at high R²' per paper Figure 2).

**Sub-axis params** (only consulted when split_shrinkage ≠ 0):
  * ``herfindahl_threshold`` = H̄ (default 0.25; smaller → deeper tree). Practice: ``{0.05, 0.1, 0.25}`` per paper.
  * ``eta_depth_step`` -- paper rule-of-thumb increases η by ``eta_depth_step·depth`` per level (default 0.0 keeps η constant).
  * ``max_depth`` -- additional safety bound on tree depth.

**Note**: sklearn ``DecisionTreeRegressor`` cannot reproduce SGT via post-fit leaf-multiplication because the *splits themselves* depend on soft weights (every row, including rule-violators, contributes to the SSE objective). The custom helper ``_SlowGrowingTree`` implements the soft-weighted CART from scratch.

**When to use**

Ablation studies; cheap non-linear baselines; SLOTH single-tree replacement for RF on small samples.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Breiman, Friedman, Stone & Olshen (1984) 'Classification and Regression Trees', CRC Press.
* Goulet Coulombe (2024) 'Slow-Growing Trees', in Machine Learning for Econometrics and Related Topics, Studies in Systems, Decision and Control 508 (Springer). doi:10.1007/978-3-031-43601-7_4.

**Related options**: [`random_forest`](#random-forest), [`extra_trees`](#extra-trees), [`gradient_boosting`](#gradient-boosting)

_Last reviewed 2026-05-04 by macroforecast author._

### `random_forest`  --  operational

Random forest (sklearn).

See [random_forest function page](../family/random_forest.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.random_forest_fit``.

### `extra_trees`  --  operational

Extremely randomized trees (sklearn).

See [extra_trees function page](../family/extra_trees.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.extra_trees_fit``.

### `gradient_boosting`  --  operational

Gradient-boosted regression trees (sklearn).

See [gradient_boosting function page](../family/gradient_boosting.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.gradient_boosting_fit``.

### `xgboost`  --  operational

XGBoost gradient-boosted trees (optional dependency).

See [xgboost function page](../family/xgboost.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.xgboost_fit``.

### `lightgbm`  --  operational

LightGBM gradient-boosted trees (optional dependency).

See [lightgbm function page](../family/lightgbm.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.lightgbm_fit``.

### `catboost`  --  operational

CatBoost gradient-boosted trees (optional dependency).

See [catboost function page](../family/catboost.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.catboost_fit``.

### `svr_linear`  --  operational

Support vector regression with linear kernel.

See [svr_linear function page](../family/svr_linear.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.svr_linear_fit``.

### `svr_rbf`  --  operational

Support vector regression with RBF kernel.

See [svr_rbf function page](../family/svr_rbf.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.svr_rbf_fit``.

### `svr_poly`  --  operational

Support vector regression with polynomial kernel.

See [svr_poly function page](../family/svr_poly.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.svr_poly_fit``.

### `kernel_ridge`  --  operational

Kernel Ridge Regression -- closed-form non-linear ridge in the dual.

See [kernel_ridge function page](../family/kernel_ridge.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.kernel_ridge_fit``.

### `mlp`  --  operational

Multi-layer perceptron (sklearn).

See [mlp function page](../family/mlp.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.mlp_fit``.

### `lstm`  --  operational

Long short-term memory recurrent NN (torch, optional).

See [lstm function page](../family/lstm.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.lstm_fit``.

### `gru`  --  operational

Gated recurrent unit RNN (torch, optional).

See [gru function page](../family/gru.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.gru_fit``.

### `transformer`  --  operational

Transformer encoder regressor (torch, optional).

See [transformer function page](../family/transformer.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.transformer_fit``.

### `knn`  --  operational

k-nearest-neighbours regression.

See [knn function page](../family/knn.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.knn_fit``.

### `bvar_minnesota`  --  operational

Bayesian VAR with Minnesota prior shrinkage.

See [bvar_minnesota function page](../family/bvar_minnesota.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.bvar_minnesota_fit``.

### `bvar_normal_inverse_wishart`  --  operational

Bayesian VAR with Normal-Inverse-Wishart prior.

See [bvar_normal_inverse_wishart function page](../family/bvar_normal_inverse_wishart.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.bvar_niw_fit``.

### `factor_augmented_var`  --  operational

Factor-augmented VAR (Bernanke-Boivin-Eliasz 2005).

See [factor_augmented_var function page](../family/factor_augmented_var.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.favar_fit``.

### `macroeconomic_random_forest`  --  operational

Goulet Coulombe (2024) MRF: random walk regularised forest with per-leaf local linear regression and Block Bayesian Bootstrap forecast ensembles.

Macroeconomic Random Forest. Each leaf fits a local linear regression of y on the state vector S; coefficient series are smoothed via random-walk regularisation (``rw_regul`` parameter); forecast ensembles use the Block Bayesian Bootstrap (Taddy 2015 extension). Surfaces Generalised Time-Varying Parameters (GTVPs) via the L7 ``mrf_gtvp`` op.

Backed by Ryan Lucas's reference implementation, vendored under ``macroforecast/_vendor/macro_random_forest/`` with surgical numpy 2.x / pandas 2.x compatibility patches (no algorithmic changes). Upstream: https://github.com/RyanLucas3/MacroRandomForest. No extra required -- the family works out of the box.

**Citation requirement**: research using this family must cite Goulet Coulombe (2024) 'The Macroeconomy as a Random Forest', Journal of Applied Econometrics (arXiv:2006.12724) and acknowledge the upstream implementation by Ryan Lucas (https://github.com/RyanLucas3/MacroRandomForest).

Tunable params: ``B`` (bootstrap iterations, default 50), ``ridge_lambda`` (default 0.1), ``rw_regul`` (RW penalty 0..1, default 0.75), ``mtry_frac`` (default 1/3), ``trend_push`` (default 1), ``quantile_rate`` (default 0.3), ``fast_rw`` (default True), ``parallelise`` (default False), ``n_cores`` (default 1).

**When to use**

Macro forecasting with non-stationary parameter dynamics; alternative to switching models.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Goulet Coulombe, P. (2024) 'The Macroeconomy as a Random Forest', Journal of Applied Econometrics. arXiv:2006.12724.
* Lucas, R. (2022) 'MacroRandomForest' (Python implementation). https://github.com/RyanLucas3/MacroRandomForest. MIT licence.

**Related options**: [`random_forest`](#random-forest), [`bvar_minnesota`](#bvar-minnesota)

_Last reviewed 2026-05-04 by macroforecast author._

### `dfm_mixed_mariano_murasawa`  --  operational

Mariano-Murasawa-style mixed-frequency dynamic factor model.

See [dfm_mixed_mariano_murasawa function page](../family/dfm_mixed_mariano_murasawa.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.dfm_fit``.

### `quantile_regression_forest`  --  operational

Meinshausen (2006) quantile regression forest.

Records the per-leaf empirical training-target distribution and forecasts arbitrary quantiles by averaging leaf-conditional CDFs. Surfaces ``forecast_intervals`` directly without a Gaussian shortcut. Pairs with ``forecast_object: quantile``.

**When to use**

Growth-at-risk / VaR studies; density forecasting.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Meinshausen (2006) 'Quantile Regression Forests', JMLR 7.

**Related options**: [`random_forest`](#random-forest), [`bagging`](#bagging)

_Last reviewed 2026-05-04 by macroforecast author._

### `bagging`  --  operational

Bootstrap-aggregating wrapper around any base family; supports Booging.

``params.base_family`` selects the base estimator; ``params.n_estimators`` (default 50) bootstrap resamples are fit; predict averages. ``predict_quantiles`` surfaces empirical bag-quantiles.

**v0.9 sub-axis ``params.strategy``** -- bag composition strategy:
* ``standard`` (default) -- plain Breiman (1996) bagging; i.i.d. bootstrap with replacement.
* ``block`` (operational v0.8.9) -- *circular* moving-block bootstrap (Künsch 1989 variant: block starts wrap at n via modulo) for serially-correlated panels (Taddy 2015 ext. used in MRF).
* ``booging`` (operational v0.9.1) -- Goulet Coulombe (2024) 'To Bag is to Prune'. Outer ``B`` bags of (intentionally over-fitted) inner Stochastic Gradient Boosted Trees + Data Augmentation: each predictor column is duplicated as ``X̃_k = X_k + N(0, (σ_k · da_noise_frac)²)``; per-bag column-drop of rate ``da_drop_rate`` (default 0.2); inner SGB at ``inner_n_estimators=500, inner_learning_rate=0.1, inner_max_depth=4, inner_subsample=0.5``. Sampling without replacement at ``max_samples=0.75``. Outer ``n_estimators=B`` (default 100) replaces tuning the boosting depth ``S`` -- the bag-prune theorem (paper §2) lets us over-fit the inner SGB and let the bag average prune. Helper ``_BoogingWrapper``.
* ``sequential_residual`` -- legacy alias for ``booging`` retained for back-compat. Pre-2026-05-07 plan sketch ("K rounds bag-on-residuals") was an inaccurate description of the same paper's algorithm; the option now routes to the outer-bagging-of-inner-SGB construction.

**When to use**

Variance reduction on noisy series; quantile bands without quantile regression; Booging / block-bootstrap recipes; over-fit-then-bag pruning.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Breiman (1996) 'Bagging Predictors', Machine Learning 24(2).
* Künsch (1989) 'The jackknife and the bootstrap for general stationary observations', Annals of Statistics 17(3) -- moving-block variant.
* Goulet Coulombe (2024) 'To Bag is to Prune', arXiv:2008.07063 -- Booging algorithm.

**Related options**: [`random_forest`](#random-forest), [`extra_trees`](#extra-trees), [`quantile_regression_forest`](#quantile-regression-forest), [`gradient_boosting`](#gradient-boosting)

_Last reviewed 2026-05-04 by macroforecast author._

### `mars`  --  operational

Multivariate Adaptive Regression Splines (Friedman 1991).

See [mars function page](../family/mars.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.mars_fit``.

### `garch11`  --  operational

GARCH(1,1) univariate conditional-variance model (Bollerslev 1986).

See [garch11 function page](../family/garch11.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.garch11_fit``.

### `egarch`  --  operational

Exponential GARCH with leverage asymmetry (Nelson 1991).

See [egarch function page](../family/egarch.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.egarch_fit``.

### `realized_garch_with_rv_exog`  --  operational

GARCH(1,1) with realised-variance series fed as the exogenous regressor (NOT Hansen-Huang-Shek 2012 joint MLE).

See [realized_garch_with_rv_exog function page](../family/realized_garch_with_rv_exog.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.realized_garch_fit``.

### `ets`  --  operational

Exponential Smoothing State-Space (Hyndman-Koehler-Ord-Snyder 2008) -- ETS family.

See [ets function page](../family/ets.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.ets_fit``.

### `theta_method`  --  operational

Theta method (Assimakopoulos-Nikolopoulos 2000) -- M3-competition winning baseline.

See [theta_method function page](../family/theta_method.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.theta_fit``.

### `holt_winters`  --  operational

Holt-Winters additive / multiplicative seasonal exponential smoothing.

See [holt_winters function page](../family/holt_winters.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.holt_winters_fit``.

### `midas_almon`  --  operational

MIDAS with Almon polynomial lag weights (Ghysels-Santa-Clara-Valkanov 2004).

See [midas_almon function page](../family/midas_almon.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.midas_almon``.

### `midas_beta`  --  operational

MIDAS with Beta distribution kernel lag weights (Ghysels-Sinko-Valkanov 2007).

See [midas_beta function page](../family/midas_beta.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.midas_beta``.

### `midas_step`  --  operational

MIDAS with piecewise-constant step-function weights, OLS (Foroni-Marcellino-Schumacher 2015).

See [midas_step function page](../family/midas_step.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.midas_step``.

### `dfm_unrestricted_midas`  --  operational

Unrestricted MIDAS (U-MIDAS) -- OLS on all HF lags (Foroni-Marcellino-Schumacher 2015).

See [dfm_unrestricted_midas function page](../family/dfm_unrestricted_midas.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.dfm_unrestricted_midas``.

### `realized_garch`  --  future

_(no schema description for `realized_garch`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l4.py`` are welcome.
