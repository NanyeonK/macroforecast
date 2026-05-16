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

- Operational: 42 option(s)
- Future: 5 option(s)

## Options

### `ar_p`  --  operational

Autoregressive AR(p) on the target.

Pure autoregression -- predictor matrix is the lagged target (no exogenous regressors). ``params.n_lag`` sets p. Useful as a non-trivial benchmark in macro forecasting where the lagged target captures most of the predictability.

**When to use**

Default benchmark in any forecasting horse race; replication of papers reporting AR baselines.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Stock & Watson (2007) 'Why Has US Inflation Become Harder to Forecast?', JMCB 39.

**Related options**: [`var`](#var), [`factor_augmented_ar`](#factor-augmented-ar)

_Last reviewed 2026-05-04 by macroforecast author._

### `ols`  --  operational

Ordinary least squares -- baseline linear regression.

Closed-form linear regression with no regularisation. Cheapest linear estimator; appropriate when p << n and predictors are well-conditioned. Returns NaN coefficients when the design matrix is rank-deficient (sklearn raises an error in that case).

**When to use**

Low-dimensional baselines; sanity-check sweeps.

**When NOT to use**

High-dimensional panels (p ≈ n) -- use ridge / lasso instead.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Greene (2018) 'Econometric Analysis', 8th ed., Pearson.

**Related options**: [`ridge`](#ridge), [`lasso`](#lasso), [`elastic_net`](#elastic-net), [`ar_p`](#ar-p)

_Last reviewed 2026-05-04 by macroforecast author._

### `ridge`  --  operational

Ridge regression (L2-regularised OLS).

See [ridge function page](../family/ridge.md) for full documentation + parameters + standalone usage. Standalone: ``mf.functions.ridge_fit``.

### `lasso`  --  operational

Lasso regression (L1-regularised OLS).

Iterative coordinate descent: minimises ``||y - Xβ||² + α||β||₁``. Forces a subset of coefficients to exactly zero, yielding a sparse solution. Uses sklearn's ``Lasso`` with ``max_iter=20000`` for stability.

**When to use**

Variable selection; sparse forecasts on high-dimensional panels.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Tibshirani (1996) 'Regression Shrinkage and Selection via the Lasso', JRSS-B 58(1).

**Related options**: [`ridge`](#ridge), [`elastic_net`](#elastic-net), [`lasso_path`](#lasso-path)

_Last reviewed 2026-05-04 by macroforecast author._

### `elastic_net`  --  operational

Elastic net (L1 + L2 hybrid).

Combines ridge and lasso penalties via ``params.l1_ratio`` (0 = ridge, 1 = lasso). Useful when predictors are correlated and pure lasso struggles with the selection.

**When to use**

Correlated predictor blocks where lasso alone gives unstable selection.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Zou & Hastie (2005) 'Regularization and variable selection via the elastic net', JRSS-B 67(2).

**Related options**: [`ridge`](#ridge), [`lasso`](#lasso)

_Last reviewed 2026-05-04 by macroforecast author._

### `lasso_path`  --  operational

Lasso with CV-selected alpha (LassoCV).

Wraps sklearn's ``LassoCV``. Picks α automatically from a regularisation path via k-fold CV (``params.cv``). Equivalent to setting ``family: lasso, search_algorithm: cv_path``.

**When to use**

When the recipe wants automatic α selection without an explicit search_algorithm.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

**Related options**: [`lasso`](#lasso), [`ridge`](#ridge)

_Last reviewed 2026-05-04 by macroforecast author._

### `bayesian_ridge`  --  operational

Bayesian ridge with empirical-Bayes prior.

sklearn ``BayesianRidge``: gamma priors on noise + coefficient precision; type-II ML estimates of both. Returns posterior mean coefficients + posterior variance. Useful when the user wants a coefficient credible interval without bootstrapping.

**When to use**

Studies that need coefficient credible intervals; default-Bayesian baselines.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

**Related options**: [`ridge`](#ridge), [`bvar_minnesota`](#bvar-minnesota)

_Last reviewed 2026-05-04 by macroforecast author._

### `glmboost`  --  operational

Componentwise L2-boosting with linear base learners.

Bühlmann-Hothorn (2007) componentwise boosting: at each iteration picks the predictor most correlated with the residual and updates only its coefficient. Approximates lasso with a boosting interpretation.

**When to use**

Transparent feature-selection pathways; alternative to lasso.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bühlmann & Hothorn (2007) 'Boosting algorithms: Regularization, prediction and model fitting', Statistical Science 22(4).

**Related options**: [`lasso`](#lasso), [`elastic_net`](#elastic-net)

_Last reviewed 2026-05-04 by macroforecast author._

### `huber`  --  operational

Huber regression (robust to outliers).

Replaces squared loss with the Huber loss: quadratic for small residuals, linear for large ones. Down-weights outliers without removing them. ``params.epsilon`` (default 1.35) sets the transition point.

**When to use**

Series with sporadic outliers that aren't worth flagging in L2.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Huber (1964) 'Robust Estimation of a Location Parameter', Annals of Mathematical Statistics 35(1).

**Related options**: [`ols`](#ols), [`ridge`](#ridge)

_Last reviewed 2026-05-04 by macroforecast author._

### `var`  --  operational

Vector autoregression VAR(p).

Joint AR(p) over the target plus its predictors. Uses statsmodels' ``VAR`` and forecasts the target component of the joint system. Captures cross-series dynamics that single-equation AR misses.

**When to use**

Multi-series joint forecasting; impulse-response decomposition (paired with L7 ``orthogonalised_irf`` for Cholesky-identified shocks; ``generalized_irf`` reserved for the future Pesaran-Shin 1998 order-invariant variant).

**When NOT to use**

High-dimensional panels (VAR scales O(p²)); use BVAR shrinkage instead.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Sims (1980) 'Macroeconomics and Reality', Econometrica 48(1).

**Related options**: [`bvar_minnesota`](#bvar-minnesota), [`factor_augmented_var`](#factor-augmented-var), [`ar_p`](#ar-p)

_Last reviewed 2026-05-04 by macroforecast author._

### `factor_augmented_ar`  --  operational

Factor-augmented AR (PCA factors + AR lags on target).

Stock-Watson (2002) FAR: extract the first ``params.n_factors`` principal components from the predictor panel, augment with AR(``params.n_lag``) lags of the target, run OLS. Standard high-dimensional macro forecasting baseline.

**When to use**

High-dimensional macro panels (FRED-MD/QD); diffusion-index baselines.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460).

**Related options**: [`factor_augmented_var`](#factor-augmented-var), [`principal_component_regression`](#principal-component-regression), [`ar_p`](#ar-p)

_Last reviewed 2026-05-04 by macroforecast author._

### `principal_component_regression`  --  operational

Principal component regression (PCA → OLS).

Identical to ``factor_augmented_ar`` without the AR lags. Useful when the target's own lags add noise (rare but happens for highly seasonal series).

**When to use**

Diffusion-index forecasts where AR augmentation hurts performance.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

**Related options**: [`factor_augmented_ar`](#factor-augmented-ar)

_Last reviewed 2026-05-04 by macroforecast author._

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

Bagged collection of decorrelated trees. ``params.n_estimators`` (default 200) controls the ensemble size; ``params.max_depth`` controls tree complexity. Standard non-linear baseline.

**When to use**

Default non-linear benchmark; non-stationary series where linear models fail.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Breiman (2001) 'Random Forests', Machine Learning 45(1).

**Related options**: [`extra_trees`](#extra-trees), [`gradient_boosting`](#gradient-boosting), [`xgboost`](#xgboost), [`macroeconomic_random_forest`](#macroeconomic-random-forest), [`quantile_regression_forest`](#quantile-regression-forest)

_Last reviewed 2026-05-04 by macroforecast author._

### `extra_trees`  --  operational

Extremely randomized trees (sklearn).

Like RF but splits at random thresholds (no greedy search). Faster than RF; sometimes lower variance.

**v0.9 sub-axis**:
* ``params.max_features`` -- number of predictors considered at each split. ``"sqrt"`` (default) matches sklearn; ``1`` (operational, v0.9) implements Coulombe (2024) 'To Bag is to Prune' Perfectly Random Forest baseline (one random feature per split, fully random structure).

**When to use**

Quick non-linear baseline; large ensemble experiments; PRF baseline (max_features=1).

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Geurts, Ernst & Wehenkel (2006) 'Extremely randomized trees', Machine Learning 63(1).

**Related options**: [`random_forest`](#random-forest), [`gradient_boosting`](#gradient-boosting)

_Last reviewed 2026-05-04 by macroforecast author._

### `gradient_boosting`  --  operational

Gradient-boosted regression trees (sklearn).

Sklearn ``GradientBoostingRegressor``. Sequential boosting with shallow trees. ``params.n_estimators`` (default 200) and ``params.learning_rate`` (default 0.05) trade variance for bias.

**When to use**

Default boosted baseline when xgboost / lightgbm are unavailable.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Friedman (2001) 'Greedy function approximation: A gradient boosting machine', Annals of Statistics 29(5).

**Related options**: [`xgboost`](#xgboost), [`lightgbm`](#lightgbm), [`catboost`](#catboost)

_Last reviewed 2026-05-04 by macroforecast author._

### `xgboost`  --  operational

XGBoost gradient-boosted trees (optional dependency).

Requires ``pip install macroforecast[xgboost]``. Histogram-based tree construction; native quantile loss; GPU support. Standard production-grade boosting library.

**When to use**

Production sweeps where xgboost's speed matters; quantile forecasting (xgb 2.0+).

**When NOT to use**

Lightweight installs (no extra installed) -- raises ImportError.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Chen & Guestrin (2016) 'XGBoost: A Scalable Tree Boosting System', KDD.

**Related options**: [`gradient_boosting`](#gradient-boosting), [`lightgbm`](#lightgbm), [`catboost`](#catboost)

_Last reviewed 2026-05-04 by macroforecast author._

### `lightgbm`  --  operational

LightGBM gradient-boosted trees (optional dependency).

Requires ``pip install macroforecast[lightgbm]``. Leaf-wise tree growth; fast on wide / categorical-heavy panels.

**When to use**

Wide categorical panels; production sweeps where lightgbm's speed matters.

**When NOT to use**

Lightweight installs (no extra installed) -- raises ImportError.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Ke et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', NeurIPS.

**Related options**: [`xgboost`](#xgboost), [`gradient_boosting`](#gradient-boosting)

_Last reviewed 2026-05-04 by macroforecast author._

### `catboost`  --  operational

CatBoost gradient-boosted trees (optional dependency).

Requires ``pip install macroforecast[catboost]``. Ordered boosting + native categorical handling.

**When to use**

Categorical-heavy panels; ordered-boosting research.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Prokhorenkova et al. (2018) 'CatBoost: unbiased boosting with categorical features', NeurIPS.

**Related options**: [`xgboost`](#xgboost), [`lightgbm`](#lightgbm)

_Last reviewed 2026-05-04 by macroforecast author._

### `svr_linear`  --  operational

Support vector regression with linear kernel.

ε-insensitive loss + L2 regularisation. Sparse in support vectors.

Configures the ``family`` axis on ``L4_A_model_selection`` (layer ``l4``); the ``svr_linear`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Robust linear baselines; comparison against ridge.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Drucker, Burges, Kaufman, Smola & Vapnik (1997) 'Support Vector Regression Machines', NeurIPS.

**Related options**: [`svr_rbf`](#svr-rbf), [`svr_poly`](#svr-poly), [`ridge`](#ridge)

_Last reviewed 2026-05-04 by macroforecast author._

### `svr_rbf`  --  operational

Support vector regression with RBF kernel.

Non-linear regression via kernel trick. Slow on large panels (O(n³)).

Configures the ``family`` axis on ``L4_A_model_selection`` (layer ``l4``); the ``svr_rbf`` value is materialised in the recipe's ``fixed_axes`` block under that sub-layer.

**When to use**

Small / medium-dim non-linear regression; kernel-method ablations.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

**Related options**: [`svr_linear`](#svr-linear), [`svr_poly`](#svr-poly), [`random_forest`](#random-forest)

_Last reviewed 2026-05-04 by macroforecast author._

### `svr_poly`  --  operational

Support vector regression with polynomial kernel.

Polynomial-kernel SVR. Useful for studies that want explicit polynomial features without manual expansion.

**When to use**

Polynomial-kernel ablations. Selecting ``svr_poly`` on ``l4.family`` activates this branch of the layer's runtime.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

**Related options**: [`svr_rbf`](#svr-rbf), [`svr_linear`](#svr-linear)

_Last reviewed 2026-05-04 by macroforecast author._

### `kernel_ridge`  --  operational

Kernel Ridge Regression -- closed-form non-linear ridge in the dual.

Ridge regression with a non-linear kernel: ``ŷ(x) = Σ_i α_i K(x, x_i) + b`` where the dual coefficients ``α = (K + λ I)⁻¹ y`` are recovered in closed form. Operational v0.9.1 dev-stage v0.9.0F (audit-fix). Surfaces as a first-class L4 family because Coulombe / Surprenant / Leroux / Stevanovic (2022 JAE) 'How is Machine Learning Useful for Macroeconomic Forecasting?' Eq. 16 / §3.1.1 uses KRR as the headline non-linearity feature in the macro horse race.

**Tunable params**: ``alpha`` (= ridge penalty λ; default 1.0); ``kernel`` ('rbf' default / 'linear' / 'poly' / 'sigmoid' / 'laplacian' / 'chi2' -- any sklearn-supported kernel); ``gamma`` (RBF bandwidth, default sklearn auto = 1/n_features); ``degree`` (poly kernel only, default 3); ``coef0`` (poly / sigmoid, default 1.0).

Distinct from ``svr_rbf`` (ε-insensitive loss, sparsity in support vectors) and from ``ridge`` (linear). The dual representation also pairs with the L7 ``dual_decomposition`` op for kernel-weighted training-target attribution.

**When to use**

Non-linear macro forecasting baselines; KRR vs SVR-RBF / RF ablations; replicating Coulombe et al. (2022) Feature 1 nonlinearity test.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Saunders, Gammerman & Vovk (1998) 'Ridge Regression Learning Algorithm in Dual Variables', ICML.
* Coulombe, Leroux, Stevanovic & Surprenant (2022) 'How is Machine Learning Useful for Macroeconomic Forecasting?', Journal of Applied Econometrics 37(5): 920-964 -- Eq. 16 + §3.1.1.

**Related options**: [`ridge`](#ridge), [`svr_rbf`](#svr-rbf), [`dual_decomposition`](#dual-decomposition)

_Last reviewed 2026-05-04 by macroforecast author._

### `mlp`  --  operational

Multi-layer perceptron (sklearn).

Feed-forward NN with ReLU activations. ``params.hidden_layer_sizes`` controls the architecture.

**v0.9 sub-axes** (apply equally to mlp / lstm / gru / transformer):
* ``params.architecture`` -- network topology. ``standard`` (default) is the standard feed-forward / sequence variant. ``hemisphere`` (future) implements Coulombe / Frenette / Klieber (2025 JAE) HNN with separate mean / variance hemispheres joined by a constraint loss.
* ``params.loss`` -- objective. ``mse`` (default), ``quantile`` (operational via forecast_object=quantile), ``volatility_emphasis`` (future, HNN constraint loss).

**When to use**

Non-linear regression baselines; ablations against deep NN.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'

**Related options**: [`lstm`](#lstm), [`gru`](#gru), [`transformer`](#transformer)

_Last reviewed 2026-05-04 by macroforecast author._

### `lstm`  --  operational

Long short-term memory recurrent NN (torch, optional).

Requires ``pip install macroforecast[deep]``. Sequence-aware RNN with input/forget/output gates. Trains on sliding windows of the lagged feature panel.

**When to use**

Sequence-modelling studies; replication of deep-NN forecasting papers.

**When NOT to use**

Without [deep] installed -- raises NotImplementedError.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hochreiter & Schmidhuber (1997) 'Long short-term memory', Neural Computation 9(8).

**Related options**: [`gru`](#gru), [`transformer`](#transformer), [`mlp`](#mlp)

_Last reviewed 2026-05-04 by macroforecast author._

### `gru`  --  operational

Gated recurrent unit RNN (torch, optional).

Requires ``pip install macroforecast[deep]``. Simpler than LSTM (one fewer gate); often comparable on macro panels.

**When to use**

Sequence-modelling baselines; LSTM ablations.

**When NOT to use**

Without [deep] installed.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Cho et al. (2014) 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation', EMNLP.

**Related options**: [`lstm`](#lstm), [`transformer`](#transformer)

_Last reviewed 2026-05-04 by macroforecast author._

### `transformer`  --  operational

Transformer encoder regressor (torch, optional).

Requires ``pip install macroforecast[deep]``. Self-attention on the lagged feature panel. Single encoder layer; suitable as a non-linear sequence-attention baseline.

**When to use**

Attention-based macro forecasting research; sequence-NN benchmark.

**When NOT to use**

Without [deep] installed.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Vaswani et al. (2017) 'Attention is all you need', NeurIPS.

**Related options**: [`lstm`](#lstm), [`gru`](#gru)

_Last reviewed 2026-05-04 by macroforecast author._

### `knn`  --  operational

k-nearest-neighbours regression.

Memorises training data; predicts via nearest-neighbour averaging. Cheap, non-parametric.

**When to use**

Non-parametric baselines; sensitivity studies.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Cover & Hart (1967) 'Nearest neighbor pattern classification', IEEE Trans. on Information Theory 13(1).

**Related options**: [`random_forest`](#random-forest), [`svr_rbf`](#svr-rbf)

_Last reviewed 2026-05-04 by macroforecast author._

### `bvar_minnesota`  --  operational

Bayesian VAR with Minnesota prior shrinkage.

Litterman (1986) Minnesota prior: shrinks each equation toward a univariate random walk. ``params.minnesota_lambda1`` controls overall tightness; ``params.minnesota_lambda_decay`` controls lag decay; ``params.minnesota_lambda_cross`` controls cross-equation shrinkage.

Returns a closed-form posterior mean -- no MCMC. Cheap and deterministic.

**When to use**

Multi-series forecasting where standard VAR overfits; macro panels with strong unit-root behaviour.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Litterman (1986) 'Forecasting With Bayesian Vector Autoregressions -- Five Years of Experience', JBES 4(1).

**Related options**: [`bvar_normal_inverse_wishart`](#bvar-normal-inverse-wishart), [`var`](#var), [`factor_augmented_var`](#factor-augmented-var)

_Last reviewed 2026-05-04 by macroforecast author._

### `bvar_normal_inverse_wishart`  --  operational

Bayesian VAR with Normal-Inverse-Wishart prior.

Conjugate Normal-IW prior on (β, Σ); the posterior mean of β has the same closed form as Minnesota but with the prior tightness scaled to reflect parameter-uncertainty inflation. Slightly less aggressive than the bare Minnesota prior.

**When to use**

Studies preferring a fully-conjugate prior over Litterman's hand-tuned shrinkage.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Kadiyala & Karlsson (1997) 'Numerical Methods for Estimation and Inference in Bayesian VAR-models', Journal of Applied Econometrics 12(2).

**Related options**: [`bvar_minnesota`](#bvar-minnesota), [`var`](#var)

_Last reviewed 2026-05-04 by macroforecast author._

### `factor_augmented_var`  --  operational

Factor-augmented VAR (Bernanke-Boivin-Eliasz 2005).

Two-stage estimator: PCA factors from the predictor panel + VAR(``params.n_lag``) on (factors, target). Captures dynamic interactions between latent factors and the target series.

Useful for monetary-policy studies where the factors stand in for unobserved economic state.

**When to use**

Monetary-policy / macro-state studies; diffusion-index VAR baselines.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bernanke, Boivin & Eliasz (2005) 'Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive Approach', QJE 120(1).

**Related options**: [`var`](#var), [`factor_augmented_ar`](#factor-augmented-ar), [`bvar_minnesota`](#bvar-minnesota)

_Last reviewed 2026-05-04 by macroforecast author._

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

Linear-Gaussian state-space model with monthly-aggregator observation equation. Routes to ``statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`` when ``params.mixed_frequency = True`` and per-column frequency tags are supplied; otherwise falls back to the single-frequency ``DynamicFactor`` estimator (Kalman MLE).

**When to use**

Mixed-frequency nowcasting (e.g., quarterly GDP from monthly indicators).

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Mariano & Murasawa (2010) 'A coincident index, common factors, and monthly real GDP', Oxford Bulletin of Economics and Statistics 72(1).

**Related options**: [`factor_augmented_ar`](#factor-augmented-ar), [`factor_augmented_var`](#factor-augmented-var)

_Last reviewed 2026-05-04 by macroforecast author._

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

Greedy forward / backward selection of piecewise-linear hinge basis functions ``max(0, x - c)`` and their products. Atomic primitive -- sklearn does not provide a MARS implementation. Runtime wraps ``pyearth`` as an optional dep; install via ``pip install macroforecast[mars]``. Required as the base learner for the Coulombe (2024) 'MARSquake' recipe (``bagging(base_family=mars, ...)``).

Operational from v0.9.0; raises ``NotImplementedError`` with an install hint when ``pyearth`` is not present (mirrors the xgboost / lightgbm / catboost optional-dep error pattern).

**When to use**

Non-linear regression with interpretable basis functions; MARSquake recipe base learner.

**When NOT to use**

Without ``[mars]`` extra installed -- raises a clear NotImplementedError.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Friedman (1991) 'Multivariate Adaptive Regression Splines', Annals of Statistics 19(1).

**Related options**: [`gradient_boosting`](#gradient-boosting), [`decision_tree`](#decision-tree), [`bagging`](#bagging)

_Last reviewed 2026-05-04 by macroforecast author._

### `garch11`  --  operational

GARCH(1,1) univariate conditional-variance model (Bollerslev 1986).

Standard GARCH(1,1) volatility model: ``σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}``. The L4 wrapper treats ``y`` as the return-like series and ignores ``X``; ``predict(X)`` returns the conditional mean (μ broadcast over ``len(X)``) and the variance forecast is exposed via ``predict_variance(h_steps)`` for L7 inspection.

**Defaults** (paper-faithful, Bollerslev 1986 §3): ``p = q = 1``, ``mean_model = 'constant'``, ``dist = 'normal'``. Wraps ``arch.arch_model`` -- requires the optional ``[arch]`` extra (``pip install macroforecast[arch]``); raises ``NotImplementedError`` with an install hint when missing.

**When to use**

Macro / financial volatility forecasting; baseline GARCH benchmark; volatility-targeting risk applications.

**When NOT to use**

Without ``[arch]`` extra installed -- raises a clear NotImplementedError.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Bollerslev (1986) 'Generalized Autoregressive Conditional Heteroskedasticity', Journal of Econometrics 31(3): 307-327.
* Engle (1982) 'Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation', Econometrica 50(4): 987-1007.

**Related options**: [`egarch`](#egarch), [`realized_garch_with_rv_exog`](#realized-garch-with-rv-exog)

_Last reviewed 2026-05-04 by macroforecast author._

### `egarch`  --  operational

Exponential GARCH with leverage asymmetry (Nelson 1991).

EGARCH(p, o, q) on log-variance: ``ln σ²_t = ω + Σ α_i (|z_{t-i}| − E|z|) + Σ γ_i z_{t-i} + Σ β_j ln σ²_{t-j}``. The asymmetry term ``γ`` captures the leverage effect (negative shocks raise volatility more than positive ones), and the log specification removes any need for non-negativity constraints on the parameters.

**Defaults** (Nelson 1991 §3): ``p = o = q = 1``, ``mean_model = 'constant'``, ``dist = 'normal'``. Wraps ``arch.arch_model(vol='EGARCH')`` -- requires ``[arch]`` extra.

**When to use**

Asymmetric / leverage volatility; equity returns where bad news amplifies vol; macro variables with sign-asymmetric volatility responses.

**When NOT to use**

Without ``[arch]`` extra installed; symmetric volatility series where GARCH(1,1) is sufficient (parsimony).

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Nelson (1991) 'Conditional Heteroskedasticity in Asset Returns: A New Approach', Econometrica 59(2): 347-370.

**Related options**: [`garch11`](#garch11), [`realized_garch_with_rv_exog`](#realized-garch-with-rv-exog)

_Last reviewed 2026-05-04 by macroforecast author._

### `realized_garch_with_rv_exog`  --  operational

GARCH(1,1) with realised-variance series fed as the exogenous regressor (NOT Hansen-Huang-Shek 2012 joint MLE).

Phase C-3 audit-fix (M9) honest rename. The L4 wrapper consumes ``params['realized_variance']`` (a column name in ``X``) as the RV series and feeds it as the **exogenous regressor** ``x=`` into a vanilla GARCH(1,1) spec. This is useful in practice (RV improves volatility forecasts), but it is **NOT** the Hansen-Huang-Shek (2012) joint return + measurement-equation MLE: there is no ``ξ``, ``φ``, ``δ_1``, ``δ_2`` measurement-equation parameters in the fitted output. The proper RealizedGARCH spec is reserved as FUTURE under the name ``realized_garch`` (awaiting native ``arch.RealizedGARCH`` API or manual joint-MLE implementation).

Returns the conditional mean as the point forecast; ``predict_variance(h_steps)`` exposes the variance path.

**Defaults**: ``mean_model = 'constant'``, ``dist = 'normal'``. Falls back to a squared-returns proxy when the RV column is unavailable.

**When to use**

Volatility forecasting when intraday realised variance is observable as a leading indicator (RV-as-exogenous improves vol forecast); honest baseline labelling for studies that need to distinguish from the proper Hansen-Huang-Shek MLE.

**When NOT to use**

When the proper joint-MLE Realized GARCH is required (the family name ``realized_garch`` is FUTURE / unrunnable until upstream supports it); without ``[arch]`` extra installed; without an RV measurement available.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hansen, Huang & Shek (2012) 'Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility', Journal of Applied Econometrics 27(6): 877-906 — the *target* spec, not implemented here.

**Related options**: [`garch11`](#garch11), [`egarch`](#egarch)

_Last reviewed 2026-05-04 by macroforecast author._

### `ets`  --  operational

Exponential Smoothing State-Space (Hyndman-Koehler-Ord-Snyder 2008) -- ETS family.

Exponential-smoothing state-space framework: ``error_trend_seasonal`` is a 3-character code ``ETS`` where ``E ∈ {A, M}`` (additive / multiplicative error), ``T ∈ {A, M, N}`` (additive / multiplicative / no trend), ``S ∈ {A, M, N}`` (additive / multiplicative / no seasonality). Wraps ``statsmodels.tsa.exponential_smoothing.ets.ETSModel`` (MLE fitting; auto-selects the closed-form initialisation per Hyndman 2008 §5.4).

**Defaults**: ``error_trend_seasonal = 'AAN'`` (additive error, additive trend, no seasonal -- the workhorse non-seasonal spec), ``seasonal_periods = 12`` (monthly), ``initialization_method = 'estimated'``. Auto-disables seasonal fitting when ``len(y) < 2 · seasonal_periods``.

**When to use**

M-competition baseline; non-seasonal / seasonal univariate forecasting where a state-space exponential-smoothing model is the natural reference.

**When NOT to use**

Multivariate or covariate-driven forecasting (ETS ignores ``X``); short series where seasonal estimation is unstable.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Hyndman, Koehler, Ord & Snyder (2008) 'Forecasting with Exponential Smoothing: The State Space Approach', Springer.
* Hyndman & Athanasopoulos (2018) 'Forecasting: Principles and Practice', 2nd ed., OTexts §7.

**Related options**: [`theta_method`](#theta-method), [`holt_winters`](#holt-winters), [`ar_p`](#ar-p)

_Last reviewed 2026-05-04 by macroforecast author._

### `theta_method`  --  operational

Theta method (Assimakopoulos-Nikolopoulos 2000) -- M3-competition winning baseline.

Hand-coded Theta(2) closed-form forecast: blends a long-run linear-trend regression with a short-run simple-exponential-smoothing (SES) level. For ``θ = 2`` (M3 winner), the h-step-ahead forecast is ``ŷ_{T+h} = 0.5 · (a + b · (T+h)) + 0.5 · ℓ_T``, where ``(a, b)`` are the OLS trend slope/intercept on time index and ``ℓ_T`` is the SES level at time T (smoothing parameter ``α`` selected via ``scipy.optimize.minimize_scalar`` minimising the in-sample 1-step MSE).

**Defaults**: ``theta = 2.0`` (M3 winner), ``seasonal = False``, ``seasonal_periods = 12``. The constructor exposes ``theta`` for forward compatibility; only the θ=2 closed form is exercised in v0.9.0 -- general θ requires a θ-line decomposition out of scope for this run.

**When to use**

M3 / M4-style univariate baselines; quick reference forecast against more elaborate models.

**When NOT to use**

Strongly seasonal series (use ``holt_winters`` or seasonally-adjusted target); covariate-driven forecasting.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Assimakopoulos & Nikolopoulos (2000) 'The theta model: a decomposition approach to forecasting', International Journal of Forecasting 16(4): 521-530.
* Hyndman & Billah (2003) 'Unmasking the Theta method', International Journal of Forecasting 19(2): 287-290.
* Petropoulos et al. (2022) 'Forecasting: theory and practice', International Journal of Forecasting 38(3): 705-871.

**Related options**: [`ets`](#ets), [`holt_winters`](#holt-winters), [`ar_p`](#ar-p)

_Last reviewed 2026-05-04 by macroforecast author._

### `holt_winters`  --  operational

Holt-Winters additive / multiplicative seasonal exponential smoothing.

Wraps ``statsmodels.tsa.holtwinters.ExponentialSmoothing``. Fits level / trend / seasonal smoothing parameters by MLE (``optimized=True``). Supports additive and multiplicative trend and seasonal components plus an optional damped trend (Hyndman et al. 2008 §3).

**Defaults**: ``seasonal = 'add'``, ``seasonal_periods = 12``, ``trend = 'add'``, ``damped_trend = False``. Auto-disables seasonal fitting when ``len(y) < 2 · seasonal_periods``.

**When to use**

Seasonal univariate baselines; M-competition style benchmarking; standard reference forecast for monthly / quarterly macro series.

**When NOT to use**

Without a clear seasonal pattern (use ``ets`` AAN instead); covariate-driven forecasting.

**References**

* macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'
* Holt (2004 / orig. 1957) 'Forecasting seasonals and trends by exponentially weighted moving averages', International Journal of Forecasting 20(1): 5-10.
* Winters (1960) 'Forecasting Sales by Exponentially Weighted Moving Averages', Management Science 6(3): 324-342.
* Hyndman & Athanasopoulos (2018) 'Forecasting: Principles and Practice', 2nd ed., OTexts §7.

**Related options**: [`ets`](#ets), [`theta_method`](#theta-method), [`ar_p`](#ar-p)

_Last reviewed 2026-05-04 by macroforecast author._

### `midas_almon`  --  future

_(no schema description for `midas_almon`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l4.py`` are welcome.

### `midas_beta`  --  future

_(no schema description for `midas_beta`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l4.py`` are welcome.

### `midas_step`  --  future

_(no schema description for `midas_step`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l4.py`` are welcome.

### `dfm_unrestricted_midas`  --  future

_(no schema description for `dfm_unrestricted_midas`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l4.py`` are welcome.

### `realized_garch`  --  future

_(no schema description for `realized_garch`)_

> TBD: option doc not yet authored for this value. The encyclopedia falls back to the bare schema description above. PRs adding a full ``OptionDoc`` entry under ``macroforecast/scaffold/option_docs/l4.py`` are welcome.
