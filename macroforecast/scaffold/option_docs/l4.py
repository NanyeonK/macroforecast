"""L4 forecasting model -- per-option documentation.

L4 is the second-largest layer by entry count: 35+ operational model
families plus axes for forecast strategy, training window, refit
policy, and tuning. Most users iterate primarily on L4 -- pick a
family, tune it, repeat.

This module ships Tier-1 docs for every operational L4 ``family``
option (35 entries) plus the four other L4 axes (forecast_strategy,
training_start_rule, refit_policy, search_algorithm). Each family
entry follows the same template: summary + algorithm description +
when_to_use + when_not_to_use + key references.
"""
from __future__ import annotations

from . import register
from .types import CodeExample, OptionDoc, Reference

_REVIEWED = "2026-05-04"
_REVIEWER = "macroforecast author"

_REF_DESIGN_L4 = Reference(
    citation="macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'",
)


def _f(option: str, summary: str, description: str, when_to_use: str,
       *, when_not_to_use: str = "", references: tuple[Reference, ...] = (_REF_DESIGN_L4,),
       related_options: tuple[str, ...] = ()) -> OptionDoc:
    return OptionDoc(
        layer="l4", sublayer="L4_A_model_selection", axis="family", option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, references=references,
        related_options=related_options,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


# Linear / regularised regression family
_F_OLS = _f(
    "ols",
    "Ordinary least squares -- baseline linear regression.",
    (
        "Closed-form linear regression with no regularisation. Cheapest "
        "linear estimator; appropriate when p << n and predictors are "
        "well-conditioned. Returns NaN coefficients when the design "
        "matrix is rank-deficient (sklearn raises an error in that "
        "case)."
    ),
    "Low-dimensional baselines; sanity-check sweeps.",
    when_not_to_use="High-dimensional panels (p ≈ n) -- use ridge / lasso instead.",
    references=(_REF_DESIGN_L4, Reference(citation="Greene (2018) 'Econometric Analysis', 8th ed., Pearson.")),
    related_options=("ridge", "lasso", "elastic_net", "ar_p"),
)

_F_RIDGE = _f(
    "ridge",
    "Ridge regression (L2-regularised OLS).",
    (
        "Closed-form ridge: ``β = (X'X + αI)⁻¹ X'y``. Shrinks coefficients "
        "toward zero proportional to the regularisation strength α "
        "(``params.alpha``).\n\n"
        "Default α = 1.0. The ``cv_path`` search algorithm uses "
        "``RidgeCV`` to pick α from a grid via leave-one-out CV; the "
        "``grid_search`` / ``random_search`` algorithms can sweep over "
        "leaf_config.tuning_grid['alpha']."
    ),
    "High-dimensional macro panels with collinear predictors; standard benchmark.",
    references=(_REF_DESIGN_L4, Reference(citation="Hoerl & Kennard (1970) 'Ridge regression: biased estimation for nonorthogonal problems', Technometrics 12(1).")),
    related_options=("lasso", "elastic_net", "lasso_path"),
)

_F_LASSO = _f(
    "lasso",
    "Lasso regression (L1-regularised OLS).",
    (
        "Iterative coordinate descent: minimises ``||y - Xβ||² + α||β||₁``. "
        "Forces a subset of coefficients to exactly zero, yielding a "
        "sparse solution. Uses sklearn's ``Lasso`` with ``max_iter=20000`` "
        "for stability."
    ),
    "Variable selection; sparse forecasts on high-dimensional panels.",
    references=(_REF_DESIGN_L4, Reference(citation="Tibshirani (1996) 'Regression Shrinkage and Selection via the Lasso', JRSS-B 58(1).")),
    related_options=("ridge", "elastic_net", "lasso_path"),
)

_F_ELASTIC_NET = _f(
    "elastic_net",
    "Elastic net (L1 + L2 hybrid).",
    (
        "Combines ridge and lasso penalties via "
        "``params.l1_ratio`` (0 = ridge, 1 = lasso). Useful when "
        "predictors are correlated and pure lasso struggles with the "
        "selection."
    ),
    "Correlated predictor blocks where lasso alone gives unstable selection.",
    references=(_REF_DESIGN_L4, Reference(citation="Zou & Hastie (2005) 'Regularization and variable selection via the elastic net', JRSS-B 67(2).")),
    related_options=("ridge", "lasso"),
)

_F_LASSO_PATH = _f(
    "lasso_path",
    "Lasso with CV-selected alpha (LassoCV).",
    (
        "Wraps sklearn's ``LassoCV``. Picks α automatically from a "
        "regularisation path via k-fold CV (``params.cv``). Equivalent "
        "to setting ``family: lasso, search_algorithm: cv_path``."
    ),
    "When the recipe wants automatic α selection without an explicit search_algorithm.",
    references=(_REF_DESIGN_L4,),
    related_options=("lasso", "ridge"),
)

_F_BAYESIAN_RIDGE = _f(
    "bayesian_ridge",
    "Bayesian ridge with empirical-Bayes prior.",
    (
        "sklearn ``BayesianRidge``: gamma priors on noise + coefficient "
        "precision; type-II ML estimates of both. Returns posterior "
        "mean coefficients + posterior variance. Useful when the user "
        "wants a coefficient credible interval without bootstrapping."
    ),
    "Studies that need coefficient credible intervals; default-Bayesian baselines.",
    references=(_REF_DESIGN_L4,),
    related_options=("ridge", "bvar_minnesota"),
)

_F_HUBER = _f(
    "huber",
    "Huber regression (robust to outliers).",
    (
        "Replaces squared loss with the Huber loss: quadratic for small "
        "residuals, linear for large ones. Down-weights outliers without "
        "removing them. ``params.epsilon`` (default 1.35) sets the "
        "transition point."
    ),
    "Series with sporadic outliers that aren't worth flagging in L2.",
    references=(_REF_DESIGN_L4, Reference(citation="Huber (1964) 'Robust Estimation of a Location Parameter', Annals of Mathematical Statistics 35(1).")),
    related_options=("ols", "ridge"),
)

_F_GLMBOOST = _f(
    "glmboost",
    "Componentwise L2-boosting with linear base learners.",
    (
        "Bühlmann-Hothorn (2007) componentwise boosting: at each "
        "iteration picks the predictor most correlated with the residual "
        "and updates only its coefficient. Approximates lasso with a "
        "boosting interpretation."
    ),
    "Transparent feature-selection pathways; alternative to lasso.",
    references=(_REF_DESIGN_L4, Reference(citation="Bühlmann & Hothorn (2007) 'Boosting algorithms: Regularization, prediction and model fitting', Statistical Science 22(4).")),
    related_options=("lasso", "elastic_net"),
)


# Time-series / autoregressive
_F_AR_P = _f(
    "ar_p",
    "Autoregressive AR(p) on the target.",
    (
        "Pure autoregression -- predictor matrix is the lagged target "
        "(no exogenous regressors). ``params.n_lag`` sets p. Useful as "
        "a non-trivial benchmark in macro forecasting where the lagged "
        "target captures most of the predictability."
    ),
    "Default benchmark in any forecasting horse race; replication of papers reporting AR baselines.",
    references=(_REF_DESIGN_L4, Reference(citation="Stock & Watson (2007) 'Why Has US Inflation Become Harder to Forecast?', JMCB 39.")),
    related_options=("var", "factor_augmented_ar"),
)

_F_VAR = _f(
    "var",
    "Vector autoregression VAR(p).",
    (
        "Joint AR(p) over the target plus its predictors. Uses statsmodels' "
        "``VAR`` and forecasts the target component of the joint system. "
        "Captures cross-series dynamics that single-equation AR misses."
    ),
    "Multi-series joint forecasting; impulse-response decomposition (paired with L7 ``generalized_irf``).",
    when_not_to_use="High-dimensional panels (VAR scales O(p²)); use BVAR shrinkage instead.",
    references=(_REF_DESIGN_L4, Reference(citation="Sims (1980) 'Macroeconomics and Reality', Econometrica 48(1).")),
    related_options=("bvar_minnesota", "factor_augmented_var", "ar_p"),
)

_F_FAR = _f(
    "factor_augmented_ar",
    "Factor-augmented AR (PCA factors + AR lags on target).",
    (
        "Stock-Watson (2002) FAR: extract the first ``params.n_factors`` "
        "principal components from the predictor panel, augment with "
        "AR(``params.n_lag``) lags of the target, run OLS. Standard "
        "high-dimensional macro forecasting baseline."
    ),
    "High-dimensional macro panels (FRED-MD/QD); diffusion-index baselines.",
    references=(_REF_DESIGN_L4, Reference(citation="Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460).")),
    related_options=("factor_augmented_var", "principal_component_regression", "ar_p"),
)

_F_PCR = _f(
    "principal_component_regression",
    "Principal component regression (PCA → OLS).",
    (
        "Identical to ``factor_augmented_ar`` without the AR lags. Useful "
        "when the target's own lags add noise (rare but happens for "
        "highly seasonal series)."
    ),
    "Diffusion-index forecasts where AR augmentation hurts performance.",
    references=(_REF_DESIGN_L4,),
    related_options=("factor_augmented_ar",),
)

_F_FAVAR = _f(
    "factor_augmented_var",
    "Factor-augmented VAR (Bernanke-Boivin-Eliasz 2005).",
    (
        "Two-stage estimator: PCA factors from the predictor panel + "
        "VAR(``params.n_lag``) on (factors, target). Captures dynamic "
        "interactions between latent factors and the target series.\n\n"
        "Useful for monetary-policy studies where the factors stand in "
        "for unobserved economic state."
    ),
    "Monetary-policy / macro-state studies; diffusion-index VAR baselines.",
    references=(_REF_DESIGN_L4, Reference(citation="Bernanke, Boivin & Eliasz (2005) 'Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive Approach', QJE 120(1).")),
    related_options=("var", "factor_augmented_ar", "bvar_minnesota"),
)

_F_BVAR_MIN = _f(
    "bvar_minnesota",
    "Bayesian VAR with Minnesota prior shrinkage.",
    (
        "Litterman (1986) Minnesota prior: shrinks each equation toward "
        "a univariate random walk. ``params.minnesota_lambda1`` controls "
        "overall tightness; ``params.minnesota_lambda_decay`` controls "
        "lag decay; ``params.minnesota_lambda_cross`` controls "
        "cross-equation shrinkage.\n\n"
        "Returns a closed-form posterior mean -- no MCMC. Cheap and "
        "deterministic."
    ),
    "Multi-series forecasting where standard VAR overfits; macro panels with strong unit-root behaviour.",
    references=(_REF_DESIGN_L4, Reference(citation="Litterman (1986) 'Forecasting With Bayesian Vector Autoregressions -- Five Years of Experience', JBES 4(1).")),
    related_options=("bvar_normal_inverse_wishart", "var", "factor_augmented_var"),
)

_F_BVAR_NIW = _f(
    "bvar_normal_inverse_wishart",
    "Bayesian VAR with Normal-Inverse-Wishart prior.",
    (
        "Conjugate Normal-IW prior on (β, Σ); the posterior mean of β "
        "has the same closed form as Minnesota but with the prior "
        "tightness scaled to reflect parameter-uncertainty inflation. "
        "Slightly less aggressive than the bare Minnesota prior."
    ),
    "Studies preferring a fully-conjugate prior over Litterman's hand-tuned shrinkage.",
    references=(_REF_DESIGN_L4, Reference(citation="Kadiyala & Karlsson (1997) 'Numerical Methods for Estimation and Inference in Bayesian VAR-models', Journal of Applied Econometrics 12(2).")),
    related_options=("bvar_minnesota", "var"),
)

_F_DFM_MM = _f(
    "dfm_mixed_mariano_murasawa",
    "Mariano-Murasawa-style mixed-frequency dynamic factor model.",
    (
        "Linear-Gaussian state-space model with monthly-aggregator "
        "observation equation. Routes to "
        "``statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`` "
        "when ``params.mixed_frequency = True`` and per-column "
        "frequency tags are supplied; otherwise falls back to the "
        "single-frequency ``DynamicFactor`` estimator (Kalman MLE)."
    ),
    "Mixed-frequency nowcasting (e.g., quarterly GDP from monthly indicators).",
    references=(_REF_DESIGN_L4, Reference(citation="Mariano & Murasawa (2010) 'A coincident index, common factors, and monthly real GDP', Oxford Bulletin of Economics and Statistics 72(1).")),
    related_options=("factor_augmented_ar", "factor_augmented_var"),
)


# Tree / boosting / forest
_F_DECISION_TREE = _f(
    "decision_tree",
    "Single decision tree (sklearn).",
    "Cheapest non-linear model. Useful as an ablation against random forests / boosting -- if a single tree matches RF performance, the ensemble isn't buying much.",
    "Ablation studies; cheap non-linear baselines.",
    references=(_REF_DESIGN_L4, Reference(citation="Breiman, Friedman, Stone & Olshen (1984) 'Classification and Regression Trees', CRC Press.")),
    related_options=("random_forest", "extra_trees", "gradient_boosting"),
)

_F_RF = _f(
    "random_forest",
    "Random forest (sklearn).",
    (
        "Bagged collection of decorrelated trees. ``params.n_estimators`` "
        "(default 200) controls the ensemble size; ``params.max_depth`` "
        "controls tree complexity. Standard non-linear baseline."
    ),
    "Default non-linear benchmark; non-stationary series where linear models fail.",
    references=(_REF_DESIGN_L4, Reference(citation="Breiman (2001) 'Random Forests', Machine Learning 45(1).")),
    related_options=("extra_trees", "gradient_boosting", "xgboost", "macroeconomic_random_forest", "quantile_regression_forest"),
)

_F_EXTRA_TREES = _f(
    "extra_trees",
    "Extremely randomized trees (sklearn).",
    "Like RF but splits at random thresholds (no greedy search). Faster than RF; sometimes lower variance.",
    "Quick non-linear baseline; large ensemble experiments.",
    references=(_REF_DESIGN_L4, Reference(citation="Geurts, Ernst & Wehenkel (2006) 'Extremely randomized trees', Machine Learning 63(1).")),
    related_options=("random_forest", "gradient_boosting"),
)

_F_GB = _f(
    "gradient_boosting",
    "Gradient-boosted regression trees (sklearn).",
    (
        "Sklearn ``GradientBoostingRegressor``. Sequential boosting with "
        "shallow trees. ``params.n_estimators`` (default 200) and "
        "``params.learning_rate`` (default 0.05) trade variance for "
        "bias."
    ),
    "Default boosted baseline when xgboost / lightgbm are unavailable.",
    references=(_REF_DESIGN_L4, Reference(citation="Friedman (2001) 'Greedy function approximation: A gradient boosting machine', Annals of Statistics 29(5).")),
    related_options=("xgboost", "lightgbm", "catboost"),
)

_F_XGB = _f(
    "xgboost",
    "XGBoost gradient-boosted trees (optional dependency).",
    (
        "Requires ``pip install macroforecast[xgboost]``. Histogram-based "
        "tree construction; native quantile loss; GPU support. Standard "
        "production-grade boosting library."
    ),
    "Production sweeps where xgboost's speed matters; quantile forecasting (xgb 2.0+).",
    when_not_to_use="Lightweight installs (no extra installed) -- raises ImportError.",
    references=(_REF_DESIGN_L4, Reference(citation="Chen & Guestrin (2016) 'XGBoost: A Scalable Tree Boosting System', KDD.")),
    related_options=("gradient_boosting", "lightgbm", "catboost"),
)

_F_LGBM = _f(
    "lightgbm",
    "LightGBM gradient-boosted trees (optional dependency).",
    (
        "Requires ``pip install macroforecast[lightgbm]``. Leaf-wise tree "
        "growth; fast on wide / categorical-heavy panels."
    ),
    "Wide categorical panels; production sweeps where lightgbm's speed matters.",
    when_not_to_use="Lightweight installs (no extra installed) -- raises ImportError.",
    references=(_REF_DESIGN_L4, Reference(citation="Ke et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', NeurIPS.")),
    related_options=("xgboost", "gradient_boosting"),
)

_F_CAT = _f(
    "catboost",
    "CatBoost gradient-boosted trees (optional dependency).",
    "Requires ``pip install macroforecast[catboost]``. Ordered boosting + native categorical handling.",
    "Categorical-heavy panels; ordered-boosting research.",
    references=(_REF_DESIGN_L4, Reference(citation="Prokhorenkova et al. (2018) 'CatBoost: unbiased boosting with categorical features', NeurIPS.")),
    related_options=("xgboost", "lightgbm"),
)

_F_MRF = _f(
    "macroeconomic_random_forest",
    "Coulombe (2024) GTVP random forest (per-leaf local linear regression).",
    (
        "Generalised Time-Varying Parameter random forest: each leaf "
        "fits a local linear regression of y on X. Forest prediction = "
        "average of leaf-local linear predictions. Captures regime-like "
        "non-linearities while preserving linear interpretability "
        "within each regime."
    ),
    "Macro forecasting with non-stationary parameter dynamics; alternative to switching models.",
    references=(_REF_DESIGN_L4, Reference(citation="Coulombe (2024) 'A Neural Phillips Curve and a Deep Output Gap', Journal of Monetary Economics, forthcoming.")),
    related_options=("random_forest", "bvar_minnesota"),
)

_F_QRF = _f(
    "quantile_regression_forest",
    "Meinshausen (2006) quantile regression forest.",
    (
        "Records the per-leaf empirical training-target distribution and "
        "forecasts arbitrary quantiles by averaging leaf-conditional CDFs. "
        "Surfaces ``forecast_intervals`` directly without a Gaussian "
        "shortcut. Pairs with ``forecast_object: quantile``."
    ),
    "Growth-at-risk / VaR studies; density forecasting.",
    references=(_REF_DESIGN_L4, Reference(citation="Meinshausen (2006) 'Quantile Regression Forests', JMLR 7.")),
    related_options=("random_forest", "bagging"),
)

_F_BAGGING = _f(
    "bagging",
    "Bootstrap-aggregating wrapper around any base family.",
    (
        "``params.base_family`` selects the base estimator; "
        "``params.n_estimators`` (default 50) bootstrap resamples are "
        "fit; predict averages. ``predict_quantiles`` surfaces empirical "
        "bag-quantiles."
    ),
    "Variance reduction on noisy series; quantile bands without quantile regression.",
    references=(_REF_DESIGN_L4, Reference(citation="Breiman (1996) 'Bagging Predictors', Machine Learning 24(2).")),
    related_options=("random_forest", "extra_trees", "quantile_regression_forest"),
)


# SVM / kNN / NN
_F_SVR_LINEAR = _f(
    "svr_linear", "Support vector regression with linear kernel.",
    "ε-insensitive loss + L2 regularisation. Sparse in support vectors.",
    "Robust linear baselines; comparison against ridge.",
    references=(_REF_DESIGN_L4, Reference(citation="Drucker, Burges, Kaufman, Smola & Vapnik (1997) 'Support Vector Regression Machines', NeurIPS.")),
    related_options=("svr_rbf", "svr_poly", "ridge"),
)

_F_SVR_RBF = _f(
    "svr_rbf", "Support vector regression with RBF kernel.",
    "Non-linear regression via kernel trick. Slow on large panels (O(n³)).",
    "Small / medium-dim non-linear regression; kernel-method ablations.",
    references=(_REF_DESIGN_L4,),
    related_options=("svr_linear", "svr_poly", "random_forest"),
)

_F_SVR_POLY = _f(
    "svr_poly", "Support vector regression with polynomial kernel.",
    "Polynomial-kernel SVR. Useful for studies that want explicit polynomial features without manual expansion.",
    "Polynomial-kernel ablations.",
    references=(_REF_DESIGN_L4,),
    related_options=("svr_rbf", "svr_linear"),
)

_F_KNN = _f(
    "knn", "k-nearest-neighbours regression.",
    "Memorises training data; predicts via nearest-neighbour averaging. Cheap, non-parametric.",
    "Non-parametric baselines; sensitivity studies.",
    references=(_REF_DESIGN_L4, Reference(citation="Cover & Hart (1967) 'Nearest neighbor pattern classification', IEEE Trans. on Information Theory 13(1).")),
    related_options=("random_forest", "svr_rbf"),
)

_F_MLP = _f(
    "mlp", "Multi-layer perceptron (sklearn).",
    "Feed-forward NN with ReLU activations. ``params.hidden_layer_sizes`` controls the architecture.",
    "Non-linear regression baselines; ablations against deep NN.",
    references=(_REF_DESIGN_L4,),
    related_options=("lstm", "gru", "transformer"),
)

_F_LSTM = _f(
    "lstm", "Long short-term memory recurrent NN (torch, optional).",
    (
        "Requires ``pip install macroforecast[deep]``. Sequence-aware RNN "
        "with input/forget/output gates. Trains on sliding windows of "
        "the lagged feature panel."
    ),
    "Sequence-modelling studies; replication of deep-NN forecasting papers.",
    when_not_to_use="Without [deep] installed -- raises NotImplementedError.",
    references=(_REF_DESIGN_L4, Reference(citation="Hochreiter & Schmidhuber (1997) 'Long short-term memory', Neural Computation 9(8).")),
    related_options=("gru", "transformer", "mlp"),
)

_F_GRU = _f(
    "gru", "Gated recurrent unit RNN (torch, optional).",
    "Requires ``pip install macroforecast[deep]``. Simpler than LSTM (one fewer gate); often comparable on macro panels.",
    "Sequence-modelling baselines; LSTM ablations.",
    when_not_to_use="Without [deep] installed.",
    references=(_REF_DESIGN_L4, Reference(citation="Cho et al. (2014) 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation', EMNLP.")),
    related_options=("lstm", "transformer"),
)

_F_TRANSFORMER = _f(
    "transformer", "Transformer encoder regressor (torch, optional).",
    "Requires ``pip install macroforecast[deep]``. Self-attention on the lagged feature panel. Single encoder layer; suitable as a non-linear sequence-attention baseline.",
    "Attention-based macro forecasting research; sequence-NN benchmark.",
    when_not_to_use="Without [deep] installed.",
    references=(_REF_DESIGN_L4, Reference(citation="Vaswani et al. (2017) 'Attention is all you need', NeurIPS.")),
    related_options=("lstm", "gru"),
)


# Other axes (forecast_strategy / training_start_rule / refit_policy / search_algorithm)
def _other(sublayer: str, axis: str, option: str, summary: str, description: str, when_to_use: str) -> OptionDoc:
    return OptionDoc(
        layer="l4", sublayer=sublayer, axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        references=(_REF_DESIGN_L4,),
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


_S_DIRECT = _other(
    "L4_B_forecast_strategy", "forecast_strategy", "direct",
    "One model per horizon (h=1, h=6, h=12, ...).",
    "Fits a separate model for each horizon h, using y_{t+h} as the target. The standard horse-race protocol: simple to implement, no error compounding, more compute.",
    "Default for most studies. Comparable across publications.",
)

_S_ITERATED = _other(
    "L4_B_forecast_strategy", "forecast_strategy", "iterated",
    "Fit h=1 model; apply recursively for h>1.",
    "Trains a single model on (y_t, X_t) → y_{t+1}, then iterates the prediction h times. Faster (one fit per cell) but errors compound.",
    "Speed-sensitive sweeps; replication of papers using iterated VAR.",
)

_S_PATH_AVG = _other(
    "L4_B_forecast_strategy", "forecast_strategy", "path_average",
    "Forecast the cumulative-average target over horizon h.",
    "Pairs with the L3 ``cumulative_average`` target-construction op. Useful for studies forecasting the *average* growth rate over horizon h rather than the level.",
    "Cumulative-growth forecasting (e.g., Stock-Watson 2002).",
)

_TS_EXPANDING = _other(
    "L4_C_training_window", "training_start_rule", "expanding",
    "Expanding window: training data grows by one observation per origin.",
    "Standard pseudo-OOS protocol. Each origin sees all data from t=0 up to that origin.",
    "Default. Comparable across publications.",
)

_TS_ROLLING = _other(
    "L4_C_training_window", "training_start_rule", "rolling",
    "Rolling window of fixed size (params.rolling_window).",
    "Drops early observations; useful for non-stationary series where parameter drift matters.",
    "Non-stationary series; structural-change studies.",
)

_TS_FIXED = _other(
    "L4_C_training_window", "training_start_rule", "fixed",
    "Fixed window with start/end pinned in leaf_config.",
    "Useful for ablation studies where every origin should see the same training sample.",
    "Replication of papers with fixed training windows.",
)

_RP_EVERY = _other(
    "L4_C_training_window", "refit_policy", "every_origin",
    "Re-fit the model at every walk-forward origin.",
    "Most expensive but most accurate -- the model's coefficients update with every new observation.",
    "Default. Standard walk-forward protocol.",
)

_RP_EVERY_N = _other(
    "L4_C_training_window", "refit_policy", "every_n_origins",
    "Re-fit every n origins (caps refit cost).",
    "Requires ``leaf_config.refit_interval``. Saves wall-clock when fits are slow but introduces stale-coefficient bias.",
    "Long sweeps with slow estimators (e.g., LSTM / xgboost on large panels).",
)

_RP_SINGLE = _other(
    "L4_C_training_window", "refit_policy", "single_fit",
    "Fit once on the full sample; use the same coefficients at every origin.",
    "Equivalent to in-sample evaluation. Useful for parameter-stability studies but does not test out-of-sample performance.",
    "In-sample studies; coefficient-stability pins.",
)

_SA_NONE = _other(
    "L4_D_tuning", "search_algorithm", "none",
    "No tuning; use the params block as-is.",
    "Default. The recipe author has already chosen the hyperparameters.",
    "Default. Studies with hand-picked hyperparameters.",
)

_SA_CV = _other(
    "L4_D_tuning", "search_algorithm", "cv_path",
    "Regularisation path via RidgeCV / LassoCV.",
    "Picks alpha from a grid via leave-one-out CV. Only applicable to ridge / lasso / elastic_net families.",
    "Quick alpha selection; comparable to published cross-validated linear baselines.",
)

_SA_GRID = _other(
    "L4_D_tuning", "search_algorithm", "grid_search",
    "Exhaustive grid over leaf_config.tuning_grid.",
    "Sklearn ``GridSearchCV`` with ``TimeSeriesSplit`` cross-validation. Requires ``leaf_config.tuning_grid``.",
    "Reproducible hyperparameter sweeps; comparison against published grid-tuned baselines.",
)

_SA_RAND = _other(
    "L4_D_tuning", "search_algorithm", "random_search",
    "Random sampling of tuning_distributions.",
    "Sklearn ``RandomizedSearchCV``. ``leaf_config.tuning_budget`` caps the iteration count.",
    "Larger search spaces; black-box hyperparameter exploration.",
)

_SA_BAYES = _other(
    "L4_D_tuning", "search_algorithm", "bayesian_optimization",
    "Optuna TPE optimisation (optional dependency).",
    "Requires ``pip install macroforecast[tuning]`` (optuna). Falls back to ``random_search`` when optuna isn't installed.",
    "Expensive estimators where each fit costs many seconds; hyperparameter spaces with smooth landscapes.",
)

_SA_GA = _other(
    "L4_D_tuning", "search_algorithm", "genetic_algorithm",
    "Tournament-selection genetic algorithm.",
    "Crossover-style evolution over hyperparameter dictionaries. ``leaf_config.genetic_algorithm_population`` and ``..._generations`` control budget.",
    "Discrete / categorical hyperparameter spaces where TPE struggles.",
)


register(
    _F_OLS, _F_RIDGE, _F_LASSO, _F_ELASTIC_NET, _F_LASSO_PATH, _F_BAYESIAN_RIDGE,
    _F_HUBER, _F_GLMBOOST,
    _F_AR_P, _F_VAR, _F_FAR, _F_PCR, _F_FAVAR, _F_BVAR_MIN, _F_BVAR_NIW, _F_DFM_MM,
    _F_DECISION_TREE, _F_RF, _F_EXTRA_TREES, _F_GB, _F_XGB, _F_LGBM, _F_CAT,
    _F_MRF, _F_QRF, _F_BAGGING,
    _F_SVR_LINEAR, _F_SVR_RBF, _F_SVR_POLY, _F_KNN,
    _F_MLP, _F_LSTM, _F_GRU, _F_TRANSFORMER,
    _S_DIRECT, _S_ITERATED, _S_PATH_AVG,
    _TS_EXPANDING, _TS_ROLLING, _TS_FIXED,
    _RP_EVERY, _RP_EVERY_N, _RP_SINGLE,
    _SA_NONE, _SA_CV, _SA_GRID, _SA_RAND, _SA_BAYES, _SA_GA,
)
