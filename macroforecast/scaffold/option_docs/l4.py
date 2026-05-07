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
    "High-dimensional macro panels with collinear predictors; standard benchmark.\n\n"
    "**v0.9 sub-axes** (default values preserve standard ridge):\n"
    "* ``params.prior`` -- prior on the coefficients. ``none`` (default) "
    "keeps standard ridge.\n"
    "  - ``random_walk`` (operational v0.9.1) -- Goulet Coulombe (2025 "
    "IJF) 'Time-Varying Parameters as Ridge Regressions' two-step "
    "closed-form estimator with a random-walk kernel on coefficient "
    "deviations. Yields per-time β path via the cumulative-sum "
    "reparametrisation β_k = C_RW · θ_k. Helper "
    "``_TwoStageRandomWalkRidge``.\n"
    "  - ``shrink_to_target`` (operational v0.9.1) -- Maximally Forward-"
    "Looking Core Inflation Albacore_comps Variant A (Goulet Coulombe "
    "/ Klieber / Barrette / Goebel 2024). ``arg min ‖y − Xw‖² + α‖w − "
    "w_target‖²`` s.t. ``w ≥ 0``, ``w'1 = 1``. Solved via scipy SLSQP. "
    "Limit cases: α=0 → unconstrained / NNLS; α→∞ → returns w_target. "
    "Helper ``_ShrinkToTargetRidge``. Sub-axis params: ``prior_target`` "
    "(default uniform 1/K), ``prior_simplex`` (default True).\n"
    "  - ``fused_difference`` (operational v0.9.1) -- Maximally FL "
    "Albacore_ranks Variant B. ``arg min ‖y − Xw‖² + α‖Dw‖²`` s.t. "
    "``w ≥ 0``, ``mean(y) = mean(Xw)``, where D is the first-difference "
    "operator. Pairs with the L3 ``asymmetric_trim`` op (B-6 v0.8.9) "
    "for rank-space transformation. Limit cases: α=0 → standard OLS / "
    "NNLS; α→∞ → uniform weights (level set by mean equality). Helper "
    "``_FusedDifferenceRidge``. Sub-axis params: ``prior_diff_order`` "
    "(default 1), ``prior_mean_equality`` (default True).\n"
    "* ``params.coefficient_constraint`` -- sign / cone constraints. "
    "``none`` (default) is unconstrained; ``nonneg`` (operational v0.8.9) "
    "implements the assemblage non-negative ridge.\n"
    "* ``params.vol_model`` (random_walk only) -- volatility model for "
    "the step-2 Ω_ε reconstruction. ``ewma`` (default; RiskMetrics "
    "λ=0.94; no extra deps) or ``garch11`` (requires ``arch>=5.0``; "
    "auto-falls-back to EWMA when missing).",
    references=(
        _REF_DESIGN_L4,
        Reference(citation="Hoerl & Kennard (1970) 'Ridge regression: biased estimation for nonorthogonal problems', Technometrics 12(1)."),
        Reference(citation="Goulet Coulombe (2025) 'Time-Varying Parameters as Ridge Regressions', International Journal of Forecasting 41:982-1002. doi:10.1016/j.ijforecast.2024.08.006."),
        Reference(citation="Goulet Coulombe / Klieber / Barrette / Goebel (2024) 'Maximally Forward-Looking Core Inflation' -- Albacore_comps (shrink_to_target Variant A) and Albacore_ranks (fused_difference Variant B)."),
    ),
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
    "Multi-series joint forecasting; impulse-response decomposition (paired with L7 ``orthogonalised_irf`` for Cholesky-identified shocks; ``generalized_irf`` reserved for the future Pesaran-Shin 1998 order-invariant variant).",
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
    "Single decision tree (sklearn) or Slow-Growing Tree (SGT, in-fit shrinkage).",
    "Cheapest non-linear model. Useful as an ablation against random forests / boosting -- if a single tree matches RF performance, the ensemble isn't buying much.\n\n"
    "**v0.9 sub-axis ``params.split_shrinkage`` = η** -- per-split soft-"
    "weight learning rate. ``0.0`` (default) keeps the standard greedy "
    "sklearn CART. Non-zero values activate **Slow-Growing Trees** "
    "(Goulet Coulombe 2024 — operational v0.9.1 dev-stage v0.9.0B-6): a "
    "*soft-weighted* tree where rows on the side that does not satisfy "
    "the split rule receive weight ``(1 − η)`` instead of 0. Implements "
    "Algorithm 1 from the paper exactly: leaf weights propagate "
    "multiplicatively through every split, the Herfindahl index "
    "``H_l = Σ(ω²)/(Σω)²`` of the leaf weight vector controls stopping, "
    "and the leaf prediction is the weighted mean.\n\n"
    "Limit cases (verified by tests):\n"
    "  * η = 1.0 → recovers standard CART (hard splits).\n"
    "  * η ≈ 0.1, H̄ ≈ 0.05 → SGT regime ('matches RF on Linear DGP at "
    "high R²' per paper Figure 2).\n\n"
    "**Sub-axis params** (only consulted when split_shrinkage ≠ 0):\n"
    "  * ``herfindahl_threshold`` = H̄ (default 0.25; smaller → deeper "
    "tree). Practice: ``{0.05, 0.1, 0.25}`` per paper.\n"
    "  * ``eta_depth_step`` -- paper rule-of-thumb increases η by "
    "``eta_depth_step·depth`` per level (default 0.0 keeps η constant).\n"
    "  * ``max_depth`` -- additional safety bound on tree depth.\n\n"
    "**Note**: sklearn ``DecisionTreeRegressor`` cannot reproduce SGT "
    "via post-fit leaf-multiplication because the *splits themselves* "
    "depend on soft weights (every row, including rule-violators, "
    "contributes to the SSE objective). The custom helper "
    "``_SlowGrowingTree`` implements the soft-weighted CART from "
    "scratch.",
    "Ablation studies; cheap non-linear baselines; SLOTH single-tree replacement for RF on small samples.",
    references=(
        _REF_DESIGN_L4,
        Reference(citation="Breiman, Friedman, Stone & Olshen (1984) 'Classification and Regression Trees', CRC Press."),
        Reference(citation="Goulet Coulombe (2024) 'Slow-Growing Trees', in Machine Learning for Econometrics and Related Topics, Studies in Systems, Decision and Control 508 (Springer). doi:10.1007/978-3-031-43601-7_4."),
    ),
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
    "Like RF but splits at random thresholds (no greedy search). Faster than RF; sometimes lower variance.\n\n"
    "**v0.9 sub-axis**:\n"
    "* ``params.max_features`` -- number of predictors considered at each "
    "split. ``\"sqrt\"`` (default) matches sklearn; ``1`` (operational, "
    "v0.9) implements Coulombe (2024) 'To Bag is to Prune' Perfectly "
    "Random Forest baseline (one random feature per split, fully random "
    "structure).",
    "Quick non-linear baseline; large ensemble experiments; PRF baseline (max_features=1).",
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
    "Goulet Coulombe (2024) MRF: random walk regularised forest with per-leaf local linear regression and Block Bayesian Bootstrap forecast ensembles.",
    (
        "Macroeconomic Random Forest. Each leaf fits a local linear "
        "regression of y on the state vector S; coefficient series are "
        "smoothed via random-walk regularisation (``rw_regul`` parameter); "
        "forecast ensembles use the Block Bayesian Bootstrap (Taddy 2015 "
        "extension). Surfaces Generalised Time-Varying Parameters (GTVPs) "
        "via the L7 ``mrf_gtvp`` op.\n\n"
        "Backed by Ryan Lucas's reference implementation, vendored under "
        "``macroforecast/_vendor/macro_random_forest/`` with surgical "
        "numpy 2.x / pandas 2.x compatibility patches (no algorithmic "
        "changes). Upstream: "
        "https://github.com/RyanLucas3/MacroRandomForest. No extra "
        "required -- the family works out of the box.\n\n"
        "**Citation requirement**: research using this family must cite "
        "Goulet Coulombe (2024) 'The Macroeconomy as a Random Forest', "
        "Journal of Applied Econometrics (arXiv:2006.12724) and "
        "acknowledge the upstream implementation by Ryan Lucas "
        "(https://github.com/RyanLucas3/MacroRandomForest).\n\n"
        "Tunable params: ``B`` (bootstrap iterations, default 50), "
        "``ridge_lambda`` (default 0.1), ``rw_regul`` (RW penalty 0..1, "
        "default 0.75), ``mtry_frac`` (default 1/3), ``trend_push`` "
        "(default 1), ``quantile_rate`` (default 0.3), ``fast_rw`` "
        "(default True), ``parallelise`` (default False), ``n_cores`` "
        "(default 1)."
    ),
    "Macro forecasting with non-stationary parameter dynamics; alternative to switching models.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation=(
                "Goulet Coulombe, P. (2024) 'The Macroeconomy as a Random "
                "Forest', Journal of Applied Econometrics. arXiv:2006.12724."
            )
        ),
        Reference(
            citation=(
                "Lucas, R. (2022) 'MacroRandomForest' (Python implementation). "
                "https://github.com/RyanLucas3/MacroRandomForest. MIT licence."
            )
        ),
    ),
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
    "Bootstrap-aggregating wrapper around any base family; supports Booging.",
    (
        "``params.base_family`` selects the base estimator; "
        "``params.n_estimators`` (default 50) bootstrap resamples are "
        "fit; predict averages. ``predict_quantiles`` surfaces empirical "
        "bag-quantiles.\n\n"
        "**v0.9 sub-axis ``params.strategy``** -- bag composition strategy:\n"
        "* ``standard`` (default) -- plain Breiman (1996) bagging; "
        "i.i.d. bootstrap with replacement.\n"
        "* ``block`` (operational v0.8.9) -- *circular* moving-block "
        "bootstrap (Künsch 1989 variant: block starts wrap at n via "
        "modulo) for serially-correlated panels (Taddy 2015 ext. used "
        "in MRF).\n"
        "* ``booging`` (operational v0.9.1) -- Goulet Coulombe (2024) "
        "'To Bag is to Prune'. Outer ``B`` bags of (intentionally over-"
        "fitted) inner Stochastic Gradient Boosted Trees + Data "
        "Augmentation: each predictor column is duplicated as ``X̃_k = "
        "X_k + N(0, (σ_k · da_noise_frac)²)``; per-bag column-drop of "
        "rate ``da_drop_rate`` (default 0.2); inner SGB at "
        "``inner_n_estimators=500, inner_learning_rate=0.1, inner_"
        "max_depth=4, inner_subsample=0.5``. Sampling without "
        "replacement at ``max_samples=0.75``. Outer ``n_estimators=B`` "
        "(default 100) replaces tuning the boosting depth ``S`` -- the "
        "bag-prune theorem (paper §2) lets us over-fit the inner SGB "
        "and let the bag average prune. Helper ``_BoogingWrapper``.\n"
        "* ``sequential_residual`` -- legacy alias for ``booging`` "
        "retained for back-compat. Pre-2026-05-07 plan sketch (\"K "
        "rounds bag-on-residuals\") was an inaccurate description of "
        "the same paper's algorithm; the option now routes to the "
        "outer-bagging-of-inner-SGB construction."
    ),
    "Variance reduction on noisy series; quantile bands without quantile regression; Booging / block-bootstrap recipes; over-fit-then-bag pruning.",
    references=(
        _REF_DESIGN_L4,
        Reference(citation="Breiman (1996) 'Bagging Predictors', Machine Learning 24(2)."),
        Reference(citation="Künsch (1989) 'The jackknife and the bootstrap for general stationary observations', Annals of Statistics 17(3) -- moving-block variant."),
        Reference(citation="Goulet Coulombe (2024) 'To Bag is to Prune', arXiv:2008.07063 -- Booging algorithm."),
    ),
    related_options=("random_forest", "extra_trees", "quantile_regression_forest", "gradient_boosting"),
)


# ---------------------------------------------------------------------------
# v0.9 Phase 2 paper-coverage atomic additions.
#
# Decomposition discipline: only the *atomic* primitives that cannot be
# expressed as a recipe pattern over existing ops + sub-axes get a new
# family entry. Paper methods that decompose (PRF / Booging / 2SRR /
# HNN / SGT / Assemblage / etc.) are captured as recipe patterns in
# ``examples/recipes/`` and as sub-axis options on existing families,
# not as standalone L4 families.
# ---------------------------------------------------------------------------

_F_MARS = _f(
    "mars",
    "Multivariate Adaptive Regression Splines (Friedman 1991).",
    (
        "Greedy forward / backward selection of piecewise-linear hinge "
        "basis functions ``max(0, x - c)`` and their products. Atomic "
        "primitive -- sklearn does not provide a MARS implementation. "
        "Runtime wraps ``pyearth`` as an optional dep; install via "
        "``pip install macroforecast[mars]``. Required as the base "
        "learner for the Coulombe (2024) 'MARSquake' recipe "
        "(``bagging(base_family=mars, ...)``).\n\n"
        "Operational from v0.9.0; raises ``NotImplementedError`` with "
        "an install hint when ``pyearth`` is not present (mirrors the "
        "xgboost / lightgbm / catboost optional-dep error pattern)."
    ),
    "Non-linear regression with interpretable basis functions; MARSquake recipe base learner.",
    when_not_to_use="Without ``[mars]`` extra installed -- raises a clear NotImplementedError.",
    references=(_REF_DESIGN_L4, Reference(citation="Friedman (1991) 'Multivariate Adaptive Regression Splines', Annals of Statistics 19(1).")),
    related_options=("gradient_boosting", "decision_tree", "bagging"),
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

_F_KERNEL_RIDGE = _f(
    "kernel_ridge",
    "Kernel Ridge Regression -- closed-form non-linear ridge in the dual.",
    (
        "Ridge regression with a non-linear kernel: "
        "``ŷ(x) = Σ_i α_i K(x, x_i) + b`` where the dual coefficients "
        "``α = (K + λ I)⁻¹ y`` are recovered in closed form. Operational "
        "v0.9.1 dev-stage v0.9.0F (audit-fix). Surfaces as a first-class "
        "L4 family because Coulombe / Surprenant / Leroux / Stevanovic "
        "(2022 JAE) 'How is Machine Learning Useful for Macroeconomic "
        "Forecasting?' Eq. 16 / §3.1.1 uses KRR as the headline "
        "non-linearity feature in the macro horse race.\n\n"
        "**Tunable params**: ``alpha`` (= ridge penalty λ; default 1.0); "
        "``kernel`` ('rbf' default / 'linear' / 'poly' / 'sigmoid' / "
        "'laplacian' / 'chi2' -- any sklearn-supported kernel); "
        "``gamma`` (RBF bandwidth, default sklearn auto = 1/n_features); "
        "``degree`` (poly kernel only, default 3); ``coef0`` (poly / "
        "sigmoid, default 1.0).\n\n"
        "Distinct from ``svr_rbf`` (ε-insensitive loss, sparsity in "
        "support vectors) and from ``ridge`` (linear). The dual "
        "representation also pairs with the L7 ``dual_decomposition`` "
        "op for kernel-weighted training-target attribution."
    ),
    "Non-linear macro forecasting baselines; KRR vs SVR-RBF / RF ablations; replicating Coulombe et al. (2022) Feature 1 nonlinearity test.",
    references=(
        _REF_DESIGN_L4,
        Reference(citation="Saunders, Gammerman & Vovk (1998) 'Ridge Regression Learning Algorithm in Dual Variables', ICML."),
        Reference(citation="Coulombe, Leroux, Stevanovic & Surprenant (2022) 'How is Machine Learning Useful for Macroeconomic Forecasting?', Journal of Applied Econometrics 37(5): 920-964 -- Eq. 16 + §3.1.1."),
    ),
    related_options=("ridge", "svr_rbf", "dual_decomposition"),
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
    "Feed-forward NN with ReLU activations. ``params.hidden_layer_sizes`` controls the architecture.\n\n"
    "**v0.9 sub-axes** (apply equally to mlp / lstm / gru / transformer):\n"
    "* ``params.architecture`` -- network topology. ``standard`` (default) "
    "is the standard feed-forward / sequence variant. ``hemisphere`` "
    "(future) implements Coulombe / Frenette / Klieber (2025 JAE) HNN "
    "with separate mean / variance hemispheres joined by a constraint "
    "loss.\n"
    "* ``params.loss`` -- objective. ``mse`` (default), ``quantile`` "
    "(operational via forecast_object=quantile), ``volatility_emphasis`` "
    "(future, HNN constraint loss).",
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
    _F_SVR_LINEAR, _F_SVR_RBF, _F_SVR_POLY, _F_KERNEL_RIDGE, _F_KNN,
    _F_MLP, _F_LSTM, _F_GRU, _F_TRANSFORMER,
    # v0.9 Phase 2 paper-coverage atomic additions
    _F_MARS,
    _S_DIRECT, _S_ITERATED, _S_PATH_AVG,
    _TS_EXPANDING, _TS_ROLLING, _TS_FIXED,
    _RP_EVERY, _RP_EVERY_N, _RP_SINGLE,
    _SA_NONE, _SA_CV, _SA_GRID, _SA_RAND, _SA_BAYES, _SA_GA,
)
