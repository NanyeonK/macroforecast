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
from .types import OptionDoc, ParameterDoc, Reference, REQUIRED

_REVIEWED = "2026-05-04"
_REVIEWER = "macroforecast author"

_REF_DESIGN_L4 = Reference(
    citation="macroforecast design Part 2, L4: 'forecasting model is the layer where every authoring iteration ends -- pick family, tune, repeat.'",
)


def _f(
    option: str,
    summary: str,
    description: str,
    when_to_use: str,
    *,
    when_not_to_use: str = "",
    references: tuple[Reference, ...] = (_REF_DESIGN_L4,),
    related_options: tuple[str, ...] = (),
    op_page: bool = False,
    op_func_name: str = "",
    parameters: tuple[ParameterDoc, ...] = (),
    data_args: tuple[ParameterDoc, ...] = (),
    return_type: str = "",
    returns_attrs: tuple[tuple[str, str, str], ...] = (),
) -> OptionDoc:
    return OptionDoc(
        layer="l4",
        sublayer="L4_A_model_selection",
        axis="family",
        option=option,
        summary=summary,
        description=description,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        references=references,
        related_options=related_options,
        op_page=op_page,
        op_func_name=op_func_name,
        parameters=parameters,
        data_args=data_args,
        return_type=return_type,
        returns_attrs=returns_attrs,
        last_reviewed=_REVIEWED,
        reviewer=_REVIEWER,
    )



# Shared data-argument docs for all L4 linear standalone callables
_L4_DATA_ARGS = (
    ParameterDoc(
        name="X",
        type="np.ndarray | pd.DataFrame",
        default=REQUIRED,
        description="Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames.",
    ),
    ParameterDoc(
        name="y",
        type="np.ndarray | pd.Series",
        default=REQUIRED,
        description="Target vector. Shape (n_samples,). Accepts numpy arrays or Series.",
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(citation="Greene (2018) 'Econometric Analysis', 8th ed., Pearson."),
    ),
    related_options=("ridge", "lasso", "elastic_net", "ar_p"),
    op_page=True,
    op_func_name="ols_fit",
    data_args=_L4_DATA_ARGS,
    return_type="OLSFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Fitted coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Fitted intercept scalar."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
        Reference(
            citation="Hoerl & Kennard (1970) 'Ridge regression: biased estimation for nonorthogonal problems', Technometrics 12(1)."
        ),
        Reference(
            citation="Goulet Coulombe (2025) 'Time-Varying Parameters as Ridge Regressions', International Journal of Forecasting 41:982-1002. doi:10.1016/j.ijforecast.2024.08.006."
        ),
        Reference(
            citation="Goulet Coulombe / Klieber / Barrette / Goebel (2024) 'Maximally Forward-Looking Core Inflation' -- Albacore_comps (shrink_to_target Variant A) and Albacore_ranks (fused_difference Variant B)."
        ),
    ),
    related_options=("lasso", "elastic_net", "lasso_path"),
    op_page=True,
    op_func_name="ridge_fit",
    parameters=(
        ParameterDoc(
            name="alpha",
            type="float",
            default=1.0,
            constraint=">=0",
            description="L2 regularisation strength. Larger values shrink coefficients more aggressively toward zero.",
        ),
        ParameterDoc(
            name="prior",
            type='str enum {"none", "random_walk", "shrink_to_target", "fused_difference"}',
            default="none",
            description=(
                "Coefficient prior. ``none`` = standard ridge. "
                "``random_walk`` = Goulet Coulombe (2025 IJF) TVP-as-ridge two-step estimator. "
                "``shrink_to_target`` = Albacore_comps Variant A (simplex non-neg + target penalty). "
                "``fused_difference`` = Albacore_ranks Variant B (fused-difference penalty)."
            ),
        ),
        ParameterDoc(
            name="coefficient_constraint",
            type='str enum {"none", "nonneg"}',
            default="none",
            description=(
                "Sign / cone constraint. ``nonneg`` enforces β >= 0 via augmented NNLS "
                "(Assemblage Regression, Coulombe et al. 2024). Ignored when ``prior`` is "
                "``shrink_to_target`` or ``fused_difference`` (those priors handle non-negativity internally)."
            ),
        ),
        ParameterDoc(
            name="vol_model",
            type='str enum {"ewma", "garch11"} | None',
            default=None,
            constraint="only used when prior='random_walk'",
            description=(
                "Volatility model for step-2 Omega_eps reconstruction in the random-walk estimator. "
                "``ewma`` = RiskMetrics lambda=0.94 (no extra deps). "
                "``garch11`` = GARCH(1,1) via the ``arch`` package; auto-falls back to EWMA if unavailable."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int | None",
            default=None,
            description="Random seed for stochastic sub-steps (currently unused in the standard ridge path; reserved for future Monte Carlo extensions).",
        ),
    ),
    data_args=(
        ParameterDoc(
            name="X",
            type="np.ndarray | pd.DataFrame",
            default=REQUIRED,
            description="Feature matrix. Shape (n_samples, n_features). Accepts numpy arrays or DataFrames.",
        ),
        ParameterDoc(
            name="y",
            type="np.ndarray | pd.Series",
            default=REQUIRED,
            description="Target vector. Shape (n_samples,). Accepts numpy arrays or Series.",
        ),
    ),
    return_type="RidgeFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Fitted coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Fitted intercept scalar."),
        (".alpha", "float", "Regularisation strength used."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Tibshirani (1996) 'Regression Shrinkage and Selection via the Lasso', JRSS-B 58(1)."
        ),
    ),
    related_options=("ridge", "elastic_net", "lasso_path"),
    op_page=True,
    op_func_name="lasso_fit",
    parameters=(
        ParameterDoc(
            name="alpha",
            type="float",
            default=1.0,
            constraint=">=0",
            description="L1 regularisation strength. Larger values force more coefficients to exactly zero.",
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=20000,
            constraint=">=1",
            description="Maximum number of coordinate descent iterations.",
        ),
    ),
    data_args=_L4_DATA_ARGS,
    return_type="LassoFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Fitted coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Fitted intercept scalar."),
        (".alpha", "float", "Regularisation strength used."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Zou & Hastie (2005) 'Regularization and variable selection via the elastic net', JRSS-B 67(2)."
        ),
    ),
    related_options=("ridge", "lasso"),
    op_page=True,
    op_func_name="elastic_net_fit",
    parameters=(
        ParameterDoc(
            name="alpha",
            type="float",
            default=1.0,
            constraint=">=0",
            description="Overall regularisation strength.",
        ),
        ParameterDoc(
            name="l1_ratio",
            type="float",
            default=0.5,
            constraint="in [0.0, 1.0]",
            description="L1/L2 mixing parameter. 0 = pure ridge, 1 = pure lasso.",
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=20000,
            constraint=">=1",
            description="Maximum number of coordinate descent iterations.",
        ),
    ),
    data_args=_L4_DATA_ARGS,
    return_type="ElasticNetFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Fitted coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Fitted intercept scalar."),
        (".alpha", "float", "Regularisation strength used."),
        (".l1_ratio", "float", "L1/L2 mixing parameter used."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
    op_page=True,
    op_func_name="lasso_path_fit",
    parameters=(
        ParameterDoc(
            name="cv",
            type="int",
            default=5,
            constraint=">=2",
            description="Number of cross-validation folds for alpha selection.",
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=20000,
            constraint=">=1",
            description="Maximum coordinate descent iterations per alpha.",
        ),
        ParameterDoc(
            name="random_state",
            type="int | None",
            default=None,
            description="Random seed for CV fold generation. None uses system entropy.",
        ),
    ),
    data_args=_L4_DATA_ARGS,
    return_type="LassoPathFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Fitted coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Fitted intercept scalar."),
        (".alpha_selected", "float", "CV-selected regularisation strength."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
    op_page=True,
    op_func_name="bayesian_ridge_fit",
    data_args=_L4_DATA_ARGS,
    return_type="BayesianRidgeFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Posterior mean coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Posterior mean intercept scalar."),
        (".alpha_", "float", "Posterior noise precision (empirical Bayes)."),
        (".lambda_", "float", "Posterior weight precision (empirical Bayes)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Huber (1964) 'Robust Estimation of a Location Parameter', Annals of Mathematical Statistics 35(1)."
        ),
    ),
    related_options=("ols", "ridge"),
    op_page=True,
    op_func_name="huber_fit",
    parameters=(
        ParameterDoc(
            name="epsilon",
            type="float",
            default=1.35,
            constraint=">1.0",
            description=(
                "Huber loss transition point. Residuals with |r| <= epsilon * scale_ "
                "are treated as inliers (quadratic loss); larger residuals are outliers "
                "(linear loss). Must be > 1.0 (sklearn requirement)."
            ),
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=1000,
            constraint=">=1",
            description="Maximum number of LBFGS iterations.",
        ),
    ),
    data_args=_L4_DATA_ARGS,
    return_type="HuberFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Fitted coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Fitted intercept scalar."),
        (".epsilon", "float", "Huber loss transition point used."),
        (".scale_", "float", "Robust scale estimate from the fitted model."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Bühlmann & Hothorn (2007) 'Boosting algorithms: Regularization, prediction and model fitting', Statistical Science 22(4)."
        ),
    ),
    related_options=("lasso", "elastic_net"),
    op_page=True,
    op_func_name="glmboost_fit",
    parameters=(
        ParameterDoc(
            name="n_iter",
            type="int",
            default=100,
            constraint=">=1",
            description="Number of boosting iterations. More iterations = finer coefficient path.",
        ),
        ParameterDoc(
            name="learning_rate",
            type="float",
            default=0.1,
            constraint=">0",
            description="Shrinkage factor applied to each coefficient update. Smaller = slower convergence, more regularisation.",
        ),
    ),
    data_args=_L4_DATA_ARGS,
    return_type="GLMBoostFitResult",
    returns_attrs=(
        (".coef_", "np.ndarray", "Fitted coefficient vector, shape (n_features,)."),
        (".intercept_", "float", "Fitted intercept scalar (initialised to mean(y))."),
        (".n_iter", "int", "Number of boosting iterations used."),
        (".learning_rate", "float", "Shrinkage factor used."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable text table of fit results."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Stock & Watson (2007) 'Why Has US Inflation Become Harder to Forecast?', JMCB 39."
        ),
    ),
    related_options=("var", "factor_augmented_ar"),
    op_page=True,
    op_func_name="ar_fit",
    data_args=_L4_DATA_ARGS,
    return_type="ARFitResult",
    returns_attrs=(
        (".n_lags", "int", "AR lag order p."),
        (".coef_", "np.ndarray", "Fitted AR coefficients, shape (n_lags,)."),
        (".intercept_", "float", "Fitted intercept."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: AR order, intercept, per-lag coefficients."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Sims (1980) 'Macroeconomics and Reality', Econometrica 48(1)."
        ),
    ),
    related_options=("bvar_minnesota", "factor_augmented_var", "ar_p"),
    op_page=True,
    op_func_name="var_fit",
    data_args=_L4_DATA_ARGS,
    return_type="VARFitResult",
    returns_attrs=(
        (".n_lags", "int", "VAR lag order p."),
        (".n_obs", "int", "Number of observations."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: lag order and observation count."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460)."
        ),
    ),
    related_options=("factor_augmented_var", "principal_component_regression", "ar_p"),
    op_page=True,
    op_func_name="far_fit",
    data_args=_L4_DATA_ARGS,
    return_type="FARFitResult",
    returns_attrs=(
        (".n_factors", "int", "Number of PCA factors extracted from X."),
        (".n_lags", "int", "AR lag order p."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: factor count and lag order."),
    ),
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
    op_page=True,
    op_func_name="pcr_fit",
    data_args=_L4_DATA_ARGS,
    return_type="PCRFitResult",
    returns_attrs=(
        (".n_components", "int", "Number of principal components used in regression."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: component count."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Bernanke, Boivin & Eliasz (2005) 'Measuring the Effects of Monetary Policy: A Factor-Augmented Vector Autoregressive Approach', QJE 120(1)."
        ),
    ),
    related_options=("var", "factor_augmented_ar", "bvar_minnesota"),
    op_page=True,
    op_func_name="favar_fit",
    data_args=_L4_DATA_ARGS,
    return_type="FAVARFitResult",
    returns_attrs=(
        (".n_factors", "int", "Number of PCA factors extracted from X."),
        (".n_lags", "int", "VAR lag order p."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: factor count and lag order."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Litterman (1986) 'Forecasting With Bayesian Vector Autoregressions -- Five Years of Experience', JBES 4(1)."
        ),
    ),
    related_options=("bvar_normal_inverse_wishart", "var", "factor_augmented_var"),
    op_page=True,
    op_func_name="bvar_minnesota_fit",
    data_args=_L4_DATA_ARGS,
    return_type="BVARMinnesotaFitResult",
    returns_attrs=(
        (".n_lags", "int", "VAR lag order p."),
        (".lambda1", "float", "Minnesota prior tightness."),
        (".n_obs", "int", "Number of observations."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: lag order, tightness, observation count."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Kadiyala & Karlsson (1997) 'Numerical Methods for Estimation and Inference in Bayesian VAR-models', Journal of Applied Econometrics 12(2)."
        ),
    ),
    related_options=("bvar_minnesota", "var"),
    op_page=True,
    op_func_name="bvar_niw_fit",
    data_args=_L4_DATA_ARGS,
    return_type="BVARNIWFitResult",
    returns_attrs=(
        (".n_lags", "int", "VAR lag order p."),
        (".n_obs", "int", "Number of observations."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: lag order and observation count."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Mariano & Murasawa (2010) 'A coincident index, common factors, and monthly real GDP', Oxford Bulletin of Economics and Statistics 72(1)."
        ),
    ),
    related_options=("factor_augmented_ar", "factor_augmented_var"),
    op_page=True,
    op_func_name="dfm_fit",
    data_args=_L4_DATA_ARGS,
    return_type="DFMFitResult",
    returns_attrs=(
        (".n_factors", "int", "Number of dynamic factors."),
        (".n_obs", "int", "Number of observations."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: factor count and observation count."),
    ),
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
        Reference(
            citation="Breiman, Friedman, Stone & Olshen (1984) 'Classification and Regression Trees', CRC Press."
        ),
        Reference(
            citation="Goulet Coulombe (2024) 'Slow-Growing Trees', in Machine Learning for Econometrics and Related Topics, Studies in Systems, Decision and Control 508 (Springer). doi:10.1007/978-3-031-43601-7_4."
        ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(citation="Breiman (2001) 'Random Forests', Machine Learning 45(1)."),
    ),
    related_options=(
        "extra_trees",
        "gradient_boosting",
        "xgboost",
        "macroeconomic_random_forest",
        "quantile_regression_forest",
    ),
    op_page=True,
    op_func_name="random_forest_fit",
    data_args=_L4_DATA_ARGS,
    return_type="RandomForestFitResult",
    returns_attrs=(
        (".feature_importances_", "np.ndarray", "Mean decrease in impurity per feature, shape (n_features,). Sums to 1.0."),
        (".n_estimators_used", "int", "Number of trees grown (= n_estimators parameter)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable table of fit results including top-3 feature importances."),
    ),
)

_F_EXTRA_TREES = _f(
    "extra_trees",
    "Extremely randomized trees (sklearn).",
    "Like RF but splits at random thresholds (no greedy search). Faster than RF; sometimes lower variance.\n\n"
    "**v0.9 sub-axis**:\n"
    "* ``params.max_features`` -- number of predictors considered at each "
    'split. ``"sqrt"`` (default) matches sklearn; ``1`` (operational, '
    "v0.9) implements Coulombe (2024) 'To Bag is to Prune' Perfectly "
    "Random Forest baseline (one random feature per split, fully random "
    "structure).",
    "Quick non-linear baseline; large ensemble experiments; PRF baseline (max_features=1).",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Geurts, Ernst & Wehenkel (2006) 'Extremely randomized trees', Machine Learning 63(1)."
        ),
    ),
    related_options=("random_forest", "gradient_boosting"),
    op_page=True,
    op_func_name="extra_trees_fit",
    data_args=_L4_DATA_ARGS,
    return_type="ExtraTreesFitResult",
    returns_attrs=(
        (".feature_importances_", "np.ndarray", "Mean decrease in impurity per feature, shape (n_features,). Sums to 1.0."),
        (".n_estimators_used", "int", "Number of trees grown (= n_estimators parameter)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable table of fit results including top-3 feature importances."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Friedman (2001) 'Greedy function approximation: A gradient boosting machine', Annals of Statistics 29(5)."
        ),
    ),
    related_options=("xgboost", "lightgbm", "catboost"),
    op_page=True,
    op_func_name="gradient_boosting_fit",
    data_args=_L4_DATA_ARGS,
    return_type="GradientBoostingFitResult",
    returns_attrs=(
        (".feature_importances_", "np.ndarray", "Feature importances from the GBM, shape (n_features,). Sums to 1.0."),
        (".n_estimators_used", "int", "Number of boosting iterations (= n_estimators parameter)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable table of fit results including top-3 feature importances."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Chen & Guestrin (2016) 'XGBoost: A Scalable Tree Boosting System', KDD."
        ),
    ),
    related_options=("gradient_boosting", "lightgbm", "catboost"),
    op_page=True,
    op_func_name="xgboost_fit",
    data_args=_L4_DATA_ARGS,
    return_type="XGBoostFitResult",
    returns_attrs=(
        (".feature_importances_", "np.ndarray", "Feature importances (gain-based) from XGBoost, shape (n_features,)."),
        (".n_estimators_used", "int", "Number of boosting rounds (= n_estimators parameter)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable table of fit results including top-3 feature importances."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Ke et al. (2017) 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', NeurIPS."
        ),
    ),
    related_options=("xgboost", "gradient_boosting"),
    op_page=True,
    op_func_name="lightgbm_fit",
    data_args=_L4_DATA_ARGS,
    return_type="LightGBMFitResult",
    returns_attrs=(
        (".feature_importances_", "np.ndarray", "Feature importances (split count) from LightGBM, shape (n_features,)."),
        (".n_estimators_used", "int", "Number of boosting rounds (= n_estimators parameter)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Human-readable table of fit results including top-3 feature importances."),
    ),
)

_F_CAT = _f(
    "catboost",
    "CatBoost gradient-boosted trees (optional dependency).",
    "Requires ``pip install macroforecast[catboost]``. Ordered boosting + native categorical handling.",
    "Categorical-heavy panels; ordered-boosting research.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Prokhorenkova et al. (2018) 'CatBoost: unbiased boosting with categorical features', NeurIPS."
        ),
    ),
    related_options=("xgboost", "lightgbm"),
    op_page=True,
    op_func_name="catboost_fit",
    data_args=_L4_DATA_ARGS,
    return_type="CatBoostFitResult",
    returns_attrs=(
        (".feature_importances_", "np.ndarray", "Feature importances from CatBoost (percentage-based), shape (n_features,)."),
        (".n_estimators_used", "int", "Number of boosting iterations (= n_estimators parameter)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, guaranteed 1-D via .ravel()."),
        (".summary()", "str", "Human-readable table of fit results including top-3 feature importances."),
    ),
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
    references=(
        _REF_DESIGN_L4,
        Reference(citation="Meinshausen (2006) 'Quantile Regression Forests', JMLR 7."),
    ),
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
        'retained for back-compat. Pre-2026-05-07 plan sketch ("K '
        'rounds bag-on-residuals") was an inaccurate description of '
        "the same paper's algorithm; the option now routes to the "
        "outer-bagging-of-inner-SGB construction."
    ),
    "Variance reduction on noisy series; quantile bands without quantile regression; Booging / block-bootstrap recipes; over-fit-then-bag pruning.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Breiman (1996) 'Bagging Predictors', Machine Learning 24(2)."
        ),
        Reference(
            citation="Künsch (1989) 'The jackknife and the bootstrap for general stationary observations', Annals of Statistics 17(3) -- moving-block variant."
        ),
        Reference(
            citation="Goulet Coulombe (2024) 'To Bag is to Prune', arXiv:2008.07063 -- Booging algorithm."
        ),
    ),
    related_options=(
        "random_forest",
        "extra_trees",
        "quantile_regression_forest",
        "gradient_boosting",
    ),
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

# ---------------------------------------------------------------------------
# v0.9 Phase C top-6 net-new methods (volatility families + ETS baselines).
# ---------------------------------------------------------------------------

_F_GARCH11 = _f(
    "garch11",
    "GARCH(1,1) univariate conditional-variance model (Bollerslev 1986).",
    (
        "Standard GARCH(1,1) volatility model: "
        "``σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}``. The L4 wrapper "
        "treats ``y`` as the return-like series and ignores ``X``; "
        "``predict(X)`` returns the conditional mean (μ broadcast over "
        "``len(X)``) and the variance forecast is exposed via "
        "``predict_variance(h_steps)`` for L7 inspection.\n\n"
        "**Defaults** (paper-faithful, Bollerslev 1986 §3): "
        "``p = q = 1``, ``mean_model = 'constant'``, ``dist = 'normal'``. "
        "Wraps ``arch.arch_model`` -- requires the optional "
        "``[arch]`` extra (``pip install macroforecast[arch]``); raises "
        "``NotImplementedError`` with an install hint when missing."
    ),
    "Macro / financial volatility forecasting; baseline GARCH benchmark; volatility-targeting risk applications.",
    when_not_to_use="Without ``[arch]`` extra installed -- raises a clear NotImplementedError.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Bollerslev (1986) 'Generalized Autoregressive Conditional Heteroskedasticity', Journal of Econometrics 31(3): 307-327."
        ),
        Reference(
            citation="Engle (1982) 'Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation', Econometrica 50(4): 987-1007."
        ),
    ),
    related_options=("egarch", "realized_garch_with_rv_exog"),
    op_page=True,
    op_func_name="garch11_fit",
    data_args=_L4_DATA_ARGS,
    return_type="GARCH11FitResult",
    returns_attrs=(
        (".conditional_mu", "float", "Fitted conditional mean mu."),
        (".n_obs", "int", "Number of non-missing observations."),
        (".params_", "dict", "Fitted GARCH parameters dict."),
        (".predict(X)", "np.ndarray", "Conditional mean broadcast over len(X) rows."),
        (".predict_variance(h)", "np.ndarray", "h-step-ahead variance forecast."),
        (".summary()", "str", "Table: conditional mean and fitted parameters."),
    ),
)

_F_EGARCH = _f(
    "egarch",
    "Exponential GARCH with leverage asymmetry (Nelson 1991).",
    (
        "EGARCH(p, o, q) on log-variance: "
        "``ln σ²_t = ω + Σ α_i (|z_{t-i}| − E|z|) + Σ γ_i z_{t-i} + Σ β_j ln σ²_{t-j}``. "
        "The asymmetry term ``γ`` captures the leverage effect (negative "
        "shocks raise volatility more than positive ones), and the log "
        "specification removes any need for non-negativity constraints "
        "on the parameters.\n\n"
        "**Defaults** (Nelson 1991 §3): ``p = o = q = 1``, "
        "``mean_model = 'constant'``, ``dist = 'normal'``. Wraps "
        "``arch.arch_model(vol='EGARCH')`` -- requires ``[arch]`` extra."
    ),
    "Asymmetric / leverage volatility; equity returns where bad news amplifies vol; macro variables with sign-asymmetric volatility responses.",
    when_not_to_use="Without ``[arch]`` extra installed; symmetric volatility series where GARCH(1,1) is sufficient (parsimony).",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Nelson (1991) 'Conditional Heteroskedasticity in Asset Returns: A New Approach', Econometrica 59(2): 347-370."
        ),
    ),
    related_options=("garch11", "realized_garch_with_rv_exog"),
    op_page=True,
    op_func_name="egarch_fit",
    data_args=_L4_DATA_ARGS,
    return_type="EGARCHFitResult",
    returns_attrs=(
        (".conditional_mu", "float", "Fitted conditional mean mu."),
        (".n_obs", "int", "Number of non-missing observations."),
        (".params_", "dict", "Fitted EGARCH parameters dict."),
        (".predict(X)", "np.ndarray", "Conditional mean broadcast over len(X) rows."),
        (".predict_variance(h)", "np.ndarray", "h-step-ahead variance forecast."),
        (".summary()", "str", "Table: conditional mean and fitted parameters."),
    ),
)

_F_REALIZED_GARCH_WITH_RV_EXOG = _f(
    "realized_garch_with_rv_exog",
    "GARCH(1,1) with realised-variance series fed as the exogenous regressor (NOT Hansen-Huang-Shek 2012 joint MLE).",
    (
        "Phase C-3 audit-fix (M9) honest rename. The L4 wrapper "
        "consumes ``params['realized_variance']`` (a column name in "
        "``X``) as the RV series and feeds it as the **exogenous "
        "regressor** ``x=`` into a vanilla GARCH(1,1) spec. This is "
        "useful in practice (RV improves volatility forecasts), but it "
        "is **NOT** the Hansen-Huang-Shek (2012) joint return + "
        "measurement-equation MLE: there is no ``ξ``, ``φ``, ``δ_1``, "
        "``δ_2`` measurement-equation parameters in the fitted output. "
        "The proper RealizedGARCH spec is reserved as FUTURE under the "
        "name ``realized_garch`` (awaiting native ``arch.RealizedGARCH`` "
        "API or manual joint-MLE implementation).\n\n"
        "Returns the conditional mean as the point forecast; "
        "``predict_variance(h_steps)`` exposes the variance path.\n\n"
        "**Defaults**: ``mean_model = 'constant'``, ``dist = 'normal'``. "
        "Falls back to a squared-returns proxy when the RV column is "
        "unavailable."
    ),
    "Volatility forecasting when intraday realised variance is observable as a leading indicator (RV-as-exogenous improves vol forecast); honest baseline labelling for studies that need to distinguish from the proper Hansen-Huang-Shek MLE.",
    when_not_to_use="When the proper joint-MLE Realized GARCH is required (the family name ``realized_garch`` is FUTURE / unrunnable until upstream supports it); without ``[arch]`` extra installed; without an RV measurement available.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Hansen, Huang & Shek (2012) 'Realized GARCH: A Joint Model for Returns and Realized Measures of Volatility', Journal of Applied Econometrics 27(6): 877-906 — the *target* spec, not implemented here."
        ),
    ),
    related_options=("garch11", "egarch"),
    op_page=True,
    op_func_name="realized_garch_fit",
    data_args=_L4_DATA_ARGS,
    return_type="RealizedGARCHFitResult",
    returns_attrs=(
        (".conditional_mu", "float", "Fitted conditional mean mu."),
        (".n_obs", "int", "Number of non-missing observations."),
        (".params_", "dict", "Fitted model parameters dict."),
        (".predict(X)", "np.ndarray", "Conditional mean broadcast over len(X) rows."),
        (".predict_variance(h)", "np.ndarray", "h-step-ahead variance forecast."),
        (".summary()", "str", "Table: conditional mean and fitted parameters."),
    ),
)

# ---------------------------------------------------------------------------
# C49 — realized_garch honesty pass (2026-05-21).
# Hansen, Huang & Shek (2012, JAE 27(6)) joint-MLE Realized GARCH promoted
# from future to operational. FUTURE_MODEL_FAMILIES is now empty.
# Distinct from realized_garch_with_rv_exog (RV-as-exogenous approximation,
# requires arch package, separately operational).
# ---------------------------------------------------------------------------

_F_REALIZED_GARCH = _f(
    "realized_garch",
    "Hansen-Huang-Shek (2012) Realized GARCH -- joint return + measurement MLE.",
    (
        "True Hansen, Huang & Shek (2012) Realized GARCH: a three-equation "
        "joint system for returns and a realized-variance measurement, estimated "
        "by maximum likelihood. All 11 parameters are recovered simultaneously "
        "via ``scipy.optimize.minimize(method='L-BFGS-B')``. No ``arch`` package "
        "dependency -- depends only on NumPy and SciPy.\n\n"
        "**Model equations** (Hansen et al. 2012, JAE 27(6): 877-906):\n"
        "* **Return**: ``r_t = mu + sqrt(h_t) * z_t``, ``z_t ~ N(0, 1)``\n"
        "* **Log-variance**: ``log(h_t) = omega + beta * log(h_{t-1}) + "
        "tau_1 * z_{t-1} + tau_2 * (z_{t-1}^2 - 1) + gamma * u_{t-1}``\n"
        "* **Measurement**: ``log(x_t) = xi + phi * log(h_t) + "
        "delta_1 * z_t + delta_2 * (z_t^2 - 1) + u_t``, "
        "``u_t ~ N(0, sigma_u^2)``\n\n"
        "**Parameter vector** (length 11): "
        "``(mu, omega, beta, tau_1, tau_2, gamma, xi, phi, delta_1, delta_2, log_sigma_u)``.\n\n"
        "**Multi-start NLS**: ``params.n_starts`` restarts (default 3). "
        "Start 0 uses canonical initialization (``h_0 = max(var(r), 1e-6)``, "
        "``u_0 = z_0 = 0``). Subsequent starts perturb theta by ``N(0, 0.05)`` "
        "via ``numpy.random.default_rng(random_state + start_index)``. "
        "Best objective selected.\n\n"
        "**RV column**: ``params.realized_variance`` specifies the column name "
        "in ``X`` containing the realized-variance series. If missing or all-NaN, "
        "falls back to ``r^2`` as a proxy (WARNING: proxy degrades estimation quality).\n\n"
        "**Numerical stability**: ``log_h_t`` clipped to ``[-30, 30]`` before "
        "``exp()``; ``h_t`` clipped to ``min 1e-8``; ``log_sigma_u`` clipped to "
        "``[-10, 10]``; ``x_t`` clipped to ``[1e-8, inf)`` before taking log.\n\n"
        "**``random_state`` propagation**: seed injected via the #215/#279 contract "
        "(``params['random_state'] = base_seed + origin_position`` at materialize "
        "time)."
    ),
    (
        "Volatility forecasting studies where high-frequency realized-variance "
        "measures (e.g., 5-minute RV, BPV, QV) are available as inputs. "
        "Produces superior volatility forecasts relative to GARCH(1,1) and EGARCH "
        "in the presence of a reliable RV series. Appropriate for equity, FX, "
        "and commodity return panels paired with corresponding RV data."
    ),
    when_not_to_use=(
        "When no realized-variance series is available (model falls back to r^2 proxy, "
        "which is equivalent to a vanilla GARCH without the measurement equation benefit). "
        "When only the approximate RV-as-exogenous approach is needed, "
        "``realized_garch_with_rv_exog`` is simpler and requires fewer parameters. "
        "When ``len(y) < 30`` -- raises ``NotImplementedError`` with an informative message."
    ),
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation=(
                "Hansen, Huang & Shek (2012) 'Realized GARCH: A Joint Model for "
                "Returns and Realized Measures of Volatility', Journal of Applied "
                "Econometrics 27(6): 877-906."
            )
        ),
        Reference(
            citation=(
                "Bollerslev (1986) 'Generalized Autoregressive Conditional "
                "Heteroskedasticity', Journal of Econometrics 31(3): 307-327."
            )
        ),
    ),
    related_options=("realized_garch_with_rv_exog", "garch11", "egarch"),
    op_page=True,
    op_func_name="realized_garch",
    parameters=(
        ParameterDoc(
            name="realized_variance",
            type="str | None",
            default=None,
            description=(
                "Column name in X containing the realized-variance series "
                "(e.g. 5-minute RV or bipower variation). If None or if the column "
                "is missing / all-NaN, falls back to r^2 as a realized-variance proxy "
                "(degrades estimation quality relative to a true RV measure)."
            ),
        ),
        ParameterDoc(
            name="mean_model",
            type='str enum {"constant"}',
            default='"constant"',
            description=(
                "Mean model for the return equation. Only 'constant' is "
                "supported in C49 (AR-mean deferred to a future cycle). "
                "Raises ValueError for any other value."
            ),
        ),
        ParameterDoc(
            name="dist",
            type='str enum {"normal"}',
            default='"normal"',
            description=(
                "Error distribution. Only 'normal' is supported in C49 "
                "(fat-tail extensions deferred). Raises ValueError for any other value."
            ),
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=2000,
            constraint=">=1",
            description="Maximum L-BFGS-B iterations per optimization start.",
        ),
        ParameterDoc(
            name="n_starts",
            type="int",
            default=3,
            constraint=">=1",
            description=(
                "Number of multi-start optimization restarts. Start 0 uses "
                "canonical initialization; subsequent starts perturb the "
                "parameter vector by N(0, 0.05). Best objective is selected."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            description=(
                "RNG seed for multi-start perturbations. Propagated from L0 "
                "via the per-origin RNG contract (#279): "
                "random_state = base_seed + origin_position."
            ),
        ),
    ),
    data_args=_L4_DATA_ARGS,
    return_type="RealizedGARCHFitResult",
    returns_attrs=(
        (".conditional_mu", "float", "Fitted conditional mean mu from the return equation."),
        (".n_obs", "int", "Number of aligned non-NaN return/RV observations used in fitting."),
        (".params_", "dict", (
            "Fitted parameter dict with keys: mu, omega, beta, tau_1, tau_2, gamma, "
            "xi, phi, delta_1, delta_2, sigma_u (all float)."
        )),
        (".conditional_volatility_", "np.ndarray | None", (
            "In-sample conditional volatility sqrt(h_t) series, shape (T_fit,). "
            "None before fit()."
        )),
        (".predict(X)", "np.ndarray", "Conditional mean broadcast over len(X) rows."),
        (".predict_variance(h)", "np.ndarray", "h-step-ahead variance forecast, shape (h,)."),
        (".summary()", "str", "Table: conditional mean, observation count, fitted parameters."),
    ),
)

_F_ETS = _f(
    "ets",
    "Exponential Smoothing State-Space (Hyndman-Koehler-Ord-Snyder 2008) -- ETS family.",
    (
        "Exponential-smoothing state-space framework: ``error_trend_seasonal`` "
        "is a 3-character code ``ETS`` where ``E ∈ {A, M}`` (additive / "
        "multiplicative error), ``T ∈ {A, M, N}`` (additive / "
        "multiplicative / no trend), ``S ∈ {A, M, N}`` (additive / "
        "multiplicative / no seasonality). Wraps "
        "``statsmodels.tsa.exponential_smoothing.ets.ETSModel`` (MLE "
        "fitting; auto-selects the closed-form initialisation per "
        "Hyndman 2008 §5.4).\n\n"
        "**Defaults**: ``error_trend_seasonal = 'AAN'`` (additive error, "
        "additive trend, no seasonal -- the workhorse non-seasonal "
        "spec), ``seasonal_periods = 12`` (monthly), "
        "``initialization_method = 'estimated'``. Auto-disables seasonal "
        "fitting when ``len(y) < 2 · seasonal_periods``."
    ),
    "M-competition baseline; non-seasonal / seasonal univariate forecasting where a state-space exponential-smoothing model is the natural reference.",
    when_not_to_use="Multivariate or covariate-driven forecasting (ETS ignores ``X``); short series where seasonal estimation is unstable.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Hyndman, Koehler, Ord & Snyder (2008) 'Forecasting with Exponential Smoothing: The State Space Approach', Springer."
        ),
        Reference(
            citation="Hyndman & Athanasopoulos (2018) 'Forecasting: Principles and Practice', 2nd ed., OTexts §7."
        ),
    ),
    related_options=("theta_method", "holt_winters", "ar_p"),
    op_page=True,
    op_func_name="ets_fit",
    data_args=_L4_DATA_ARGS,
    return_type="ETSFitResult",
    returns_attrs=(
        (".error_trend_seasonal", "str", "3-character ETS code, e.g. AAN."),
        (".n_obs", "int", "Number of observations."),
        (".predict(X)", "np.ndarray", "Forecast, len(X) steps ahead."),
        (".summary()", "str", "Table: ETS code and observation count."),
    ),
)

_F_THETA_METHOD = _f(
    "theta_method",
    "Theta method (Assimakopoulos-Nikolopoulos 2000) -- M3-competition winning baseline.",
    (
        "Hand-coded Theta(2) closed-form forecast: blends a long-run "
        "linear-trend regression with a short-run simple-exponential-"
        "smoothing (SES) level. For ``θ = 2`` (M3 winner), the "
        "h-step-ahead forecast is "
        "``ŷ_{T+h} = 0.5 · (a + b · (T+h)) + 0.5 · ℓ_T``, "
        "where ``(a, b)`` are the OLS trend slope/intercept on time "
        "index and ``ℓ_T`` is the SES level at time T (smoothing "
        "parameter ``α`` selected via ``scipy.optimize.minimize_scalar`` "
        "minimising the in-sample 1-step MSE).\n\n"
        "**Defaults**: ``theta = 2.0`` (M3 winner), ``seasonal = False``, "
        "``seasonal_periods = 12``. The constructor exposes ``theta`` "
        "for forward compatibility; only the θ=2 closed form is "
        "exercised in v0.9.0 -- general θ requires a θ-line "
        "decomposition out of scope for this run."
    ),
    "M3 / M4-style univariate baselines; quick reference forecast against more elaborate models.",
    when_not_to_use="Strongly seasonal series (use ``holt_winters`` or seasonally-adjusted target); covariate-driven forecasting.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Assimakopoulos & Nikolopoulos (2000) 'The theta model: a decomposition approach to forecasting', International Journal of Forecasting 16(4): 521-530."
        ),
        Reference(
            citation="Hyndman & Billah (2003) 'Unmasking the Theta method', International Journal of Forecasting 19(2): 287-290."
        ),
        Reference(
            citation="Petropoulos et al. (2022) 'Forecasting: theory and practice', International Journal of Forecasting 38(3): 705-871."
        ),
    ),
    related_options=("ets", "holt_winters", "ar_p"),
    op_page=True,
    op_func_name="theta_fit",
    data_args=_L4_DATA_ARGS,
    return_type="ThetaFitResult",
    returns_attrs=(
        (".theta", "float", "Theta parameter (default 2.0 = M3 winner)."),
        (".alpha_", "float", "Fitted SES smoothing parameter."),
        (".n_obs", "int", "Number of observations."),
        (".predict(X)", "np.ndarray", "Forecast, len(X) steps ahead."),
        (".summary()", "str", "Table: theta, alpha, observation count."),
    ),
)

_F_HOLT_WINTERS = _f(
    "holt_winters",
    "Holt-Winters additive / multiplicative seasonal exponential smoothing.",
    (
        "Wraps ``statsmodels.tsa.holtwinters.ExponentialSmoothing``. "
        "Fits level / trend / seasonal smoothing parameters by MLE "
        "(``optimized=True``). Supports additive and multiplicative "
        "trend and seasonal components plus an optional damped trend "
        "(Hyndman et al. 2008 §3).\n\n"
        "**Defaults**: ``seasonal = 'add'``, ``seasonal_periods = 12``, "
        "``trend = 'add'``, ``damped_trend = False``. Auto-disables "
        "seasonal fitting when ``len(y) < 2 · seasonal_periods``."
    ),
    "Seasonal univariate baselines; M-competition style benchmarking; standard reference forecast for monthly / quarterly macro series.",
    when_not_to_use="Without a clear seasonal pattern (use ``ets`` AAN instead); covariate-driven forecasting.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Holt (2004 / orig. 1957) 'Forecasting seasonals and trends by exponentially weighted moving averages', International Journal of Forecasting 20(1): 5-10."
        ),
        Reference(
            citation="Winters (1960) 'Forecasting Sales by Exponentially Weighted Moving Averages', Management Science 6(3): 324-342."
        ),
        Reference(
            citation="Hyndman & Athanasopoulos (2018) 'Forecasting: Principles and Practice', 2nd ed., OTexts §7."
        ),
    ),
    related_options=("ets", "theta_method", "ar_p"),
    op_page=True,
    op_func_name="holt_winters_fit",
    data_args=_L4_DATA_ARGS,
    return_type="HoltWintersFitResult",
    returns_attrs=(
        (".seasonal", "str", "Seasonal component type (add or mul)."),
        (".seasonal_periods", "int", "Number of periods per season."),
        (".n_obs", "int", "Number of observations."),
        (".predict(X)", "np.ndarray", "Forecast, len(X) steps ahead."),
        (".summary()", "str", "Table: seasonal type, periods, observation count."),
    ),
)


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
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Friedman (1991) 'Multivariate Adaptive Regression Splines', Annals of Statistics 19(1)."
        ),
    ),
    related_options=("gradient_boosting", "decision_tree", "bagging"),
    op_page=True,
    op_func_name="mars_fit",
    data_args=_L4_DATA_ARGS,
    return_type="MARSFitResult",
    returns_attrs=(
        (".n_terms", "int", "Number of MARS basis terms."),
        (".n_features_in_", "int", "Number of input features."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: term count and feature count."),
    ),
)


# SVM / kNN / NN
_F_SVR_LINEAR = _f(
    "svr_linear",
    "Support vector regression with linear kernel.",
    "ε-insensitive loss + L2 regularisation. Sparse in support vectors.",
    "Robust linear baselines; comparison against ridge.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Drucker, Burges, Kaufman, Smola & Vapnik (1997) 'Support Vector Regression Machines', NeurIPS."
        ),
    ),
    related_options=("svr_rbf", "svr_poly", "ridge"),
    op_page=True,
    op_func_name="svr_linear_fit",
    data_args=_L4_DATA_ARGS,
    return_type="SVRLinearFitResult",
    returns_attrs=(
        (".C", "float", "Regularisation parameter used."),
        (".n_support_vectors", "int", "Number of support vectors."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: C and support vector count."),
    ),
)

_F_SVR_RBF = _f(
    "svr_rbf",
    "Support vector regression with RBF kernel.",
    "Non-linear regression via kernel trick. Slow on large panels (O(n³)).",
    "Small / medium-dim non-linear regression; kernel-method ablations.",
    references=(_REF_DESIGN_L4,),
    related_options=("svr_linear", "svr_poly", "random_forest"),
    op_page=True,
    op_func_name="svr_rbf_fit",
    data_args=_L4_DATA_ARGS,
    return_type="SVRRBFFitResult",
    returns_attrs=(
        (".C", "float", "Regularisation parameter used."),
        (".gamma", "str|float", "RBF bandwidth parameter."),
        (".n_support_vectors", "int", "Number of support vectors."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: C, gamma, support vector count."),
    ),
)

_F_SVR_POLY = _f(
    "svr_poly",
    "Support vector regression with polynomial kernel.",
    "Polynomial-kernel SVR. Useful for studies that want explicit polynomial features without manual expansion.",
    "Polynomial-kernel ablations.",
    references=(_REF_DESIGN_L4,),
    related_options=("svr_rbf", "svr_linear"),
    op_page=True,
    op_func_name="svr_poly_fit",
    data_args=_L4_DATA_ARGS,
    return_type="SVRPolyFitResult",
    returns_attrs=(
        (".C", "float", "Regularisation parameter used."),
        (".degree", "int", "Polynomial degree."),
        (".n_support_vectors", "int", "Number of support vectors."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: C, degree, support vector count."),
    ),
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
        Reference(
            citation="Saunders, Gammerman & Vovk (1998) 'Ridge Regression Learning Algorithm in Dual Variables', ICML."
        ),
        Reference(
            citation="Coulombe, Leroux, Stevanovic & Surprenant (2022) 'How is Machine Learning Useful for Macroeconomic Forecasting?', Journal of Applied Econometrics 37(5): 920-964 -- Eq. 16 + §3.1.1."
        ),
    ),
    related_options=("ridge", "svr_rbf", "dual_decomposition"),
    op_page=True,
    op_func_name="kernel_ridge_fit",
    data_args=_L4_DATA_ARGS,
    return_type="KernelRidgeFitResult",
    returns_attrs=(
        (".alpha", "float", "Ridge regularisation strength."),
        (".kernel", "str", "Kernel type (e.g. rbf, linear)."),
        (".n_features_in_", "int", "Number of input features."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: alpha, kernel, feature count."),
    ),
)

_F_KNN = _f(
    "knn",
    "k-nearest-neighbours regression.",
    "Memorises training data; predicts via nearest-neighbour averaging. Cheap, non-parametric.",
    "Non-parametric baselines; sensitivity studies.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Cover & Hart (1967) 'Nearest neighbor pattern classification', IEEE Trans. on Information Theory 13(1)."
        ),
    ),
    related_options=("random_forest", "svr_rbf"),
    op_page=True,
    op_func_name="knn_fit",
    data_args=_L4_DATA_ARGS,
    return_type="KNNFitResult",
    returns_attrs=(
        (".n_neighbors", "int", "Neighbour count k requested."),
        (".n_neighbors_used", "int", "Actual k used (clipped to training set size)."),
        (".n_features_in_", "int", "Number of input features."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Table: neighbour counts and feature count."),
    ),
)

_F_MLP = _f(
    "mlp",
    "Multi-layer perceptron (sklearn).",
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
    op_page=True,
    op_func_name="mlp_fit",
    data_args=_L4_DATA_ARGS,
    return_type="MLPFitResult",
    returns_attrs=(
        (".n_params", "int", "Total number of trainable parameters (weights + biases)."),
        (".n_features_in_", "int", "Number of input features seen during fit."),
        (".hidden_layer_sizes", "tuple", "Tuple of hidden layer widths, e.g. (32, 16)."),
        (".epochs_used", "int", "Number of optimiser iterations completed."),
        (".final_loss", "float", "Training MSE at the end of fitting."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Arch metadata table: model_type, hidden_layer_sizes, n_features, n_params, epochs_used, final_loss."),
    ),
)

_F_LSTM = _f(
    "lstm",
    "Long short-term memory recurrent NN (torch, optional).",
    (
        "Requires ``pip install macroforecast[deep]``. Sequence-aware RNN "
        "with input/forget/output gates. Trains on sliding windows of "
        "the lagged feature panel."
    ),
    "Sequence-modelling studies; replication of deep-NN forecasting papers.",
    when_not_to_use="Without [deep] installed -- raises NotImplementedError.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Hochreiter & Schmidhuber (1997) 'Long short-term memory', Neural Computation 9(8)."
        ),
    ),
    related_options=("gru", "transformer", "mlp"),
    op_page=True,
    op_func_name="lstm_fit",
    data_args=_L4_DATA_ARGS,
    return_type="LSTMFitResult",
    returns_attrs=(
        (".n_params", "int", "Total number of trainable parameters in LSTM + head."),
        (".n_features_in_", "int", "Number of input features seen during fit."),
        (".hidden_size", "int", "Width of the LSTM hidden state."),
        (".epochs_used", "int", "Number of training epochs completed."),
        (".final_loss", "float", "Training MSE via no-grad forward pass after fitting."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Arch metadata table: model_type, hidden_size, n_features, n_params, epochs_used, final_loss."),
    ),
)

_F_GRU = _f(
    "gru",
    "Gated recurrent unit RNN (torch, optional).",
    "Requires ``pip install macroforecast[deep]``. Simpler than LSTM (one fewer gate); often comparable on macro panels.",
    "Sequence-modelling baselines; LSTM ablations.",
    when_not_to_use="Without [deep] installed.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Cho et al. (2014) 'Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation', EMNLP."
        ),
    ),
    related_options=("lstm", "transformer"),
    op_page=True,
    op_func_name="gru_fit",
    data_args=_L4_DATA_ARGS,
    return_type="GRUFitResult",
    returns_attrs=(
        (".n_params", "int", "Total number of trainable parameters in GRU + head."),
        (".n_features_in_", "int", "Number of input features seen during fit."),
        (".hidden_size", "int", "Width of the GRU hidden state."),
        (".epochs_used", "int", "Number of training epochs completed."),
        (".final_loss", "float", "Training MSE via no-grad forward pass after fitting."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Arch metadata table: model_type, hidden_size, n_features, n_params, epochs_used, final_loss."),
    ),
)

_F_TRANSFORMER = _f(
    "transformer",
    "Transformer encoder regressor (torch, optional).",
    "Requires ``pip install macroforecast[deep]``. Self-attention on the lagged feature panel. Single encoder layer; suitable as a non-linear sequence-attention baseline.",
    "Attention-based macro forecasting research; sequence-NN benchmark.",
    when_not_to_use="Without [deep] installed.",
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Vaswani et al. (2017) 'Attention is all you need', NeurIPS."
        ),
    ),
    related_options=("lstm", "gru"),
    op_page=True,
    op_func_name="transformer_fit",
    data_args=_L4_DATA_ARGS,
    return_type="TransformerFitResult",
    returns_attrs=(
        (".n_params", "int", "Total trainable parameters in Transformer encoder + head."),
        (".n_features_in_", "int", "Number of input features seen during fit (= d_model)."),
        (".hidden_size", "int", "dim_feedforward of the single TransformerEncoderLayer."),
        (".epochs_used", "int", "Number of training epochs completed."),
        (".final_loss", "float", "Training MSE via no-grad forward pass after fitting."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_samples,)."),
        (".summary()", "str", "Arch metadata table: model_type, hidden_size, n_features, n_params, epochs_used, final_loss."),
    ),
)


# Other axes (forecast_strategy / training_start_rule / refit_policy / search_algorithm)
def _other(
    sublayer: str,
    axis: str,
    option: str,
    summary: str,
    description: str,
    when_to_use: str,
) -> OptionDoc:
    return OptionDoc(
        layer="l4",
        sublayer=sublayer,
        axis=axis,
        option=option,
        summary=summary,
        description=description,
        when_to_use=when_to_use,
        references=(_REF_DESIGN_L4,),
        last_reviewed=_REVIEWED,
        reviewer=_REVIEWER,
    )


_S_DIRECT = _other(
    "L4_B_forecast_strategy",
    "forecast_strategy",
    "direct",
    "One model per horizon (h=1, h=6, h=12, ...).",
    "Fits a separate model for each horizon h, using y_{t+h} as the target. The standard horse-race protocol: simple to implement, no error compounding, more compute.",
    "Default for most studies. Comparable across publications.",
)

_S_ITERATED = _other(
    "L4_B_forecast_strategy",
    "forecast_strategy",
    "iterated",
    "Fit h=1 model; apply recursively for h>1.",
    "Trains a single model on (y_t, X_t) → y_{t+1}, then iterates the prediction h times. Faster (one fit per cell) but errors compound.",
    "Speed-sensitive sweeps; replication of papers using iterated VAR.",
)

_S_PATH_AVG = _other(
    "L4_B_forecast_strategy",
    "forecast_strategy",
    "path_average",
    "Forecast the cumulative-average target over horizon h.",
    "Pairs with the L3 ``cumulative_average`` target-construction op. Useful for studies forecasting the *average* growth rate over horizon h rather than the level.",
    "Cumulative-growth forecasting (e.g., Stock-Watson 2002).",
)

_TS_EXPANDING = _other(
    "L4_C_training_window",
    "training_start_rule",
    "expanding",
    "Expanding window: training data grows by one observation per origin.",
    "Standard pseudo-OOS protocol. Each origin sees all data from t=0 up to that origin.",
    "Default. Comparable across publications.",
)

_TS_ROLLING = _other(
    "L4_C_training_window",
    "training_start_rule",
    "rolling",
    "Rolling window of fixed size (params.rolling_window).",
    "Drops early observations; useful for non-stationary series where parameter drift matters.",
    "Non-stationary series; structural-change studies.",
)

_TS_FIXED = _other(
    "L4_C_training_window",
    "training_start_rule",
    "fixed",
    "Fixed window with start/end pinned in leaf_config.",
    "Useful for ablation studies where every origin should see the same training sample.",
    "Replication of papers with fixed training windows.",
)

_RP_EVERY = _other(
    "L4_C_training_window",
    "refit_policy",
    "every_origin",
    "Re-fit the model at every walk-forward origin.",
    "Most expensive but most accurate -- the model's coefficients update with every new observation.",
    "Default. Standard walk-forward protocol.",
)

_RP_EVERY_N = _other(
    "L4_C_training_window",
    "refit_policy",
    "every_n_origins",
    "Re-fit every n origins (caps refit cost).",
    "Requires ``leaf_config.refit_interval``. Saves wall-clock when fits are slow but introduces stale-coefficient bias.",
    "Long sweeps with slow estimators (e.g., LSTM / xgboost on large panels).",
)

_RP_SINGLE = _other(
    "L4_C_training_window",
    "refit_policy",
    "single_fit",
    "Fit once on the full sample; use the same coefficients at every origin.",
    "Equivalent to in-sample evaluation. Useful for parameter-stability studies but does not test out-of-sample performance.",
    "In-sample studies; coefficient-stability pins.",
)

_SA_NONE = _other(
    "L4_D_tuning",
    "search_algorithm",
    "none",
    "No tuning; use the params block as-is.",
    "Default. The recipe author has already chosen the hyperparameters.",
    "Default. Studies with hand-picked hyperparameters.",
)

_SA_CV = _other(
    "L4_D_tuning",
    "search_algorithm",
    "cv_path",
    "Regularisation path via RidgeCV / LassoCV.",
    "Picks alpha from a grid via leave-one-out CV. Only applicable to ridge / lasso / elastic_net families.",
    "Quick alpha selection; comparable to published cross-validated linear baselines.",
)

_SA_GRID = _other(
    "L4_D_tuning",
    "search_algorithm",
    "grid_search",
    "Exhaustive grid over leaf_config.tuning_grid.",
    "Sklearn ``GridSearchCV`` with ``TimeSeriesSplit`` cross-validation. Requires ``leaf_config.tuning_grid``.",
    "Reproducible hyperparameter sweeps; comparison against published grid-tuned baselines.",
)

_SA_RAND = _other(
    "L4_D_tuning",
    "search_algorithm",
    "random_search",
    "Random sampling of tuning_distributions.",
    "Sklearn ``RandomizedSearchCV``. ``leaf_config.tuning_budget`` caps the iteration count.",
    "Larger search spaces; black-box hyperparameter exploration.",
)

_SA_BAYES = _other(
    "L4_D_tuning",
    "search_algorithm",
    "bayesian_optimization",
    "Optuna TPE optimisation (optional dependency).",
    "Requires ``pip install macroforecast[tuning]`` (optuna). Falls back to ``random_search`` when optuna isn't installed.",
    "Expensive estimators where each fit costs many seconds; hyperparameter spaces with smooth landscapes.",
)

_SA_GA = _other(
    "L4_D_tuning",
    "search_algorithm",
    "genetic_algorithm",
    "Tournament-selection genetic algorithm.",
    "Crossover-style evolution over hyperparameter dictionaries. ``leaf_config.genetic_algorithm_population`` and ``..._generations`` control budget.",
    "Discrete / categorical hyperparameter spaces where TPE struggles.",
)


# v0.9 Phase C M12 pi_correction axis (Bai-Ng 2006).
#
# Surfaced through introspect.py under the virtual ``L4_E_predict`` sub-layer
# (the runtime ``predict`` op reads ``params['pi_correction']`` directly).
# Phase C-2 HOLD-cosmetic resolution registers the OptionDoc entries below so
# the v1.0 release gate ("every operational (axis, option) tuple has a Tier-1
# entry") stays green when the axis is registered in introspect.
#
# References:
# - Bai & Ng (2006) "Confidence Intervals for Diffusion Index Forecasts and
#   Inference for Factor-Augmented Regressions", Econometrica 74(4): 1133-1150.

_PI_NONE = OptionDoc(
    layer="l4",
    sublayer="L4_E_predict",
    axis="pi_correction",
    option="none",
    summary="No PI correction; standard Gaussian-residual sigma.",
    description=(
        "Default predict-op behaviour: prediction-interval bands derive "
        "from the fitted family's residual variance σ²_ε (Gaussian "
        "approximation around the point forecast). This treats factor "
        "regressors and parameter estimates as if they were observed "
        "exactly. Appropriate for non-factor-augmented families (OLS, "
        "ridge, AR_p, etc.) or when factor estimation noise is "
        "negligible relative to residual variance."
    ),
    when_to_use=(
        "Default for any family that does not estimate latent factors as "
        "regressors -- the residual-variance band is honest in that case."
    ),
    when_not_to_use=(
        "Factor-augmented forecasts where estimated factors enter the "
        "regression -- use ``bai_ng`` to inflate the band for the "
        "factor-estimation noise."
    ),
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Bai & Ng (2006) 'Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions', Econometrica 74(4): 1133-1150.",
        ),
    ),
    related_options=("bai_ng",),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)

_PI_BAI_NG = OptionDoc(
    layer="l4",
    sublayer="L4_E_predict",
    axis="pi_correction",
    option="bai_ng",
    summary="Bai-Ng (2006) generated-regressor PI correction.",
    description=(
        "Activates the Bai-Ng (2006) Theorem 3 + Corollary 1 correction "
        "to the prediction-interval sigma. The corrected sigma reflects "
        "(a) factor-estimation noise V₂/N where V₂ = β̂_F^T (Λ̂ diag(Σ̂_e) "
        "Λ̂^T / N) β̂_F, (b) parameter-estimation noise V₁/T from the "
        "OLS coefficient covariance evaluated at the last training "
        "factor row, and (c) the residual variance σ²_ε. Active only "
        "when the upstream fitted family is ``factor_augmented_ar``; "
        "for any other family the predict op falls through to the "
        "uncorrected Gaussian-residual sigma."
    ),
    when_to_use=(
        "Factor-augmented forecasts (FAR / FAVAR-style) where the band "
        "should be honest about factor-estimation noise on top of the "
        "usual parameter and residual uncertainty."
    ),
    when_not_to_use=(
        "Non-factor families -- the correction is a no-op there. Use "
        "``none`` to keep the predict op's default behaviour."
    ),
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation="Bai & Ng (2006) 'Confidence Intervals for Diffusion Index Forecasts and Inference for Factor-Augmented Regressions', Econometrica 74(4): 1133-1150.",
        ),
    ),
    related_options=("none",),
    last_reviewed=_REVIEWED,
    reviewer=_REVIEWER,
)


# ---------------------------------------------------------------------------
# C48 — MIDAS family honesty pass (2026-05-21).
# Four families promoted from future to operational.
# References:
#   Ghysels, Santa-Clara & Valkanov (2004) "The MIDAS Touch"
#   Ghysels, Sinko & Valkanov (2007) "MIDAS Regressions"
#   Foroni, Marcellino & Schumacher (2015) "Unrestricted Mixed Data Sampling"
# ---------------------------------------------------------------------------

_F_MIDAS_ALMON = _f(
    "midas_almon",
    "MIDAS with Almon polynomial lag weights (Ghysels-Santa-Clara-Valkanov 2004).",
    (
        "Estimates mixed-frequency regressions via non-linear least squares "
        "(Nelder-Mead) on Almon polynomial lag weights. The weight function "
        "``b(k; θ) = Σ_{q=0}^{Q} θ_q · k^q`` (Ghysels, Santa-Clara & "
        "Valkanov 2004, §2 eq. 3) maps lag index ``k`` to a scalar weight; "
        "weights are clamped to non-negative values and optionally normalized "
        "to sum to one. The full parameter vector is ``(θ_0, ..., θ_Q, μ, β)`` "
        "where ``μ`` is the intercept and ``β`` is the overall slope multiplier.\n\n"
        "``params.freq_ratio`` (default 1) governs the mixed-frequency "
        "contract. When ``freq_ratio > 1`` the model internally calls "
        "``_midas_lag_stack`` to build the high-frequency lag design matrix "
        "from a raw HF DataFrame. When ``freq_ratio = 1`` (default) X is "
        "treated as already low-frequency aligned -- this is the primary "
        "usage path following an upstream L3 ``midas`` or ``u_midas`` op.\n\n"
        "Multi-start NLS uses ``params.n_starts`` restarts (default 5): "
        "start 0 is canonical (θ_0 = 1, rest zero → flat weights); remaining "
        "starts perturb θ from ``N(0, 0.1)``. Seed propagated from L0 via "
        "the per-origin RNG contract (#279).\n\n"
        "Two weight attributes are maintained: ``_w_hat`` (length "
        "``n_lags_high``, zero-padded when ``K_eff < n_lags_high``) for "
        "external inspection; ``_w_hat_effective`` (length ``K_eff``) for "
        "internal predict matmul. This resolves the ``n_lags_high != "
        "X.shape[1]`` edge case at ``freq_ratio = 1``."
    ),
    (
        "Mixed-frequency macro forecasting where monthly or weekly predictors "
        "inform a quarterly target. Parsimonious alternative to U-MIDAS when "
        "K is large relative to T."
    ),
    when_not_to_use=(
        "Very short samples (T < K + Q + 3) -- the model falls back to "
        "uniform weights and mean intercept. When frequency alignment has "
        "already been handled by an upstream L3 op and ``freq_ratio = 1`` is "
        "sufficient, prefer ``midas_step`` (OLS, cheaper) or "
        "``dfm_unrestricted_midas`` (U-MIDAS, more flexible)."
    ),
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation=(
                "Ghysels, Santa-Clara & Valkanov (2004) 'The MIDAS Touch: "
                "Mixed Data Sampling Regression Models', CIRANO Working Paper."
            )
        ),
    ),
    related_options=("midas_beta", "midas_step", "dfm_unrestricted_midas"),
    op_page=True,
    op_func_name="midas_almon",
    parameters=(
        ParameterDoc(
            name="freq_ratio",
            type="int",
            default=1,
            constraint=">=1",
            description=(
                "High-frequency periods per low-frequency period (m). "
                "1 = X is already LF-aligned (primary path). "
                ">1 = model internally lag-stacks X via ``_midas_lag_stack``."
            ),
        ),
        ParameterDoc(
            name="n_lags_high",
            type="int",
            default=12,
            constraint=">=1",
            description="Number of high-frequency lags K to include in the weight vector.",
        ),
        ParameterDoc(
            name="polynomial_order",
            type="int",
            default=2,
            constraint=">=0",
            description="Almon polynomial degree Q. Number of θ hyperparameters is Q+1.",
        ),
        ParameterDoc(
            name="sum_to_one",
            type="bool",
            default=True,
            description="Normalize lag weights to sum to 1 after non-negativity clamp.",
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=200,
            description="Maximum Nelder-Mead iterations per start.",
        ),
        ParameterDoc(
            name="n_starts",
            type="int",
            default=5,
            constraint=">=1",
            description="Number of NLS restarts. Start 0 is canonical; remaining starts perturb θ.",
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            description="RNG seed for perturbed NLS starts. Propagated from L0 via per-origin RNG contract (#279).",
        ),
    ),
    data_args=(
        ParameterDoc(
            name="X",
            type="pd.DataFrame",
            default=REQUIRED,
            description=(
                "Feature matrix as a DataFrame. When ``freq_ratio = 1``: "
                "LF-aligned lag columns (one column per lag feature). "
                "When ``freq_ratio > 1``: raw HF DataFrame that is internally "
                "lag-stacked. Shape (n_samples, n_features)."
            ),
        ),
        ParameterDoc(
            name="y",
            type="pd.Series",
            default=REQUIRED,
            description="Low-frequency target series aligned to the LF index.",
        ),
    ),
    return_type="_MidasAlmonModel",
    returns_attrs=(
        ("._w_hat", "np.ndarray", "Estimated Almon lag weights, shape (n_lags_high,). Zero-padded when K_eff < n_lags_high."),
        ("._theta_hat", "np.ndarray", "Estimated Almon polynomial coefficients, shape (Q+1,)."),
        ("._intercept", "float", "Fitted intercept mu."),
        ("._slope", "float", "Fitted overall slope beta."),
        ("._converged", "bool", "NLS convergence flag (True if any start converged)."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_rows,)."),
    ),
)

_F_MIDAS_BETA = _f(
    "midas_beta",
    "MIDAS with Beta distribution kernel lag weights (Ghysels-Sinko-Valkanov 2007).",
    (
        "Estimates mixed-frequency regressions via non-linear least squares "
        "(Nelder-Mead) on Beta distribution kernel weights. The weight "
        "function ``b(k) ∝ x_k^{a-1} · (1-x_k)^{b-1}`` (Ghysels, Sinko & "
        "Valkanov 2007, §2) maps normalized lag positions "
        "``x_k = (k+1)/(K+1) ∈ (0,1)`` to scalar weights. Only 2 shape "
        "parameters ``(a, b)`` control the entire lag profile, making this "
        "the most parsimonious MIDAS variant.\n\n"
        "Frequency contract and multi-start NLS are identical to "
        "``midas_almon``. Start 0: ``[1, 1]`` (uniform Beta = equal "
        "weights). Perturbed starts: ``a, b ~ Gamma(2, 1)`` to keep both "
        "parameters positive naturally. Post-optimization clamp: "
        "``a, b >= 1e-3``.\n\n"
        "The two-attribute design (``_w_hat`` zero-padded to ``n_lags_high``; "
        "``_w_hat_effective`` of length ``K_eff``) is identical to "
        "``midas_almon``."
    ),
    (
        "Parsimonious mixed-frequency forecasting -- only 2 shape parameters "
        "regardless of K. Suitable when T is small relative to K, making the "
        "Almon polynomial over-parameterized."
    ),
    when_not_to_use=(
        "When the lag weight profile is known to be non-Beta-shaped (e.g., "
        "step-function or flat). Use ``midas_step`` or ``midas_almon`` instead."
    ),
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation=(
                "Ghysels, Sinko & Valkanov (2007) 'MIDAS Regressions: "
                "Further Results and New Directions', Econometric Reviews 26(1)."
            )
        ),
    ),
    related_options=("midas_almon", "midas_step", "dfm_unrestricted_midas"),
    op_page=True,
    op_func_name="midas_beta",
    parameters=(
        ParameterDoc(
            name="freq_ratio",
            type="int",
            default=1,
            constraint=">=1",
            description="High-frequency periods per low-frequency period. 1 = X already LF-aligned.",
        ),
        ParameterDoc(
            name="n_lags_high",
            type="int",
            default=12,
            constraint=">=1",
            description="Number of high-frequency lags K.",
        ),
        ParameterDoc(
            name="sum_to_one",
            type="bool",
            default=True,
            description="Normalize weights to sum to 1 (always True by construction of Beta kernel; parameter retained for API symmetry with midas_almon).",
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=200,
            description="Maximum Nelder-Mead iterations per start.",
        ),
        ParameterDoc(
            name="n_starts",
            type="int",
            default=5,
            constraint=">=1",
            description="Number of NLS restarts. Start 0 uses [1,1]; remaining starts draw a, b from Gamma(2,1).",
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            description="RNG seed for perturbed NLS starts. Propagated from L0 via per-origin RNG contract (#279).",
        ),
    ),
    data_args=(
        ParameterDoc(
            name="X",
            type="pd.DataFrame",
            default=REQUIRED,
            description=(
                "Feature matrix as a DataFrame. When ``freq_ratio = 1``: "
                "LF-aligned lag columns. When ``freq_ratio > 1``: raw HF "
                "DataFrame internally lag-stacked. Shape (n_samples, n_features)."
            ),
        ),
        ParameterDoc(
            name="y",
            type="pd.Series",
            default=REQUIRED,
            description="Low-frequency target series aligned to the LF index.",
        ),
    ),
    return_type="_MidasBetaModel",
    returns_attrs=(
        ("._w_hat", "np.ndarray", "Estimated Beta kernel lag weights, shape (n_lags_high,). Zero-padded when K_eff < n_lags_high."),
        ("._theta_hat", "np.ndarray", "Estimated Beta shape parameters [a, b], shape (2,)."),
        ("._intercept", "float", "Fitted intercept mu."),
        ("._slope", "float", "Fitted overall slope beta."),
        ("._converged", "bool", "NLS convergence flag."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_rows,)."),
    ),
)

_F_MIDAS_STEP = _f(
    "midas_step",
    "MIDAS with piecewise-constant step-function weights, OLS (Foroni-Marcellino-Schumacher 2015).",
    (
        "Restricted MIDAS with a step-function lag weight profile. The K "
        "HF lags are partitioned into S equal-size groups; within each "
        "group all lags share one coefficient estimated by OLS on the "
        "group-mean aggregate (Foroni, Marcellino & Schumacher 2015, "
        "§2.2). Group boundaries: group s covers lags in "
        "``[s*K//S, (s+1)*K//S)``.\n\n"
        "No stochastic initialization (OLS is deterministic). Closed-form "
        "via ``numpy.linalg.lstsq(rcond=None)``.\n\n"
        "``params.n_steps`` defaults to ``freq_ratio`` (one group per HF "
        "sub-period). ``params.freq_ratio = 1`` treats X columns as a flat "
        "lag sequence grouped by position index."
    ),
    (
        "Fast, interpretable mixed-frequency baseline. Useful when K is "
        "large and only the coarse lag structure matters. No NLS overhead."
    ),
    when_not_to_use=(
        "When the lag weight profile is smooth (use ``midas_almon`` or "
        "``midas_beta``) or when the fully-flexible U-MIDAS parameterization "
        "is affordable (use ``dfm_unrestricted_midas``)."
    ),
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation=(
                "Foroni, Marcellino & Schumacher (2015) 'Unrestricted Mixed "
                "Data Sampling (MIDAS): MIDAS Regressions with Unrestricted "
                "Lag Polynomials', Journal of the Royal Statistical Society: "
                "Series A 178(1)."
            )
        ),
    ),
    related_options=("midas_almon", "midas_beta", "dfm_unrestricted_midas"),
    op_page=True,
    op_func_name="midas_step",
    parameters=(
        ParameterDoc(
            name="freq_ratio",
            type="int",
            default=1,
            constraint=">=1",
            description="High-frequency periods per low-frequency period. 1 = X already LF-aligned.",
        ),
        ParameterDoc(
            name="n_lags_high",
            type="int",
            default=12,
            constraint=">=1",
            description="Number of high-frequency lags K to include.",
        ),
        ParameterDoc(
            name="n_steps",
            type="int",
            default="freq_ratio",
            constraint=">=1",
            description=(
                "Number of piecewise-constant groups S. Defaults to "
                "``freq_ratio`` (one group per HF sub-period). Determines "
                "the coarseness of the lag weight profile."
            ),
        ),
    ),
    data_args=(
        ParameterDoc(
            name="X",
            type="pd.DataFrame",
            default=REQUIRED,
            description=(
                "Feature matrix as a DataFrame. When ``freq_ratio = 1``: "
                "LF-aligned lag columns grouped by position. When "
                "``freq_ratio > 1``: raw HF DataFrame internally lag-stacked."
            ),
        ),
        ParameterDoc(
            name="y",
            type="pd.Series",
            default=REQUIRED,
            description="Low-frequency target series aligned to the LF index.",
        ),
    ),
    return_type="_MidasStepModel",
    returns_attrs=(
        ("._step_coef", "np.ndarray", "OLS step-group coefficients, shape (n_steps,)."),
        ("._intercept", "float", "OLS intercept."),
        ("._group_boundaries", "list[tuple[int,int]]", "Lag index boundaries per group, e.g. [(0,4),(4,8),(8,12)]."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_rows,)."),
    ),
)

_F_DFM_UNRESTRICTED_MIDAS = _f(
    "dfm_unrestricted_midas",
    "Unrestricted MIDAS (U-MIDAS) -- OLS on all HF lags (Foroni-Marcellino-Schumacher 2015).",
    (
        "Unrestricted mixed-data sampling regression: every HF lag enters "
        "linearly with its own OLS coefficient ``ψ_k`` (Foroni, Marcellino "
        "& Schumacher 2015, §3 eq. 7). The lag polynomial is fully flexible "
        "with no shape restriction, making U-MIDAS equivalent to OLS on the "
        "lag-stacked design matrix.\n\n"
        "``params.n_lags_high`` accepts an integer (fixed K), ``'bic'``, or "
        "``'aic'`` for information-criterion lag selection (Marcellino & "
        "Schumacher 2010; the pre-existing ``_bic_select_k`` helper is "
        "reused). When ``freq_ratio = 1``, BIC/AIC selection defaults to "
        "``K = X.shape[1]`` (all available columns).\n\n"
        "``params.include_y_lag = True`` augments the design matrix with the "
        "lagged dependent variable ``y_{t-1}`` (eq. 20), yielding a "
        "mixed-frequency ADL specification. In predict, the last observed "
        "``y`` value is used as the y-lag.\n\n"
        "OLS via ``numpy.linalg.lstsq(rcond=None)``. No stochastic step; "
        "``random_state`` is accepted for API symmetry but unused."
    ),
    (
        "Flexible mixed-frequency benchmark when T is large relative to K. "
        "BIC/AIC lag selection avoids manual K tuning. Pairs well with "
        "upstream L3 ``u_midas`` for feature preprocessing."
    ),
    when_not_to_use=(
        "When T is small relative to K (use ``midas_almon`` or "
        "``midas_beta`` for parsimonious alternatives). The 'dfm_' prefix "
        "is a historical naming artifact -- this is not a dynamic factor model."
    ),
    references=(
        _REF_DESIGN_L4,
        Reference(
            citation=(
                "Foroni, Marcellino & Schumacher (2015) 'Unrestricted Mixed "
                "Data Sampling (MIDAS): MIDAS Regressions with Unrestricted "
                "Lag Polynomials', Journal of the Royal Statistical Society: "
                "Series A 178(1)."
            )
        ),
        Reference(
            citation=(
                "Marcellino & Schumacher (2010) 'Factor MIDAS for Nowcasting "
                "and Forecasting with Ragged-Edge Data', Journal of Applied "
                "Econometrics 25(7)."
            )
        ),
    ),
    related_options=("midas_almon", "midas_beta", "midas_step", "dfm_mixed_mariano_murasawa"),
    op_page=True,
    op_func_name="dfm_unrestricted_midas",
    parameters=(
        ParameterDoc(
            name="freq_ratio",
            type="int",
            default=1,
            constraint=">=1",
            description="High-frequency periods per low-frequency period. 1 = X already LF-aligned.",
        ),
        ParameterDoc(
            name="n_lags_high",
            type='int | str enum {"bic", "aic"}',
            default='"bic"',
            description=(
                "Lag order K. Integer fixes K; ``'bic'`` or ``'aic'`` selects K "
                "via information criterion (``_bic_select_k``). When "
                "``freq_ratio = 1`` and IC selected, defaults to ``X.shape[1]``."
            ),
        ),
        ParameterDoc(
            name="include_y_lag",
            type="bool",
            default=False,
            description=(
                "Include lagged dependent variable y_{t-1} as an additional "
                "predictor (eq. 20 in Foroni et al. 2015). Yields a "
                "mixed-frequency ADL specification."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            description="Accepted for API symmetry with NLS families; unused (OLS is deterministic).",
        ),
    ),
    data_args=(
        ParameterDoc(
            name="X",
            type="pd.DataFrame",
            default=REQUIRED,
            description=(
                "Feature matrix as a DataFrame. When ``freq_ratio = 1``: "
                "LF-aligned lag columns. When ``freq_ratio > 1``: raw HF "
                "DataFrame internally lag-stacked to K lags."
            ),
        ),
        ParameterDoc(
            name="y",
            type="pd.Series",
            default=REQUIRED,
            description="Low-frequency target series aligned to the LF index.",
        ),
    ),
    return_type="_UnrestrictedMidasModel",
    returns_attrs=(
        ("._coef", "np.ndarray", "OLS coefficient vector, shape (K_eff + 1,) with intercept at index 0."),
        ("._intercept", "float", "OLS intercept (= _coef[0])."),
        ("._K_fit", "int", "Resolved number of HF lags K used in fitting."),
        (".predict(X)", "np.ndarray", "Predictions for new data X, shape (n_rows,)."),
    ),
)


register(
    _F_OLS,
    _F_RIDGE,
    _F_LASSO,
    _F_ELASTIC_NET,
    _F_LASSO_PATH,
    _F_BAYESIAN_RIDGE,
    _F_HUBER,
    _F_GLMBOOST,
    _F_AR_P,
    _F_VAR,
    _F_FAR,
    _F_PCR,
    _F_FAVAR,
    _F_BVAR_MIN,
    _F_BVAR_NIW,
    _F_DFM_MM,
    _F_DECISION_TREE,
    _F_RF,
    _F_EXTRA_TREES,
    _F_GB,
    _F_XGB,
    _F_LGBM,
    _F_CAT,
    _F_MRF,
    _F_QRF,
    _F_BAGGING,
    _F_SVR_LINEAR,
    _F_SVR_RBF,
    _F_SVR_POLY,
    _F_KERNEL_RIDGE,
    _F_KNN,
    _F_MLP,
    _F_LSTM,
    _F_GRU,
    _F_TRANSFORMER,
    # v0.9 Phase 2 paper-coverage atomic additions
    _F_MARS,
    # v0.9 Phase C top-6 net-new families
    _F_GARCH11,
    _F_EGARCH,
    _F_REALIZED_GARCH_WITH_RV_EXOG,
    _F_ETS,
    _F_THETA_METHOD,
    _F_HOLT_WINTERS,
    _S_DIRECT,
    _S_ITERATED,
    _S_PATH_AVG,
    _TS_EXPANDING,
    _TS_ROLLING,
    _TS_FIXED,
    _RP_EVERY,
    _RP_EVERY_N,
    _RP_SINGLE,
    _SA_NONE,
    _SA_CV,
    _SA_GRID,
    _SA_RAND,
    _SA_BAYES,
    _SA_GA,
    # v0.9 Phase C M12 pi_correction (Bai-Ng 2006)
    _PI_NONE,
    _PI_BAI_NG,
    # C48 MIDAS family honesty pass (2026-05-21)
    _F_MIDAS_ALMON,
    _F_MIDAS_BETA,
    _F_MIDAS_STEP,
    _F_DFM_UNRESTRICTED_MIDAS,
    # C49 realized_garch honesty pass (2026-05-21)
    _F_REALIZED_GARCH,
)
