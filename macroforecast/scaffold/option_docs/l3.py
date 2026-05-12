"""L3 feature-engineering DAG -- per-option documentation.

L3 is the cascading feature-engineering layer: 37 operational ops
across stationary / lag / aggregation / scale / reduction / spectral /
detrend / expansion / auxiliary / target / selection / combine
families. Each op is a node in the DAG; ops chain via ``inputs`` and
the cascade-depth gate (``cascade_max_depth``) bounds recursion.

This module ships Tier-1 docs for every operational L3 ``op`` choice.
The only L3 axis exposed via :data:`introspect.operational_options` is
``L3.A.op``; the ``inputs`` / ``params`` / ``temporal_rule`` keys are
operator-specific configuration that lives on the node body, not as
schema axes.
"""

from __future__ import annotations

from . import register
from .types import OptionDoc, Reference

_REVIEWED = "2026-05-05"
_REVIEWER = "macroforecast author"

_REF_DESIGN_L3 = Reference(
    citation="macroforecast design Part 2, L3: 'feature engineering is a DAG of typed transforms; cascade-depth bounds the longest chain at cascade_max_depth.'",
)
_REF_MCCRACKEN_NG = Reference(
    citation="McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4): 574-589.",
    url="https://doi.org/10.1080/07350015.2015.1086655",
)
_REF_STOCK_WATSON_2002 = Reference(
    citation="Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.",
)


def _o(
    option: str,
    summary: str,
    description: str,
    when_to_use: str,
    *,
    when_not_to_use: str = "",
    references: tuple[Reference, ...] = (_REF_DESIGN_L3,),
    related_options: tuple[str, ...] = (),
) -> OptionDoc:
    return OptionDoc(
        layer="l3",
        sublayer="L3_A_step_op",
        axis="op",
        option=option,
        summary=summary,
        description=description,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        references=references,
        related_options=related_options,
        last_reviewed=_REVIEWED,
        reviewer=_REVIEWER,
    )


# ---------------------------------------------------------------------------
# Stationary / level-change family (5 ops)
# ---------------------------------------------------------------------------

_OP_LEVEL = _o(
    "level",
    "Pass-through: keep the series at its original level.",
    (
        "No-op transform; the column flows through unchanged. Used as "
        "an explicit anchor in the DAG so downstream ops can reference "
        "the level form even when the L2 transform_policy converted to "
        "log-differences. Useful when you want both ``level`` and ``diff`` "
        "branches in the same recipe."
    ),
    "Authoring branches that need the level form alongside a transformed branch.",
    related_options=("diff", "log", "log_diff", "pct_change"),
)

_OP_DIFF = _o(
    "diff",
    "First difference: ``y_t - y_{t-1}``.",
    (
        "Computes the simple first difference on the input column. The "
        "first observation becomes NaN. Combine with ``lag`` to recover "
        "level features when the L2 layer already differenced the panel."
    ),
    "I(1) variables that need a stationary representation in addition to the L2-applied tcode.",
    when_not_to_use="When the panel is already differenced by L2.B (avoids double-differencing).",
    related_options=("level", "log_diff", "pct_change"),
)

_OP_LOG = _o(
    "log",
    "Natural logarithm: ``ln(y)``.",
    (
        "Element-wise natural log. Strictly positive series only; raises "
        "if any input is non-positive. Often paired with ``diff`` to "
        "produce log-changes (which are approximately equal to "
        "percentage changes for small movements)."
    ),
    "Strictly-positive macro series (price levels, employment counts, GDP) before differencing.",
    when_not_to_use="Series that can be negative or zero (rates, growth rates, balances).",
    related_options=("log_diff", "level", "pct_change"),
)

_OP_LOG_DIFF = _o(
    "log_diff",
    "Log first difference: ``ln(y_t) - ln(y_{t-1})``.",
    (
        "Composite of ``log`` then ``diff``. The standard FRED-MD "
        "transformation code 5/6 representation; produces a stationary "
        "approximation of the percentage change and is symmetric in "
        "expansions vs contractions."
    ),
    "Strictly-positive trending series (real GDP, employment, prices); FRED-MD tcode 5/6 default.",
    when_not_to_use="Series that take non-positive values.",
    references=(_REF_DESIGN_L3, _REF_MCCRACKEN_NG),
    related_options=("log", "diff", "pct_change"),
)

_OP_PCT_CHANGE = _o(
    "pct_change",
    "Period-over-period percentage change: ``(y_t / y_{t-1}) - 1``.",
    (
        "Strict simple growth rate; not equivalent to log_diff for "
        "large movements. Returns NaN where the previous observation "
        "is zero or NaN."
    ),
    "When a literal percentage interpretation is required (returns, inflation rates).",
    when_not_to_use="Trend-following analysis where log_diff's symmetry is preferable.",
    related_options=("log_diff", "diff"),
)


# ---------------------------------------------------------------------------
# Lag / seasonal-lag family (2 ops)
# ---------------------------------------------------------------------------

_OP_SEASONAL_LAG = _o(
    "seasonal_lag",
    "Lag at a seasonal period (e.g. y_{t-12} for monthly data).",
    (
        "Standard ``lag`` op restricted to the seasonal index "
        "(``params.lag = 12`` for monthly, ``4`` for quarterly). "
        "Useful for year-over-year features and seasonal AR terms."
    ),
    "Capturing year-over-year persistence; seasonal AR baselines.",
    when_not_to_use="When season_dummy or X-13 deseasonalisation is preferred over lag-based seasonality.",
    related_options=("season_dummy", "ma_window"),
)


# ---------------------------------------------------------------------------
# Aggregation / moving-window family (3 ops)
# ---------------------------------------------------------------------------

_OP_MA_WINDOW = _o(
    "ma_window",
    "Trailing moving average over a fixed window.",
    (
        "Computes ``mean(y_{t-w+1..t})`` for a user-specified window "
        "``params.window``. ``temporal_rule`` controls expanding vs "
        "rolling vs block-wise refit semantics. The first ``w-1`` rows "
        "are NaN."
    ),
    "Smoothing noisy series; building short / medium / long-term momentum features.",
    related_options=("ma_increasing_order", "diff", "scale"),
)

_OP_MA_INCREASING = _o(
    "ma_increasing_order",
    "MARX -- moving averages of increasing order (Coulombe 2024).",
    (
        "Stacks moving averages with windows ``[1, 2, 4, 8, ..., w_max]`` "
        "into a multi-column block. Captures multi-scale persistence "
        "in a single op; popular feature in macroeconomic random "
        "forest pipelines.\n\n"
        "Implements the MARX (Moving-Average-of-Random-eXogeneous) "
        "trick from Coulombe (2024)."
    ),
    "Tree / RF models that benefit from multi-scale temporal features without manual lag selection.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Coulombe (2024) 'The Macroeconomic Random Forest', Journal of Applied Econometrics 39(7): 1190-1209."
        ),
    ),
    related_options=("ma_window", "lag"),
)

_OP_CUMSUM = _o(
    "cumsum",
    "Cumulative sum of a series.",
    (
        "Running total ``Σ_{s ≤ t} y_s``. Inverts ``diff`` (modulo "
        "an initial constant). Used to recover level forecasts from "
        "differenced predictions or to build cumulative-shock features."
    ),
    "Building cumulative-impact features; recovering levels from differenced forecasts.",
    related_options=("diff", "level"),
)


# ---------------------------------------------------------------------------
# Scale family (1 op)
# ---------------------------------------------------------------------------

_OP_SCALE = _o(
    "scale",
    "Standardise to zero mean and unit variance.",
    (
        "Computes ``(y - μ) / σ`` over the temporal_rule window "
        "(``expanding_window_per_origin`` by default to avoid look-ahead). "
        "Required pre-step for distance-based learners (kNN, SVM, NN); "
        "ridge/lasso also benefit when columns are on different scales."
    ),
    "Pre-conditioning for distance-based or regularised learners; mandatory for SVM/NN.",
    when_not_to_use="Tree-based models (RF/XGBoost/LightGBM) -- scale-invariant by construction.",
    related_options=("pca", "kernel_features"),
)


# ---------------------------------------------------------------------------
# Reduction / factor family (7 ops)
# ---------------------------------------------------------------------------

_OP_PCA = _o(
    "pca",
    "Principal component analysis -- linear factor extraction.",
    (
        "Eigendecomposition of the column covariance; returns the "
        "top ``params.n_components`` principal components. Implements "
        "the Stock-Watson (2002) diffusion-index workflow used "
        "throughout FRED-MD applications.\n\n"
        "Combine with ``factor_augmented_ar`` or ``factor_augmented_var`` "
        "at L4 to build the diffusion-index forecaster. ``temporal_rule`` "
        "controls whether components are re-fit per origin "
        "(default: ``expanding_window_per_origin``)."
    ),
    "Reducing FRED-MD's 100+ predictors to a handful of latent factors; factor-augmented forecasts.",
    references=(_REF_DESIGN_L3, _REF_STOCK_WATSON_2002),
    related_options=(
        "sparse_pca",
        "scaled_pca",
        "varimax",
        "dfm",
        "partial_least_squares",
    ),
)

_OP_SPARSE_PCA = _o(
    "sparse_pca",
    "Sparse PCA -- L1-penalised factor loadings (sklearn / Zou-Hastie-Tibshirani 2006).",
    (
        "Variant of PCA where loadings are pushed toward zero by an "
        "L1 penalty (``params.alpha``). Yields more interpretable "
        "factors at the cost of a small reconstruction loss; uses "
        "sklearn's ``SparsePCA``."
    ),
    "When you want factor loadings to map cleanly onto a small subset of original predictors (interpretability).",
    when_not_to_use="When pure variance maximisation is more important than interpretability -- use plain ``pca``. For the Chen-Rohe (2023) SCA variant used in Zhou-Rapach (2025) Sparse Macro-Finance Factors, use ``sparse_pca_chen_rohe`` instead.",
    related_options=("pca", "scaled_pca", "sparse_pca_chen_rohe", "supervised_pca"),
)

_OP_SPARSE_PCA_CHEN_ROHE = _o(
    "sparse_pca_chen_rohe",
    "Chen-Rohe (2023) Sparse Component Analysis -- non-diagonal D variant.",
    (
        "Sparse component analysis solving "
        "``min_{Z,D,Θ} ‖X − Z D Θ'‖_F`` s.t. ``Z ∈ S(T,J)``, "
        "``Θ ∈ S(M,J)``, ``‖Θ‖_1 ≤ ζ`` (Chen-Rohe 2023; Rapach & Zhou "
        "2025 eq. 3). Differs from ``sparse_pca`` (sklearn / Zou-"
        "Hastie-Tibshirani 2006) in two ways: (1) the central matrix "
        "D is *not* restricted to be diagonal, which lets SCA explain "
        "more total variation for a given sparsity budget; (2) the "
        "single hyperparameter ``ζ ∈ [J, J√M]`` enters as an ℓ_1 "
        "budget *constraint* rather than a Lagrangian penalty.\n\n"
        "Implementation: alternating maximisation of the equivalent "
        "bilinear convex-hull form ``max_{Z,Θ} ‖Z' X Θ‖_F`` over "
        "``H(T,J) × H(M,J)`` (Zhou-Rapach 2025 eq. 4), iterating SVD-"
        "projection of Z and L1-budget projection of Θ. Used as the "
        "macro-side stage in Rapach & Zhou (2025) Sparse Macro-"
        "Finance Factors. Operational v0.9.1 dev-stage v0.9.0C-3.\n\n"
        "Hyperparams: ``n_components`` (= J; default 4), ``zeta`` "
        "(= L1 budget; ``0.0`` defaults to J = most-binding boundary "
        "the paper finds optimal in CV), ``max_iter`` (default 200), "
        "``random_state``."
    ),
    "Sparse macro-finance factor extraction with non-diagonal D; the Rapach-Zhou (2025) macro-side procedure.",
    when_not_to_use="When sklearn-style L1-penalised loadings are sufficient -- prefer the cheaper ``sparse_pca``.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Chen & Rohe (2023) 'A New Basis for Sparse Principal Component Analysis', Journal of Computational and Graphical Statistics. arXiv:2007.00596."
        ),
        Reference(
            citation="Rapach & Zhou (2025) 'Sparse Macro-Finance Factors' working paper -- §2.1 eqs. (3)-(4)."
        ),
    ),
    related_options=("sparse_pca", "supervised_pca", "scaled_pca", "pca"),
)

_OP_SUPERVISED_PCA = _o(
    "supervised_pca",
    "Supervised PCA (Giglio-Xiu-Zhang 2025) -- screen-then-PCA on a target panel.",
    (
        "Two-stage supervised reduction:\n"
        "  1. For each target column ``g``, rank panel columns by "
        "univariate correlation with ``g`` and keep the top "
        "``⌊q · M⌋`` (q ∈ (0, 1] hyperparameter; default 0.5);\n"
        "  2. Run PCA on the screened sub-panel, returning P "
        "supervised components.\n\n"
        "Refinement of Giglio-Xiu (2021) three-pass: screening makes "
        "the construction robust to weak factors and omitted-variable "
        "bias. Used as the asset-side stage of Rapach & Zhou (2025) "
        "Sparse Macro-Finance Factors for risk-premium estimation. "
        "Distinct from ``partial_least_squares`` (PLS uses covariance-"
        "maximising NIPALS over all columns; SPCA uses correlation-"
        "screened PCA on a sub-panel) and from ``scaled_pca`` (Huang-"
        "Jiang-Tu-Zhou 2022 weights every column; SPCA hard-screens).\n\n"
        "Operational v0.9.1 dev-stage v0.9.0C-4. Hyperparams: "
        "``n_components`` (= P; default 4), ``q`` (screening rate; "
        "default 0.5)."
    ),
    "Cross-sectional asset-pricing factor extraction; weak-factor-robust supervised reduction; Rapach-Zhou (2025) replication.",
    when_not_to_use="When the supervisory signal is dense (every panel column matters) -- prefer ``scaled_pca`` or ``partial_least_squares``.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Giglio, Xiu & Zhang (2025) 'Test Assets and Weak Factors', Journal of Finance, forthcoming."
        ),
        Reference(
            citation="Giglio & Xiu (2021) 'Asset Pricing with Omitted Factors', Journal of Political Economy 129(7): 1947-1990."
        ),
        Reference(
            citation="Rapach & Zhou (2025) 'Sparse Macro-Finance Factors' working paper -- §2.2 eqs. (5)-(8)."
        ),
    ),
    related_options=(
        "partial_least_squares",
        "scaled_pca",
        "sparse_pca_chen_rohe",
        "pca",
    ),
)

_OP_SCALED_PCA = _o(
    "scaled_pca",
    "Scaled / weighted PCA (target-aware factor extraction).",
    (
        "Weights each column by its predictive correlation with the "
        "target before performing PCA. Implements the Huang-Jiang-"
        "Tu-Zhou (2022) scaled PCA for forecasting macro variables.\n\n"
        "Reduces to plain PCA when all weights are equal."
    ),
    "When standard PCA's leading factor is dominated by predictively-irrelevant variance.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Huang, Jiang, Tu & Zhou (2022) 'Scaled PCA: A New Approach to Dimension Reduction', Management Science 68(3): 1678-1695."
        ),
    ),
    related_options=("pca", "partial_least_squares"),
)

_OP_DFM = _o(
    "dfm",
    "Dynamic factor model -- Kalman state-space factor extraction.",
    (
        "statsmodels ``DynamicFactor`` MLE estimate of latent factors "
        "with idiosyncratic AR(1) errors. Differs from ``pca`` in that "
        "factors are smoothed via the Kalman filter and respect a "
        "factor-VAR transition.\n\n"
        "When the panel is mixed-frequency (FRED-SD), the runtime "
        "auto-routes to ``DynamicFactorMQ`` (Mariano-Murasawa 2003)."
    ),
    "Smoothed factors with an explicit dynamic; mixed-frequency panels (FRED-SD).",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443."
        ),
    ),
    related_options=("pca", "scaled_pca"),
)

_OP_VARIMAX = _o(
    "varimax",
    "Varimax-rotated factors (orthogonal rotation for interpretability).",
    (
        "Applies a varimax rotation to PCA loadings, maximising the "
        "variance of squared loadings within each factor. Produces "
        "factors that load heavily on a small subset of original "
        "predictors -- useful for naming / labelling factors."
    ),
    "Factor analysis where downstream interpretation requires distinct, well-named factors.",
    related_options=("pca", "sparse_pca"),
)

_OP_VARIMAX_ROTATION = _o(
    "varimax_rotation",
    "Alias for ``varimax`` -- rotation step in a multi-stage factor pipeline.",
    (
        "Identical operation to ``varimax`` but registered separately "
        "so a cascading L3 pipeline can declare ``pca → varimax_rotation`` "
        "as two visible nodes in its lineage."
    ),
    "Multi-stage pipelines that explicitly separate factor extraction from rotation.",
    related_options=("varimax", "pca"),
)

_OP_PARTIAL_LEAST_SQUARES = _o(
    "partial_least_squares",
    "Partial least squares regression -- supervised factor extraction.",
    (
        "Computes orthogonal latent components that maximise the "
        "covariance with the target (not just predictor variance, "
        "as in PCA). sklearn's ``PLSRegression``; ``params.n_components``."
    ),
    "When a target-supervised reduction is preferable to PCA's unsupervised projection.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Wold, Sjöström & Eriksson (2001) 'PLS-regression: a basic tool of chemometrics', Chemometrics and Intelligent Laboratory Systems 58(2): 109-130."
        ),
    ),
    related_options=("pca", "scaled_pca"),
)

_OP_RANDOM_PROJECTION = _o(
    "random_projection",
    "Johnson-Lindenstrauss random Gaussian projection.",
    (
        "Reduces dimensionality by multiplying with a random Gaussian "
        "matrix scaled to (approximately) preserve pairwise distances. "
        "Cheap baseline for dimensionality reduction; sklearn's "
        "``GaussianRandomProjection``."
    ),
    "Sweep baselines / sanity checks against PCA's structured reduction.",
    related_options=("pca", "kernel_features"),
)


# ---------------------------------------------------------------------------
# Spectral family (2 ops)
# ---------------------------------------------------------------------------

_OP_FOURIER = _o(
    "fourier",
    "Fourier basis features -- sin/cos at fixed harmonics.",
    (
        "Generates sin/cos pairs at harmonic frequencies of the "
        "calendar period (``params.period``, ``params.n_harmonics``). "
        "Captures smooth periodic patterns without the indicator-"
        "explosion of season_dummy."
    ),
    "Smooth seasonality (annual / weekly cycles) where dummies would over-fit.",
    related_options=("season_dummy", "wavelet"),
)

_OP_WAVELET = _o(
    "wavelet",
    "Discrete wavelet transform -- multi-scale time-frequency features.",
    (
        "Decomposes the series into wavelet detail and approximation "
        "coefficients at several scales (``params.wavelet``, "
        "``params.level``). Captures localised time-frequency patterns "
        "that Fourier basis cannot."
    ),
    "Series with localised oscillations or non-stationary cycles (financial / climate macro).",
    when_not_to_use="Smooth seasonal patterns -- use ``fourier`` instead.",
    related_options=("fourier",),
)


# ---------------------------------------------------------------------------
# Detrend family (2 ops)
# ---------------------------------------------------------------------------

_OP_HP_FILTER = _o(
    "hp_filter",
    "Hodrick-Prescott filter -- trend / cycle decomposition.",
    (
        "statsmodels ``hpfilter`` with smoothing parameter "
        "``params.lamb`` (1600 for quarterly, 129600 for monthly per "
        "Ravn-Uhlig 2002). Returns the cyclical component by default; "
        "the trend can also be returned via ``params.return = 'trend'``."
    ),
    "Extracting business-cycle gaps from trending series.",
    when_not_to_use="Real-time / one-sided forecasting -- HP introduces look-ahead bias unless restricted to ``expanding_window_per_origin``.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Hodrick & Prescott (1997) 'Postwar U.S. Business Cycles: An Empirical Investigation', JMCB 29(1): 1-16."
        ),
    ),
    related_options=("hamilton_filter", "diff"),
)

_OP_HAMILTON_FILTER = _o(
    "hamilton_filter",
    "Hamilton (2018) regression-based detrend (HP-filter alternative).",
    (
        "Regression-based two-sided alternative to the HP filter "
        "advocated by Hamilton (2018) for its better real-time "
        "properties. Default lookback h = 8 (quarterly) / 24 (monthly). "
        "Uses statsmodels ``hamilton_filter``."
    ),
    "Real-time / one-sided detrending where HP's two-sided smoothing is inappropriate.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Hamilton (2018) 'Why You Should Never Use the Hodrick-Prescott Filter', RES 100(5): 831-843."
        ),
    ),
    related_options=("hp_filter",),
)


# ---------------------------------------------------------------------------
# Expansion family (4 ops)
# ---------------------------------------------------------------------------

_OP_POLYNOMIAL = _o(
    "polynomial",
    "Polynomial basis expansion -- degree-d powers of input.",
    (
        "sklearn ``PolynomialFeatures`` of degree ``params.degree``. "
        "Includes interaction terms by default; set "
        "``params.interaction_only=True`` for products without pure "
        "powers."
    ),
    "Capturing low-order non-linearity for linear / kernel models.",
    when_not_to_use="High dimension (degree > 3 with many predictors) -- explodes the design matrix; use kernel methods instead.",
    related_options=("interaction", "kernel_features", "polynomial_expansion"),
)

_OP_POLYNOMIAL_EXPANSION = _o(
    "polynomial_expansion",
    "Alias for ``polynomial`` -- explicit expansion node in cascade pipelines.",
    (
        "Identical to ``polynomial`` but with a name that reads more "
        "clearly as a stage in a multi-step expansion pipeline."
    ),
    "Pipelines that explicitly stage `expand → reduce` sequences.",
    related_options=("polynomial",),
)

_OP_INTERACTION = _o(
    "interaction",
    "Pairwise interaction terms only (no pure powers).",
    (
        "Subset of polynomial degree-2 features that contains only "
        "pairwise products ``x_i · x_j`` for ``i ≠ j``. Cheaper than "
        "full polynomial expansion when interaction structure (not "
        "non-linearity in single inputs) is the target."
    ),
    "Capturing predictor-pair complementarities in linear models.",
    related_options=("polynomial",),
)

_OP_KERNEL = _o(
    "kernel",
    "Kernel-feature pre-step (Random Fourier / Nyström handle).",
    (
        "Generic handle for an explicit kernel-feature embedding; "
        "concrete dispatch is determined by ``params.kernel`` "
        "(``rbf`` / ``poly`` / ``laplacian``). For named variants use "
        "``kernel_features`` (RBF Random Fourier) or ``nystroem``."
    ),
    "Kernel-augmented linear / SVM pipelines.",
    related_options=("kernel_features", "nystroem"),
)

_OP_KERNEL_FEATURES = _o(
    "kernel_features",
    "Random Fourier features -- approximate RBF kernel via random projection.",
    (
        "sklearn ``RBFSampler``: maps inputs to ``params.n_components`` "
        "random Fourier features whose dot product approximates the "
        "RBF kernel. Enables linear models to fit RBF-kernelised "
        "responses at training-set-size linear cost."
    ),
    "Kernel-augmented ridge / SVM at scale (n > 10k).",
    when_not_to_use="Small-sample problems where exact kernel SVM is feasible.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Rahimi & Recht (2007) 'Random Features for Large-Scale Kernel Machines', NeurIPS."
        ),
    ),
    related_options=("kernel", "nystroem"),
)

_OP_NYSTROEM = _o(
    "nystroem",
    "Nyström kernel approximation -- subset-based feature map.",
    (
        "sklearn ``Nystroem`` constructs a low-rank approximation of "
        "an arbitrary kernel matrix using a random subsample of "
        "training points. More accurate than Random Fourier features "
        "for non-RBF kernels but with a larger memory footprint."
    ),
    "Non-RBF kernel-augmented linear models (poly / sigmoid).",
    related_options=("kernel_features", "kernel"),
)

_OP_NYSTROEM_FEATURES = _o(
    "nystroem_features",
    "Alias for ``nystroem`` -- explicit feature-stage name.",
    (
        "Identical to ``nystroem``; preferred when a multi-stage "
        "pipeline names its kernel approximation explicitly in the "
        "lineage graph."
    ),
    "Multi-stage pipelines that separate kernel approximation from downstream linear fits.",
    related_options=("nystroem",),
)


# ---------------------------------------------------------------------------
# Auxiliary / calendar family (4 ops)
# ---------------------------------------------------------------------------

_OP_REGIME_INDICATOR = _o(
    "regime_indicator",
    "Discrete regime / state indicator from L1.G.",
    (
        "Lifts the L1.G regime sink (``regime_indicator``) into the "
        "feature panel as a categorical column. Required when L4 is "
        "configured for ``regime_use = predictor`` or "
        "``conditional_intercept``."
    ),
    "Regime-conditional forecasts where the model needs explicit access to the state.",
    related_options=("season_dummy", "time_trend"),
)

_OP_SEASON_DUMMY = _o(
    "season_dummy",
    "Calendar dummy variables (month-of-year, quarter-of-year).",
    (
        "Generates ``params.n - 1`` 0/1 indicators for the calendar "
        "period (drops one to avoid multicollinearity with intercept). "
        "Standard frequentist seasonality control."
    ),
    "Capturing calendar seasonality in linear models when a smooth Fourier basis would over-shrink discrete jumps.",
    related_options=("fourier", "seasonal_lag"),
)

_OP_TIME_TREND = _o(
    "time_trend",
    "Deterministic linear time trend (``t = 1, 2, ...``).",
    (
        "Adds a column ``t`` to the panel; with ``params.degree > 1`` "
        "appends polynomial trends. Deterministic complement to "
        "stochastic detrending (HP / Hamilton)."
    ),
    "Trend-stationary linear models where a deterministic trend is part of the DGP.",
    when_not_to_use="Series with structural breaks -- use ``regime_indicator`` or stochastic detrending instead.",
    related_options=("hp_filter", "hamilton_filter"),
)

_OP_HOLIDAY = _o(
    "holiday",
    "Holiday / event dummy variables.",
    (
        "0/1 indicators for calendar holidays (US federal by default; "
        "``params.country`` selects locale via the ``holidays`` "
        "package). For business / financial macro series."
    ),
    "Daily / weekly business-cycle series where holidays create discrete level shifts.",
    when_not_to_use="Pure macro series at monthly+ frequency where holidays are absorbed by ``season_dummy``.",
    related_options=("season_dummy",),
)


# ---------------------------------------------------------------------------
# Target family (1 op)
# ---------------------------------------------------------------------------

_OP_TARGET_CONSTRUCTION = _o(
    "target_construction",
    "Build the supervised target (``y``) from the panel.",
    (
        "Constructs the regression target according to "
        "``forecast_strategy`` (direct / iterated / cumulative_average) "
        "and the L1.F horizon set. Outputs the ``y`` artifact that L4 "
        "fit_model nodes consume.\n\n"
        "Required as the leaf of every L3 DAG; the runtime auto-injects "
        "it when the user does not."
    ),
    "Always required -- runtime auto-inserts when missing.",
    related_options=("level", "diff", "log_diff"),
)


# ---------------------------------------------------------------------------
# Selection family (1 op)
# ---------------------------------------------------------------------------

_OP_FEATURE_SELECTION = _o(
    "feature_selection",
    "Filter columns by variance / correlation / lasso pre-screen.",
    (
        "Drops columns failing one of three criteria configured via "
        "``params.method``:\n\n"
        "* ``variance`` -- drop columns with variance below "
        "``params.threshold``.\n"
        "* ``correlation`` -- drop columns with pairwise correlation "
        "above ``params.threshold`` (keeps the first).\n"
        "* ``lasso`` -- fit a Lasso pre-screen and keep columns with "
        "non-zero coefficients."
    ),
    "Trimming the panel before expensive downstream estimators (NN, SVM, kernel) when high-dim noise dominates.",
    when_not_to_use="Tree models -- they handle irrelevant features natively.",
    related_options=("scale", "pca"),
)


# ---------------------------------------------------------------------------
# Combine / DAG sentinels (2 ops -- internal nodes used by the cascade machinery)
# ---------------------------------------------------------------------------

_OP_L3_FEATURE_BUNDLE = _o(
    "l3_feature_bundle",
    "Internal sentinel: bundle column-producing children into the X matrix.",
    (
        "Synthetic node emitted by the runtime to collect the outputs "
        "of all leaf transforms into a single ``X`` artifact consumed "
        "by L4. Authors do not write this node by hand; the cascade "
        "builder inserts it.\n\n"
        "Surfaced in ``operational_options`` so the schema completeness "
        "test stays consistent."
    ),
    "Never authored directly -- inserted by the cascade builder.",
    related_options=("l3_metadata_build",),
)

_OP_L3_METADATA_BUILD = _o(
    "l3_metadata_build",
    "Internal sentinel: assemble L3 metadata sink (lineage / pipeline definitions).",
    (
        "Synthetic node that produces the ``column_lineage`` and "
        "``pipeline_definitions`` sinks from the cascade graph. Powers "
        "L7 ``lineage_attribution`` and ``transformation_attribution`` "
        "downstream."
    ),
    "Never authored directly -- inserted by the cascade builder when L7 lineage hooks are active.",
    related_options=("l3_feature_bundle",),
)


# ---------------------------------------------------------------------------
# v0.9 Phase 2 paper-coverage atomic primitives.
# ---------------------------------------------------------------------------

_OP_SAVITZKY_GOLAY_FILTER = _o(
    "savitzky_golay_filter",
    "Polynomial-fit smoothing filter (Savitzky & Golay 1964).",
    (
        "Local polynomial regression smoothing: each output value is the "
        "polynomial-fit centre value of a moving window. ``window_length`` "
        "(default 5) and ``polyorder`` (default 2) parameterise the "
        "kernel. Operational: runtime delegates to "
        "``scipy.signal.savgol_filter`` (scipy is a hard dependency).\n\n"
        "Used as the fixed-window baseline against which Goulet Coulombe "
        "& Klieber (2025) AlbaMA's adaptive-window estimator is compared "
        "in the v0.9.x replication recipe."
    ),
    "Smoothing macro indicator series for monitoring; AlbaMA replication baseline.",
    when_not_to_use="Series with strong non-linear trends -- the polynomial fit smooths them out.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Savitzky & Golay (1964) 'Smoothing and Differentiation of Data by Simplified Least Squares Procedures', Analytical Chemistry 36(8)."
        ),
    ),
    related_options=("hp_filter", "hamilton_filter", "ma_window", "adaptive_ma_rf"),
)

_OP_ASYMMETRIC_TRIM = _o(
    "asymmetric_trim",
    "Albacore-family rank-space transformation (Goulet Coulombe et al. 2024).",
    (
        "Per-period sort: panel ``Π`` of shape ``(T, K)`` is mapped to "
        "``O`` where ``O[t, r] = sort(Π[t, :])[r]`` (ascending). "
        "Asymmetric trimming emerges in the *downstream* nonneg ridge "
        "(``ridge(coefficient_constraint=nonneg)``) that learns "
        "rank-position weights -- this op does the rank-space "
        "transformation only.\n\n"
        "Optional ``smooth_window > 0`` applies a centred moving "
        "average to each rank-position time series (paper §3 mentions "
        "3-month MA for noisy components; users can chain ``ma_window`` "
        "explicitly when they want a different window).\n\n"
        "Operational from v0.8.9 (B-6). Layer scope ``(l2, l3)`` so "
        "the L3 DAG can dispatch it at recipe time. Algorithm spec: "
        "``docs/replications/maximally_forward_looking_algorithm_notes.md``."
    ),
    "Building Albacore_ranks-style core inflation indicators; supervised asymmetric trimming where the band is learned from data.",
    when_not_to_use="Symmetric trimmed-mean targets (use a fixed-window ``ma_window`` instead).",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Goulet Coulombe, Klieber, Barrette & Goebel (2024) 'Maximally Forward-Looking Core Inflation', technical report (R package: assemblage)."
        ),
    ),
    related_options=("ma_window", "ma_increasing_order", "scaled_pca"),
)


_OP_ADAPTIVE_MA_RF = _o(
    "adaptive_ma_rf",
    "AlbaMA -- RF-driven adaptive moving average smoother for a single time series.",
    (
        "Goulet Coulombe & Klieber (2025) 'Adaptive Moving Average for "
        "Macroeconomic Monitoring' (arXiv:2501.13222 §2). A random "
        "forest fit with a *single* regressor -- the time index -- on "
        "the target series ``y`` (i.e. ``RF(y_t ~ t)``). Per-"
        "observation leaf membership induces a weight matrix ``w_τt`` "
        "whose row sums to 1, so the smoother is a learned-bandwidth "
        "moving average of ``y``; the realised window adapts to local "
        "volatility / regime. Paper p.8 defaults: ``n_estimators = "
        "B = 500``, ``min_samples_leaf = 40``, ``max_features = 1``. "
        "``sided = 'two'`` (default) fits one forest on the full "
        "sample (retrospective smoother); ``sided = 'one'`` fits an "
        "expanding-window forest per ``t`` (real-time nowcasting "
        "variant, paper §3.3 / p.10).\n\n"
        "Atomic primitive: existing ``ma_window`` uses a fixed length; "
        "``hamilton_filter`` is a regression on lags rather than a "
        "moving average; neither composes into AlbaMA without a "
        "learned window selector."
    ),
    "Replicating AlbaMA recipes; macro indicator monitoring under regime shifts.",
    when_not_to_use="Multivariate denoising of a predictor panel (AlbaMA smooths a single target series).",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Goulet Coulombe & Klieber (2025) 'An Adaptive Moving Average for Macroeconomic Monitoring', arXiv:2501.13222."
        ),
    ),
    related_options=(
        "savitzky_golay_filter",
        "hamilton_filter",
        "hp_filter",
        "ma_window",
    ),
)


# ---------------------------------------------------------------------------
# v0.9 Phase C top-6 net-new methods (mixed-frequency + supervised reduction).
# ---------------------------------------------------------------------------

_OP_U_MIDAS = _o(
    "u_midas",
    "Unrestricted MIDAS lag stack -- mixed-frequency aggregation primitive (Foroni-Marcellino-Schumacher 2015).",
    (
        "Atomic mixed-frequency primitive: stacks ``K = n_lags_high`` "
        "high-frequency lags of each predictor at low-frequency dates "
        "without imposing any weighting structure. For HF column "
        "``col`` and LF index position ``t·m``, emits "
        "``col_lag0, col_lag1, …, col_lag{K-1}`` where "
        "``col_lagk[t] = frame[col].iloc[t·m − k]``. The downstream L4 "
        "ridge / OLS / lasso recovers data-driven lag coefficients, "
        "i.e. the *unrestricted* MIDAS regression ``y_t = α + Σ_k β_k "
        "x_{t·m − k} + ε_t``.\n\n"
        "**Defaults**: ``freq_ratio = 3`` (quarterly target / monthly "
        "HF), ``n_lags_high = 6`` (≈ 2·m); ``target_freq = 'low'`` "
        "subsamples the LF anchor dates. ``temporal_rule`` is required "
        "and rejects ``full_sample_once`` so the aggregation respects "
        "walk-forward boundaries.\n\n"
        "Surfaces the Borup-Rapach-Schütte (2023) mixed-frequency "
        "ML-nowcasting workflow as a 1-line recipe via "
        "``paper_methods.u_midas(...)``."
    ),
    "Macro nowcasting with monthly predictors and quarterly targets; mixed-frequency feature engineering when no parametric weight kernel is desired.",
    when_not_to_use="When ``n_lags_high · n_HF_columns`` exceeds T (use ``midas`` parametric weighting instead, or pair with downstream lasso).",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Foroni, Marcellino & Schumacher (2015) 'Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions With Unrestricted Lag Polynomials', JRSS-A 178(1): 57-82."
        ),
        Reference(
            citation="Borup, Rapach & Schütte (2023) 'Mixed-frequency machine learning: Nowcasting and backcasting weekly initial claims with daily internet search-volume data', International Journal of Forecasting 39(3): 1122-1144."
        ),
    ),
    related_options=("midas", "lag", "ma_window"),
)


_OP_MIDAS = _o(
    "midas",
    "MIDAS Almon / Exp-Almon / Beta weighted lag polynomial (Ghysels-Sinko-Valkanov 2007).",
    (
        "Parametric mixed-frequency aggregation: emits one column per HF "
        "predictor whose value is the weighted sum "
        "``Σ_k ω_k(θ̂) · x_{t·m − k}``. The weight kernel ``ω_k(θ)`` is "
        "fitted by NLS against the LF ``target_signal`` input.\n\n"
        "**Three weighting families**:\n"
        "* ``almon`` -- polynomial basis ``ω_k = Σ_q θ_q · k^q`` of "
        "order ``polynomial_order`` (default 2). Optional sum-to-one "
        "normalisation.\n"
        "* ``exp_almon`` (default) -- ``ω_k ∝ exp(θ_1 k + θ_2 k²)``; "
        "numerically stable and the GSV 2007 §3 default.\n"
        "* ``beta`` -- ``ω_k ∝ k_norm^{θ_1−1} (1 − k_norm)^{θ_2−1}`` for "
        "``k_norm = (k+1)/(K+1)``; flexible monotone / hump kernel.\n\n"
        "Defaults: ``freq_ratio = 3``, ``n_lags_high = 12``, "
        "``sum_to_one = True``, ``max_iter = 200``. Optimiser: "
        "``scipy.optimize.minimize`` (Nelder-Mead). Per-predictor "
        "``theta_hat`` / ``weights`` / ``converged`` stashed in "
        "``result.attrs['midas_fit']`` for L7 inspection.\n\n"
        "Requires a ``target_signal`` input port -- shares routing with "
        "``scaled_pca``."
    ),
    "Mixed-frequency nowcasting where parametric lag weights are desired; reproducing GSV 2007 macro / asset-pricing applications.",
    when_not_to_use="High-noise predictors where NLS optimiser diverges (use ``u_midas`` + downstream regularised regression instead).",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Ghysels, Sinko & Valkanov (2007) 'MIDAS Regressions: Further Results and New Directions', Econometric Reviews 26(1): 53-90."
        ),
        Reference(
            citation="Ghysels, Santa-Clara & Valkanov (2004) 'The MIDAS Touch: Mixed Data Sampling Regression Models', UCLA / UNC working paper."
        ),
    ),
    related_options=("u_midas", "scaled_pca", "lag"),
)


_OP_MAF_PER_VARIABLE_PCA = _o(
    "maf_per_variable_pca",
    "Per-variable MAF via PCA on lag-panels -- Coulombe et al. (2021 IJF) Eq. (7).",
    (
        "Implements the paper-exact Moving Average Factor (MAF) construction "
        "from Coulombe, Leroux, Stevanovic & Surprenant (2021 IJF) §2.2 "
        "Eq. (7). For each variable ``k = 1..K`` in the input panel:\n\n"
        "1. Build the ``T × (n_lags + 1)`` lag-panel "
        "``[X_{t,k}, L X_{t,k}, ..., L^{n_lags} X_{t,k}]``.\n"
        "2. Run PCA retaining ``n_components_per_var`` components "
        "(paper default: 2).\n"
        "3. Append the resulting factor columns to the output.\n\n"
        "Output shape: ``(T, K · n_components_per_var)``. With defaults "
        "``n_lags=12``, ``n_components_per_var=2`` the output is ``(T, 2K)`` "
        "-- paper footnote 11: 'We keep two MAFs for each series and they are "
        "obtained by PCA.'\n\n"
        "**Distinction from existing ``ma_increasing_order → pca(4)`` path**: "
        "the existing stacked-PCA MAF cell runs a single PCA over all MA "
        "columns at once (stacked, 4 global components). This op runs "
        "separate PCA per variable, yielding ``2K`` locally-structured "
        "factors rather than 4 global ones. Use this op when paper-Eq.7-exact "
        "replication is required.\n\n"
        "First ``n_lags`` rows per variable are NaN (lag-panel boundary). "
        "``temporal_rule`` is required; ``full_sample_once`` is rejected to "
        "enforce walk-forward boundaries.\n\n"
        "Operational from v0.9.0 (phase-f16)."
    ),
    "Paper-exact replication of Coulombe et al. (2021 IJF) MAF construction; when per-variable PCA factors are preferred over global stacked-PCA (4-component) path.",
    when_not_to_use="When the 16-cell horse-race stacked-PCA MAF cell is sufficient (use ``ma_increasing_order`` → ``pca`` instead); or when K is large and 2K output columns would exceed T.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Coulombe, Leroux, Stevanovic & Surprenant (2021) 'Macroeconomic Data Transformations Matter', International Journal of Forecasting 37(4): 1338-1354.",
            url="https://doi.org/10.1016/j.ijforecast.2021.05.005",
        ),
    ),
    related_options=("maf", "ma_increasing_order", "ma_window"),
)


_OP_SLICED_INVERSE_REGRESSION = _o(
    "sliced_inverse_regression",
    "sSUFF / Sliced inverse regression (scaled) -- supervised dimension reduction (Huang-Jiang-Li-Tong-Zhou 2022).",
    (
        "Supervised dimension reduction extending ``scaled_pca`` to "
        "non-linear y → X dependence. Pipeline: (1) standardise X; "
        "(2) optional column-wise predictive scaling (``scaling_method`` "
        "= ``scaled_pca`` reuses the Huang-Zhou OLS-slope; "
        "``marginal_R2`` uses sign(β_j)·√R²_j; ``none`` skips); "
        "(3) sort rows by y and partition into ``n_slices`` H "
        "contiguous slices; (4) compute weighted between-slice "
        "covariance ``Σ_S = Σ_h (n_h/n) · m̄_h · m̄_h^⊤``; (5) take the "
        "top-``n_components`` eigenvectors as factor loadings; "
        "(6) project the full panel onto these directions. The sSUFF "
        "augmentation (Huang-Zhou-Tong 2022) recovers latent factors "
        "with higher correlation than plain SIR in the macro-panel "
        "regime where signals are sparse over predictors.\n\n"
        "Defaults: ``n_components = 2``, ``n_slices = 10``, "
        "``scaling_method = 'scaled_pca'``. Requires a "
        "``target_signal`` input port; ``temporal_rule`` is required "
        "and rejects ``full_sample_once``."
    ),
    "Supervised factor extraction from macro panels with non-linear y → X structure; alternative to ``scaled_pca`` when the predictive direction is non-monotone.",
    when_not_to_use="Very small T (need ≥ 5·n_slices observations after dropping NaN); strictly linear y → X relationship (``scaled_pca`` is sufficient).",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Huang, Jiang, Li, Tong & Zhou (2022) 'Scaled PCA: A New Approach to Dimension Reduction', Management Science 68(3): 1678-1695."
        ),
        Reference(
            citation="Fan, Xue & Yao (2017) 'Sufficient forecasting using factor models', Journal of Econometrics 201(2): 292-306."
        ),
        Reference(
            citation="Li (1991) 'Sliced Inverse Regression for Dimension Reduction', JASA 86(414): 316-327."
        ),
    ),
    related_options=("scaled_pca", "supervised_pca", "partial_least_squares"),
)


register(
    _OP_LEVEL,
    _OP_DIFF,
    _OP_LOG,
    _OP_LOG_DIFF,
    _OP_PCT_CHANGE,
    _OP_SEASONAL_LAG,
    _OP_MA_WINDOW,
    _OP_MA_INCREASING,
    _OP_CUMSUM,
    _OP_SCALE,
    _OP_PCA,
    _OP_SPARSE_PCA,
    _OP_SPARSE_PCA_CHEN_ROHE,
    _OP_SUPERVISED_PCA,
    _OP_SCALED_PCA,
    _OP_DFM,
    _OP_VARIMAX,
    _OP_VARIMAX_ROTATION,
    _OP_PARTIAL_LEAST_SQUARES,
    _OP_RANDOM_PROJECTION,
    _OP_FOURIER,
    _OP_WAVELET,
    _OP_HP_FILTER,
    _OP_HAMILTON_FILTER,
    _OP_POLYNOMIAL,
    _OP_POLYNOMIAL_EXPANSION,
    _OP_INTERACTION,
    _OP_KERNEL,
    _OP_KERNEL_FEATURES,
    _OP_NYSTROEM,
    _OP_NYSTROEM_FEATURES,
    _OP_REGIME_INDICATOR,
    _OP_SEASON_DUMMY,
    _OP_TIME_TREND,
    _OP_HOLIDAY,
    _OP_TARGET_CONSTRUCTION,
    _OP_FEATURE_SELECTION,
    _OP_L3_FEATURE_BUNDLE,
    _OP_L3_METADATA_BUILD,
    # v0.9 Phase 2 paper-coverage atomic primitives
    _OP_SAVITZKY_GOLAY_FILTER,
    _OP_ADAPTIVE_MA_RF,
    _OP_ASYMMETRIC_TRIM,
    # v0.9 Phase C top-6 net-new methods
    _OP_U_MIDAS,
    _OP_MIDAS,
    _OP_SLICED_INVERSE_REGRESSION,
    # v0.9.0 phase-f16: per-variable PCA MAF (Coulombe et al. 2021 IJF Eq. 7)
    _OP_MAF_PER_VARIABLE_PCA,
)
