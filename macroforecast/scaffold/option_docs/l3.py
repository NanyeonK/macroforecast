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
from .types import CodeExample, OptionDoc, Reference

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


def _o(option: str, summary: str, description: str, when_to_use: str,
       *, when_not_to_use: str = "",
       references: tuple[Reference, ...] = (_REF_DESIGN_L3,),
       related_options: tuple[str, ...] = ()) -> OptionDoc:
    return OptionDoc(
        layer="l3", sublayer="L3_A_step_op", axis="op", option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, references=references,
        related_options=related_options,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
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
    references=(_REF_DESIGN_L3, Reference(citation="Coulombe (2024) 'The Macroeconomic Random Forest', Journal of Applied Econometrics 39(7): 1190-1209.")),
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
    related_options=("sparse_pca", "scaled_pca", "varimax", "dfm", "partial_least_squares"),
)

_OP_SPARSE_PCA = _o(
    "sparse_pca",
    "Sparse PCA -- L1-penalised factor loadings.",
    (
        "Variant of PCA where loadings are pushed toward zero by an "
        "L1 penalty (``params.alpha``). Yields more interpretable "
        "factors at the cost of a small reconstruction loss; uses "
        "sklearn's ``SparsePCA``."
    ),
    "When you want factor loadings to map cleanly onto a small subset of original predictors (interpretability).",
    when_not_to_use="When pure variance maximisation is more important than interpretability -- use plain ``pca``.",
    related_options=("pca", "scaled_pca"),
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
    references=(_REF_DESIGN_L3, Reference(citation="Huang, Jiang, Tu & Zhou (2022) 'Scaled PCA: A New Approach to Dimension Reduction', Management Science 68(3): 1678-1695.")),
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
    references=(_REF_DESIGN_L3, Reference(citation="Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.")),
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
    references=(_REF_DESIGN_L3, Reference(citation="Wold, Sjöström & Eriksson (2001) 'PLS-regression: a basic tool of chemometrics', Chemometrics and Intelligent Laboratory Systems 58(2): 109-130.")),
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
    references=(_REF_DESIGN_L3, Reference(citation="Hodrick & Prescott (1997) 'Postwar U.S. Business Cycles: An Empirical Investigation', JMCB 29(1): 1-16.")),
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
    references=(_REF_DESIGN_L3, Reference(citation="Hamilton (2018) 'Why You Should Never Use the Hodrick-Prescott Filter', RES 100(5): 831-843.")),
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
    references=(_REF_DESIGN_L3, Reference(citation="Rahimi & Recht (2007) 'Random Features for Large-Scale Kernel Machines', NeurIPS.")),
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


register(
    _OP_LEVEL, _OP_DIFF, _OP_LOG, _OP_LOG_DIFF, _OP_PCT_CHANGE,
    _OP_SEASONAL_LAG,
    _OP_MA_WINDOW, _OP_MA_INCREASING, _OP_CUMSUM,
    _OP_SCALE,
    _OP_PCA, _OP_SPARSE_PCA, _OP_SCALED_PCA, _OP_DFM, _OP_VARIMAX,
    _OP_VARIMAX_ROTATION, _OP_PARTIAL_LEAST_SQUARES, _OP_RANDOM_PROJECTION,
    _OP_FOURIER, _OP_WAVELET,
    _OP_HP_FILTER, _OP_HAMILTON_FILTER,
    _OP_POLYNOMIAL, _OP_POLYNOMIAL_EXPANSION, _OP_INTERACTION,
    _OP_KERNEL, _OP_KERNEL_FEATURES, _OP_NYSTROEM, _OP_NYSTROEM_FEATURES,
    _OP_REGIME_INDICATOR, _OP_SEASON_DUMMY, _OP_TIME_TREND, _OP_HOLIDAY,
    _OP_TARGET_CONSTRUCTION,
    _OP_FEATURE_SELECTION,
    _OP_L3_FEATURE_BUNDLE, _OP_L3_METADATA_BUILD,
)
