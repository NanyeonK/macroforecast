"""L3 feature-engineering DAG -- per-option documentation.

L3 is the cascading feature-engineering layer: 37 operational ops
across stationary / lag / aggregation / scale / reduction / spectral /
detrend / expansion / auxiliary / target / selection / combine
families. Each op is a node in the DAG; ops chain via ``inputs`` and
the cascade-depth gate (``cascade_max_depth``) bounds recursion.

Cycle 30: 10 basic transform ops updated with op_page=True, op_func_name,
data_args, and return_type to support per-op encyclopedia pages.

This module ships Tier-1 docs for every operational L3 ``op`` choice.
The only L3 axis exposed via :data:`introspect.operational_options` is
``L3.A.op``; the ``inputs`` / ``params`` / ``temporal_rule`` keys are
operator-specific configuration that lives on the node body, not as
schema axes.
"""

from __future__ import annotations

from . import register
from .types import OptionDoc, ParameterDoc, Reference, REQUIRED

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


# Shared data-argument doc for all L3 panel-transform standalone callables (Cycle 30).
_L3_PANEL_DATA_ARG: tuple[ParameterDoc, ...] = (
    ParameterDoc(
        name="panel",
        type="pd.DataFrame",
        default=REQUIRED,
        description=(
            "Input panel. Each column is a variable; rows are time periods. "
            "Series is promoted to a single-column DataFrame internally."
        ),
    ),
)

# Cycle 32: shared data-argument docs for supervised L3 ops (panel + required target).
_panel_arg = ParameterDoc(
    name="panel",
    type="pd.DataFrame",
    default=REQUIRED,
    description=(
        "Input panel. Each column is a variable; rows are time periods. "
        "Series is promoted to a single-column DataFrame internally."
    ),
)
_target_arg = ParameterDoc(
    name="target",
    type="pd.Series",
    default=REQUIRED,
    description=(
        "Supervisory signal aligned to the panel index. "
        "Must share at least one index value with panel; raises ValueError if the "
        "intersection is empty."
    ),
)
_optional_target_arg = ParameterDoc(
    name="target",
    type="pd.Series | None",
    default=None,
    description=(
        "Optional supervisory signal. Required when method is 'correlation' or 'lasso'; "
        "ignored for method='variance'."
    ),
)

# Panel + required target (4 supervised ops).
_L3_SUPERVISED_DATA_ARGS: tuple[ParameterDoc, ...] = (_panel_arg, _target_arg)

# Panel + optional target (feature_selection_transform).
_L3_OPTIONAL_TARGET_DATA_ARGS: tuple[ParameterDoc, ...] = (
    _panel_arg,
    _optional_target_arg,
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
    op_page: bool = False,
    op_func_name: str = "",
    data_args: tuple[ParameterDoc, ...] = (),
    return_type: str = "",
    returns_attrs: tuple[tuple[str, str, str], ...] = (),
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
        op_page=op_page,
        op_func_name=op_func_name,
        data_args=data_args,
        return_type=return_type,
        returns_attrs=returns_attrs,
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
    op_page=True,
    op_func_name="diff_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="periods",
            type="int",
            default=1,
            constraint=">= 1",
            description="Number of lag periods to difference.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="log_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="log_diff_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="periods",
            type="int",
            default=1,
            constraint=">= 1",
            description="Number of lag periods to difference after taking logs.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="pct_change_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="periods",
            type="int",
            default=1,
            constraint=">= 1",
            description="Number of lag periods for the percentage change.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)


# ---------------------------------------------------------------------------
# Lag / seasonal-lag family (2 ops)
# ---------------------------------------------------------------------------

_OP_LAG = _o(
    "lag",
    "Lagged target/predictor block.",
    (
        "Constructs a lagged matrix from inputs. ``params.n_lag`` sets "
        "the lag depth. Standard predictor for autoregressive baselines."
    ),
    "Always when building AR features or lagged-X feature blocks.",
    when_not_to_use="When the target itself is already differenced/lagged in L2 -- avoid double-lagging.",
    related_options=("seasonal_lag", "target_construction"),
    op_page=True,
    op_func_name="lag_matrix",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_lag",
            type="int",
            default=4,
            constraint=">= 1",
            description="Number of lags. Output has K * n_lag columns.",
        ),
        ParameterDoc(
            name="include_contemporaneous",
            type="bool",
            default=False,
            description="If True, also include lag 0 (the contemporaneous column).",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)


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
    op_page=True,
    op_func_name="seasonal_lag_matrix",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="seasonal_period",
            type="int",
            default=12,
            constraint=">= 2",
            description="Seasonal cycle length (12 for monthly, 4 for quarterly).",
        ),
        ParameterDoc(
            name="n_seasonal_lags",
            type="int",
            default=1,
            constraint=">= 1",
            description="Number of seasonal lags. Shifts by seasonal_period * i.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="ma_window_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="window",
            type="int",
            default=3,
            constraint=">= 1",
            description="Rolling window size in periods. First window-1 rows are NaN.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="ma_increasing_order_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="max_order",
            type="int",
            default=12,
            constraint=">= 2",
            description="Maximum window order. Generates windows 2, 3, ..., max_order.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="cumsum_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="scale_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="method",
            type='str enum {"zscore", "standard", "standardize", "robust", "minmax"}',
            default="zscore",
            description='Scaling method. "zscore"/"standard"/"standardize" for zero-mean/unit-std; "robust" for median/IQR; "minmax" for [0, 1] range.',
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="pca_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_components",
            type="int | str",
            default=3,
            constraint=">= 1 or 'all'",
            description="Number of principal components to extract. Clamped to min(T, K) - 1 internally. Sentinel 'all' extracts full effective rank min(T, K).",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="sparse_pca_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=8,
            constraint=">= 1",
            description=(
                "Number of sparse principal components to extract. "
                "Clamped internally to min(T_clean, K) - 1."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="sparse_pca_chen_rohe_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=4,
            constraint=">= 1",
            description=(
                "Number of sparse components (= J in the SCA objective). "
                "Clamped internally to min(T_clean, K)."
            ),
        ),
        ParameterDoc(
            name="zeta",
            type="float",
            default=0.0,
            constraint=">= 0",
            description=(
                "L1 budget for loadings Theta. 0.0 routes to "
                "zeta = n_components (paper CV-optimal boundary)."
            ),
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=200,
            constraint=">= 1",
            description="Maximum alternating-maximisation iterations.",
        ),
        ParameterDoc(
            name="var_innovations",
            type="bool",
            default="False",
            description=(
                "If True, fit VAR(1) on SCA scores and return residuals "
                "as sparse macro-finance factors (Rapach-Zhou 2025 step 2)."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            description="Seed for NumPy RNG used in Z/Theta initialisation.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="supervised_pca_transform",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=3,
            constraint=">= 1",
            description=(
                "Number of supervised principal components (P). Clamped internally "
                "to the number of columns kept after correlation screening."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="scaled_pca_transform",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=3,
            constraint=">= 1",
            description=(
                "Number of principal components to extract. Clamped internally "
                "to min(T_clean, K) - 1."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="dfm_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_factors",
            type="int",
            default=3,
            constraint=">= 1",
            description=(
                "Number of latent dynamic factors to extract. Clamped internally "
                "to min(T_clean, K) - 1."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="varimax_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_VARIMAX_ROTATION = _o(
    "varimax_rotation",
    "Alias for ``varimax`` -- rotation step in a multi-stage factor pipeline.",
    (
        "**Alias** -- no dedicated function page. See canonical "
        "``varimax`` (`op/varimax.md`) for full documentation + standalone usage.\n\n"
        "Identical operation to ``varimax`` but registered separately "
        "so a cascading L3 pipeline can declare ``pca → varimax_rotation`` "
        "as two visible nodes in its lineage."
    ),
    "Multi-stage pipelines that explicitly separate factor extraction from rotation.",
    related_options=("varimax", "pca"),
    op_page=False,
    op_func_name="varimax_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="partial_least_squares_transform",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=3,
            constraint=">= 1",
            description=(
                "Number of PLS latent components. Clamped internally to "
                "min(T_clean - 1, K_clean)."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="random_projection_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=8,
            constraint=">= 1",
            description=(
                "Number of random projection output dimensions. "
                "Clamped internally to min(n_components, K)."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="fourier_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_terms",
            type="int",
            default=4,
            constraint=">= 1",
            description="Number of harmonic pairs (sin + cos) to generate. Total output columns: 2 * n_terms.",
        ),
        ParameterDoc(
            name="period",
            type="int",
            default=12,
            constraint=">= 1",
            description="Fundamental period of the seasonal pattern (e.g., 12 for monthly annual cycle, 4 for quarterly).",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="wavelet_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="wavelet",
            type="str",
            default='"db4"',
            description="Wavelet family name (e.g., \"db4\", \"haar\"). Accepted for API consistency; runtime uses a rolling-mean low-pass approximation.",
        ),
        ParameterDoc(
            name="n_levels",
            type="int",
            default=3,
            constraint=">= 1",
            description="Number of decomposition levels. Each level produces an approximation (_wA{level}) and detail (_wD{level}) pair.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="hp_filter_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="lambda_",
            type="float",
            default=1600,
            constraint="> 0",
            description="HP smoothing parameter. Convention: 1600 for quarterly, 129600 for monthly (Ravn-Uhlig 2002).",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="hamilton_filter_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="h",
            type="int",
            default=8,
            constraint=">= 1",
            description="Forecast horizon (periods ahead). Hamilton (2018) recommends h=8 for quarterly, h=24 for monthly.",
        ),
        ParameterDoc(
            name="p",
            type="int",
            default=4,
            constraint=">= 1",
            description="Number of lags used in the regression.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)


# ---------------------------------------------------------------------------
# Expansion family (4 ops)
# ---------------------------------------------------------------------------

_OP_POLYNOMIAL = _o(
    "polynomial",
    "Polynomial basis expansion -- degree-d powers of input.",
    (
        "**Alias** -- no dedicated function page. See canonical "
        "``polynomial_expansion`` (`op/polynomial_expansion.md`) for full documentation + standalone usage.\n\n"
        "sklearn ``PolynomialFeatures`` of degree ``params.degree``. "
        "Includes interaction terms by default; set "
        "``params.interaction_only=True`` for products without pure "
        "powers."
    ),
    "Capturing low-order non-linearity for linear / kernel models.",
    when_not_to_use="High dimension (degree > 3 with many predictors) -- explodes the design matrix; use kernel methods instead.",
    related_options=("interaction", "kernel_features", "polynomial_expansion"),
    op_page=False,
    op_func_name="polynomial_expansion_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="polynomial_expansion_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="degree",
            type="int",
            default=2,
            constraint=">= 1",
            description="Maximum polynomial degree. Degree 1 returns the panel unchanged; degree 2 appends _pow2 columns; etc.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="interaction_terms_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_KERNEL = _o(
    "kernel",
    "Kernel-feature pre-step (Random Fourier / Nyström handle).",
    (
        "**Alias** -- no dedicated function page. See canonical "
        "``kernel_features`` (`op/kernel_features.md`) for full documentation + standalone usage.\n\n"
        "Generic handle for an explicit kernel-feature embedding; "
        "concrete dispatch is determined by ``params.kernel`` "
        "(``rbf`` / ``poly`` / ``laplacian``). For named variants use "
        "``kernel_features`` (RBF Random Fourier) or ``nystroem``."
    ),
    "Kernel-augmented linear / SVM pipelines.",
    related_options=("kernel_features", "nystroem"),
    op_page=False,
    op_func_name="kernel_features_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="kernel_features_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="kind",
            type='str enum {"rbf", "polynomial"}',
            default='"rbf"',
            description=(
                "Kernel type. 'rbf' for Gaussian kernel; 'polynomial' "
                "for degree-2 polynomial kernel."
            ),
        ),
        ParameterDoc(
            name="gamma",
            type="float",
            default=1.0,
            constraint="> 0",
            description=(
                "Kernel bandwidth. For rbf: exp(-gamma * ||x-z||^2). "
                "For polynomial: (gamma * <x,z> + 1)^2."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="nystroem_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=32,
            constraint=">= 1",
            description=(
                "Number of landmark points for Nystroem approximation. "
                "Clamped internally to min(n_components, T_clean)."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_NYSTROEM_FEATURES = _o(
    "nystroem_features",
    "Alias for ``nystroem`` -- explicit feature-stage name.",
    (
        "**Alias** -- no dedicated function page. See canonical "
        "``nystroem`` (`op/nystroem.md`) for full documentation + standalone usage.\n\n"
        "Identical to ``nystroem``; preferred when a multi-stage "
        "pipeline names its kernel approximation explicitly in the "
        "lineage graph."
    ),
    "Multi-stage pipelines that separate kernel approximation from downstream linear fits.",
    related_options=("nystroem",),
    op_page=False,
    op_func_name="nystroem_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="season_dummy_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="season",
            type='str enum {"quarter", "month"}',
            default='"quarter"',
            description='Seasonal granularity hint. Accepted values: "quarter" and "month". Currently validated but has no effect on output (deprecated -- kept for API compatibility). Non-DatetimeIndex inputs produce season_* columns; DatetimeIndex inputs produce month_* columns.',
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_TIME_TREND = _o(
    "time_trend",
    "Deterministic linear time trend (``t = 1, 2, ...``).",
    (
        "Adds a column ``time_trend`` to the panel; with ``params.degree > 1`` "
        "appends polynomial trends. Deterministic complement to "
        "stochastic detrending (HP / Hamilton)."
    ),
    "Trend-stationary linear models where a deterministic trend is part of the DGP.",
    when_not_to_use="Series with structural breaks -- use ``regime_indicator`` or stochastic detrending instead.",
    related_options=("hp_filter", "hamilton_filter"),
    op_page=True,
    op_func_name="time_trend_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="holiday_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
# Selection family (6 ops)
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
    op_page=True,
    op_func_name="feature_selection_transform",
    data_args=_L3_OPTIONAL_TARGET_DATA_ARGS + (
        ParameterDoc(
            name="n_features",
            type="int | float",
            default=0.5,
            constraint="int >= 1 or float in (0, 1]",
            description=(
                "Number of features to keep. If a float in (0, 1], treated as a "
                "fraction of total columns. If an integer, used as a direct count "
                "clamped to [1, K]."
            ),
        ),
        ParameterDoc(
            name="method",
            type="str",
            default='"variance"',
            constraint='"variance" | "correlation" | "lasso"',
            description=(
                "Selection criterion. 'variance' keeps highest-variance columns "
                "(no target needed). 'correlation' keeps columns most correlated "
                "with target. 'lasso' fits LassoCV and keeps largest-coefficient "
                "columns. 'correlation' and 'lasso' require target."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)


_OP_BORUTA_SELECTION = _o(
    "boruta_selection",
    "All-relevant feature selection via shadow-feature random forest (Kursa-Rudnicki 2010).",
    (
        "Boruta identifies all features that carry statistically relevant "
        "predictive information by comparing each feature's importance to the "
        "maximum importance achieved by random (shadow) copies. Algorithm:\n\n"
        "1. Append shuffled shadow copies of every column to the panel.\n"
        "2. Fit a random forest (``n_estimators_rf`` trees) and record mean "
        "impurity-reduction importance for each real and shadow feature.\n"
        "3. Use a two-sided binomial test (threshold ``alpha``) to classify each "
        "real feature as confirmed, rejected, or tentative.\n"
        "4. Remove rejected features and their shadows; repeat up to "
        "``max_iter`` rounds or until no tentative features remain.\n"
        "5. Return the sub-panel of confirmed (and optionally tentative, "
        "``include_tentative=True``) features.\n\n"
        "Because the null is the importance of random noise, Boruta is an "
        "all-relevant selector: it keeps every feature that beats chance, "
        "not just the minimal predictive set. ``temporal_rule`` controls "
        "whether the forest is fit once per origin (``expanding_window_per_origin``). "
        "``full_sample_once`` is rejected by a hard rule."
    ),
    "Macro panels where many predictors may matter but standard Lasso-type selectors over-shrink; best before tree or neural forecasters where high-dim panels are acceptable.",
    when_not_to_use="Very wide panels (K >> 500) -- shadow copies double memory; prefer lasso_path_selection or stability_selection for computational cost.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Kursa, M.B. & Rudnicki, W.R. (2010) 'Feature Selection with the Boruta Package', Journal of Statistical Software 36(11): 1-13.",
            url="https://doi.org/10.18637/jss.v036.i11",
        ),
    ),
    related_options=("feature_selection", "stability_selection", "recursive_feature_elimination"),
    op_page=True,
    op_func_name="boruta_selection",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_estimators_rf",
            type="int",
            default=100,
            constraint=">= 1",
            description=(
                "Number of trees in each random forest fit. Larger values stabilise "
                "importance rankings at the cost of compute."
            ),
        ),
        ParameterDoc(
            name="max_iter",
            type="int",
            default=100,
            constraint=">= 1",
            description=(
                "Maximum number of Boruta iterations. Each iteration removes at "
                "least one rejected feature; convergence typically occurs well "
                "before the limit."
            ),
        ),
        ParameterDoc(
            name="alpha",
            type="float",
            default=0.05,
            constraint="in (0, 1)",
            description=(
                "Two-sided binomial test significance level for classifying features "
                "as confirmed or rejected. Lower values are more conservative."
            ),
        ),
        ParameterDoc(
            name="include_tentative",
            type="bool",
            default=False,
            constraint="",
            description=(
                "If True, tentative features (not yet confirmed or rejected within "
                "max_iter rounds) are retained in the output panel."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            constraint="",
            description="Random seed for the random forest and shadow-feature shuffles.",
        ),
        ParameterDoc(
            name="temporal_rule",
            type="str",
            default='"expanding_window_per_origin"',
            constraint='"expanding_window_per_origin" | "rolling_window_per_origin"',
            description=(
                "Controls when the random forest is refitted relative to each "
                "forecast origin. ``full_sample_once`` is hard-rejected."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_RECURSIVE_FEATURE_ELIMINATION = _o(
    "recursive_feature_elimination",
    "Backward stepwise feature pruning via estimator importance (Guyon et al. 2002).",
    (
        "Recursively eliminates the weakest features according to coefficients "
        "or feature importances of a base estimator until the target count is "
        "reached. Algorithm:\n\n"
        "1. Fit the base estimator (``ridge``, ``lasso``, or ``svr_linear``) on "
        "the full feature set aligned to the target.\n"
        "2. Rank features by absolute coefficient magnitude (linear models) or "
        "impurity reduction (trees).\n"
        "3. Remove the bottom ``step`` features (an int) or fraction (a float).\n"
        "4. Repeat until exactly ``n_features_to_select`` remain.\n"
        "5. If ``use_cv=True``, wrap in cross-validated RFECV with ``cv_folds`` "
        "time-series folds to auto-select the optimal count.\n\n"
        "``temporal_rule`` governs refitting per forecast origin; "
        "``full_sample_once`` is rejected by a hard rule."
    ),
    "Trimming macro panels to a compact predictor set before linear or penalised forecasters; especially useful when coefficient-based ranking aligns with the forecasting objective.",
    when_not_to_use="When the estimator's coefficient ranking is a poor proxy for marginal predictive value (e.g. highly correlated groups); prefer stability_selection or boruta_selection.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Guyon, I., Weston, J., Barnhill, S. & Vapnik, V. (2002) 'Gene Selection for Cancer Classification using Support Vector Machines', Machine Learning 46(1-3): 389-422.",
            url="https://doi.org/10.1023/A:1012487302797",
        ),
    ),
    related_options=("feature_selection", "boruta_selection", "lasso_path_selection"),
    op_page=True,
    op_func_name="recursive_feature_elimination",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_features_to_select",
            type="int | float",
            default=0.5,
            constraint="int >= 1 or float in (0, 1]",
            description=(
                "Number of features to retain. A float in (0, 1] is treated as a "
                "fraction of total columns; an integer is used directly."
            ),
        ),
        ParameterDoc(
            name="step",
            type="int | float",
            default=1,
            constraint="int >= 1 or float in (0, 1)",
            description=(
                "Features removed per iteration. An integer removes that many; "
                "a float removes that fraction of the remaining features."
            ),
        ),
        ParameterDoc(
            name="estimator",
            type="str",
            default='"ridge"',
            constraint='"ridge" | "lasso" | "svr_linear"',
            description=(
                "Base estimator whose coefficient magnitudes rank feature importance. "
                "``svr_linear`` uses the SVM weight vector."
            ),
        ),
        ParameterDoc(
            name="use_cv",
            type="bool",
            default=False,
            constraint="",
            description=(
                "If True, wrap RFE in cross-validated RFECV; ``n_features_to_select`` "
                "is ignored and the optimal count is determined by CV."
            ),
        ),
        ParameterDoc(
            name="cv_folds",
            type="int",
            default=5,
            constraint=">= 2",
            description="Number of time-series cross-validation folds when use_cv=True.",
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            constraint="",
            description="Random seed for estimators that require it (e.g. SVR with RBF).",
        ),
        ParameterDoc(
            name="temporal_rule",
            type="str",
            default='"expanding_window_per_origin"',
            constraint='"expanding_window_per_origin" | "rolling_window_per_origin"',
            description=(
                "Controls when the base estimator is refitted per forecast origin. "
                "``full_sample_once`` is hard-rejected."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_LASSO_PATH_SELECTION = _o(
    "lasso_path_selection",
    "Feature selection along the Lasso regularisation path (Efron et al. 2004).",
    (
        "Traces the full Lasso regularisation path from lambda_max down to a "
        "lambda that retains approximately ``n_features_to_select`` columns, "
        "then returns the sub-panel of surviving predictors. Algorithm:\n\n"
        "1. Optionally standardise each column to unit variance "
        "(``normalize_features=True``, default).\n"
        "2. Compute the full Lasso path via LARS or coordinate descent.\n"
        "3. Identify the regularisation value where the number of non-zero "
        "coefficients first reaches ``n_features_to_select``.\n"
        "4. Return the columns with non-zero coefficients at that lambda.\n\n"
        "Unlike ``feature_selection`` with ``method='lasso'``, this op "
        "traverses the entire path so the selection threshold adapts to the "
        "data geometry rather than a fixed penalty. ``temporal_rule`` controls "
        "refitting per forecast origin; ``full_sample_once`` is rejected."
    ),
    "Compact, theory-grounded feature selection for linear macro forecasting models where a path-based threshold is preferable to a manually tuned penalty.",
    when_not_to_use="When the number of desired features is unknown and cross-validation is required -- use recursive_feature_elimination with use_cv=True instead.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Efron, B., Hastie, T., Johnstone, I. & Tibshirani, R. (2004) 'Least Angle Regression', Annals of Statistics 32(2): 407-499.",
            url="https://doi.org/10.1214/009053604000000067",
        ),
    ),
    related_options=("feature_selection", "recursive_feature_elimination", "stability_selection"),
    op_page=True,
    op_func_name="lasso_path_selection",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_features_to_select",
            type="int | float",
            default=0.5,
            constraint="int >= 1 or float in (0, 1]",
            description=(
                "Target number of features. A float in (0, 1] is treated as a "
                "fraction of total columns; an integer is used directly. The path "
                "is traced until this count is first reached."
            ),
        ),
        ParameterDoc(
            name="normalize_features",
            type="bool",
            default=True,
            constraint="",
            description=(
                "If True, standardise each column to zero mean and unit variance "
                "before computing the Lasso path so coefficients are comparable."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            constraint="",
            description="Random seed for any stochastic components of the path solver.",
        ),
        ParameterDoc(
            name="temporal_rule",
            type="str",
            default='"expanding_window_per_origin"',
            constraint='"expanding_window_per_origin" | "rolling_window_per_origin"',
            description=(
                "Controls when the Lasso path is recomputed per forecast origin. "
                "``full_sample_once`` is hard-rejected."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_STABILITY_SELECTION = _o(
    "stability_selection",
    "Feature selection by subsampling stability -- selection probability threshold (Meinshausen-Bühlmann 2010).",
    (
        "Estimates the probability that each feature would be selected by a "
        "sparse estimator on a random subsample, then retains features whose "
        "selection probability exceeds ``pi_thr``. Algorithm:\n\n"
        "1. Draw ``n_subsamples`` subsamples of size ``subsample_fraction * T`` "
        "from the aligned (panel, target) observations.\n"
        "2. On each subsample, fit the base estimator (``lasso`` or "
        "``elastic_net``) with regularisation ``alpha`` and record which "
        "features receive non-zero coefficients.\n"
        "3. Compute the empirical selection frequency for each feature across "
        "all subsamples.\n"
        "4. Return the sub-panel of features with frequency >= ``pi_thr``.\n\n"
        "Stability selection provides FWER control under mild assumptions on the "
        "regularisation path (Meinshausen-Bühlmann Theorem 1). "
        "``temporal_rule`` governs refitting per origin; "
        "``full_sample_once`` is rejected."
    ),
    "Macro panels where robustness of selection across data perturbations is more important than computational speed; pairs well with L4 ridge or elastic_net forecasters.",
    when_not_to_use="Short time series (T < 100) where subsampling leaves too few observations per draw; prefer lasso_path_selection for small T.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Meinshausen, N. & Bühlmann, P. (2010) 'Stability selection', Journal of the Royal Statistical Society Series B 72(4): 417-473.",
            url="https://doi.org/10.1111/j.1467-9868.2010.00740.x",
        ),
    ),
    related_options=("feature_selection", "boruta_selection", "lasso_path_selection"),
    op_page=True,
    op_func_name="stability_selection",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_subsamples",
            type="int",
            default=100,
            constraint=">= 10",
            description=(
                "Number of subsampling rounds. More rounds give stable frequency "
                "estimates at higher compute cost."
            ),
        ),
        ParameterDoc(
            name="subsample_fraction",
            type="float",
            default=0.5,
            constraint="in (0, 1)",
            description=(
                "Fraction of observations in each subsample. Meinshausen-Bühlmann "
                "recommend 0.5 for the theoretical FWER bound to apply."
            ),
        ),
        ParameterDoc(
            name="pi_thr",
            type="float",
            default=0.6,
            constraint="in (0.5, 1]",
            description=(
                "Selection-probability threshold. Features with empirical frequency "
                "above this value are retained. Values near 0.9 are very conservative."
            ),
        ),
        ParameterDoc(
            name="base_estimator",
            type="str",
            default='"lasso"',
            constraint='"lasso" | "elastic_net"',
            description=(
                "Sparse base estimator applied to each subsample. ``elastic_net`` "
                "can improve stability when predictors are highly correlated."
            ),
        ),
        ParameterDoc(
            name="alpha",
            type="float",
            default=0.01,
            constraint="> 0",
            description=(
                "Regularisation strength passed to the base estimator on each "
                "subsample. Controls sparsity per subsample draw."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            constraint="",
            description="Random seed for reproducible subsampling.",
        ),
        ParameterDoc(
            name="temporal_rule",
            type="str",
            default='"expanding_window_per_origin"',
            constraint='"expanding_window_per_origin" | "rolling_window_per_origin"',
            description=(
                "Controls when subsampling is rerun per forecast origin. "
                "``full_sample_once`` is hard-rejected."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)

_OP_GENETIC_ALGORITHM_SELECTION = _o(
    "genetic_algorithm_selection",
    "Evolutionary feature subset search via genetic algorithm (Goldberg 1989).",
    (
        "Evolves a population of binary feature-inclusion masks to maximise "
        "cross-validated forecast accuracy. Algorithm:\n\n"
        "1. Initialise ``population_size`` random binary masks (each bit "
        "indicates inclusion of one feature).\n"
        "2. Evaluate each mask by fitting ``fitness_estimator`` on the "
        "sub-panel and computing ``cv_folds``-fold time-series CV MSE.\n"
        "3. Select parents by tournament selection proportional to CV fitness.\n"
        "4. Apply uniform crossover (rate ``crossover_prob``) and bit-flip "
        "mutation to produce the next generation.\n"
        "5. Repeat for ``n_generations`` generations; return the feature "
        "subset of the fittest mask in the final population.\n\n"
        "Genetic search is useful when the inclusion/exclusion objective is "
        "non-convex and greedy backward/forward procedures get stuck. "
        "``temporal_rule`` governs when the GA is re-run per origin; "
        "``full_sample_once`` is rejected."
    ),
    "Feature selection when the predictive objective is highly non-linear or when combinations of individually weak predictors form strong subsets inaccessible to greedy methods.",
    when_not_to_use="Large panels (K > 200) or long time series where CV evaluation of many masks is computationally prohibitive -- prefer lasso_path_selection or boruta_selection.",
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation="Goldberg, D.E. (1989) Genetic Algorithms in Search, Optimization and Machine Learning. Addison-Wesley, Reading, MA.",
        ),
    ),
    related_options=("feature_selection", "boruta_selection", "recursive_feature_elimination"),
    op_page=True,
    op_func_name="genetic_algorithm_selection",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="population_size",
            type="int",
            default=30,
            constraint=">= 4",
            description=(
                "Number of candidate feature masks in each generation. Larger "
                "populations explore more of the subset space per generation."
            ),
        ),
        ParameterDoc(
            name="n_generations",
            type="int",
            default=50,
            constraint=">= 1",
            description=(
                "Number of evolutionary generations. More generations allow finer "
                "convergence at higher compute cost."
            ),
        ),
        ParameterDoc(
            name="crossover_prob",
            type="float",
            default=0.8,
            constraint="in (0, 1]",
            description=(
                "Probability of applying uniform crossover to a parent pair. "
                "The complement probability reproduces a parent unchanged."
            ),
        ),
        ParameterDoc(
            name="fitness_estimator",
            type="str",
            default='"ridge"',
            constraint='"ridge" | "lasso" | "ols"',
            description=(
                "Estimator used to evaluate each feature subset's CV accuracy. "
                "``ridge`` is recommended for high-dim panels; ``ols`` for small subsets."
            ),
        ),
        ParameterDoc(
            name="cv_folds",
            type="int",
            default=3,
            constraint=">= 2",
            description=(
                "Number of time-series cross-validation folds per fitness evaluation. "
                "Higher values are more reliable but slow."
            ),
        ),
        ParameterDoc(
            name="random_state",
            type="int",
            default=0,
            constraint="",
            description="Random seed for population initialisation and genetic operators.",
        ),
        ParameterDoc(
            name="temporal_rule",
            type="str",
            default='"expanding_window_per_origin"',
            constraint='"expanding_window_per_origin" | "rolling_window_per_origin"',
            description=(
                "Controls when the genetic search is re-run per forecast origin. "
                "``full_sample_once`` is hard-rejected."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="savitzky_golay_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="window",
            type="int",
            default=7,
            constraint=">= 3",
            description="Length of the smoothing window. If even, rounded up to next odd integer (scipy requirement).",
        ),
        ParameterDoc(
            name="polyorder",
            type="int",
            default=3,
            description="Degree of the polynomial fit within each window. Must be < window.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="asymmetric_trim_transform",
    data_args=_L3_PANEL_DATA_ARG,
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="adaptive_ma_rf_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_estimators",
            type="int",
            default=100,
            constraint=">= 1",
            description="Number of trees in the RF ensemble. Paper recommends 500; 100 is the default for speed.",
        ),
        ParameterDoc(
            name="min_samples_leaf",
            type="int",
            default=40,
            constraint=">= 1",
            description="Minimum samples per leaf; lower-bounds the effective adaptive window length (paper default: 40).",
        ),
        ParameterDoc(
            name="sided",
            type="str",
            default="two",
            constraint="'two' or 'one'",
            description="'two' fits one forest on the full sample (retrospective); 'one' fits an expanding-window forest per time index t (real-time variant, O(T) RF fits).",
        ),
        ParameterDoc(
            name="random_state",
            type="int | None",
            default=0,
            constraint="int or None",
            description="RNG seed for sklearn RandomForestRegressor. None gives non-reproducible results.",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
        "``col_lagk[t] = frame[col].iloc[t·m − k]``.\n\n"
        "The downstream L4 OLS estimator recovers data-driven lag "
        "coefficients -- the *unrestricted* MIDAS regression "
        "(paper §3.2 eq.(20)):\n\n"
        "``y_{t×k} = μ₀ + μ₁ y_{t×k−k} + ψ₀ x_{t×k−1} + ψ₁ x_{t×k−2} "
        "+ … + ψ_K x_{t×k−K} + ε_{t×k}``\n\n"
        "where μ₀, μ₁, and ψ(L) are estimated by OLS (paper §3.2 p.11). "
        "Ridge regularisation is available as an explicit opt-in via "
        "``regularization='ridge'`` in the ``paper_methods.u_midas(...)`` "
        "recipe; it deviates from the paper's estimator choice and is "
        "not the default.\n\n"
        "**Lag-order selection**: ``n_lags_high='bic'`` (default) runs "
        "BIC over K ∈ {1, …, ceil(1.5 × freq_ratio)}, fitting OLS at "
        "each candidate and selecting K* = argmin BIC (paper §3.2 p.11 + "
        "§3.5). Pass an integer to fix K. ``'aic'`` is also accepted.\n\n"
        "**AR(1) y-lag**: the ``paper_methods.u_midas(...)`` helper sets "
        "``include_y_lag=True`` by default, prepending the lagged target "
        "``y_lag1`` as the leftmost design-matrix column (μ₁ term of "
        "eq.(20)). Set ``include_y_lag=False`` to match the simplified "
        "§2.3 eq.(14) form with no AR component.\n\n"
        "**Defaults**: ``freq_ratio = 3`` (quarterly target / monthly "
        "HF), ``n_lags_high = 'bic'``; ``target_freq = 'low'`` "
        "subsamples the LF anchor dates. ``temporal_rule`` is required "
        "and rejects ``full_sample_once`` so the aggregation respects "
        "walk-forward boundaries.\n\n"
        "Surfaces the Borup-Rapach-Schütte (2023) mixed-frequency "
        "ML-nowcasting workflow as a 1-line recipe via "
        "``paper_methods.u_midas(...)``."
    ),
    "Macro nowcasting with monthly predictors and quarterly targets; mixed-frequency feature engineering when no parametric weight kernel is desired.",
    when_not_to_use=(
        "When ``n_lags_high · n_HF_columns`` is large relative to T even "
        "after BIC selects a small K -- BIC penalises over-parameterisation "
        "but cannot reduce the number of predictor columns. Use "
        "``midas`` parametric weighting instead, or pair with downstream "
        "lasso / ridge (set ``regularization='ridge'``) to handle wide "
        "design matrices."
    ),
    references=(
        _REF_DESIGN_L3,
        Reference(
            citation=(
                "Foroni, Marcellino & Schumacher (2011/2015) "
                "'Unrestricted Mixed Data Sampling (MIDAS): MIDAS Regressions "
                "With Unrestricted Lag Polynomials'. "
                "Bundesbank Discussion Paper Series 1, No. 35/2011; "
                "published as JRSS-A 178(1): 57-82. "
                "DOI 10.1111/rssa.12043."
            )
        ),
        Reference(
            citation=(
                "Borup, Rapach & Schütte (2023) "
                "'Mixed-frequency machine learning: Nowcasting and backcasting "
                "weekly initial claims with daily internet search-volume data', "
                "International Journal of Forecasting 39(3): 1122-1144."
            )
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
    op_page=True,
    op_func_name="maf_per_variable_pca_transform",
    data_args=_L3_PANEL_DATA_ARG + (
        ParameterDoc(
            name="n_lags",
            type="int",
            default=12,
            constraint=">= 1",
            description="Number of lags in the per-variable lag-panel. Paper default: 12 (monthly data).",
        ),
        ParameterDoc(
            name="n_components_per_var",
            type="int",
            default=2,
            constraint=">= 1",
            description="Number of PCA components per variable. Paper default: 2 (footnote 11).",
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
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
    op_page=True,
    op_func_name="sliced_inverse_regression_transform",
    data_args=_L3_SUPERVISED_DATA_ARGS + (
        ParameterDoc(
            name="n_components",
            type="int",
            default=3,
            constraint=">= 1",
            description=(
                "Number of SIR directions (effective rank of the between-slice "
                "covariance matrix)."
            ),
        ),
        ParameterDoc(
            name="n_slices",
            type="int",
            default=10,
            constraint=">= 2",
            description=(
                "Number of contiguous slices of the target distribution. "
                "Clamped internally to the number of aligned clean rows."
            ),
        ),
    ),
    return_type="pd.DataFrame",
    returns_attrs=(),
)


register(
    _OP_LEVEL,
    _OP_DIFF,
    _OP_LOG,
    _OP_LOG_DIFF,
    _OP_PCT_CHANGE,
    _OP_LAG,
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
    # Selection family (6 ops)
    _OP_FEATURE_SELECTION,
    _OP_BORUTA_SELECTION,
    _OP_RECURSIVE_FEATURE_ELIMINATION,
    _OP_LASSO_PATH_SELECTION,
    _OP_STABILITY_SELECTION,
    _OP_GENETIC_ALGORITHM_SELECTION,
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
