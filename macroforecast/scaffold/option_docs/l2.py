"""L2 preprocessing -- per-option documentation.

L2 implements the McCracken-Ng cleaning pipeline: A) FRED-SD frequency
alignment → B) transform (t-codes) → C) outlier handling →
D) imputation → E) frame-edge trimming. Each stage has a primary axis
plus a scope axis controlling which series the stage applies to
(target / predictors / both).
"""
from __future__ import annotations

from . import register
from .types import CodeExample, OptionDoc, Reference

_REVIEWED = "2026-05-04"
_REVIEWER = "macroforecast author"

_REF_DESIGN_L2 = Reference(
    citation="macroforecast design Part 2, L2: 'preprocessing is the only layer with a strict A→B→C→D→E execution order; every cell follows the same pipeline.'",
)
_REF_MCCRACKEN_NG_2016 = Reference(
    citation="McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4).",
    doi="10.1080/07350015.2015.1086655",
)
_REF_STOCK_WATSON_2002 = Reference(
    citation="Stock & Watson (2002) 'Macroeconomic Forecasting Using Diffusion Indexes', JBES 20(2).",
)
_REF_TUKEY_1977 = Reference(
    citation="Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.",
)
_REF_CHOW_LIN_1971 = Reference(
    citation="Chow & Lin (1971) 'Best Linear Unbiased Interpolation, Distribution, and Extrapolation of Time Series by Related Series', RES 53(4).",
)


def _e(sublayer: str, axis: str, option: str, summary: str, description: str, when_to_use: str,
       *, when_not_to_use: str = "", references: tuple[Reference, ...] = (_REF_DESIGN_L2,),
       related_options: tuple[str, ...] = (), examples: tuple[CodeExample, ...] = ()) -> OptionDoc:
    return OptionDoc(
        layer="l2", sublayer=sublayer, axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        references=references, when_not_to_use=when_not_to_use,
        related_options=related_options, examples=examples,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


# L2.B transform_policy
_L2B_OFFICIAL = _e(
    "l2_b", "transform_policy", "apply_official_tcode",
    "Apply McCracken-Ng's series-by-series stationarity transforms.",
    (
        "Each FRED-MD/QD series ships with a transformation code (1-7) "
        "mapping to a stationarity transform. ``apply_official_tcode`` runs "
        "the canonical mapping per series:\n\n"
        "* 1 = level\n"
        "* 2 = first difference\n"
        "* 3 = second difference\n"
        "* 4 = log\n"
        "* 5 = first difference of log (≈ growth rate)\n"
        "* 6 = second difference of log\n"
        "* 7 = log diff of (1 + growth rate)\n\n"
        "Applied per-origin within walk-forward to avoid look-ahead."
    ),
    "Default for FRED-based studies. Canonical replication path.",
    references=(_REF_DESIGN_L2, _REF_MCCRACKEN_NG_2016),
    related_options=("no_transform", "custom_tcode", "transform_scope"),
)

_L2B_NO_TRANSFORM = _e(
    "l2_b", "transform_policy", "no_transform",
    "Skip transforms; pass raw levels through.",
    (
        "Useful for tree-based / ranking models that don't need stationarity, "
        "or for studies that apply alternative transforms in L3 (Hodrick-"
        "Prescott filter, Hamilton (2018) detrender, etc.)."
    ),
    "Tree / forest models; alternative-transform studies; custom panels with already-transformed data.",
    references=(_REF_DESIGN_L2,),
    related_options=("apply_official_tcode", "custom_tcode"),
)

_L2B_CUSTOM = _e(
    "l2_b", "transform_policy", "custom_tcode",
    "User-supplied per-series t-code map.",
    (
        "Requires ``leaf_config.custom_tcode_map: {series_name: int}``. "
        "Macrocast applies the same 1-7 mapping as ``apply_official_tcode`` "
        "but reads codes from the user's dict instead of the bundled FRED "
        "metadata. Useful for custom panels where the user wants the "
        "McCracken-Ng transform vocabulary."
    ),
    "Custom panels with user-defined stationarity codes.",
    references=(_REF_DESIGN_L2,),
    related_options=("apply_official_tcode", "no_transform"),
)


# L2.C outlier_policy
_L2C_MCCRACKEN_IQR = _e(
    "l2_c", "outlier_policy", "mccracken_ng_iqr",
    "McCracken-Ng's published IQR-multiple outlier rule.",
    (
        "Flags any observation more than ``leaf_config.outlier_iqr_threshold`` "
        "(default 10) IQRs from the per-series median. The 10×IQR threshold "
        "is the published McCracken-Ng default and matches their replication "
        "scripts.\n\n"
        "Pairs with an L2.C ``outlier_action`` to specify what happens to "
        "flagged observations (replace with NaN by default, then L2.D "
        "imputation fills them)."
    ),
    "Default for FRED-based studies. Canonical replication path.",
    references=(_REF_DESIGN_L2, _REF_MCCRACKEN_NG_2016),
    related_options=("winsorize", "zscore_threshold", "none", "outlier_action"),
)

_L2C_WINSORIZE = _e(
    "l2_c", "outlier_policy", "winsorize",
    "Cap observations at user-supplied quantile thresholds.",
    (
        "Truncates each series at ``leaf_config.winsorize_lower_quantile`` "
        "(default 0.01) and ``leaf_config.winsorize_upper_quantile`` "
        "(default 0.99). Less aggressive than the McCracken-Ng IQR rule "
        "and preserves more of the tail."
    ),
    "Studies that want a bounded but non-NaN outlier handler; alternative-rule comparisons.",
    references=(_REF_DESIGN_L2, _REF_TUKEY_1977),
    related_options=("mccracken_ng_iqr", "zscore_threshold"),
)

_L2C_ZSCORE = _e(
    "l2_c", "outlier_policy", "zscore_threshold",
    "Flag observations beyond a z-score threshold.",
    (
        "Computes the rolling z-score per series and flags ``|z|`` > "
        "``leaf_config.zscore_threshold_value`` (default 3.0). Simpler "
        "than IQR but assumes approximately Gaussian residuals."
    ),
    "Approximately-Gaussian series; quick sanity-check sweeps.",
    references=(_REF_DESIGN_L2,),
    related_options=("mccracken_ng_iqr", "winsorize"),
)

_L2C_NONE = _e(
    "l2_c", "outlier_policy", "none",
    "Skip outlier handling.",
    "Pass series through unchanged. Useful when L1 already cleaned outliers (``raw_outlier_policy``) or when the study wants to compare against a no-cleaning baseline.",
    "Custom panels already cleaned upstream; no-cleaning ablations.",
    references=(_REF_DESIGN_L2,),
    related_options=("mccracken_ng_iqr", "winsorize", "zscore_threshold"),
)


# L2.C outlier_action
_L2C_ACTION_NAN = _e(
    "l2_c", "outlier_action", "flag_as_nan",
    "Replace flagged outliers with NaN; let L2.D imputation fill them.",
    (
        "Default for the McCracken-Ng pipeline. Outliers become missing "
        "values, then EM-factor imputation in L2.D recovers a smoothed "
        "value from the cross-series factor structure."
    ),
    "Default. Pairs with em_factor / em_multivariate imputation.",
    references=(_REF_DESIGN_L2, _REF_MCCRACKEN_NG_2016),
    related_options=("replace_with_median", "replace_with_cap_value", "keep_with_indicator"),
)

_L2C_ACTION_MEDIAN = _e(
    "l2_c", "outlier_action", "replace_with_median",
    "Replace flagged outliers with the per-series median.",
    "Simpler than imputation; useful when L2.D is set to ``none_propagate``.",
    "Studies that want a deterministic, no-imputation outlier handler.",
    references=(_REF_DESIGN_L2,),
    related_options=("flag_as_nan", "replace_with_cap_value"),
)

_L2C_ACTION_CAP = _e(
    "l2_c", "outlier_action", "replace_with_cap_value",
    "Replace outliers with the cap value (winsorize-style cap).",
    "Caps at the threshold rather than the median. Pairs with the winsorize / iqr policies.",
    "Bounded-output studies; portfolios with hard limits on extreme values.",
    references=(_REF_DESIGN_L2,),
    related_options=("flag_as_nan", "replace_with_median"),
)

# keep_with_indicator is status=future in v1.0 -- not registered.


# L2.D imputation_policy
_L2D_EM_FACTOR = _e(
    "l2_d", "imputation_policy", "em_factor",
    "EM-factor imputation (McCracken-Ng default).",
    (
        "Iterative EM algorithm: alternates between (1) fitting a factor "
        "model to the currently-imputed panel and (2) imputing missing "
        "cells from the factor model's prediction. Converges to a "
        "low-rank fill consistent with the cross-series factor structure.\n\n"
        "Used per-origin under ``imputation_temporal_rule = "
        "expanding_window_per_origin`` so the imputation respects the "
        "walk-forward information set."
    ),
    "Default for FRED-MD/QD high-dimensional panels.",
    references=(_REF_DESIGN_L2, _REF_STOCK_WATSON_2002),
    related_options=("em_multivariate", "mean", "forward_fill", "linear_interpolation"),
)

_L2D_EM_MULTI = _e(
    "l2_d", "imputation_policy", "em_multivariate",
    "Multivariate-Gaussian EM imputation.",
    (
        "Models the full panel as multivariate Gaussian and imputes "
        "missing cells via Schur-complement conditioning. More flexible "
        "than ``em_factor`` (no rank cap) but more expensive on large "
        "panels (O(p²) per iteration)."
    ),
    "Smaller panels (≤ 50 series) where the full covariance is tractable.",
    references=(_REF_DESIGN_L2,),
    related_options=("em_factor", "mean"),
)

_L2D_MEAN = _e(
    "l2_d", "imputation_policy", "mean",
    "Replace missing cells with the per-series rolling mean.",
    "Simple, fast, deterministic. No iteration. Useful when the missing pattern is sparse.",
    "Sparse missingness; quick smoke tests.",
    references=(_REF_DESIGN_L2,),
    related_options=("em_factor", "forward_fill"),
)

_L2D_FFILL = _e(
    "l2_d", "imputation_policy", "forward_fill",
    "Carry the last observed value forward.",
    "Standard pandas ffill. Appropriate for series where the most recent observation is the best forecast of the missing value.",
    "Slowly-moving series (interest rates, ratios); release-lag handling.",
    references=(_REF_DESIGN_L2,),
    related_options=("linear_interpolation", "em_factor"),
)

_L2D_LINEAR = _e(
    "l2_d", "imputation_policy", "linear_interpolation",
    "Linear interpolation between adjacent observations.",
    "Smooths over isolated missing observations. Not appropriate for leading / trailing missings.",
    "Interior missing observations in well-behaved series.",
    references=(_REF_DESIGN_L2, _REF_CHOW_LIN_1971),
    related_options=("forward_fill", "em_factor"),
)

_L2D_NONE = _e(
    "l2_d", "imputation_policy", "none_propagate",
    "Pass NaN through; downstream layers handle it.",
    "Useful when the recipe wants L3 / L4 to see the missing pattern (e.g., for missingness-as-feature studies).",
    "Studies that treat missingness as informative; or panels with no missings.",
    references=(_REF_DESIGN_L2,),
    related_options=("em_factor", "mean", "forward_fill"),
)


# L2.E frame_edge_policy
_L2E_TRUNCATE = _e(
    "l2_e", "frame_edge_policy", "truncate_to_balanced",
    "Trim leading / trailing rows until every series is observed.",
    (
        "Makes the panel rectangular by removing rows where any predictor "
        "(or the target, depending on scope) is missing. Standard for "
        "factor-model-style studies that need a balanced panel."
    ),
    "Default for high-dimensional studies; pairs with em_factor imputation for the interior.",
    references=(_REF_DESIGN_L2, _REF_STOCK_WATSON_2002),
    related_options=("drop_unbalanced_series", "keep_unbalanced", "zero_fill_leading"),
)

_L2E_DROP_SERIES = _e(
    "l2_e", "frame_edge_policy", "drop_unbalanced_series",
    "Drop predictor columns that aren't observed across the full sample.",
    "Trades predictor count for sample length. Useful when the recipe wants to keep early observations and is willing to lose late-arrival series.",
    "Long-history studies (1959-) where late-introduction series should be excluded.",
    references=(_REF_DESIGN_L2,),
    related_options=("truncate_to_balanced", "keep_unbalanced"),
)

_L2E_KEEP_UNBAL = _e(
    "l2_e", "frame_edge_policy", "keep_unbalanced",
    "Keep the panel's natural unbalanced shape.",
    "Lets L4 estimators handle missingness directly. Required for some L4 families (LSTM/GRU/transformer) and for partial-data robustness studies.",
    "Custom panels with intentional unbalanced structure; missing-data-robust models.",
    references=(_REF_DESIGN_L2,),
    related_options=("truncate_to_balanced", "drop_unbalanced_series"),
)

_L2E_ZERO_FILL = _e(
    "l2_e", "frame_edge_policy", "zero_fill_leading",
    "Zero-fill leading missing predictor cells; preserve the rest.",
    "Useful when leading NaN values block early-sample fits but interior NaN should remain visible to imputation.",
    "Studies that want the early sample but accept zero-fill on leading edges.",
    references=(_REF_DESIGN_L2,),
    related_options=("truncate_to_balanced", "keep_unbalanced"),
)


# L2.A FRED-SD frequency rules (one example each — most users default)
_L2A_QM_BACKWARD = _e(
    "l2_a", "quarterly_to_monthly_rule", "step_backward",
    "Step-function: each month inherits the most-recent published quarterly value.",
    (
        "When a quarterly series needs to align with a monthly target, "
        "macroforecast holds the quarterly observation constant for all three "
        "months of the quarter (with a 1-quarter publication lag where "
        "appropriate). Conservative: no smoothing, no extrapolation."
    ),
    "Default for FRED-SD mixed-frequency studies.",
    references=(_REF_DESIGN_L2,),
    related_options=("step_forward", "linear_interpolation", "chow_lin"),
)

_L2A_QM_LINEAR = _e(
    "l2_a", "quarterly_to_monthly_rule", "linear_interpolation",
    "Linear interpolation between quarterly observations.",
    "Smoother than step_backward but introduces look-ahead unless used per-origin.",
    "Studies with smooth quarterly series and per-origin alignment.",
    references=(_REF_DESIGN_L2,),
    related_options=("step_backward", "chow_lin"),
)

_L2A_QM_FORWARD = _e(
    "l2_a", "quarterly_to_monthly_rule", "step_forward",
    "Step-function: each month inherits the next-published quarterly value.",
    "Use when later observations are informative for current state (rare in real-time work).",
    "Hindsight-feasible studies (e.g., counterfactual nowcasts).",
    references=(_REF_DESIGN_L2,),
    related_options=("step_backward", "linear_interpolation"),
)

# chow_lin is status=future in v1.0 -- not registered. Tracked at issue
# #255 (real Chow-Lin disaggregation landed in v0.25 at the runtime
# level but the schema option remains future-flagged pending validator
# acceptance work in v1.0+).


_L2A_MQ_AVG = _e(
    "l2_a", "monthly_to_quarterly_rule", "quarterly_average",
    "Aggregate to quarterly via mean of the three monthly observations.",
    "Standard NIPA aggregation for stocks / averages.",
    "Default. Stock variables (interest rates, prices, employment levels).",
    references=(_REF_DESIGN_L2,),
    related_options=("quarterly_endpoint", "quarterly_sum"),
)

_L2A_MQ_END = _e(
    "l2_a", "monthly_to_quarterly_rule", "quarterly_endpoint",
    "Aggregate via the end-of-quarter observation.",
    "Use for series that snap to a quarter-end (e.g., balance-sheet data).",
    "End-of-period stocks (M2 month-end, balance-sheet series).",
    references=(_REF_DESIGN_L2,),
    related_options=("quarterly_average", "quarterly_sum"),
)

_L2A_MQ_SUM = _e(
    "l2_a", "monthly_to_quarterly_rule", "quarterly_sum",
    "Aggregate via the sum of the three monthly observations.",
    "Standard for flow variables (production, sales, payroll growth).",
    "Flow variables; cumulative-quantity series.",
    references=(_REF_DESIGN_L2,),
    related_options=("quarterly_average", "quarterly_endpoint"),
)


# ---------------------------------------------------------------------------
# Auxiliary axes (sd_series_frequency_filter / *_scope / imputation_temporal_rule)
# ---------------------------------------------------------------------------

# L2.A FRED-SD series-frequency filter
register(
    _e("l2_a", "sd_series_frequency_filter", "monthly_only",
       "Drop quarterly series; retain monthly FRED-SD only.",
       (
           "Filter applied before any L2.A frequency-alignment rule. "
           "Useful when the user wants a strictly monthly FRED-SD panel "
           "and would prefer dropping the quarterly variables to keeping "
           "them and accepting the alignment rule's interpolation."
       ),
       "Strict monthly panels; avoiding quarterly-to-monthly interpolation.",
       when_not_to_use="When quarterly variables are central to the analysis.",
       related_options=("quarterly_only", "both")),
    _e("l2_a", "sd_series_frequency_filter", "quarterly_only",
       "Drop monthly series; retain quarterly FRED-SD only.",
       (
           "Inverse of ``monthly_only``: keeps quarterly variables and "
           "drops monthly variables. Used when comparing to a "
           "quarterly benchmark (e.g. real GDP) and monthly variables "
           "would inflate the panel without contributing forecast skill."
       ),
       "Quarterly-target studies; FRED-QD style analyses on FRED-SD data.",
       related_options=("monthly_only", "both")),
    _e("l2_a", "sd_series_frequency_filter", "both",
       "Keep both monthly and quarterly FRED-SD series.",
       (
           "Default; defers to the L2.A frequency-alignment rules "
           "(``monthly_to_quarterly_rule`` / ``quarterly_to_monthly_rule``) "
           "to render the mixed-frequency panel into a single grid."
       ),
       "Default for FRED-SD recipes; mixed-frequency panels.",
       related_options=("monthly_only", "quarterly_only")),
)


# L2.{B,C,D,E} scope axes -- target / predictors / both / not_applicable
_SCOPE_DOCS = {
    "target_and_predictors": (
        "Apply the rule to target and all predictors.",
        (
            "Default scope: every series in the panel passes through the "
            "stage. Maintains consistency between target and predictors "
            "(e.g. both differenced, both winsorised)."
        ),
        "Default; matches McCracken-Ng's convention.",
    ),
    "predictors_only": (
        "Apply only to predictors; leave the target untouched.",
        (
            "Used when the target's transform / cleaning policy is "
            "controlled separately (e.g. user already applied a tcode "
            "to the target via raw_panel)."
        ),
        "When the target enters the pipeline already cleaned.",
    ),
    "target_only": (
        "Apply only to the target.",
        (
            "Rare scope; used when predictors are pre-engineered and "
            "do not need this stage (e.g. PCA scores are already "
            "stationary)."
        ),
        "Pre-engineered predictor panels.",
    ),
    "not_applicable": (
        "Skip the stage entirely (gate inactive).",
        (
            "Used when an upstream stage already produced the desired "
            "form. Equivalent in effect to selecting the no-op option "
            "on the primary axis."
        ),
        "Pipelines that bypass this stage by construction.",
    ),
}
for _axis, _sub in (
    ("transform_scope",   "l2_b"),
    ("outlier_scope",     "l2_c"),
    ("imputation_scope",  "l2_d"),
    ("frame_edge_scope",  "l2_e"),
):
    for _opt, (_summary, _desc, _when) in _SCOPE_DOCS.items():
        register(_e(
            _sub, _axis, _opt, _summary, _desc, _when,
            related_options=tuple(k for k in _SCOPE_DOCS if k != _opt),
        ))


# L2.D imputation_temporal_rule
register(
    _e("l2_d", "imputation_temporal_rule", "expanding_window_per_origin",
       "Re-fit the imputation model on every expanding window.",
       (
           "Default temporal_rule: at each OOS origin, the imputation "
           "model is fit on all data from the sample start through "
           "the origin date. Avoids look-ahead while ensuring the "
           "model has access to maximum data at each step."
       ),
       "Default; OOS-safe imputation.",
       when_not_to_use="When per-origin re-fits are too expensive -- consider ``block_recompute``.",
       related_options=("rolling_window_per_origin", "block_recompute")),
    _e("l2_d", "imputation_temporal_rule", "rolling_window_per_origin",
       "Re-fit the imputation model on a fixed-length rolling window.",
       (
           "Fits the imputation model on the most-recent "
           "``params.window`` observations only. Useful when the "
           "underlying covariance structure is non-stationary and "
           "old data should not influence current imputations."
       ),
       "Non-stationary panels where covariance drifts.",
       when_not_to_use="When the panel is stationary -- expanding window uses more information.",
       related_options=("expanding_window_per_origin", "block_recompute")),
    _e("l2_d", "imputation_temporal_rule", "block_recompute",
       "Re-fit the imputation model every N origins.",
       (
           "Fits the imputation model once every "
           "``leaf_config.imputation_recompute_interval`` origins; "
           "intermediate origins reuse the cached fit. Cheap "
           "approximation to ``expanding_window_per_origin``."
       ),
       "Long sweeps where per-origin re-fits are computationally infeasible.",
       when_not_to_use="When precise OOS-safe imputation is critical.",
       related_options=("expanding_window_per_origin", "rolling_window_per_origin")),
)


register(
    _L2B_OFFICIAL, _L2B_NO_TRANSFORM, _L2B_CUSTOM,
    _L2C_MCCRACKEN_IQR, _L2C_WINSORIZE, _L2C_ZSCORE, _L2C_NONE,
    _L2C_ACTION_NAN, _L2C_ACTION_MEDIAN, _L2C_ACTION_CAP,
    _L2D_EM_FACTOR, _L2D_EM_MULTI, _L2D_MEAN, _L2D_FFILL, _L2D_LINEAR, _L2D_NONE,
    _L2E_TRUNCATE, _L2E_DROP_SERIES, _L2E_KEEP_UNBAL, _L2E_ZERO_FILL,
    _L2A_QM_BACKWARD, _L2A_QM_LINEAR, _L2A_QM_FORWARD,
    _L2A_MQ_AVG, _L2A_MQ_END, _L2A_MQ_SUM,
)


# ---------------------------------------------------------------------------
# v0.8.5: L2.A mixed_frequency_representation + L2.B sd_tcode_policy
# ---------------------------------------------------------------------------
# Generalises the FRED-SD-specific mixed-frequency rules to any panel that
# carries mixed-frequency columns (either FRED-SD or a custom panel that
# declares per-column native frequency). The two new axes are:
#
#   * ``mixed_frequency_representation`` (L2.A): five options ranging from
#     the default ``calendar_aligned_frame`` to a model-adapter contract
#     (``mixed_frequency_model_adapter``) that carries native-frequency
#     block metadata to MIDAS-style downstream models.
#   * ``sd_tcode_policy`` (L2.B): orthogonal to ``transform_policy``; lets
#     the user opt into the FRED-SD inferred (national-analog) or
#     empirical (stationarity-audit) t-code maps. Default ``none``
#     leaves FRED-SD source values as published.

_L2A_MFR_CALENDAR = _e(
    "l2_a", "mixed_frequency_representation", "calendar_aligned_frame",
    "Default: keep selected mixed-frequency columns on the experiment calendar.",
    (
        "When a panel mixes monthly and quarterly columns (FRED-SD by "
        "default; any custom panel that declares per-column native "
        "frequency in metadata), the default representation flattens all "
        "columns to the experiment calendar via the L2.A "
        "``quarterly_to_monthly_rule`` / ``monthly_to_quarterly_rule`` "
        "alignment rules. The panel emerges as a single rectangular "
        "frame; downstream layers see a uniform sampling grid."
    ),
    "Default for mixed-frequency studies; pairs with the canonical L2.A alignment rules.",
    references=(_REF_DESIGN_L2,),
    related_options=(
        "drop_unknown_native_frequency",
        "drop_non_target_native_frequency",
        "native_frequency_block_payload",
        "mixed_frequency_model_adapter",
    ),
)

_L2A_MFR_DROP_UNKNOWN = _e(
    "l2_a", "mixed_frequency_representation", "drop_unknown_native_frequency",
    "Drop columns whose native frequency cannot be inferred.",
    (
        "Restricts the panel to columns whose native sampling rate is "
        "either declared in the L1 metadata or detectable from the "
        "FRED-SD workbook. Columns with unknown native frequency are "
        "dropped before any frequency-alignment rule fires."
    ),
    "Studies that demand strict provenance over per-column native frequency.",
    references=(_REF_DESIGN_L2,),
    related_options=("calendar_aligned_frame", "drop_non_target_native_frequency"),
)

_L2A_MFR_DROP_NON_TARGET = _e(
    "l2_a", "mixed_frequency_representation", "drop_non_target_native_frequency",
    "Keep only columns whose native frequency matches the experiment frequency.",
    (
        "Restricts the panel to columns whose native sampling rate "
        "equals the L1 ``frequency``. For a monthly experiment the "
        "quarterly columns are dropped (and vice versa). Useful when "
        "the user wants a strict single-frequency panel without any "
        "interpolation artifacts."
    ),
    "Strict monthly-only or quarterly-only panels; single-frequency benchmarks.",
    references=(_REF_DESIGN_L2,),
    related_options=("calendar_aligned_frame", "drop_unknown_native_frequency"),
)

_L2A_MFR_BLOCK_PAYLOAD = _e(
    "l2_a", "mixed_frequency_representation", "native_frequency_block_payload",
    "Emit per-frequency block metadata for downstream models.",
    (
        "Keeps the panel intact (no alignment / drop) and instead "
        "publishes a ``fred_sd_native_frequency_block_payload.json`` "
        "manifest entry that lists each column's native frequency. "
        "Models that consume mixed-frequency input directly (e.g. "
        "MIDAS, mixed-frequency factor models) can read this metadata "
        "from ``context['auxiliary_payloads']``."
    ),
    "Researcher-owned MIDAS / mixed-frequency factor model studies.",
    references=(_REF_DESIGN_L2,),
    related_options=("mixed_frequency_model_adapter", "calendar_aligned_frame"),
)

_L2A_MFR_ADAPTER = _e(
    "l2_a", "mixed_frequency_representation", "mixed_frequency_model_adapter",
    "Block payload + a model-adapter contract for MIDAS-style fits.",
    (
        "Strictest option: emits the per-frequency block payload "
        "(see ``native_frequency_block_payload``) plus a model-adapter "
        "contract that the L4 model_family must honour. The adapter "
        "validates that the registered ``model_family`` either declares "
        "MIDAS-style mixed-frequency support or registers via "
        "``mf.custom_model`` with the appropriate ``auxiliary_payloads`` "
        "consumption. Runtime writes "
        "``fred_sd_mixed_frequency_model_adapter.json`` with the "
        "adapter contract details."
    ),
    "Built-in MIDAS families (``midas_almon``, ``midasr``) or registered custom mixed-frequency models.",
    references=(_REF_DESIGN_L2,),
    related_options=("native_frequency_block_payload", "calendar_aligned_frame"),
)

# L2.B sd_tcode_policy
_L2B_SDT_NONE = _e(
    "l2_b", "sd_tcode_policy", "none",
    "Default: no FRED-SD-specific t-code applied.",
    (
        "FRED-SD does not publish official transformation codes. The "
        "default ``none`` policy leaves FRED-SD source values as "
        "published and applies whatever ``transform_policy`` the user "
        "selected (default ``apply_official_tcode`` only operates on "
        "FRED-MD/QD columns; FRED-SD columns pass through). Use this "
        "option whenever the study does not depend on a particular "
        "FRED-SD stationarity transform."
    ),
    "Default for FRED-SD recipes; canonical baseline.",
    references=(_REF_DESIGN_L2,),
    related_options=("inferred", "empirical"),
)

_L2B_SDT_INFERRED = _e(
    "l2_b", "sd_tcode_policy", "inferred",
    "Apply the inferred SD t-code map (national-analog research layer).",
    (
        "Opt-in: applies the package-shipped inferred t-code map for "
        "FRED-SD columns. The map is derived by taking the FRED-MD/QD "
        "national analog of each FRED-SD variable and inheriting that "
        "analog's published t-code. The manifest records "
        "``data_reports.sd_inferred_tcodes`` with ``official: false``, "
        "the map version, and the allowed review statuses."
    ),
    "Studies that want a published (but non-official) t-code path; replication of national-analog research.",
    references=(_REF_DESIGN_L2,),
    related_options=("none", "empirical"),
)

_L2B_SDT_EMPIRICAL = _e(
    "l2_b", "sd_tcode_policy", "empirical",
    "Apply the empirical stationarity-audit t-code map.",
    (
        "Opt-in: applies an empirical t-code map derived from a "
        "stationarity audit of the FRED-SD panel. Two ``unit`` modes:\n\n"
        "* ``variable_global`` -- one t-code per FRED-SD variable, "
        "  shared across states\n"
        "* ``state_series`` -- one t-code per (variable, state) pair; "
        "  requires ``leaf_config.sd_tcode_code_map`` and "
        "  ``sd_tcode_audit_uri``\n\n"
        "The manifest records ``official: false`` plus the audit URI "
        "and chosen unit."
    ),
    "Stationarity-audit driven research; per-state t-code policies.",
    references=(_REF_DESIGN_L2,),
    related_options=("none", "inferred"),
)

register(
    _L2A_MFR_CALENDAR,
    _L2A_MFR_DROP_UNKNOWN,
    _L2A_MFR_DROP_NON_TARGET,
    _L2A_MFR_BLOCK_PAYLOAD,
    _L2A_MFR_ADAPTER,
    _L2B_SDT_NONE,
    _L2B_SDT_INFERRED,
    _L2B_SDT_EMPIRICAL,
)
