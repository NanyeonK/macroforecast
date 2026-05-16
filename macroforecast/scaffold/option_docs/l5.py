"""L5 evaluation -- per-option documentation.

L5 turns forecasts into evaluation tables. The single L5.A axis exposes
five metric lists (primary / point / density / direction / relative);
each is a multi-select choice from a controlled vocabulary.

Every entry below ships hand-written description + literature reference
to the canonical source (Diebold for forecasting; Hyndman-Koehler for
scale-free metrics; Gneiting-Raftery for proper density scores;
Pesaran-Timmermann for directional accuracy; Theil for inequality
coefficients).
"""
from __future__ import annotations

from . import register
from .types import OptionDoc, ParameterDoc, Reference

_REVIEWED = "2026-05-05"
_REVIEWER = "macroforecast author"

_REF_DESIGN_L5 = Reference(
    citation="macroforecast design Part 3, L5: 'evaluation = (metric × benchmark × aggregation × decomposition × ranking).'",
)
_REF_DIEBOLD_2017 = Reference(
    citation="Diebold (2017) 'Forecasting in Economics, Business, Finance and Beyond', University of Pennsylvania (free online).",
    url="https://www.sas.upenn.edu/~fdiebold/Textbooks.html",
)
_REF_HYNDMAN_KOEHLER_2006 = Reference(
    citation="Hyndman & Koehler (2006) 'Another look at measures of forecast accuracy', International Journal of Forecasting 22(4): 679-688.",
    doi="10.1016/j.ijforecast.2006.03.001",
)
_REF_GNEITING_RAFTERY_2007 = Reference(
    citation="Gneiting & Raftery (2007) 'Strictly Proper Scoring Rules, Prediction, and Estimation', JASA 102(477): 359-378.",
    doi="10.1198/016214506000001437",
)
_REF_GNEITING_KATZFUSS_2014 = Reference(
    citation="Gneiting & Katzfuss (2014) 'Probabilistic Forecasting', Annual Review of Statistics and Its Application 1: 125-151.",
)
_REF_PESARAN_TIMMERMANN_1992 = Reference(
    citation="Pesaran & Timmermann (1992) 'A simple nonparametric test of predictive performance', JBES 10(4): 461-465.",
)
_REF_THEIL_1966 = Reference(
    citation="Theil (1966) 'Applied Economic Forecasting', North-Holland (Chapter 2: Inequality coefficients).",
)
_REF_CAMPBELL_THOMPSON_2008 = Reference(
    citation="Campbell & Thompson (2008) 'Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?', Review of Financial Studies 21(4): 1509-1531.",
    doi="10.1093/rfs/hhm055",
)
_REF_WINKLER_1972 = Reference(
    citation="Winkler (1972) 'A Decision-Theoretic Approach to Interval Estimation', JASA 67(337): 187-191.",
)


def _e(axis: str, option: str, summary: str, description: str, when_to_use: str,
       *, when_not_to_use: str = "",
       references: tuple[Reference, ...] = (_REF_DESIGN_L5,),
       related: tuple[str, ...] = (),
       op_page: bool = False,
       op_func_name: str = "",
       parameters: tuple[ParameterDoc, ...] = ()) -> OptionDoc:
    return OptionDoc(
        layer="l5", sublayer="L5_A_metric_specification", axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, references=references,
        related_options=related,
        op_page=op_page,
        op_func_name=op_func_name,
        parameters=parameters,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


# ---------------------------------------------------------------------------
# Point-forecast metrics
# ---------------------------------------------------------------------------

_DOC_MSE = (
    "Mean squared error -- ``(1/N) Σ (y_t - ŷ_t)²``.",
    (
        "The classical quadratic-loss metric. Optimal under "
        "Gaussian-residual / squared-loss decision theory; the L4 fit "
        "objective for OLS / ridge / elastic net is its in-sample "
        "version. MSE penalises large residuals super-linearly, so a "
        "single outlier in the OOS sample can dominate the score."
    ),
    "Default for Gaussian-residual problems; horse-race ranking under squared-loss decision rules.",
    "Heavy-tailed forecast errors -- a single outlier dominates the score; consider MAE or MedAE instead.",
)

_DOC_RMSE = (
    "Root mean squared error -- ``√MSE``.",
    (
        "Same ranking as MSE but expressed in target units (rather than "
        "squared target units). Standard reporting metric in macro / "
        "finance papers; pairs naturally with confidence-band charts "
        "since RMSE has the same units as the prediction interval."
    ),
    "Reporting forecast accuracy in target units.",
    "Heavy-tailed errors -- inherits MSE's outlier sensitivity.",
)

_DOC_MAE = (
    "Mean absolute error -- ``(1/N) Σ |y_t - ŷ_t|``.",
    (
        "L1 loss; robust alternative to MSE. Equally weighs every "
        "absolute residual rather than penalising large errors super-"
        "linearly. The implicit decision rule under MAE is the median "
        "of the predictive distribution (vs the mean for MSE)."
    ),
    "Heavy-tailed targets where extreme errors should not dominate; reporting in target units.",
    "When the squared-loss decision rule is what the user actually faces.",
)

_DOC_MEDAE = (
    "Median absolute error -- ``median |y_t - ŷ_t|``.",
    (
        "Maximally robust point-forecast metric: substitution by median "
        "completely insulates the score from a constant-share of "
        "extreme residuals. Common in robust-statistics papers; rarer "
        "in mainstream forecasting."
    ),
    "Pathologically heavy-tailed errors (financial crises, regime shifts).",
    "Standard reporting -- mean-based metrics are the convention.",
)

_DOC_MAPE = (
    "Mean absolute percentage error -- ``(100/N) Σ |y_t - ŷ_t| / |y_t|``.",
    (
        "Scale-free percentage version of MAE. Allows comparing forecasts "
        "for targets on different scales (US GDP vs Korean GDP). "
        "Pathological when targets can be zero or near-zero -- the "
        "metric blows up. Hyndman & Koehler (2006) recommend MASE / "
        "sMAPE in those cases."
    ),
    "Cross-target / cross-country comparisons; reporting forecast accuracy in percentage terms.",
    "Targets that can be near zero (rates, growth rates) -- division by tiny ``|y_t|`` makes the metric explode.",
)

_DOC_THEIL_U1 = (
    "Theil's U1 inequality coefficient -- bounded in ``[0, 1]``.",
    (
        "``U₁ = √MSE / (√(1/N Σ y²) + √(1/N Σ ŷ²))``. Bounded between 0 "
        "(perfect forecast) and 1 (worst possible). Theil's original "
        "1966 metric; less commonly used today than U2 because the "
        "denominator's interpretation is less intuitive."
    ),
    "Long-run macro forecasting tradition; comparability with Theil-1966-era papers.",
    "Modern reporting -- U2 is more interpretable as a ratio against the no-change benchmark.",
)

_DOC_THEIL_U2 = (
    "Theil's U2 inequality coefficient -- ratio of forecast MSE to no-change MSE.",
    (
        "``U₂ = √(Σ (ŷ_t - y_t)² / Σ (y_{t-1} - y_t)²)``. ``U₂ < 1`` "
        "means the forecast beats the random-walk benchmark. Standard "
        "sanity-check ratio in macro forecasting -- if ``U₂ ≥ 1`` the "
        "model is no better than 'tomorrow looks like today'."
    ),
    "Sanity-checking against the random-walk benchmark; macro-forecasting tradition.",
    "When a custom benchmark (not random walk) is preferred -- use ``relative_mse`` instead.",
)

POINT = {
    "mse":       _DOC_MSE,
    "rmse":      _DOC_RMSE,
    "mae":       _DOC_MAE,
    "medae":     _DOC_MEDAE,
    "mape":      _DOC_MAPE,
    "theil_u1":  _DOC_THEIL_U1,
    "theil_u2":  _DOC_THEIL_U2,
}


# ---------------------------------------------------------------------------
# Relative metrics
# ---------------------------------------------------------------------------

_DOC_REL_MSE = (
    "Forecast MSE divided by the L4 ``is_benchmark`` model's MSE.",
    (
        "``MSE_model / MSE_benchmark``. The standard horse-race ratio. "
        "Below 1 means the candidate beats the benchmark; the L5.E "
        "ranking tables surface this column by default. Requires "
        "exactly one L4 model with ``is_benchmark = true`` (validator "
        "hard-rejects 0 or > 1 benchmarks)."
    ),
    "Default reporting metric in horse-race tables; comparing candidate models against a fixed benchmark.",
)

_DOC_REL_MAE = (
    "Forecast MAE divided by the L4 ``is_benchmark`` model's MAE.",
    (
        "L1-loss analogue of ``relative_mse``. Below 1 means the "
        "candidate beats the benchmark on absolute-loss criterion. "
        "Robust to heavy-tailed forecast errors."
    ),
    "Heavy-tailed targets where MSE is too sensitive to outliers.",
)

_DOC_MSE_REDUCTION = (
    "``1 - relative_mse`` -- positive means the candidate beats the benchmark.",
    (
        "Convenience reformulation that flips the sign so positive "
        "numbers indicate improvement. Common in macro-forecasting "
        "papers (e.g. Stock-Watson 2002 reports MSE reduction in %). "
        "Equivalent to ``1 - MSE_model / MSE_benchmark``."
    ),
    "Default reporting in horse-race tables when 'positive = better' is preferred.",
)

_DOC_R2_OOS = (
    "Out-of-sample R² (Campbell-Thompson 2008) -- ``1 - SSE_model / SSE_benchmark``.",
    (
        "Standard return-predictability metric in finance (and "
        "increasingly in macro). Identical formula to ``mse_reduction`` "
        "when the benchmark is the historical mean. Campbell & "
        "Thompson (2008) popularised the metric for the empirical-"
        "asset-pricing literature."
    ),
    "Macro / financial forecasting tradition; literature-compatibility with CT-2008-era papers.",
)

RELATIVE = {
    "relative_mse": _DOC_REL_MSE,
    "relative_mae": _DOC_REL_MAE,
    "mse_reduction": _DOC_MSE_REDUCTION,
    "r2_oos":        _DOC_R2_OOS,
}


# ---------------------------------------------------------------------------
# Density / interval metrics
# ---------------------------------------------------------------------------

_DOC_LOG_SCORE = (
    "Logarithmic predictive density score -- ``log f̂(y_t)``.",
    (
        "The strictly-proper scoring rule recommended by Gneiting & "
        "Raftery (2007). Equivalent to the Bayesian predictive "
        "log-likelihood. Larger = better. Requires "
        "``forecast_object = density / quantile`` from L4.\n\n"
        "When the predictive density is parametric (e.g. Gaussian) "
        "the score reduces to a closed-form involving the predictive "
        "mean / variance."
    ),
    "Default scoring rule for Bayesian forecasts; probabilistic horse-race ranking.",
)

_DOC_CRPS = (
    "Continuous ranked probability score -- generalisation of MAE to densities.",
    (
        "``CRPS = ∫ (F̂(y) - 1{y ≥ y_obs})² dy``. Strictly-proper, "
        "expressed in the same units as the target. Reduces to MAE "
        "when the predictive distribution is a point mass at the "
        "predicted value. Standard density-score in weather / macro "
        "forecasting (Gneiting-Katzfuss 2014)."
    ),
    "Distributional forecasts; comparing point and density forecasts on a common scale.",
)

_DOC_INTERVAL_SCORE = (
    "Winkler (1972) interval score -- jointly penalises miscoverage + interval width.",
    (
        "For a nominal-α interval ``[L, U]``: "
        "``IS_α = (U - L) + (2/α)(L - y) 1{y < L} + (2/α)(y - U) 1{y > U}``. "
        "Lower = better. Strictly-proper for the α-level prediction "
        "interval; the natural metric when L4 emits "
        "``forecast_object = interval``."
    ),
    "Prediction-interval evaluation; balancing tightness against coverage.",
)

_DOC_COVERAGE = (
    "Empirical coverage rate -- share of OOS observations falling within the nominal-α interval.",
    (
        "Should equal α (1 - α miscoverage) if the model is well-"
        "calibrated. Deviations indicate miscalibration: low "
        "coverage = intervals too narrow; high coverage = intervals "
        "too wide. Pair with ``interval_score`` to capture both "
        "calibration and sharpness."
    ),
    "Interval-calibration audits; reporting alongside interval_score.",
)

DENSITY = {
    "log_score":      _DOC_LOG_SCORE,
    "crps":           _DOC_CRPS,
    "interval_score": _DOC_INTERVAL_SCORE,
    "coverage_rate":  _DOC_COVERAGE,
}


# ---------------------------------------------------------------------------
# Direction metrics
# ---------------------------------------------------------------------------

_DOC_SUCCESS_RATIO = (
    "Hit-rate of correct directional forecasts -- ``(1/N) Σ 1{sign(ŷ_t) = sign(y_t)}``.",
    (
        "Naive directional accuracy, bounded in ``[0, 1]``. Does not "
        "adjust for the unconditional direction frequency, so a "
        "constant 'always positive' forecast can score 0.7 on a "
        "growth target. For statistical significance, pair with "
        "``pesaran_timmermann_metric`` and the L6.F PT test."
    ),
    "Quick directional-accuracy reporting; reporting the raw hit-rate alongside the PT statistic.",
    "Standalone significance testing -- needs PT correction for unconditional direction frequency.",
)

_DOC_PT_METRIC = (
    "Pesaran-Timmermann (1992) directional-accuracy statistic.",
    (
        "Adjusts the success ratio for the joint probability of "
        "agreement under independence (so a constant-sign forecast "
        "no longer scores high). Asymptotically standard normal "
        "under the null of no directional skill; the L6.F test "
        "computes the corresponding p-value."
    ),
    "Formal directional-accuracy reporting (paired with the L6 PT test).",
    "",
)

DIRECTION = {
    "success_ratio":             _DOC_SUCCESS_RATIO,
    "pesaran_timmermann_metric": _DOC_PT_METRIC,
}


# ---------------------------------------------------------------------------
# Build OptionDoc entries.
# ---------------------------------------------------------------------------

# Primary metric (single-select)
_PRIMARY_REFS_BY_OPTION: dict[str, tuple[Reference, ...]] = {
    "mse":           (_REF_DESIGN_L5, _REF_DIEBOLD_2017),
    "rmse":          (_REF_DESIGN_L5, _REF_DIEBOLD_2017),
    "mae":           (_REF_DESIGN_L5, _REF_DIEBOLD_2017),
    "medae":         (_REF_DESIGN_L5,),
    "mape":          (_REF_DESIGN_L5, _REF_HYNDMAN_KOEHLER_2006),
    "theil_u1":      (_REF_DESIGN_L5, _REF_THEIL_1966),
    "theil_u2":      (_REF_DESIGN_L5, _REF_THEIL_1966),
    "relative_mse":  (_REF_DESIGN_L5, _REF_DIEBOLD_2017),
    "relative_mae":  (_REF_DESIGN_L5,),
    "mse_reduction": (_REF_DESIGN_L5, _REF_CAMPBELL_THOMPSON_2008),
    "r2_oos":        (_REF_DESIGN_L5, _REF_CAMPBELL_THOMPSON_2008),
    "log_score":     (_REF_DESIGN_L5, _REF_GNEITING_RAFTERY_2007),
    "crps":          (_REF_DESIGN_L5, _REF_GNEITING_RAFTERY_2007, _REF_GNEITING_KATZFUSS_2014),
}

_ALL = {**POINT, **RELATIVE, **DENSITY}

_PRIMARY = []
for option, doc in _ALL.items():
    if option not in _PRIMARY_REFS_BY_OPTION:
        continue
    summary, desc, when, *rest = doc + ("",) * (4 - len(doc))
    when_not = rest[0] if rest else ""
    _PRIMARY.append(_e(
        "primary_metric", option, summary,
        f"Primary metric used for the L5.A summary table and the L5.E ranking. {desc}",
        when, when_not_to_use=when_not,
        references=_PRIMARY_REFS_BY_OPTION[option],
        related=tuple(k for k in _PRIMARY_REFS_BY_OPTION if k != option)[:4]))


# ParameterDocs for theil_u1 function-op (Cycle 22 POC).
_THEIL_U1_PARAMS: tuple[ParameterDoc, ...] = (
    ParameterDoc(
        name="y_true",
        type="np.ndarray | pd.Series",
        default=None,
        description="Actual (realised) values. 1-D array of length N.",
    ),
    ParameterDoc(
        name="y_pred",
        type="np.ndarray | pd.Series",
        default=None,
        description="Forecast values. Must be the same length as y_true.",
    ),
)

# Multi-select axes
_POINT = []
for option, (summary, desc, when, when_not) in POINT.items():
    refs: tuple[Reference, ...] = (_REF_DESIGN_L5, _REF_DIEBOLD_2017)
    if option == "mape":
        refs = refs + (_REF_HYNDMAN_KOEHLER_2006,)
    if option in ("theil_u1", "theil_u2"):
        refs = (_REF_DESIGN_L5, _REF_THEIL_1966)
    # Cycle 22 POC: theil_u1 gets a dedicated per-op encyclopedia page.
    _op_page = option == "theil_u1"
    _op_func_name = "theil_u1" if option == "theil_u1" else ""
    _params: tuple[ParameterDoc, ...] = _THEIL_U1_PARAMS if option == "theil_u1" else ()
    _POINT.append(_e("point_metrics", option, summary,
        f"Point-forecast metric ``{option}``. {desc}",
        when, when_not_to_use=when_not, references=refs,
        related=tuple(k for k in POINT if k != option),
        op_page=_op_page, op_func_name=_op_func_name, parameters=_params))

_DENSITY = []
for option, (summary, desc, when) in DENSITY.items():
    density_refs: tuple[Reference, ...] = (_REF_DESIGN_L5, _REF_GNEITING_RAFTERY_2007, _REF_GNEITING_KATZFUSS_2014)
    if option == "interval_score":
        density_refs = density_refs + (_REF_WINKLER_1972,)
    _DENSITY.append(_e("density_metrics", option, summary,
        f"Density-forecast metric ``{option}``. {desc}",
        when, references=density_refs,
        related=tuple(k for k in DENSITY if k != option)))

_DIRECTION = [
    _e("direction_metrics", option, summary,
       f"Directional-accuracy metric ``{option}``. {desc}",
       when, when_not_to_use=when_not,
       references=(_REF_DESIGN_L5, _REF_PESARAN_TIMMERMANN_1992),
       related=tuple(k for k in DIRECTION if k != option))
    for option, (summary, desc, when, when_not) in DIRECTION.items()
]

_RELATIVE = []
for option, (summary, desc, when) in RELATIVE.items():
    relative_refs: tuple[Reference, ...] = (_REF_DESIGN_L5, _REF_DIEBOLD_2017)
    if option in ("mse_reduction", "r2_oos"):
        relative_refs = (_REF_DESIGN_L5, _REF_CAMPBELL_THOMPSON_2008)
    _RELATIVE.append(_e("relative_metrics", option, summary,
        f"Relative-loss metric ``{option}``. {desc}",
        when, references=relative_refs,
        related=tuple(k for k in RELATIVE if k != option)))

register(*_PRIMARY, *_POINT, *_DENSITY, *_DIRECTION, *_RELATIVE)
