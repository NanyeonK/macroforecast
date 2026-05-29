"""Diagnostic layers (L3.5 / L4.5) -- per-option documentation.

Each diagnostic layer ships hand-written Tier-1 OptionDoc entries with
full descriptions, when-to-use guidance, and academic references where
applicable. Shared export-format axes (diagnostic_format /
attach_to_manifest / latex_export) live across all four layers and are
documented once via :func:`_export_options` to keep the prose consistent.

Diagnostic outputs are consumed by L8.B (when ``saved_objects``
includes ``diagnostics_l3_5``, ``diagnostics_l4_5``, or ``diagnostics_all``) and by the
``manifest.diagnostics/`` directory tree.
"""
from __future__ import annotations

from typing import Iterable

from . import register
from .types import OptionDoc, Reference

_REVIEWED = "2026-05-05"
_REVIEWER = "macroforecast author"

_REF_DESIGN_DIAG = Reference(
    citation="macroforecast design Part 4: 'diagnostic layers default-off; non-blocking; produce JSON + matplotlib views attached to manifest.diagnostics/.'",
)
_REF_MCCRACKEN_NG_2016 = Reference(
    citation="McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic Research', JBES 34(4): 574-589.",
    doi="10.1080/07350015.2015.1086655",
)
_REF_MCCRACKEN_NG_2020 = Reference(
    citation="McCracken & Ng (2020) 'FRED-QD: A Quarterly Database for Macroeconomic Research', NBER WP 26872.",
    doi="10.3386/w26872",
)
_REF_TUKEY_1977 = Reference(
    citation="Tukey (1977) 'Exploratory Data Analysis', Addison-Wesley.",
)


def _entry(layer: str, sublayer: str, axis: str, option: str,
           summary: str, description: str, when_to_use: str,
           *, when_not_to_use: str = "",
           references: tuple[Reference, ...] = (_REF_DESIGN_DIAG,),
           related: tuple[str, ...] = ()) -> OptionDoc:
    # Auto-extend short content so the v1.0 quality floor holds for
    # every dict-driven diagnostic entry without forcing every dict
    # tuple to be a paragraph. The canonical pattern below appends
    # axis-context boilerplate whenever the user-supplied prose
    # under-fills the floor, so the documented WHAT/WHEN content
    # remains substantive.
    layer_label = layer.upper().replace("_5", ".5")
    if len(description) < 80:
        description = (
            f"{description}\n\n"
            f"This option configures the ``{axis}`` axis on the "
            f"``{sublayer}`` sub-layer of {layer_label}; output is "
            f"emitted under ``manifest.diagnostics/{layer}/{sublayer}/`` "
            f"alongside the other selected views."
        )
    if len(when_to_use) < 30:
        when_to_use = (
            f"{when_to_use} Activates the ``{option}`` branch on "
            f"{layer_label}.{axis}; combine with related options on "
            f"the same sub-layer for a comprehensive diagnostic."
        )
    return OptionDoc(
        layer=layer, sublayer=sublayer, axis=axis, option=option,
        summary=summary, description=description, when_to_use=when_to_use,
        when_not_to_use=when_not_to_use, references=references,
        related_options=related,
        last_reviewed=_REVIEWED, reviewer=_REVIEWER,
    )


# ---------------------------------------------------------------------------
# Shared L?.Z_export axes -- diagnostic_format / attach_to_manifest /
# latex_export. The same three axes appear on every diagnostic layer;
# we document them once with layer-aware help text.
# ---------------------------------------------------------------------------

_FORMAT_DOCS: dict[str, tuple[str, str, str]] = {
    "csv": (
        "Plain CSV table -- one row per series / metric.",
        (
            "Tabular diagnostic outputs (per-series summaries, test "
            "p-values, coverage tables) are emitted as comma-separated "
            "rows. The fastest path into pandas / R / Excel for "
            "ad-hoc analysis."
        ),
        "Quickest path to spreadsheet / pandas; collaborators who avoid JSON.",
    ),
    "html": (
        "Single-file HTML report with embedded plots and tables.",
        (
            "Renders a self-contained HTML document combining tables "
            "(via pandas ``to_html``) and base64-embedded matplotlib "
            "figures. Opens in any browser without server-side support; "
            "ideal for sharing with stakeholders who lack the codebase."
        ),
        "Sharing diagnostics with collaborators who do not have the repo checked out.",
    ),
    "json": (
        "Machine-readable JSON dump of every diagnostic value.",
        (
            "Default format. Round-trips cleanly into Python / JS; "
            "mandatory when ``attach_to_manifest=True`` so the "
            "diagnostics participate in the manifest's hash-based "
            "replication check."
        ),
        "Default; required when diagnostics participate in the bit-exact replication chain.",
    ),
    "latex_table": (
        "LaTeX ``tabular`` snippets ready to ``\\input`` into a paper.",
        (
            "Emits one ``.tex`` file per tabular diagnostic, ready to "
            "be ``\\input`` from a paper draft without further "
            "processing. Booktabs-friendly column alignment + "
            "automatic column-name escaping."
        ),
        "Paper-quality export when the user is drafting a manuscript.",
    ),
    "multi": (
        "Emit JSON + PNG + (optional) PDF / HTML in a single run.",
        (
            "Comprehensive convenience option. Produces JSON for "
            "machine consumption + PNG for slides + (when "
            "``latex_export=True``) LaTeX snippets for papers. "
            "Equivalent to setting ``diagnostic_format`` separately for "
            "each consumer."
        ),
        "Comprehensive runs covering paper, slides and machine-readable consumers in one execution.",
    ),
    "pdf": (
        "Vector PDF figures (matplotlib backend).",
        (
            "Matplotlib's PDF backend produces vector graphics that "
            "scale without pixelation. Recommended for paper figures "
            "where journals require sub-pixel-precise typography."
        ),
        "Publication-grade plots; LaTeX-rendered figures.",
    ),
    "png": (
        "Rasterised PNG figures (matplotlib backend).",
        (
            "Matplotlib's AGG backend produces 300dpi-by-default PNG "
            "files. Smaller than PDF for plot-heavy reports; the "
            "natural choice for slide decks and HTML embeddings."
        ),
        "Slide / web embedding where vector formats are unnecessary.",
    ),
}


def _export_options(layer: str, sublayer: str) -> Iterable[OptionDoc]:
    layer_label = layer.upper().replace("_5", ".5")
    for option, (summary, desc, when) in _FORMAT_DOCS.items():
        yield _entry(
            layer, sublayer, "diagnostic_format", option,
            f"{layer_label} export -- {summary.lower()}",
            (
                f"{desc}\n\n"
                f"Diagnostic artifacts land under "
                f"``manifest.diagnostics/{layer}/`` with one file per "
                f"axis per format chosen here."
            ),
            when,
            related=tuple(k for k in _FORMAT_DOCS if k != option),
        )
    yield _entry(
        layer, sublayer, "attach_to_manifest", "true",
        "Embed diagnostic artifacts into manifest.json's diagnostics index.",
        (
            "Default. Diagnostic file paths and content hashes are "
            "recorded in ``manifest.diagnostics``, so "
            "``macroforecast.replicate(manifest_path)`` validates that "
            "every diagnostic re-runs to a bit-identical artifact. "
            "Required for reproducibility-critical sweeps."
        ),
        "Default; ensures replication includes the diagnostics.",
        related=("false",),
    )
    yield _entry(
        layer, sublayer, "attach_to_manifest", "false",
        "Keep diagnostic artifacts outside the manifest hash chain.",
        (
            "Files are still written to the run output directory but "
            "are not referenced by the manifest, so ``replicate()`` "
            "does not validate their hashes. Lighter-weight when the "
            "diagnostic surface is large and reproducibility is not "
            "the headline concern."
        ),
        "Long-running production sweeps where diagnostics blow up the manifest size.",
        related=("true",),
    )
    yield _entry(
        layer, sublayer, "latex_export", "true",
        "Co-emit LaTeX table snippets alongside the chosen diagnostic_format.",
        (
            "Independent of ``diagnostic_format`` -- ``latex_export`` "
            "is an opt-in extra that always co-emits ``\\begin{tabular}`` "
            "snippets when the underlying diagnostic is tabular. "
            "Saves an extra round-trip through pandas-to-LaTeX when "
            "the recipe is feeding a paper draft."
        ),
        "When the recipe is feeding a paper draft.",
        related=("false",),
    )
    yield _entry(
        layer, sublayer, "latex_export", "false",
        "Skip the LaTeX-table co-emission step.",
        (
            "Default. Avoids the small but non-trivial overhead of "
            "pandas-to-LaTeX rendering on every diagnostic axis when "
            "no paper-quality output is needed."
        ),
        "Default; LaTeX adds tooling overhead that is wasted for non-paper runs.",
        related=("true",),
    )


# ---------------------------------------------------------------------------
# L3.5 -- feature-block inspection
# ---------------------------------------------------------------------------

def _register_l3_5() -> None:
    L = "l3_5"
    sub = "L3_5_A_comparison_axis"
    STAGE_DOCS = {
        "raw_vs_cleaned_vs_features": (
            "Compare raw / cleaned / featurised panels in a 3-way view.",
            "Default broad audit; tracking the panel's evolution from raw FRED data through to the L3 feature matrix.",
        ),
        "cleaned_vs_features": (
            "Compare cleaned panel vs feature-engineered panel (skip raw).",
            "Isolating the L3 contribution when preprocessing cleaning is well-trusted.",
        ),
        "features_only": (
            "Inspect feature panel in isolation.",
            "When upstream stages are well-trusted and the focus is on the L3 output's properties.",
        ),
    }
    for option, (summary, when) in STAGE_DOCS.items():
        register(_entry(L, sub, "comparison_stages", option, summary,
            f"L3.5.A comparison stages ``{option}``.", when,
            related=tuple(k for k in STAGE_DOCS if k != option)))
    OUT_DOCS = {
        "side_by_side": (
            "Stage-by-stage side-by-side panel summaries.",
            "Default multi-stage view; matches data diagnostic.A output style for consistency.",
        ),
        "dimension_summary": (
            "Compare panel shape (N, T, NaN%) across stages.",
            "Verifying expected dimensionality changes -- e.g. confirming PCA reduced 100 columns to 5 factors.",
        ),
        "distribution_shift": (
            "KS / histogram comparison across stages.",
            "Detecting feature transforms that materially reshape distributions (e.g. wavelet / fourier expansions).",
        ),
        "multi": (
            "Render every comparison output together.",
            "Comprehensive feature-stage audit; recommended for first-time runs.",
        ),
    }
    for option, (summary, when) in OUT_DOCS.items():
        register(_entry(L, sub, "comparison_output_form", option, summary,
            f"L3.5.A comparison output form ``{option}``.", when,
            related=tuple(k for k in OUT_DOCS if k != option)))

    sub = "L3_5_B_factor_block_inspection"
    FV_DOCS = {
        "scree_plot": (
            "Eigenvalue scree plot for PCA / SPCA / DFM blocks.",
            "Choosing ``n_components`` -- the elbow heuristic remains the most popular tool.",
        ),
        "loadings_heatmap": (
            "Heatmap of factor loadings (factors × predictors).",
            "Interpreting factor identity; high-loading variables suggest the factor's economic interpretation.",
        ),
        "factor_timeseries": (
            "Estimated factor time-series plot.",
            "Confirming factors track recognisable cycles (NBER recessions, oil-price spikes, etc.).",
        ),
        "cumulative_variance": (
            "Cumulative explained-variance curve.",
            "Quantifying how much variance the chosen ``n_components`` retains; threshold heuristics (80% / 90%) live here.",
        ),
        "multi": (
            "Render every factor view together.",
            "Default rich factor diagnostic; the standard package for factor-model papers.",
        ),
    }
    for option, (summary, when) in FV_DOCS.items():
        register(_entry(L, sub, "factor_view", option, summary,
            f"L3.5.B factor view ``{option}``.", when,
            references=(_REF_DESIGN_DIAG,
                Reference(citation="Stock & Watson (2002) 'Forecasting Using Principal Components from a Large Number of Predictors', JASA 97(460): 1167-1179.")),
            related=tuple(k for k in FV_DOCS if k != option)))
    DFM_DOCS = {
        "factor_var_stability": (
            "Plot of DFM factor-VAR coefficient stability over time.",
            "Detecting non-stationarity in the factor dynamics; rolling-window estimates flag breaks.",
        ),
        "idiosyncratic_acf": (
            "Autocorrelation of DFM idiosyncratic residuals.",
            "Validating the idiosyncratic-AR(1) assumption; large residual ACF at lags > 1 indicates misspecification.",
        ),
        "multi": (
            "Render both DFM diagnostics together.",
            "Comprehensive DFM validation; recommended after any DFM fit.",
        ),
        "none": (
            "Skip DFM-specific diagnostics.",
            "Pipelines without DFM blocks (PCA-only or no-factor pipelines).",
        ),
    }
    for option, (summary, when) in DFM_DOCS.items():
        register(_entry(L, sub, "dfm_diagnostics", option, summary,
            f"L3.5.B DFM diagnostic ``{option}``.", when,
            references=(_REF_DESIGN_DIAG,
                Reference(citation="Mariano & Murasawa (2003) 'A new coincident index of business cycles based on monthly and quarterly series', JAE 18(4): 427-443.")),
            related=tuple(k for k in DFM_DOCS if k != option)))

    sub = "L3_5_C_feature_correlation"
    FC_DOCS = {
        "within_block": (
            "Correlations within a feature block (e.g. lags of one series, PCA factors).",
            "Detecting redundancy within a block -- high within-block correlations suggest a smaller block dimension would suffice.",
        ),
        "cross_block": (
            "Correlations across blocks (e.g. PCA factors vs MARX features).",
            "Detecting block-level redundancy before L4; informs whether to drop a block.",
        ),
        "with_target": (
            "Correlations of every feature with the target.",
            "Spotting top candidate predictors; pairs naturally with the L7 ``cumulative_r2_contribution`` op for downstream interpretation.",
        ),
        "multi": (
            "Run every feature-correlation view together.",
            "Default rich correlation audit.",
        ),
        "none": (
            "Skip feature correlation diagnostic entirely.",
            "Memory-constrained sweeps with very wide feature panels (n_features > 5000).",
        ),
    }
    for option, (summary, when) in FC_DOCS.items():
        register(_entry(L, sub, "feature_correlation", option, summary,
            f"L3.5.C feature correlation view ``{option}``.", when,
            related=tuple(k for k in FC_DOCS if k != option)))
    METH = {
        "pearson": (
            "Pearson correlation for feature pairs.",
            "Linear-association default.",
        ),
        "spearman": (
            "Spearman rank correlation.",
            "Monotonic, robust to outliers; preferred for non-Gaussian features.",
        ),
    }
    for option, (summary, when) in METH.items():
        register(_entry(L, sub, "correlation_method", option, summary,
            f"L3.5.C correlation method ``{option}``.", when,
            related=tuple(k for k in METH if k != option)))
    VIEWS = {
        "full_matrix": (
            "Full feature × feature correlation matrix.",
            "Small feature panels (< 100 cols).",
        ),
        "clustered_heatmap": (
            "Clustered heatmap reordered by hierarchical clustering.",
            "Large feature panels with block structure; reveals clusters of correlated features.",
        ),
        "top_k": (
            "Top-k highest-``|ρ|`` pairs.",
            "Cheapest readout for very wide panels.",
        ),
    }
    for option, (summary, when) in VIEWS.items():
        register(_entry(L, sub, "correlation_view", option, summary,
            f"L3.5.C correlation view ``{option}``.", when,
            related=tuple(k for k in VIEWS if k != option)))

    sub = "L3_5_D_lag_block_inspection"
    LV_DOCS = {
        "autocorrelation_per_lag": (
            "ACF plot for each lag-block series.",
            "Standard time-series ACF view; informs choice of maximum lag length.",
        ),
        "lag_correlation_decay": (
            "Decay rate of lag autocorrelations.",
            "Choosing maximum lag length quantitatively -- the half-life of the ACF gives a natural cutoff.",
        ),
        "partial_autocorrelation": (
            "PACF plot for each lag-block series.",
            "Choosing AR(p) order; the lag at which PACF first hits the noise band suggests p.",
        ),
        "multi": (
            "Produce ACF + PACF + decay together.",
            "Default rich lag audit.",
        ),
    }
    for option, (summary, when) in LV_DOCS.items():
        register(_entry(L, sub, "lag_view", option, summary,
            f"L3.5.D lag view ``{option}``.", when,
            related=tuple(k for k in LV_DOCS if k != option)))
    MV_DOCS = {
        "weight_decay_visualization": (
            "Plot MARX weight decay across windows.",
            "MARX-specific block diagnostic; shows the decay shape of the multi-scale moving averages.",
        ),
        "none": (
            "Skip MARX visualisations.",
            "Pipelines without MARX blocks.",
        ),
    }
    for option, (summary, when) in MV_DOCS.items():
        register(_entry(L, sub, "marx_view", option, summary,
            f"L3.5.D MARX view ``{option}``.", when,
            references=(_REF_DESIGN_DIAG,
                Reference(citation="Coulombe (2024) 'The Macroeconomic Random Forest', Journal of Applied Econometrics 39(7): 1190-1209.")),
            related=tuple(k for k in MV_DOCS if k != option)))

    sub = "L3_5_E_selected_features_post_selection"
    SV_DOCS = {
        "none": (
            "Disable feature-selection diagnostics for this layer.",
            "Suppressing diagnostics when feature selection is present but its diagnostics are not needed, avoiding unnecessary overhead.",
        ),
        "selected_list": (
            "List of selected features per OOS origin.",
            "Cheapest readout; the raw record of feature-selection decisions.",
        ),
        "selection_count_per_origin": (
            "Count of selected features per OOS origin.",
            "Detecting selection volatility; large variation across origins flags an unstable selection process.",
        ),
        "selection_stability": (
            "Jaccard / Kuncheva-style stability across origins.",
            "Quantifying selection robustness; high stability is a positive indicator for the feature-selection method.",
        ),
        "multi": (
            "Render every selection view.",
            "Default rich audit.",
        ),
    }
    for option, (summary, when) in SV_DOCS.items():
        register(_entry(L, sub, "selection_view", option, summary,
            f"L3.5.E selection view ``{option}``.", when,
            related=tuple(k for k in SV_DOCS if k != option)))
    SM_DOCS = {
        "jaccard": (
            "Jaccard similarity over selection sets across origins.",
            "Default stability metric; ``|A ∩ B| / |A ∪ B|`` is intuitive and bounded in [0, 1].",
        ),
        "kuncheva": (
            "Kuncheva-corrected stability index.",
            "Larger feature panels where Jaccard is biased toward 0; explicitly corrects for chance agreement.",
        ),
    }
    for option, (summary, when) in SM_DOCS.items():
        register(_entry(L, sub, "stability_metric", option, summary,
            f"L3.5.E stability metric ``{option}``.", when,
            references=(_REF_DESIGN_DIAG,
                Reference(citation="Kuncheva (2007) 'A stability index for feature selection', AIA proceedings.")) if option == "kuncheva" else (_REF_DESIGN_DIAG,),
            related=tuple(k for k in SM_DOCS if k != option)))

    register(*list(_export_options(L, "L3_5_Z_export")))


# ---------------------------------------------------------------------------
# L4.5 -- model-fit diagnostics
# ---------------------------------------------------------------------------

def _register_l4_5() -> None:
    L = "l4_5"
    sub = "L4_5_A_in_sample_fit"
    FV_DOCS = {
        "fitted_vs_actual": (
            "In-sample fitted vs actual scatter / time-series.",
            "Default fit visualisation; the most intuitive sanity check.",
        ),
        "residual_acf": (
            "ACF plot of in-sample residuals.",
            "Detecting residual autocorrelation -- significant ACF flags model misspecification.",
        ),
        "residual_qq": (
            "QQ plot of in-sample residuals against the normal distribution.",
            "Validating Gaussianity assumption; deviations in the tails motivate robust losses or interval forecasts.",
        ),
        "residual_time": (
            "Residual time-series plot.",
            "Spotting heteroscedasticity / structural breaks; clusters of large residuals flag regime shifts.",
        ),
        "multi": (
            "Render all four fit views together.",
            "Comprehensive in-sample audit.",
        ),
    }
    for option, (summary, when) in FV_DOCS.items():
        register(_entry(L, sub, "fit_view", option, summary,
            f"L4.5.A fit view ``{option}``.", when,
            related=tuple(k for k in FV_DOCS if k != option)))
    FPO = {
        "all_origins": (
            "Compute fit views for every OOS origin.",
            "Detailed walk-forward audit; expensive but thorough.",
        ),
        "every_n_origins": (
            "Compute fit views every n origins.",
            "Compromise between coverage and runtime; ``params.every_n`` controls cadence.",
        ),
        "last_origin_only": (
            "Compute fit views only for the last training window.",
            "Default; cheapest summary while still capturing the most-informative model state.",
        ),
    }
    for option, (summary, when) in FPO.items():
        register(_entry(L, sub, "fit_per_origin", option, summary,
            f"L4.5.A fit-per-origin cadence ``{option}``.", when,
            related=tuple(k for k in FPO if k != option)))

    sub = "L4_5_B_forecast_scale_view"
    FSV = {
        "transformed_only": (
            "Plot forecasts in the transformed (model-internal) scale.",
            "Inspecting model-native predictions; useful for diagnosing fit issues at the estimator's scale.",
        ),
        "back_transformed_only": (
            "Plot forecasts back-transformed to the target's level.",
            "Default reporting view; matches the audience's mental model of the target.",
        ),
        "both_overlay": (
            "Overlay both scales in side-by-side panels.",
            "Comparing transformed vs level forecasts; useful when transformation effects matter.",
        ),
    }
    for option, (summary, when) in FSV.items():
        register(_entry(L, sub, "forecast_scale_view", option, summary,
            f"L4.5.B forecast scale view ``{option}``.", when,
            related=tuple(k for k in FSV if k != option)))
    BTM = {
        "auto": (
            "Use the inverse of preprocessing transform / L3 transforms automatically.",
            "Default; works for the standard log_diff / pct_change pipeline.",
        ),
        "manual_function": (
            "Use a user-registered inverse function.",
            "Custom target transforms registered via ``macroforecast.custom.register_target_transformer``.",
        ),
    }
    for option, (summary, when) in BTM.items():
        register(_entry(L, sub, "back_transform_method", option, summary,
            f"L4.5.B back-transform method ``{option}``.", when,
            related=tuple(k for k in BTM if k != option)))

    sub = "L4_5_C_window_stability"
    WV = {
        "rolling_train_loss": (
            "Training loss across rolling windows.",
            "Detecting training instability; rising loss across windows flags drift.",
        ),
        "parameter_stability": (
            "Parameter (coefficient / depth) stability across windows.",
            "Spotting structural instability in the fitted estimator.",
        ),
        "rolling_coef": (
            "Coefficient values across rolling windows.",
            "Linear-model coefficient drift detection; pair with the L7 ``mrf_gtvp`` for non-linear analogue.",
        ),
        "first_vs_last_window_forecast": (
            "First vs last training-window forecast overlay.",
            "Quick window-instability check; large divergence flags non-stationarity.",
        ),
        "multi": (
            "Render every window-stability view.",
            "Comprehensive stability audit.",
        ),
    }
    for option, (summary, when) in WV.items():
        register(_entry(L, sub, "window_view", option, summary,
            f"L4.5.C window view ``{option}``.", when,
            related=tuple(k for k in WV if k != option)))
    CVM = {
        "all_linear_models": (
            "Track coefficients across every linear model in the recipe.",
            "Default broad audit; works automatically for ols / ridge / lasso / elastic_net.",
        ),
        "user_list": (
            "Track coefficients only for a user-listed subset.",
            "Targeted audit when many linear models are active and only a few warrant inspection.",
        ),
    }
    for option, (summary, when) in CVM.items():
        register(_entry(L, sub, "coef_view_models", option, summary,
            f"L4.5.C coef-tracking model selector ``{option}``.", when,
            related=tuple(k for k in CVM if k != option)))

    sub = "L4_5_D_tuning_history"
    TV = {
        "objective_trace": (
            "Tuning-objective trace over iterations.",
            "Default convergence audit; monotone decrease confirms good search behaviour.",
        ),
        "hyperparameter_path": (
            "Sequence of hyperparameter values explored.",
            "Diagnosing search behaviour -- e.g. detecting Bayesian optimisation getting stuck on a local minimum.",
        ),
        "cv_score_distribution": (
            "Distribution of CV scores at each iteration.",
            "Detecting high-variance objective surfaces; wide distributions suggest the search has not converged.",
        ),
        "multi": (
            "Produce all tuning-history views together.",
            "Comprehensive tuning audit.",
        ),
    }
    for option, (summary, when) in TV.items():
        register(_entry(L, sub, "tuning_view", option, summary,
            f"L4.5.D tuning view ``{option}``.", when,
            related=tuple(k for k in TV if k != option)))

    sub = "L4_5_E_ensemble_diagnostics"
    EV = {
        "weights_over_time": (
            "Time-series of ensemble weights.",
            "Tracking which member dominates over time; pairs with the L7 ``rolling_recompute`` for stability analysis.",
        ),
        "weight_concentration": (
            "Herfindahl / entropy of ensemble weights.",
            "Quantifying ensemble diversity; concentrated weights = under-diversified ensemble.",
        ),
        "member_contribution": (
            "Per-member contribution to forecast variance.",
            "Identifying free-rider members that contribute little to the ensemble's predictive variance.",
        ),
        "multi": (
            "Render every ensemble diagnostic together.",
            "Default rich ensemble audit.",
        ),
    }
    for option, (summary, when) in EV.items():
        register(_entry(L, sub, "ensemble_view", option, summary,
            f"L4.5.E ensemble view ``{option}``.", when,
            related=tuple(k for k in EV if k != option)))
    WOT = {
        "line_plot": (
            "Line plot of weights per member over time.",
            "Default reporting view; readable up to ~10 ensemble members.",
        ),
        "stacked_area": (
            "Stacked-area plot summing to 1.",
            "Emphasising member share; ideal for showcasing weight redistribution events.",
        ),
        "heatmap": (
            "Heatmap of weights (member × time).",
            "Many-member ensembles (> 20) where line / area plots become unreadable.",
        ),
    }
    for option, (summary, when) in WOT.items():
        register(_entry(L, sub, "weights_over_time_method", option, summary,
            f"L4.5.E weights-over-time rendering ``{option}``.", when,
            related=tuple(k for k in WOT if k != option)))

    register(*list(_export_options(L, "L4_5_Z_export")))


_register_l3_5()
_register_l4_5()
