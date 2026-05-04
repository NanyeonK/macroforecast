from __future__ import annotations

from dataclasses import replace

from .registry import _OPS, register_op
from ..types import L3FeaturesArtifact, L3MetadataArtifact, L4ForecastsArtifact, L4ModelArtifactsArtifact, L5EvaluationArtifact, L6TestsArtifact, L7ImportanceArtifact, L7TransformationAttributionArtifact

FIGURE_TYPES = (
    "bar_global", "bar_grouped", "bar_grouped_by_pipeline", "beeswarm", "force_plot", "pdp_line", "ale_line", "heatmap",
    "feature_heatmap_over_time", "attribution_heatmap", "inclusion_heatmap", "pip_bar", "shapley_waterfall",
    "shap_dependence_scatter", "importance_by_horizon_bar", "lasso_path_inclusion_order", "irf_with_confidence_band",
    "historical_decomp_stacked_bar",
)

def _mccracken_ng_md_groups() -> dict[str, tuple[str, ...]]:
    """McCracken & Ng (2016) FRED-MD canonical 8-group taxonomy.

    Issue #260 -- complete the column membership instead of leaving it empty.
    Source: McCracken & Ng (2016) "FRED-MD: A Monthly Database for
    Macroeconomic Research" appendix Table B.1 (column ``group``).
    """

    return {
        "output_and_income": ("RPI", "W875RX1", "INDPRO", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "CUMFNS"),
        "labor_market": ("UNRATE", "UEMPMEAN", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "CLAIMSx", "PAYEMS", "USGOOD", "CES1021000001", "USCONS", "MANEMP", "DMANEMP", "NDMANEMP", "SRVPRD", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "CES0600000007", "AWOTMAN", "AWHMAN", "CES0600000008", "CES2000000008", "CES3000000008", "HOABS"),
        "housing": ("HOUST", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMIT", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW"),
        "consumption_orders_inventories": ("DPCERA3M086SBEA", "CMRMTSPLx", "RETAILx", "ACOGNO", "AMDMNOx", "ANDENOx", "AMDMUOx", "BUSINVx", "ISRATIOx", "UMCSENTx"),
        "money_and_credit": ("M1SL", "M2SL", "M2REAL", "BOGMBASE", "TOTRESNS", "NONBORRES", "BUSLOANS", "REALLN", "NONREVSL", "CONSPI", "S&P 500", "S&P: indust", "S&P div yield", "S&P PE ratio"),
        "interest_and_exchange_rates": ("FEDFUNDS", "CP3Mx", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA", "COMPAPFFx", "TB3SMFFM", "TB6SMFFM", "T1YFFM", "T5YFFM", "T10YFFM", "AAAFFM", "BAAFFM", "TWEXAFEGSMTHx", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx"),
        "prices": ("WPSFD49207", "WPSFD49502", "WPSID61", "WPSID62", "OILPRICEx", "PPICMM", "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAD", "CUSR0000SAS", "CPIULFSL", "CUSR0000SA0L2", "CUSR0000SA0L5", "PCEPI", "DDURRG3M086SBEA", "DNDGRG3M086SBEA", "DSERRG3M086SBEA"),
        "stock_market": ("S&P 500", "S&P: indust", "S&P div yield", "S&P PE ratio", "VIXCLSx"),
    }


def _mccracken_ng_qd_groups() -> dict[str, tuple[str, ...]]:
    """McCracken & Ng (2020) FRED-QD canonical 14-group taxonomy.

    Issue #260. Source: McCracken & Ng (2020) FRED-QD appendix Table B.1.
    """

    return {
        "nipa": ("GDPC1", "PCECC96", "PCDGx", "PCESVx", "PCNDx", "GPDIC1", "FPIx", "Y033RC1Q027SBEAx", "PNFIx", "PRFIx", "A014RE1Q156NBEA", "GCEC1", "A823RL1Q225SBEA", "FGRECPTx", "SLCEx", "EXPGSC1", "IMPGSC1"),
        "industrial_production": ("INDPRO", "IPFPNSS", "IPFINAL", "IPCONGD", "IPDCONGD", "IPNCONGD", "IPBUSEQ", "IPMAT", "IPDMAT", "IPNMAT", "IPMANSICS", "IPB51222S", "IPFUELS", "CUMFNS"),
        "employment_unemployment": ("UNRATE", "PAYEMS", "USGOOD", "USCONS", "MANEMP", "USTPU", "USWTRADE", "USTRADE", "USFIRE", "USGOVT", "USEHS", "USPBS", "USSERV", "USMINE", "CE16OV", "CIVPART", "UEMPMEAN", "UEMPLT5", "UEMP5TO14", "UEMP15OV", "UEMP15T26", "UEMP27OV", "CLAIMSx"),
        "housing": ("HOUST", "HOUSTNE", "HOUSTMW", "HOUSTS", "HOUSTW", "PERMIT", "PERMITNE", "PERMITMW", "PERMITS", "PERMITW"),
        "inventories_orders_sales": ("CMRMTSPLx", "RSAFSx", "AMDMNOx", "AMDMUOx", "BUSINVx", "ISRATIOx"),
        "prices": ("WPSFD49207", "PPIACO", "WPSID61", "OILPRICEx", "CPIAUCSL", "CPIAPPSL", "CPITRNSL", "CPIMEDSL", "CUSR0000SAC", "CUSR0000SAS", "CPIULFSL", "PCEPI", "GDPCTPI"),
        "earnings_productivity": ("CES0600000008", "CES2000000008", "CES3000000008", "COMPRMS", "ULCBS", "ULCMFG", "PRS84006221", "PRS85006023", "PRS85006221", "OPHNFB"),
        "interest_rates": ("FEDFUNDS", "TB3MS", "TB6MS", "GS1", "GS5", "GS10", "AAA", "BAA"),
        "money_credit": ("M1SL", "M2SL", "M2REAL", "BOGMBASE", "TOTRESNS", "NONBORRES", "BUSLOANS", "REALLN", "NONREVSL"),
        "household_balance_sheets": ("TABSHNOx", "TLBSHNOx", "TFAABSHNOx", "VANRRESHNOx", "TARESAx", "HNOREMQ027Sx", "OEHRENWBSHNOx", "TVCKSHNOx"),
        "exchange_rates": ("TWEXAFEGSMTHx", "EXSZUSx", "EXJPUSx", "EXUSUKx", "EXCAUSx"),
        "stock_markets": ("S&P 500", "S&P: indust", "S&P div yield", "S&P PE ratio"),
        "non_household_balance_sheets": ("TABSNNCBx", "TLBSNNCBx"),
        "other": ("UMCSENTx", "VIXCLSx"),
    }


def _fred_sd_states_block() -> tuple[str, ...]:
    """All 50 US state postal codes + DC, used by FRED-SD geographic
    visualisation. Issue #260."""

    return (
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC",
    )


PRE_DEFINED_BLOCKS = {
    # Issue #260: backfilled from McCracken & Ng (2016) FRED-MD Table B.1 +
    # (2020) FRED-QD Table B.1 + Fed economist documentation.
    "mccracken_ng_md_groups": _mccracken_ng_md_groups(),
    "mccracken_ng_qd_groups": _mccracken_ng_qd_groups(),
    "fred_sd_states": _fred_sd_states_block(),
    "nber_real_activity": ("INDPRO", "PAYEMS", "RPI", "CMRMTSPL", "CMRMTSPLx"),
    "taylor_rule_block": ("CPIAUCSL", "GDPC1", "FEDFUNDS", "UNRATE", "PCEPI"),
    "term_structure_block": ("TB3MS", "TB6MS", "GS1", "GS5", "GS10", "T1YFFM", "T5YFFM", "T10YFFM"),
    "credit_spread_block": ("BAA", "AAA", "BAAFFM", "AAAFFM", "TB3SMFFM", "TB6SMFFM"),
    "financial_conditions_block": ("NFCI", "VIXCLSx", "S&P 500", "TB3SMFFM", "BAAFFM"),
}

DEFAULT_FIGURE_MAPPING = {
    "permutation_importance": "bar_global",
    "lofo": "bar_global",
    "model_native_linear_coef": "bar_global",
    "model_native_tree_importance": "bar_global",
    "mrf_gtvp": "feature_heatmap_over_time",
    "shap_tree": ["beeswarm", "force_plot"],
    "shap_kernel": ["beeswarm", "force_plot"],
    "shap_linear": ["beeswarm", "force_plot"],
    "shap_deep": ["beeswarm", "force_plot"],
    "shap_interaction": "heatmap",
    "partial_dependence": "pdp_line",
    "accumulated_local_effect": "ale_line",
    "friedman_h_interaction": "heatmap",
    "integrated_gradients": "attribution_heatmap",
    "saliency_map": "attribution_heatmap",
    "deep_lift": "attribution_heatmap",
    "gradient_shap": "attribution_heatmap",
    "lasso_inclusion_frequency": "inclusion_heatmap",
    "bvar_pip": "pip_bar",
    "cumulative_r2_contribution": "bar_global",
    "fevd": "historical_decomp_stacked_bar",
    "historical_decomposition": "historical_decomp_stacked_bar",
    "generalized_irf": "irf_with_confidence_band",
    "forecast_decomposition": "historical_decomp_stacked_bar",
    "group_aggregate": "bar_grouped",
    "lineage_attribution": "bar_grouped_by_pipeline",
    "rolling_recompute": "feature_heatmap_over_time",
    "bootstrap_jackknife": "bar_global",
    "transformation_attribution": "shapley_waterfall",
}

# Ops whose v0.1 runtime did not faithfully implement the design's named
# procedure. Demoted to ``future`` by PR-C of the v0.1 honesty pass; the
# OpSpec status causes the L7 layer validator (and the universal op-status
# rule in :func:`macrocast.core.validator`) to hard-reject these at recipe
# validation time. Real implementations land per-op via the v0.2 issue
# tracker; see ``plans/design/part3_l5_l6_l7_l8.md`` for the gap.
#
# - fevd / historical_decomposition / generalized_irf: returned a flat
#   coefficient mean (for VAR fits) or fell back to ``tree_importance``
#   for non-VAR models. Real Cholesky / generalized-Pesaran-Shin
#   orthogonalised IRF + variance-decomposition output is missing.
# - mrf_gtvp: returned ``RandomForestRegressor.feature_importances_``
#   (a single static ranking) instead of a Coulombe (2024) GTVP
#   coefficient time series.
# - lasso_inclusion_frequency: returned ``(|coef| > 1e-9).astype(float)``
#   from a single fit. Real frequency requires resampling (rolling /
#   bootstrap / sub-sample lasso path).
# - accumulated_local_effect: bin endpoint prediction-difference sum.
#   Real ALE per Apley & Zhu (2020) needs centred local effects via
#   derivative integration.
# - friedman_h_interaction: variance-ratio surrogate. Real Friedman &
#   Popescu (2008) H statistic uses bivariate vs marginal partial
#   dependence ratios.
# - gradient_shap / integrated_gradients / saliency_map / deep_lift:
#   gradient-based attributions. v0.1 falls back to a SHAP proxy --
#   different attribution method, not the gradient-based one named.
HONESTY_DEMOTED_L7_OPS: tuple[str, ...] = ()
# Every honesty-pass-demoted L7 op has been re-promoted in v0.2:
# - #189: fevd / historical_decomposition / generalized_irf
# - #190: mrf_gtvp
# - #191: lasso_inclusion_frequency
# - #192: accumulated_local_effect
# - #193: friedman_h_interaction
# - #194: gradient_shap / integrated_gradients / saliency_map / deep_lift
#         (operational when ``[deep]`` extra installed; runtime raises
#         NotImplementedError otherwise -- mirrors lstm/gru/transformer
#         torch-missing pattern).

OPERATIONAL_OPS = tuple(name for name in DEFAULT_FIGURE_MAPPING if name not in HONESTY_DEMOTED_L7_OPS)
FUTURE_OPS = HONESTY_DEMOTED_L7_OPS + (
    "attention_weights",
    "lstm_hidden_state",
    "boruta_selection",
    "recursive_feature_elimination",
    "lasso_path_selection",
    "stability_selection",
)


def _stub(name: str):
    """Forward to :func:`macrocast.core.runtime._execute_l7_step` so the L7
    importance ops registered here actually compute (rather than raise)."""

    def run(inputs, params):
        from ..runtime import _execute_l7_step
        from ..types import L3FeaturesArtifact, L3MetadataArtifact, L5EvaluationArtifact

        l3_features = next((item for item in inputs if isinstance(item, L3FeaturesArtifact)), None)
        l3_metadata = next((item for item in inputs if isinstance(item, L3MetadataArtifact)), None)
        l5_eval = next((item for item in inputs if isinstance(item, L5EvaluationArtifact)), None)
        if l3_features is None or l3_metadata is None or l5_eval is None:
            # Fall back to a structured payload if upstream context is incomplete.
            return {"op": name, "inputs": list(inputs), "params": dict(params)}
        return _execute_l7_step(name, list(inputs), dict(params), l3_features, l3_metadata, l5_eval)

    run.__name__ = name
    return run


def _schema(name: str) -> dict[str, dict]:
    if name == "group_aggregate":
        return {"grouping": {"options": tuple(PRE_DEFINED_BLOCKS) + ("user_defined",)}, "aggregation": {"options": ("sum", "mean", "max_abs", "signed_sum"), "default": "sum"}}
    if name == "lineage_attribution":
        return {"level": {"options": ("pipeline_name", "step_op", "source_node"), "default": "pipeline_name"}, "aggregation": {"options": ("sum", "mean", "max_abs", "signed_sum"), "default": "sum"}}
    if name == "rolling_recompute":
        return {"window": {"options": ("expanding", "rolling"), "default": "expanding"}, "step_size": {"default": 1}, "recompute_step": {"default": "shap_tree"}}
    if name == "transformation_attribution":
        return {"decomposition_method": {"options": ("shapley_over_pipelines", "marginal_addition", "leave_one_out_pipeline"), "default": "shapley_over_pipelines"}}
    return {}


for _name in OPERATIONAL_OPS:
    _output = L7TransformationAttributionArtifact if _name == "transformation_attribution" else L7ImportanceArtifact
    register_op(
        name=_name,
        layer_scope=("l7",),
        input_types={"default": (L4ModelArtifactsArtifact, L4ForecastsArtifact, L3FeaturesArtifact, L3MetadataArtifact, L5EvaluationArtifact, L6TestsArtifact, L7ImportanceArtifact)},
        output_type=_output,
        params_schema=_schema(_name),
        default_figure_type=DEFAULT_FIGURE_MAPPING[_name],
    )(_stub(_name))

# v0.1 honesty pass: register the 11 demoted ops with the same input/output
# contract as the operational ops so existing recipes pretty-print, but
# carry ``status="future"`` so the validator hard-rejects them.
for _name in HONESTY_DEMOTED_L7_OPS:
    register_op(
        name=_name,
        layer_scope=("l7",),
        input_types={"default": (L4ModelArtifactsArtifact, L4ForecastsArtifact, L3FeaturesArtifact, L3MetadataArtifact, L5EvaluationArtifact, L6TestsArtifact, L7ImportanceArtifact)},
        output_type=L7ImportanceArtifact,
        params_schema=_schema(_name),
        default_figure_type=DEFAULT_FIGURE_MAPPING[_name],
        status="future",
    )(_stub(_name))

# Tail: design-future ops that were never operational.
for _name in FUTURE_OPS:
    if _name in HONESTY_DEMOTED_L7_OPS:
        continue  # already registered above
    if _name in _OPS:
        spec = _OPS[_name]
        scope = spec.layer_scope if isinstance(spec.layer_scope, tuple) else ()
        if "l7" not in scope:
            _OPS[_name] = replace(spec, layer_scope=tuple(scope) + ("l7",))
    else:
        register_op(name=_name, layer_scope=("l7",), input_types={"default": L7ImportanceArtifact}, output_type=L7ImportanceArtifact, status="future")(_stub(_name))
