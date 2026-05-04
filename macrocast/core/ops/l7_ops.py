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

PRE_DEFINED_BLOCKS = {
    "mccracken_ng_md_groups": (),
    "mccracken_ng_qd_groups": (),
    "fred_sd_states": (),
    "nber_real_activity": ("INDPRO", "PAYEMS", "RPI", "CMRMTSPL"),
    "taylor_rule_block": ("CPIAUCSL", "GDPC1", "FEDFUNDS"),
    "term_structure_block": ("TB3MS", "GS1", "GS5", "GS10", "T10Y3M"),
    "credit_spread_block": ("BAA", "AAA", "BAAFFM", "AAAFFM"),
    "financial_conditions_block": ("NFCI",),
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
HONESTY_DEMOTED_L7_OPS: tuple[str, ...] = (
    # v0.2 promoted: fevd / historical_decomposition / generalized_irf (#189),
    # mrf_gtvp (#190), lasso_inclusion_frequency (#191),
    # accumulated_local_effect (#192), friedman_h_interaction (#193).
    # The remaining 4 are the gradient-based attributions tracked by #194.
    "gradient_shap",
    "integrated_gradients",
    "saliency_map",
    "deep_lift",
)

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
