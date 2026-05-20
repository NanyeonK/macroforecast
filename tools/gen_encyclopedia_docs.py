"""Introspection-based encyclopedia page generator for macroforecast.

Produces one .md file per registered op (or per virtual op for L5/L6) into
``docs/encyclopedia/<layer>/<subdir>/<op_name>.md``, mirroring the existing
per-op page structure exactly.

Reuses patterns from ``tools/gen_standalone_docs.py`` (Cycle 41): same
sys.path bootstrap, same ``inspect.signature`` extraction, same forward-ref
quote stripping.

Usage::

    # Dry-run all layers (print paths + first 5 lines, no writes):
    python tools/gen_encyclopedia_docs.py --layer all --dry-run

    # Generate L3 into docs/encyclopedia/ with a fixed date for idempotency:
    python tools/gen_encyclopedia_docs.py --layer L3 --review-date 2026-05-21

    # Diff L7 output against existing pages without writing:
    python tools/gen_encyclopedia_docs.py --layer L7 --diff-against docs/encyclopedia/

    # Generate a single layer into a temp directory:
    python tools/gen_encyclopedia_docs.py --layer L4 --out /tmp/enc_l4 --force
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import inspect
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Step 1.1 — Resolve repo root and insert into sys.path
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Step 1.3 — Import all ops modules to populate the registry
# (deferred until main() to allow --help without triggering imports)
# ---------------------------------------------------------------------------


def _bootstrap_ops() -> None:
    """Import all ops modules so the registry is fully populated."""
    import macroforecast.core.ops.l3_ops  # noqa: F401
    import macroforecast.core.ops.l4_ops  # noqa: F401
    import macroforecast.core.ops.l5_ops  # noqa: F401
    import macroforecast.core.ops.l6_ops  # noqa: F401
    import macroforecast.core.ops.l7_ops  # noqa: F401
    import macroforecast.core.ops.l8_ops  # noqa: F401
    import macroforecast.core.ops.diagnostic_ops  # noqa: F401
    import macroforecast.core.ops.universal  # noqa: F401


# ---------------------------------------------------------------------------
# Step 2.2 — CLI layer normalization table
# ---------------------------------------------------------------------------

LAYER_CLI_TO_INTERNAL: dict[str, str] = {
    "L0": "l0",
    "L1": "l1",
    "L1.5": "l1_5",
    "L2": "l2",
    "L2.5": "l2_5",
    "L3": "l3",
    "L3.5": "l3_5",
    "L4": "l4",
    "L4.5": "l4_5",
    "L5": "l5",
    "L6": "l6",
    "L7": "l7",
    "L8": "l8",
}

ALL_LAYERS: tuple[str, ...] = (
    "l0", "l1", "l1_5", "l2", "l2_5", "l3", "l3_5",
    "l4", "l4_5", "l5", "l6", "l7", "l8",
)

# Human-readable layer labels used in page content
LAYER_LABEL: dict[str, str] = {
    "l0": "L0", "l1": "L1", "l1_5": "L1.5",
    "l2": "L2", "l2_5": "L2.5", "l3": "L3", "l3_5": "L3.5",
    "l4": "L4", "l4_5": "L4.5", "l5": "L5",
    "l6": "L6", "l7": "L7", "l8": "L8",
}

# ---------------------------------------------------------------------------
# Step 4.1 — Layer output map
# Determines the subdirectory rule for each layer.
# None = no per-op pages for this layer.
# "__axis_subdir__" = look up axis via OP_TO_AXIS_SUBDIR.
# "diagnostic_ops" = half-layer diagnostic ops subdir.
# ---------------------------------------------------------------------------

LAYER_OUTPUT_MAP: dict[str, str | None] = {
    # Layers with no per-op pages (axes only):
    "l0": None,
    "l1": None,
    "l8": None,
    # Layers with a single op subdir:
    "l3": "op",
    "l7": "op",
    # L4 uses virtual family-based ops (not registry ops):
    "l4": "__family__",
    # Layers with axis-based subdirs:
    "l2": "__axis_subdir__",
    "l5": "__virtual__",  # L5 metrics are virtual ops not in registry
    "l6": "__virtual__",  # L6 tests are virtual ops not in registry
    # Half-layers: diagnostic ops go into diagnostic_ops/
    "l1_5": "diagnostic_ops",
    "l2_5": "diagnostic_ops",
    "l3_5": "diagnostic_ops",
    "l4_5": "diagnostic_ops",
}

# ---------------------------------------------------------------------------
# Step 4.3 — OP_TO_AXIS_SUBDIR mapping
# Maps op name to encyclopedia subdirectory for axis-based layers (L2, L5, L6).
# ---------------------------------------------------------------------------

OP_TO_AXIS_SUBDIR: dict[str, str] = {
    # L2 ops:
    "apply_official_tcode": "transform_policy",
    "no_transform": "transform_policy",
    "custom_tcode": "transform_policy",
    "diff": "transform_policy",
    "log": "transform_policy",
    "log_diff": "transform_policy",
    "pct_change": "transform_policy",
    "level": "transform_policy",
    "asymmetric_trim": "transform_policy",
    "mccracken_ng_iqr": "outlier_policy",
    "winsorize": "outlier_policy",
    "zscore_threshold": "outlier_policy",
    "em_factor": "imputation_policy",
    "em_multivariate": "imputation_policy",
    "forward_fill": "imputation_policy",
    "linear_interpolation": "imputation_policy",
    "mean": "imputation_policy",
    "drop_unbalanced_series": "frame_edge_policy",
    "truncate_to_balanced": "frame_edge_policy",
    "zero_fill_leading": "frame_edge_policy",
    "quarterly_average": "monthly_to_quarterly_rule",
    "step_backward": "quarterly_to_monthly_rule",
    "chow_lin_disaggregation": "quarterly_to_monthly_rule",
    # L5 ops (virtual — from OP_TO_STANDALONE keys that map to l5):
    "rmse": "point_metrics",
    "mse": "point_metrics",
    "mae": "point_metrics",
    "medae": "point_metrics",
    "mape": "point_metrics",
    "theil_u1": "point_metrics",
    "theil_u2": "point_metrics",
    "interval_score": "density_metrics",
    "coverage_rate": "density_metrics",
    "pesaran_timmermann_metric": "direction_metrics",
    "success_ratio": "direction_metrics",
    "mse_reduction": "relative_metrics",
    "r2_oos": "relative_metrics",
    "relative_mse": "relative_metrics",
    "relative_mae": "relative_metrics",
    # L6 ops (virtual — from OP_TO_STANDALONE keys that map to l6):
    "dm_diebold_mariano": "equal_predictive_test",
    "gw_giacomini_white": "equal_predictive_test",
    "dmp_multi_horizon": "equal_predictive_test",
    "harvey_newbold_encompassing": "equal_predictive_test",
    "clark_west": "nested_test",
    "enc_new": "nested_test",
    "enc_t": "nested_test",
}

# ---------------------------------------------------------------------------
# Step 5.1 — OP_TO_STANDALONE mapping
# Maps registry op name (or virtual op name for L5/L6) to the
# mf.functions.<callable> name. Ops absent from this dict are recipe-only.
# ---------------------------------------------------------------------------

OP_TO_STANDALONE: dict[str, str] = {
    # L3 ops:
    "lag": "lag_matrix",
    "seasonal_lag": "seasonal_lag_matrix",
    "ma_window": "ma_window_transform",
    "ma_increasing_order": "ma_increasing_order_transform",
    "diff": "diff_transform",
    "log": "log_transform",
    "log_diff": "log_diff_transform",
    "pct_change": "pct_change_transform",
    "cumsum": "cumsum_transform",
    "scale": "scale_transform",
    "hp_filter": "hp_filter_transform",
    "hamilton_filter": "hamilton_filter_transform",
    "savitzky_golay_filter": "savitzky_golay_transform",
    "polynomial": "polynomial_expansion_transform",
    "polynomial_expansion": "polynomial_expansion_transform",
    "interaction": "interaction_terms_transform",
    "pca": "pca_transform",
    "maf_per_variable_pca": "maf_per_variable_pca_transform",
    "adaptive_ma_rf": "adaptive_ma_rf_transform",
    "wavelet": "wavelet_transform",
    "fourier": "fourier_transform",
    "asymmetric_trim": "asymmetric_trim_transform",
    "season_dummy": "season_dummy_transform",
    "scaled_pca": "scaled_pca_transform",
    "supervised_pca": "supervised_pca_transform",
    "partial_least_squares": "partial_least_squares_transform",
    "sliced_inverse_regression": "sliced_inverse_regression_transform",
    "dfm": "dfm_transform",
    "feature_selection": "feature_selection_transform",
    "sparse_pca": "sparse_pca_transform",
    "sparse_pca_chen_rohe": "sparse_pca_chen_rohe_transform",
    "varimax": "varimax_transform",
    "random_projection": "random_projection_transform",
    "kernel_features": "kernel_features_transform",
    "nystroem": "nystroem_transform",
    "time_trend": "time_trend_transform",
    "holiday": "holiday_transform",
    # L2 ops:
    "apply_official_tcode": "apply_tcode_transform",
    "mccracken_ng_iqr": "iqr_outlier_clean",
    "zscore_threshold": "zscore_outlier_clean",
    "winsorize": "winsorize_clean",
    "em_factor": "em_factor_impute_clean",
    "em_multivariate": "em_multivariate_impute_clean",
    "mean": "mean_impute_clean",
    "forward_fill": "forward_fill_clean",
    "linear_interpolation": "linear_interpolate_clean",
    "truncate_to_balanced": "truncate_to_balanced_clean",
    "drop_unbalanced_series": "drop_unbalanced_series_clean",
    "zero_fill_leading": "zero_fill_leading_clean",
    "quarterly_average": "freq_align_monthly_to_quarterly_clean",
    "step_backward": "freq_align_quarterly_to_monthly_clean",
    # L4 virtual family ops (encyclopedia pages, not registry ops):
    "ar_p": "ar_fit",
    "ols": "ols_fit",
    "ridge": "ridge_fit",
    "lasso": "lasso_fit",
    "lasso_path": "lasso_path_fit",
    "elastic_net": "elastic_net_fit",
    "bayesian_ridge": "bayesian_ridge_fit",
    "huber": "huber_fit",
    "glmboost": "glmboost_fit",
    "random_forest": "random_forest_fit",
    "extra_trees": "extra_trees_fit",
    "gradient_boosting": "gradient_boosting_fit",
    "xgboost": "xgboost_fit",
    "lightgbm": "lightgbm_fit",
    "catboost": "catboost_fit",
    "mlp": "mlp_fit",
    "lstm": "lstm_fit",
    "gru": "gru_fit",
    "transformer": "transformer_fit",
    "var": "var_fit",
    "bvar_minnesota": "bvar_minnesota_fit",
    "bvar_normal_inverse_wishart": "bvar_niw_fit",
    "factor_augmented_ar": "far_fit",
    "principal_component_regression": "pcr_fit",
    "factor_augmented_var": "favar_fit",
    "knn": "knn_fit",
    "svr_linear": "svr_linear_fit",
    "svr_rbf": "svr_rbf_fit",
    "svr_poly": "svr_poly_fit",
    "kernel_ridge": "kernel_ridge_fit",
    "mars": "mars_fit",
    "garch11": "garch11_fit",
    "egarch": "egarch_fit",
    "realized_garch_with_rv_exog": "realized_garch_fit",
    "ets": "ets_fit",
    "theta_method": "theta_fit",
    "holt_winters": "holt_winters_fit",
    "dfm_mixed_mariano_murasawa": "dfm_fit",
    # L5 virtual metric ops:
    "rmse": "rmse",
    "mse": "mse",
    "mae": "mae",
    "medae": "medae",
    "mape": "mape",
    "mse_reduction": "mse_reduction",
    "r2_oos": "r2_oos",
    "relative_mse": "relative_mse",
    "relative_mae": "relative_mae",
    "interval_score": "interval_score",
    "coverage_rate": "coverage_rate",
    "success_ratio": "success_ratio",
    "pesaran_timmermann_metric": "pesaran_timmermann_metric",
    "theil_u1": "theil_u1",
    "theil_u2": "theil_u2",
    # L6 virtual test ops:
    "dm_diebold_mariano": "dm_test",
    "gw_giacomini_white": "gw_test",
    "dmp_multi_horizon": "dmp_test",
    "harvey_newbold_encompassing": "hn_test",
    "clark_west": "cw_test",
    "enc_new": "enc_new_test",
    "enc_t": "enc_t_test",
    # L7 ops:
    "model_native_linear_coef": "model_native_linear_coef_importance",
    "model_native_tree_importance": "model_native_tree_importance",
    "permutation_importance": "permutation_importance",
    "permutation_importance_strobl": "cond_permutation_importance",
    "cond_permutation_importance": "cond_permutation_importance",
    "partial_dependence": "partial_dependence_importance",
    "accumulated_local_effect": "ale_importance",
    "shap_tree": "shap_tree_importance",
    "shap_linear": "shap_linear_importance",
}

# ---------------------------------------------------------------------------
# Step 7 — EXCLUSION_LIST: internal pipeline ops excluded from page generation
# ---------------------------------------------------------------------------

EXCLUSION_LIST: set[str] = {
    # Universal ops (not layer-specific recipe options):
    "identity",
    "concat",
    "interact",
    "layer_meta_aggregate",
    "hierarchical_pca",  # universal, not layer-specific
    "simple_average",    # universal combiner
    "weighted_concat",   # universal combiner
    # L3 internal pipeline ops:
    "l3_feature_bundle",
    "l3_metadata_build",
    # L3 internal aliases and future ops excluded from page generation:
    "varimax_rotation",    # alias for varimax — deduplicate with varimax
    "nystroem_features",   # alias for nystroem — deduplicate with nystroem
    "u_midas",             # future internal alias
    "midas",               # future internal op
    "kernel",              # internal alias for kernel_features
    "polynomial",          # alias for polynomial_expansion; use polynomial_expansion
    # level: not in existing encyclopedia pages (no page generated)
    "level",
    # L4 internal pipeline ops:
    "fit_model",           # internal dispatch op — family pages are virtual
    "predict",             # internal prediction op
    "l4_model_artifacts_collect",
    "l4_training_metadata_build",
    # L4 family-level virtual ops — handled via __family__ subdir, not registry
    # (these are not in registry; listed here to prevent double-generation if
    # a future registry migration adds them)
    # L5 internal:
    "l5_collect_inputs",
    "blocked_oob_reality_check",
    "metric_compute",
    "benchmark_relative",
    "aggregate",
    "slice_and_decompose",
    "rank_and_report",
    # L6 internal:
    "l6_collect_inputs",
    "L6_A_equal_predictive",
    "L6_B_nested",
    "L6_C_cpa",
    "L6_D_multiple_model",
    "L6_E_density_interval",
    "L6_F_direction",
    "L6_G_residual",
    "multiple_model_test_step_m_romano_wolf",
    # L8 internal:
    "l8_artifact_granularity",
    "l8_collect_inputs",
    "l8_export_format",
    "l8_provenance",
    "l8_saved_objects",
    # Diagnostic collect ops (internal aggregators):
    "diagnostic_collect_l1",
    "diagnostic_collect_l2",
    "diagnostic_collect_l3",
    "diagnostic_collect_l4",
    # L4 future families with no standalone callable:
    # midas_almon, midas_beta, midas_step, dfm_unrestricted_midas, realized_garch
    # These are in FUTURE_MODEL_FAMILIES and handled by the family virtual-op path.
    # L3 future ops (future-status but still user-facing in some cases):
    "boruta_selection",          # future
    "genetic_algorithm_selection",  # future
    "lasso_path_selection",      # future
    "recursive_feature_elimination",  # future
    "stability_selection",       # future
    "target_construction",       # internal target construction
    "regime_indicator",          # internal regime feature
    # L7 ops that are in the registry but do NOT have encyclopedia pages yet.
    # These ops are operational/future but not yet documented in the per-op
    # encyclopedia surface. Generate pages only for the 8 ops that already
    # have pages in docs/encyclopedia/l7/op/.
    "attention_weights",
    "bootstrap_jackknife",
    "bvar_pip",
    "cumulative_r2_contribution",
    "deep_lift",
    "dual_decomposition",
    "fevd",
    "forecast_decomposition",
    "friedman_h_interaction",
    "generalized_irf",
    "gradient_shap",
    "group_aggregate",
    "historical_decomposition",
    "integrated_gradients",
    "lasso_inclusion_frequency",
    "lineage_attribution",
    "lofo",
    "lstm_hidden_state",
    "mrf_gtvp",
    "orthogonalised_irf",
    "oshapley_vi",
    "pbsv",
    "rolling_recompute",
    "saliency_map",
    "shap_deep",
    "shap_interaction",
    "shap_kernel",
    "transformation_attribution",
    # Also boruta_selection / lasso_path_selection / recursive_feature_elimination /
    # stability_selection appear in both L3 and L7 scope — they're future L3 ops
    # covered by the L3 exclusion list above; exclude their L7 occurrences too.
    # (They're already in EXCLUSION_LIST under L3 future ops section.)
}

# ---------------------------------------------------------------------------
# L2, L4, L5, and L6 virtual ops: these have encyclopedia pages but use
# different names from registry ops (or aren't in the registry at all).
# These are enumerated here directly, from OP_TO_AXIS_SUBDIR.
# ---------------------------------------------------------------------------

# L2 virtual ops: the encyclopedia page names for L2 ops
# These come from OP_TO_AXIS_SUBDIR keys that map to L2 subdirs
L2_VIRTUAL_OPS_ORDERED: list[tuple[str, str]] = [
    # (op_name, status) — status is operational unless known future
    ("apply_official_tcode", "operational"),
    ("mccracken_ng_iqr", "operational"),
    ("winsorize", "operational"),
    ("zscore_threshold", "operational"),
    ("em_factor", "operational"),
    ("em_multivariate", "operational"),
    ("forward_fill", "operational"),
    ("linear_interpolation", "operational"),
    ("mean", "operational"),
    ("drop_unbalanced_series", "operational"),
    ("truncate_to_balanced", "operational"),
    ("zero_fill_leading", "operational"),
    ("quarterly_average", "operational"),
    ("step_backward", "operational"),
    ("chow_lin_disaggregation", "future"),
]

# L5 virtual ops: each is a standalone callable serving as a recipe axis value
L5_VIRTUAL_OPS: dict[str, str] = {
    # op_name -> status
    "rmse": "operational",
    "mse": "operational",
    "mae": "operational",
    "medae": "operational",
    "mape": "operational",
    "theil_u1": "operational",
    "theil_u2": "operational",
    "interval_score": "operational",
    "coverage_rate": "operational",
    "pesaran_timmermann_metric": "operational",
    "success_ratio": "operational",
    "mse_reduction": "operational",
    "r2_oos": "operational",
    "relative_mse": "operational",
    "relative_mae": "operational",
}

# L5 virtual op sub-layer labels
L5_AXIS_TO_SUBLAYER: dict[str, str] = {
    "point_metrics": "L5_A_metric_specification",
    "density_metrics": "L5_A_metric_specification",
    "direction_metrics": "L5_A_metric_specification",
    "relative_metrics": "L5_A_metric_specification",
}

# L6 virtual ops: each corresponds to a standalone test callable
L6_VIRTUAL_OPS: dict[str, str] = {
    "dm_diebold_mariano": "operational",
    "gw_giacomini_white": "operational",
    "dmp_multi_horizon": "operational",
    "harvey_newbold_encompassing": "operational",
    "clark_west": "operational",
    "enc_new": "operational",
    "enc_t": "operational",
}

# L6 virtual op sub-layer labels
L6_AXIS_TO_SUBLAYER: dict[str, str] = {
    "equal_predictive_test": "L6_A_equal_predictive",
    "nested_test": "L6_B_nested",
}

# L4 virtual family ops — from OPERATIONAL_MODEL_FAMILIES + FUTURE_MODEL_FAMILIES
# Sub-layer label for L4 family pages
L4_SUBLAYER = "L4_A_model_selection"

# Sub-layer labels for L7 ops
L7_SUBLAYER = "L7_A_importance_dag_body"

# Sub-layer labels for L2 ops
L2_AXIS_TO_SUBLAYER: dict[str, str] = {
    "transform_policy": "l2_b",
    "outlier_policy": "l2_c",
    "imputation_policy": "l2_d",
    "frame_edge_policy": "l2_e",
    "monthly_to_quarterly_rule": "l2_a",
    "quarterly_to_monthly_rule": "l2_a",
}

# Sub-layer label for L3 ops
L3_SUBLAYER = "L3_A_step_op"

# Sub-layer labels for diagnostic half-layer ops
HALF_LAYER_SUBLABEL: dict[str, str] = {
    "l1_5": "L1.5_diagnostic",
    "l2_5": "L2.5_diagnostic",
    "l3_5": "L3.5_diagnostic",
    "l4_5": "L4.5_diagnostic",
}

# Axis key used in recipe YAML for each layer/subdir
AXIS_KEY_FOR_LAYER: dict[str, dict[str, str]] = {
    "l3": {"op": "op"},
    "l4": {"family": "family"},
    "l5": {
        "point_metrics": "point_metrics",
        "density_metrics": "density_metrics",
        "direction_metrics": "direction_metrics",
        "relative_metrics": "relative_metrics",
    },
    "l6": {
        "equal_predictive_test": "equal_predictive_test",
        "nested_test": "nested_test",
    },
    "l7": {"op": "op"},
    "l2": {
        "transform_policy": "transform_policy",
        "outlier_policy": "outlier_policy",
        "imputation_policy": "imputation_policy",
        "frame_edge_policy": "frame_edge_policy",
        "monthly_to_quarterly_rule": "monthly_to_quarterly_rule",
        "quarterly_to_monthly_rule": "quarterly_to_monthly_rule",
    },
}


# ---------------------------------------------------------------------------
# Helper: extract first non-empty line of docstring
# ---------------------------------------------------------------------------


def first_doc_line(obj: Any) -> str:
    """Return the first non-empty line of obj's docstring, or empty string."""
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def strip_quotes(s: str) -> str:
    """Strip forward-reference single quotes from a type annotation string."""
    return s.replace("'", "")


def render_sig_line(callable_name: str, obj: Any) -> str:
    """Render a multi-line function signature block for the .md page."""
    sig = inspect.signature(obj)
    params = list(sig.parameters.values())
    return_anno = sig.return_annotation

    # Render return annotation
    if return_anno is inspect.Parameter.empty:
        ret_str = ""
    elif hasattr(return_anno, "__name__"):
        ret_str = return_anno.__name__
    else:
        ret_str = strip_quotes(str(return_anno))

    # Render parameters, one per line, stripped of forward-ref quotes
    param_lines = []
    for p in params:
        anno_str = ""
        if p.annotation is not inspect.Parameter.empty:
            anno = p.annotation
            anno_str = f": {strip_quotes(str(anno) if not hasattr(anno, '__name__') else anno.__name__)}"
        default_str = ""
        if p.default is not inspect.Parameter.empty:
            default_str = f" = {p.default!r}"
        param_lines.append(f"    {p.name}{anno_str}{default_str},")

    params_block = "\n".join(param_lines)
    arrow = f" -> {ret_str}" if ret_str else ""
    return (
        f"```python\n"
        f"mf.functions.{callable_name}(\n"
        f"{params_block}\n"
        f"){arrow}\n"
        f"```"
    )


def render_params_table(obj: Any) -> str:
    """Render the parameters markdown table from inspect.signature."""
    sig = inspect.signature(obj)
    doc = inspect.getdoc(obj) or ""
    param_descs = _extract_param_descriptions(doc)

    rows = []
    for p in sig.parameters.values():
        anno = p.annotation
        if anno is inspect.Parameter.empty:
            type_str = "—"
        elif hasattr(anno, "__name__"):
            type_str = f"`{anno.__name__}`"
        else:
            type_str = f"`{strip_quotes(str(anno))}`"

        default = p.default
        if default is inspect.Parameter.empty:
            default_str = "—"
        else:
            default_str = f"`{default!r}`"

        desc = param_descs.get(p.name, "—")
        rows.append(f"| `{p.name}` | {type_str} | {default_str} | — | {desc} |")

    header = "| name | type | default | constraint | description |\n|---|---|---|---|---|"
    return header + "\n" + "\n".join(rows)


def _extract_param_descriptions(doc: str) -> dict[str, str]:
    """Best-effort extraction of parameter descriptions from a numpy/Google docstring."""
    descriptions: dict[str, str] = {}
    in_params = False
    current_param: str | None = None

    for line in doc.splitlines():
        stripped = line.strip()
        # Detect "Parameters" section header
        if stripped in ("Parameters", "Parameters:", "Args", "Args:"):
            in_params = True
            continue
        # Detect next section header (exits params section)
        if in_params and stripped and not line.startswith(" ") and not line.startswith("\t"):
            if stripped.endswith(":") and not stripped.startswith("|"):
                in_params = False
                continue
        if not in_params:
            continue

        # Detect parameter name lines: "    param_name : type" or "    param_name (type):"
        if line.startswith("    ") and not line.startswith("        "):
            parts = stripped.split(":", 1)
            if len(parts) >= 1:
                param_candidate = parts[0].split("(")[0].strip()
                # Simple heuristic: param name is a valid Python identifier
                if param_candidate.isidentifier():
                    current_param = param_candidate
                    # Description may start on same line after colon
                    if len(parts) > 1:
                        desc_start = parts[1].strip()
                        if desc_start:
                            descriptions[current_param] = desc_start
                    continue
        # Continuation lines (indented more deeply) append to current param description
        if current_param and line.startswith("        "):
            existing = descriptions.get(current_param, "")
            if existing:
                descriptions[current_param] = existing + " " + stripped
            else:
                descriptions[current_param] = stripped

    return descriptions


def get_return_annotation_str(obj: Any) -> str:
    """Return a human-readable return annotation string."""
    sig = inspect.signature(obj)
    ret = sig.return_annotation
    if ret is inspect.Parameter.empty:
        return "—"
    if hasattr(ret, "__name__"):
        return ret.__name__
    return strip_quotes(str(ret))


def extract_references(doc: str) -> list[str]:
    """Extract lines from a 'References' section in the docstring."""
    refs: list[str] = []
    in_refs = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("references"):
            in_refs = True
            continue
        if in_refs:
            if stripped == "" and refs:
                # Blank line after first reference ends section
                break
            if stripped.startswith("*"):
                refs.append(stripped)
    return refs


def get_behavior_paragraph(doc: str) -> str:
    """Extract the first paragraph of the docstring (behavior block)."""
    lines = doc.splitlines()
    # Skip the first non-empty line (one-liner description already used)
    started = False
    para_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not started:
            if stripped:
                started = True
            continue
        # After skipping first line, collect until we find a section header
        if stripped.lower() in ("parameters", "parameters:", "args", "args:",
                                 "returns", "returns:", "references", "references:",
                                 "examples", "examples:", "notes", "notes:"):
            break
        para_lines.append(line)

    # Trim trailing blank lines
    while para_lines and not para_lines[-1].strip():
        para_lines.pop()

    if not para_lines:
        return "(See standalone callable docstring.)"

    return "\n".join(para_lines).strip()


# ---------------------------------------------------------------------------
# Core page renderer
# ---------------------------------------------------------------------------


class VirtualOpSpec:
    """Lightweight stand-in for OpSpec used with virtual (non-registry) ops."""

    def __init__(
        self,
        name: str,
        layer_id: str,
        status: str,
        sublayer: str,
    ) -> None:
        self.name = name
        self.layer_id = layer_id
        self.status = status
        self.sublayer = sublayer


def render_page(
    op_name: str,
    layer_id: str,
    subdir: str,
    status: str,
    sublayer: str,
    review_date: str,
    callable_name: str | None,
    mf_functions: Any,
    axis_key: str,
) -> str:
    """Render the full .md content for one op encyclopedia page.

    Parameters
    ----------
    op_name : str
        Registry or virtual op name (e.g., "rmse", "ar_p", "lag").
    layer_id : str
        Internal layer ID (e.g., "l3", "l5").
    subdir : str
        Subdirectory under the layer dir (e.g., "op", "point_metrics").
    status : str
        "operational" or "future".
    sublayer : str
        Sub-layer label for the blockquote line.
    review_date : str
        Date string for the footer.
    callable_name : str or None
        Name of the mf.functions callable, or None if recipe-only.
    mf_functions : module
        The macroforecast.functions module (passed to avoid re-import cost).
    axis_key : str
        The YAML key for the recipe context section (e.g., "op", "family",
        "point_metrics").
    """
    layer_label = LAYER_LABEL[layer_id]

    # Resolve callable object
    callable_obj = None
    if callable_name:
        callable_obj = getattr(mf_functions, callable_name, None)

    # --- Header line ---
    if callable_obj is not None:
        one_liner = first_doc_line(callable_obj)
    else:
        one_liner = f"{op_name} op."

    # --- Back-links ---
    # For ops with an axis subdir, link back to axes/<subdir>.md and index.md
    is_half_layer = layer_id in ("l1_5", "l2_5", "l3_5", "l4_5")

    if is_half_layer:
        # Half-layers: no axis link
        backlinks = (
            f"[Back to {layer_label}](../index.md) | "
            f"[Browse all options](../../browse_by_option.md)"
        )
    elif subdir in ("op", "family", "diagnostic_ops"):
        backlinks = (
            f"[Back to `{subdir}` axis](../axes/{subdir}.md) | "
            f"[Back to {layer_label}](../index.md) | "
            f"[Browse all options](../../browse_by_option.md)"
        )
    else:
        # Axis-based subdir (e.g., "point_metrics", "transform_policy")
        backlinks = (
            f"[Back to `{subdir}` axis](../axes/{subdir}.md) | "
            f"[Back to {layer_label}](../index.md) | "
            f"[Browse all options](../../browse_by_option.md)"
        )

    # --- Status label ---
    status_label = "Operational" if status == "operational" else "Future"

    # --- Blockquote ---
    blockquote_lines = [
        f"> {status_label} op under axis `{axis_key}`, "
        f"sub-layer `{sublayer}`, layer `{layer_id}`."
    ]
    if callable_name:
        blockquote_lines.append(
            f"> Standalone callable: `mf.functions.{callable_name}`."
        )
    blockquote = "\n".join(blockquote_lines)

    # --- Build page sections ---
    sections: list[str] = []

    # Title
    sections.append(f"# `{op_name}` -- {one_liner}")
    sections.append("")
    sections.append(backlinks)
    sections.append("")
    sections.append(blockquote)

    # Signature, parameters, returns (only when callable exists)
    if callable_obj is not None:
        sections.append("")
        sections.append("## Function signature")
        sections.append("")
        sections.append(render_sig_line(callable_name, callable_obj))
        sections.append("")
        sections.append("## Parameters")
        sections.append("")
        sections.append(render_params_table(callable_obj))
        sections.append("")
        sections.append("## Returns")
        sections.append("")
        ret_str = get_return_annotation_str(callable_obj)
        sections.append(f"`{ret_str}` — scalar result.")

        # Behavior section (from docstring body)
        doc = inspect.getdoc(callable_obj) or ""
        behavior = get_behavior_paragraph(doc)
        sections.append("")
        sections.append("## Behavior")
        sections.append("")
        sections.append(behavior)

    # In recipe context section
    sections.append("")
    sections.append("## In recipe context")
    sections.append("")
    sections.append(
        f"Set ``params.{axis_key} = \"{op_name}\"`` in the relevant layer "
        f"to activate this op within a recipe:"
    )
    sections.append("")
    sections.append(f"```yaml")
    sections.append(f"# Layer {layer_label} recipe fragment")
    sections.append(f"params:")
    sections.append(f"  {axis_key}: {op_name}")
    sections.append(f"```")

    # References section
    sections.append("")
    sections.append("## References")
    sections.append("")
    if callable_obj is not None:
        doc = inspect.getdoc(callable_obj) or ""
        refs = extract_references(doc)
        if refs:
            for ref in refs:
                sections.append(ref)
        else:
            sections.append(
                f"* macroforecast design, {layer_label}: "
                f"see design docs for {op_name}."
            )
    else:
        sections.append(
            f"* macroforecast design, {layer_label}: "
            f"see design docs for {op_name}."
        )

    # Related ops section (boilerplate — lists siblings from same axis)
    sections.append("")
    sections.append("## Related ops")
    sections.append("")
    siblings = _find_sibling_ops(op_name, layer_id, subdir)
    if siblings:
        sibling_str = ", ".join(f"`{s}`" for s in siblings)
        sections.append(f"See also: {sibling_str} (on the same axis).")
    else:
        sections.append("See the layer index for related ops.")

    # Footer
    sections.append("")
    sections.append(f"_Last reviewed {review_date} by macroforecast author._")
    sections.append("")

    return "\n".join(sections)


def _find_sibling_ops(op_name: str, layer_id: str, subdir: str) -> list[str]:
    """Find other op names in the same layer/subdir bucket (from OP_TO_AXIS_SUBDIR)."""
    siblings: list[str] = []
    for name, axis in sorted(OP_TO_AXIS_SUBDIR.items()):
        if name == op_name:
            continue
        if axis == subdir:
            # Check if this op belongs to the same layer by checking OP_TO_STANDALONE
            # (L5 ops are all in l5, L6 in l6, L2 in l2)
            if layer_id in ("l5", "l6"):
                siblings.append(name)
            elif layer_id == "l2":
                siblings.append(name)
    # For l3/l7/l4 single-subdir layers, use OP_TO_STANDALONE keys for that layer
    if not siblings and layer_id in ("l3", "l7"):
        for name, callable_name in sorted(OP_TO_STANDALONE.items()):
            if name == op_name:
                continue
            # Rough heuristic: l3 ops end in _transform or are known l3 names
            if layer_id == "l3":
                import macroforecast.core.ops.registry as reg
                ops = reg.list_ops()
                spec = ops.get(name)
                if spec and layer_id in spec.layer_scope:
                    siblings.append(name)
    return siblings[:5]  # Limit to 5 siblings for readability


# ---------------------------------------------------------------------------
# Step 3 — Collect work items per layer
# ---------------------------------------------------------------------------


def collect_work_items(
    selected_layers: list[str],
    ops: dict[str, Any],
    mf_functions: Any,
) -> list[tuple[str, str, str, str, str, str | None]]:
    """Build the list of (layer_id, op_name, subdir, status, sublayer, callable_name).

    Layer handling:

    - L0, L1, L8: no per-op pages (axes only). Skipped.
    - L2: virtual ops from L2_VIRTUAL_OPS_ORDERED + OP_TO_AXIS_SUBDIR.
      The encyclopedia page names for L2 are NOT registry op names;
      they are the named axis values (e.g., "mccracken_ng_iqr" not a registry op).
    - L3: registry ops with l3 in layer_scope, excluding EXCLUSION_LIST.
      Multi-layer ops (e.g., diff with scope ('l2', 'l3')) ARE included for L3.
    - L4: virtual family ops from OPERATIONAL_MODEL_FAMILIES + FUTURE_MODEL_FAMILIES.
      The registry has a single "fit_model" op with a "family" param;
      the encyclopedia pages correspond to each family value.
    - L5: virtual metric ops from L5_VIRTUAL_OPS. These are standalone callable
      names that serve as recipe axis values, not registry op names.
    - L6: virtual test ops from L6_VIRTUAL_OPS. Same rationale as L5.
    - L7: registry ops with l7 in layer_scope that have a standalone callable.
    - Half-layers (l1_5, l2_5, l3_5, l4_5): registry ops for those layers,
      excluding EXCLUSION_LIST (the diagnostic_collect ops).
    """
    work: list[tuple[str, str, str, str, str, str | None]] = []

    # Lazily import L4 family lists (available after _bootstrap_ops)
    from macroforecast.core.ops.l4_ops import (
        OPERATIONAL_MODEL_FAMILIES,
        FUTURE_MODEL_FAMILIES,
    )

    for layer_id in selected_layers:
        subdir_rule = LAYER_OUTPUT_MAP.get(layer_id)

        if subdir_rule is None:
            # No per-op pages for this layer (L0, L1, L8)
            print(
                f"[INFO] Layer {LAYER_LABEL[layer_id]} has no per-op pages — skipping.",
                file=sys.stderr,
            )
            continue

        if layer_id == "l2":
            # L2: use virtual ops enumerated in L2_VIRTUAL_OPS_ORDERED
            for op_name, status in L2_VIRTUAL_OPS_ORDERED:
                if op_name in EXCLUSION_LIST:
                    continue
                subdir = OP_TO_AXIS_SUBDIR.get(op_name)
                if subdir is None:
                    print(
                        f"[WARN] L2 virtual op {op_name!r} not in OP_TO_AXIS_SUBDIR — skipping.",
                        file=sys.stderr,
                    )
                    continue
                sublayer = L2_AXIS_TO_SUBLAYER.get(subdir, "l2_b")
                callable_name = OP_TO_STANDALONE.get(op_name)
                work.append((layer_id, op_name, subdir, status, sublayer, callable_name))

        elif layer_id == "l4":
            # L4: enumerate model families as virtual ops
            for family in sorted(OPERATIONAL_MODEL_FAMILIES):
                if family in EXCLUSION_LIST:
                    continue
                callable_name = OP_TO_STANDALONE.get(family)
                work.append((layer_id, family, "family", "operational", L4_SUBLAYER, callable_name))
            for family in sorted(FUTURE_MODEL_FAMILIES):
                if family in EXCLUSION_LIST:
                    continue
                callable_name = OP_TO_STANDALONE.get(family)
                work.append((layer_id, family, "family", "future", L4_SUBLAYER, callable_name))

        elif layer_id == "l5":
            # L5: enumerate virtual metric ops
            for op_name, status in sorted(L5_VIRTUAL_OPS.items()):
                if op_name in EXCLUSION_LIST:
                    continue
                subdir = OP_TO_AXIS_SUBDIR.get(op_name, "point_metrics")
                sublayer = L5_AXIS_TO_SUBLAYER.get(subdir, "L5_A_metric_specification")
                callable_name = OP_TO_STANDALONE.get(op_name)
                work.append((layer_id, op_name, subdir, status, sublayer, callable_name))

        elif layer_id == "l6":
            # L6: enumerate virtual test ops
            for op_name, status in sorted(L6_VIRTUAL_OPS.items()):
                if op_name in EXCLUSION_LIST:
                    continue
                subdir = OP_TO_AXIS_SUBDIR.get(op_name, "equal_predictive_test")
                sublayer = L6_AXIS_TO_SUBLAYER.get(subdir, "L6_A_equal_predictive")
                callable_name = OP_TO_STANDALONE.get(op_name)
                work.append((layer_id, op_name, subdir, status, sublayer, callable_name))

        elif layer_id in ("l1_5", "l2_5", "l3_5", "l4_5"):
            # Half-layers: enumerate registry ops for this layer
            # EXCLUDE the diagnostic_collect ops (they are internal aggregators)
            for op_name, op_spec in sorted(ops.items()):
                if op_name in EXCLUSION_LIST:
                    continue
                if op_spec.layer_scope == "universal":
                    continue
                if layer_id not in op_spec.layer_scope:
                    continue
                sublayer = HALF_LAYER_SUBLABEL.get(layer_id, layer_id)
                callable_name = OP_TO_STANDALONE.get(op_name)
                work.append((
                    layer_id, op_name, "diagnostic_ops",
                    str(op_spec.status), sublayer, callable_name,
                ))

        elif layer_id == "l3":
            # L3: registry ops with l3 in layer_scope (including multi-layer ops)
            # Multi-layer ops like diff, log, lag with scope ('l2', 'l3')
            # ARE included for L3 (they have encyclopedia pages in L3).
            for op_name, op_spec in sorted(ops.items()):
                if op_name in EXCLUSION_LIST:
                    continue
                if op_spec.layer_scope == "universal":
                    continue
                if layer_id not in op_spec.layer_scope:
                    continue
                callable_name = OP_TO_STANDALONE.get(op_name)
                work.append((
                    layer_id, op_name, "op",
                    str(op_spec.status), L3_SUBLAYER, callable_name,
                ))

        elif layer_id == "l7":
            # L7: registry ops with l7 in layer_scope that are user-facing
            # (i.e., have encyclopedia pages — checked via non-exclusion)
            for op_name, op_spec in sorted(ops.items()):
                if op_name in EXCLUSION_LIST:
                    continue
                if op_spec.layer_scope == "universal":
                    continue
                if layer_id not in op_spec.layer_scope:
                    continue
                callable_name = OP_TO_STANDALONE.get(op_name)
                work.append((
                    layer_id, op_name, "op",
                    str(op_spec.status), L7_SUBLAYER, callable_name,
                ))

        else:
            print(
                f"[WARN] Layer {layer_id!r} not handled — skipping.",
                file=sys.stderr,
            )

    # Sort deterministically by (layer_id, op_name)
    work.sort(key=lambda x: (x[0], x[1]))
    return work


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for gen_encyclopedia_docs.py.

    Returns 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="gen_encyclopedia_docs.py",
        description="Generate macroforecast encyclopedia op pages from introspection.",
    )
    parser.add_argument(
        "--layer",
        default="all",
        choices=["all", "L0", "L1", "L1.5", "L2", "L2.5",
                 "L3", "L3.5", "L4", "L4.5", "L5", "L6", "L7", "L8"],
        help="Layer(s) to generate. Default: all.",
    )
    parser.add_argument(
        "--out",
        default="docs/encyclopedia/",
        metavar="DIR",
        help="Output base directory. Default: docs/encyclopedia/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print file paths and first 5 lines of content; do not write.",
    )
    parser.add_argument(
        "--diff-against",
        metavar="PATH",
        help="After generating to a temp dir, diff against PATH. Implies --dry-run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files without checking for changes.",
    )
    parser.add_argument(
        "--review-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Date string for the '_Last reviewed' footer. Default: today.",
    )

    args = parser.parse_args()

    # Validate --review-date
    if args.review_date is not None:
        try:
            datetime.date.fromisoformat(args.review_date)
        except ValueError:
            print(
                f"ERROR: --review-date {args.review_date!r} is not a valid YYYY-MM-DD date.",
                file=sys.stderr,
            )
            return 1
        review_date = args.review_date
    else:
        review_date = datetime.date.today().isoformat()

    # --diff-against implies no writes to --out
    effective_dry_run = args.dry_run or bool(args.diff_against)

    # Determine output directory
    if args.diff_against:
        # Validate --diff-against is an existing directory
        diff_base = Path(args.diff_against)
        if not diff_base.is_dir():
            print(
                f"ERROR: --diff-against {args.diff_against!r} is not an existing directory.",
                file=sys.stderr,
            )
            return 1
        # Write to a temp dir for diffing
        tmp_dir = Path(tempfile.mkdtemp(prefix="mf_enc_gen_"))
        out_base = tmp_dir
    else:
        out_base = Path(args.out)

    if not effective_dry_run:
        # Validate or create --out directory
        try:
            out_base.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Cannot create output directory {out_base}: {e}", file=sys.stderr)
            return 1

    # Bootstrap ops registry
    try:
        _bootstrap_ops()
    except Exception as e:
        print(f"ERROR: Failed to import ops modules: {e}", file=sys.stderr)
        return 1

    # Import standalone functions namespace
    try:
        import macroforecast.functions as mf_functions
    except Exception as e:
        print(f"ERROR: Failed to import macroforecast.functions: {e}", file=sys.stderr)
        return 1

    # Import registry
    from macroforecast.core.ops.registry import list_ops

    ops = list_ops()

    # Determine selected layers
    if args.layer == "all":
        selected_layers = list(ALL_LAYERS)
    else:
        internal_id = LAYER_CLI_TO_INTERNAL[args.layer]
        selected_layers = [internal_id]

    # Collect work items
    work = collect_work_items(selected_layers, ops, mf_functions)

    # Track statistics
    n_generated = 0
    n_unchanged = 0
    n_updated = 0

    for layer_id, op_name, subdir, status, sublayer, callable_name in work:
        # Resolve axis_key for recipe context section
        axis_keys_for_layer = AXIS_KEY_FOR_LAYER.get(layer_id, {})
        axis_key = axis_keys_for_layer.get(subdir, subdir)

        # Render the page content
        try:
            content = render_page(
                op_name=op_name,
                layer_id=layer_id,
                subdir=subdir,
                status=status,
                sublayer=sublayer,
                review_date=review_date,
                callable_name=callable_name,
                mf_functions=mf_functions,
                axis_key=axis_key,
            )
        except Exception as e:
            print(
                f"[ERROR] Failed to render page for {op_name} (layer {layer_id}): {e}",
                file=sys.stderr,
            )
            continue

        # Compute output file path
        out_path = out_base / layer_id / subdir / f"{op_name}.md"

        if effective_dry_run and not args.diff_against:
            # Print path and first 5 lines
            first_5 = "\n".join(content.splitlines()[:5])
            print(f"[DRY-RUN] {out_path}")
            print(textwrap.indent(first_5, "  "))
            print()
            n_generated += 1
            continue

        if args.diff_against:
            # Write to temp dir for later diffing
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(content, encoding="utf-8")
            n_generated += 1
            continue

        # Write to --out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")

        if out_path.exists() and not args.force:
            existing = out_path.read_bytes()
            if existing == encoded:
                print(f"UNCHANGED {out_path}")
                n_unchanged += 1
                n_generated += 1
                continue
            else:
                out_path.write_bytes(encoded)
                print(f"UPDATED {out_path}")
                n_updated += 1
        else:
            out_path.write_bytes(encoded)
            print(f"WROTE {out_path}")
            n_generated += 1

    # Handle --diff-against: diff temp dir against reference
    if args.diff_against:
        diff_base = Path(args.diff_against)
        print(f"\n[DIFF] Comparing generated output against {diff_base}\n")
        n_changed = 0
        n_new = 0
        n_missing = 0

        # Walk the temp dir for generated files
        for gen_file in sorted(out_base.rglob("*.md")):
            rel = gen_file.relative_to(out_base)
            ref_file = diff_base / rel
            gen_content = gen_file.read_text(encoding="utf-8")

            if not ref_file.exists():
                print(f"[NEW]     {rel}")
                n_new += 1
                continue

            ref_content = ref_file.read_text(encoding="utf-8")
            if gen_content != ref_content:
                diff = list(
                    difflib.unified_diff(
                        ref_content.splitlines(keepends=True),
                        gen_content.splitlines(keepends=True),
                        fromfile=str(rel),
                        tofile=str(rel) + " (generated)",
                        n=2,
                    )
                )
                print(f"[CHANGED] {rel} ({len(diff)} diff lines)")
                n_changed += 1
            else:
                print(f"[SAME]    {rel}")

        print(
            f"\nDiff summary: {n_changed} changed, {n_new} new, "
            f"{n_missing} missing from generated set."
        )

    # Print final summary
    if not args.diff_against:
        total_str = f"Generated {n_generated} pages"
        if not effective_dry_run:
            total_str += f", {n_unchanged} unchanged, {n_updated} updated."
        print(total_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())
