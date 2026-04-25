from __future__ import annotations

from collections.abc import Mapping

IMPORTANCE_CONTRACT_VERSION = "layer7_importance_split_v1"

IMPORTANCE_AXIS_NAMES = (
    "importance_model_native",
    "importance_model_agnostic",
    "importance_shap",
    "importance_local_surrogate",
    "importance_partial_dependence",
    "importance_grouped",
    "importance_stability",
)

IMPORTANCE_META_AXIS_NAMES = (
    "importance_scope",
    "importance_aggregation",
    "importance_output_style",
    "importance_temporal",
    "importance_gradient_path",
)

DEFAULT_IMPORTANCE_SPEC: dict[str, str] = {
    "importance_method": "none",
    "importance_scope": "global",
    "importance_model_native": "none",
    "importance_model_agnostic": "none",
    "importance_shap": "none",
    "importance_local_surrogate": "none",
    "importance_partial_dependence": "none",
    "importance_grouped": "none",
    "importance_stability": "none",
    "importance_aggregation": "mean_abs",
    "importance_output_style": "ranked_table",
    "importance_temporal": "static_snapshot",
    "importance_gradient_path": "none",
}

LEGACY_IMPORTANCE_METHOD_TO_AXIS = {
    "minimal_importance": ("importance_model_native", "minimal_importance"),
    "tree_shap": ("importance_shap", "tree_shap"),
    "kernel_shap": ("importance_shap", "kernel_shap"),
    "linear_shap": ("importance_shap", "linear_shap"),
    "permutation_importance": ("importance_model_agnostic", "permutation_importance"),
    "lime": ("importance_local_surrogate", "lime"),
    "feature_ablation": ("importance_local_surrogate", "feature_ablation"),
    "pdp": ("importance_partial_dependence", "pdp"),
    "ice": ("importance_partial_dependence", "ice"),
    "ale": ("importance_partial_dependence", "ale"),
    "grouped_permutation": ("importance_grouped", "grouped_permutation"),
    "importance_stability": ("importance_stability", "importance_stability"),
}

IMPORTANCE_FILE_NAMES = {
    "minimal_importance": "importance_minimal.json",
    "tree_shap": "importance_tree_shap.json",
    "kernel_shap": "importance_kernel_shap.json",
    "linear_shap": "importance_linear_shap.json",
    "permutation_importance": "importance_permutation_importance.json",
    "lime": "importance_lime.json",
    "feature_ablation": "importance_feature_ablation.json",
    "pdp": "importance_pdp.json",
    "ice": "importance_ice.json",
    "ale": "importance_ale.json",
    "grouped_permutation": "importance_grouped_permutation.json",
    "importance_stability": "importance_stability.json",
}

LOCAL_IMPORTANCE_METHODS = frozenset({"kernel_shap", "lime", "feature_ablation"})


def canonicalize_importance_spec(raw_spec: Mapping[str, object] | None) -> dict[str, str]:
    raw = dict(raw_spec or {})
    spec = dict(DEFAULT_IMPORTANCE_SPEC)
    for key, value in raw.items():
        if key in spec and value is not None:
            spec[key] = str(value)

    legacy_method = spec.get("importance_method", "none")
    if legacy_method != "none" and legacy_method in LEGACY_IMPORTANCE_METHOD_TO_AXIS:
        axis, value = LEGACY_IMPORTANCE_METHOD_TO_AXIS[legacy_method]
        if spec.get(axis, "none") == "none":
            spec[axis] = value
    scope_explicit = "importance_scope" in raw and raw.get("importance_scope") is not None
    methods = {spec[axis] for axis in IMPORTANCE_AXIS_NAMES if spec.get(axis, "none") != "none"}
    if methods and not scope_explicit and methods <= LOCAL_IMPORTANCE_METHODS:
        spec["importance_scope"] = "local"
    return spec


def active_importance_axes(raw_spec: Mapping[str, object] | None) -> dict[str, str]:
    spec = canonicalize_importance_spec(raw_spec)
    return {axis: spec[axis] for axis in IMPORTANCE_AXIS_NAMES if spec.get(axis, "none") != "none"}


def active_importance_methods(raw_spec: Mapping[str, object] | None) -> tuple[str, ...]:
    methods: list[str] = []
    for method in active_importance_axes(raw_spec).values():
        if method not in methods:
            methods.append(method)
    return tuple(methods)
