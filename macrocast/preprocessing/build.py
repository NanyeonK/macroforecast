from __future__ import annotations

from .errors import PreprocessValidationError
from .types import PreprocessContract

_TARGET = {
    "raw_level",
    "tcode_transformed",
    "custom_target_transform",
}
_X = {
    "raw_level",
    "dataset_tcode_transformed",
    "custom_x_transform",
}
_TCODE = {
    "raw_only",
    "tcode_only",
    "tcode_then_extra_preprocess",
    "extra_preprocess_without_tcode",
    "extra_then_tcode",
    "custom_transform_pipeline",
}
_REPRESENTATION_POLICY = {
    "raw_only",
    "tcode_only",
    "custom_transform_only",
}
_TCODE_APPLICATION_SCOPE = {
    "apply_tcode_to_target",
    "apply_tcode_to_X",
    "apply_tcode_to_both",
    "apply_tcode_to_none",
}
_MISSING = {
    "none",
    "drop",
    "em_impute",
    "mean_impute",
    "median_impute",
    "ffill",
    "interpolate_linear",
    "drop_rows",
    "drop_columns",
    "drop_if_above_threshold",
    "missing_indicator",
    "custom",
}
_OUTLIER = {
    "none",
    "clip",
    "outlier_to_nan",
    "winsorize",
    "trim",
    "iqr_clip",
    "mad_clip",
    "zscore_clip",
    "outlier_to_missing",
    "custom",
}
_SCALING = {
    "none",
    "standard",
    "robust",
    "minmax",
    "demean_only",
    "unit_variance_only",
    "rank_scale",
    "custom",
}
_DIMRED = {
    "none",
    "pca",
    "static_factor",
    "ipca",
    "custom",
}
_FEATURE_SELECTION = {
    "none",
    "correlation_filter",
    "lasso_select",
    "mutual_information_screen",
    "custom",
}
_ORDER = {
    "none",
    "tcode_only",
    "extra_only",
    "tcode_then_extra",
    "extra_then_tcode",
    "custom",
}
_FIT_SCOPE = {
    "not_applicable",
    "train_only",
    "expanding_train_only",
    "rolling_train_only",
}
_INVERSE = {
    "none",
    "target_only",
    "forecast_scale_only",
    "custom",
}
_EVAL_SCALE = {
    "raw_level",
    "original_scale",
    "transformed_scale",
    "both",
}
_TARGET_TRANSFORM = {
    "level",
    "difference",
    "log",
    "log_difference",
    "growth_rate",
}
_TARGET_NORMALIZATION = {
    "none",
    "zscore_train_only",
    "robust_zscore",
    "minmax",
    "unit_variance",
}
_TARGET_DOMAIN = {
    "unconstrained",
    "nonnegative",
    "bounded_0_1",
    "integer_count",
    "probability_target",
}
_SCALING_SCOPE = {
    "columnwise",
    "datewise_cross_sectional",
    "groupwise",
    "categorywise",
    "global_train_only",
}
_ADDITIONAL = {
    "none",
    "smoothing_ma",
    "ema",
    "hp_filter",
    "bandpass_filter",
}
_X_LAG = {
    "no_x_lags",
    "fixed_x_lags",
    "cv_selected_x_lags",
    "variable_specific_lags",
    "category_specific_lags",
}
_FEATURE_GROUPING = {
    "none",
    "fred_category_group",
    "economic_theme_group",
    "lag_group",
    "factor_group",
}


def build_preprocess_contract(
    *,
    target_transform_policy: str,
    x_transform_policy: str,
    tcode_policy: str,
    target_missing_policy: str,
    x_missing_policy: str,
    target_outlier_policy: str,
    x_outlier_policy: str,
    scaling_policy: str,
    dimensionality_reduction_policy: str,
    feature_selection_policy: str,
    preprocess_order: str,
    preprocess_fit_scope: str,
    inverse_transform_policy: str,
    evaluation_scale: str,
    representation_policy: str = "raw_only",
    tcode_application_scope: str = "apply_tcode_to_none",
    target_transform: str = "level",
    target_normalization: str = "none",
    target_domain: str = "unconstrained",
    scaling_scope: str = "columnwise",
    additional_preprocessing: str = "none",
    x_lag_creation: str = "no_x_lags",
    feature_grouping: str = "none",
) -> PreprocessContract:
    allowed_map = {
        "target_transform_policy": _TARGET,
        "x_transform_policy": _X,
        "tcode_policy": _TCODE,
        "target_missing_policy": _MISSING,
        "x_missing_policy": _MISSING,
        "target_outlier_policy": _OUTLIER,
        "x_outlier_policy": _OUTLIER,
        "scaling_policy": _SCALING,
        "dimensionality_reduction_policy": _DIMRED,
        "feature_selection_policy": _FEATURE_SELECTION,
        "preprocess_order": _ORDER,
        "preprocess_fit_scope": _FIT_SCOPE,
        "inverse_transform_policy": _INVERSE,
        "evaluation_scale": _EVAL_SCALE,
        "representation_policy": _REPRESENTATION_POLICY,
        "tcode_application_scope": _TCODE_APPLICATION_SCOPE,
        "target_transform": _TARGET_TRANSFORM,
        "target_normalization": _TARGET_NORMALIZATION,
        "target_domain": _TARGET_DOMAIN,
        "scaling_scope": _SCALING_SCOPE,
        "additional_preprocessing": _ADDITIONAL,
        "x_lag_creation": _X_LAG,
        "feature_grouping": _FEATURE_GROUPING,
    }
    selected = {
        "target_transform_policy": target_transform_policy,
        "x_transform_policy": x_transform_policy,
        "tcode_policy": tcode_policy,
        "target_missing_policy": target_missing_policy,
        "x_missing_policy": x_missing_policy,
        "target_outlier_policy": target_outlier_policy,
        "x_outlier_policy": x_outlier_policy,
        "scaling_policy": scaling_policy,
        "dimensionality_reduction_policy": dimensionality_reduction_policy,
        "feature_selection_policy": feature_selection_policy,
        "preprocess_order": preprocess_order,
        "preprocess_fit_scope": preprocess_fit_scope,
        "inverse_transform_policy": inverse_transform_policy,
        "evaluation_scale": evaluation_scale,
        "representation_policy": representation_policy,
        "tcode_application_scope": tcode_application_scope,
        "target_transform": target_transform,
        "target_normalization": target_normalization,
        "target_domain": target_domain,
        "scaling_scope": scaling_scope,
        "additional_preprocessing": additional_preprocessing,
        "x_lag_creation": x_lag_creation,
        "feature_grouping": feature_grouping,
    }
    for field_name, value in selected.items():
        if value not in allowed_map[field_name]:
            raise PreprocessValidationError(f"unknown {field_name}={value!r}")
    return PreprocessContract(**selected)


def _has_extra_preprocessing(contract: PreprocessContract) -> bool:
    return any(
        value != "none"
        for value in (
            contract.target_missing_policy,
            contract.x_missing_policy,
            contract.target_outlier_policy,
            contract.x_outlier_policy,
            contract.scaling_policy,
            contract.dimensionality_reduction_policy,
            contract.feature_selection_policy,
            contract.additional_preprocessing,
        )
    )


def _supported_train_only_extra(contract: PreprocessContract) -> bool:
    allowed_x_missing = {
        "none", "em_impute", "mean_impute", "median_impute", "ffill", "interpolate_linear",
        "drop", "drop_rows", "drop_columns", "drop_if_above_threshold", "missing_indicator",
    }
    allowed_x_outlier = {
        "none", "winsorize", "iqr_clip", "zscore_clip",
        "trim", "mad_clip", "outlier_to_missing",
    }
    allowed_scaling = {
        "none", "standard", "robust", "minmax",
        "demean_only", "unit_variance_only",
    }
    allowed_dimred = {"none", "pca", "static_factor"}
    allowed_feature_selection = {"none", "correlation_filter", "lasso_select"}
    if contract.target_missing_policy != "none":
        return False
    if contract.target_outlier_policy != "none":
        return False
    if contract.representation_policy != "raw_only":
        return False
    if contract.tcode_application_scope != "apply_tcode_to_none":
        return False
    if contract.tcode_policy != "extra_preprocess_without_tcode":
        return False
    if contract.preprocess_order != "extra_only":
        return False
    if contract.preprocess_fit_scope != "train_only":
        return False
    if contract.inverse_transform_policy != "none":
        return False
    if contract.evaluation_scale not in {"raw_level", "original_scale"}:
        return False
    if contract.scaling_scope not in {"columnwise", "global_train_only"}:
        return False
    if contract.additional_preprocessing not in {"none", "hp_filter"}:
        return False
    if contract.x_lag_creation not in {"no_x_lags", "fixed_x_lags"}:
        return False
    if contract.feature_grouping != "none":
        return False
    if contract.target_transform not in {"level", "difference", "log", "log_difference", "growth_rate"}:
        return False
    if contract.target_normalization not in {"none", "zscore_train_only", "robust_zscore"}:
        return False
    if contract.target_domain != "unconstrained":
        return False
    if contract.x_missing_policy not in allowed_x_missing:
        return False
    if contract.x_outlier_policy not in allowed_x_outlier:
        return False
    if contract.scaling_policy not in allowed_scaling:
        return False
    if contract.dimensionality_reduction_policy not in allowed_dimred:
        return False
    if contract.feature_selection_policy not in allowed_feature_selection:
        return False
    if contract.dimensionality_reduction_policy != "none" and contract.feature_selection_policy != "none":
        return False
    return True


def is_operational_preprocess_contract(contract: PreprocessContract) -> bool:
    from dataclasses import replace as _replace
    raw_only = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="raw_only",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="none",
        preprocess_fit_scope="not_applicable",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    # Normalize evaluation_scale: original_scale and raw_level are aliases for back-compat (Phase 3 rename).
    eval_norm = "raw_level" if contract.evaluation_scale in {"raw_level", "original_scale"} else contract.evaluation_scale
    contract_norm = _replace(contract, evaluation_scale=eval_norm)
    tcode_only = build_preprocess_contract(
        target_transform_policy="tcode_transformed",
        x_transform_policy="dataset_tcode_transformed",
        tcode_policy="tcode_only",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="tcode_only",
        preprocess_fit_scope="not_applicable",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
        representation_policy="tcode_only",
        tcode_application_scope="apply_tcode_to_both",
    )
    tcode_only_norm = _replace(tcode_only, evaluation_scale=eval_norm)
    return contract_norm in (raw_only, tcode_only_norm) or _supported_train_only_extra(contract)


def check_preprocess_governance(
    contract: PreprocessContract,
    *,
    preprocessing_sweep: bool = False,
    model_sweep: bool = False,
) -> None:
    extra_present = _has_extra_preprocessing(contract)

    if contract.representation_policy == "raw_only":
        if contract.tcode_application_scope != "apply_tcode_to_none":
            raise PreprocessValidationError("raw_only representation requires tcode_application_scope='apply_tcode_to_none'")
        if contract.target_transform_policy != "raw_level" or contract.x_transform_policy != "raw_level":
            raise PreprocessValidationError("raw_only requires raw-level target and x representation")
        if contract.tcode_policy == "tcode_only":
            raise PreprocessValidationError("raw_only representation cannot pair with tcode_only ordering")

    if contract.representation_policy == "tcode_only":
        if contract.tcode_application_scope != "apply_tcode_to_both":
            raise PreprocessValidationError("tcode_only representation currently requires tcode_application_scope='apply_tcode_to_both'")
        if contract.target_transform_policy != "tcode_transformed" or contract.x_transform_policy != "dataset_tcode_transformed":
            raise PreprocessValidationError("tcode_only representation requires tcode-transformed target and x representation")
        if contract.tcode_policy not in {"tcode_only", "tcode_then_extra_preprocess", "extra_then_tcode"}:
            raise PreprocessValidationError("tcode_only representation requires explicit tcode ordering semantics")

    if contract.tcode_application_scope == "apply_tcode_to_none":
        if contract.target_transform_policy == "tcode_transformed" or contract.x_transform_policy == "dataset_tcode_transformed":
            raise PreprocessValidationError("apply_tcode_to_none cannot use tcode-transformed representation")

    if contract.tcode_policy == "raw_only":
        if contract.target_transform_policy != "raw_level" or contract.x_transform_policy != "raw_level":
            raise PreprocessValidationError("raw_only requires raw-level target and x representation")
        if contract.preprocess_order != "none":
            raise PreprocessValidationError("raw_only requires preprocess_order='none'")
        if extra_present:
            raise PreprocessValidationError("raw_only cannot carry extra preprocessing")

    if contract.tcode_policy == "tcode_only":
        if contract.target_transform_policy != "tcode_transformed":
            raise PreprocessValidationError("tcode_only requires target_transform_policy='tcode_transformed'")
        if contract.x_transform_policy != "dataset_tcode_transformed":
            raise PreprocessValidationError("tcode_only requires x_transform_policy='dataset_tcode_transformed'")
        if contract.preprocess_order != "tcode_only":
            raise PreprocessValidationError("tcode_only requires preprocess_order='tcode_only'")
        if extra_present:
            raise PreprocessValidationError("tcode_only cannot carry extra preprocessing")

    if contract.tcode_policy == "tcode_then_extra_preprocess":
        if contract.preprocess_order != "tcode_then_extra":
            raise PreprocessValidationError("tcode_then_extra_preprocess requires preprocess_order='tcode_then_extra'")
        if contract.target_transform_policy != "tcode_transformed" or contract.x_transform_policy != "dataset_tcode_transformed":
            raise PreprocessValidationError("tcode_then_extra_preprocess requires tcode-transformed target and x representation")
        if not extra_present:
            raise PreprocessValidationError("tcode_then_extra_preprocess requires at least one extra preprocessing policy")

    if contract.tcode_policy == "extra_preprocess_without_tcode":
        if contract.preprocess_order != "extra_only":
            raise PreprocessValidationError("extra_preprocess_without_tcode requires preprocess_order='extra_only'")
        if contract.target_transform_policy == "tcode_transformed" or contract.x_transform_policy == "dataset_tcode_transformed":
            raise PreprocessValidationError("extra_preprocess_without_tcode cannot use tcode-transformed representation")
        if contract.tcode_application_scope != "apply_tcode_to_none":
            raise PreprocessValidationError("extra_preprocess_without_tcode requires tcode_application_scope='apply_tcode_to_none'")
        if not extra_present:
            raise PreprocessValidationError("extra_preprocess_without_tcode requires at least one extra preprocessing policy")

    if contract.tcode_policy == "extra_then_tcode":
        if contract.preprocess_order != "extra_then_tcode":
            raise PreprocessValidationError("extra_then_tcode requires preprocess_order='extra_then_tcode'")
        if contract.target_transform_policy != "tcode_transformed" or contract.x_transform_policy != "dataset_tcode_transformed":
            raise PreprocessValidationError("extra_then_tcode requires tcode-transformed target and x representation")
        if not extra_present:
            raise PreprocessValidationError("extra_then_tcode requires at least one extra preprocessing policy")

    if not extra_present and contract.preprocess_fit_scope != "not_applicable":
        raise PreprocessValidationError("preprocess_fit_scope can be non-trivial only when extra preprocessing is explicit")
    if extra_present and contract.preprocess_fit_scope == "not_applicable":
        raise PreprocessValidationError("extra preprocessing requires explicit preprocess_fit_scope")
    if contract.scaling_policy != "none" and contract.preprocess_fit_scope not in {"train_only", "expanding_train_only", "rolling_train_only"}:
        raise PreprocessValidationError("scaling requires explicit train-only fit scope")
    if contract.x_missing_policy in {"em_impute", "mean_impute", "median_impute"} and contract.preprocess_fit_scope not in {"train_only", "expanding_train_only", "rolling_train_only"}:
        raise PreprocessValidationError("imputation requires explicit train-only fit scope")
    if preprocessing_sweep and model_sweep:
        raise PreprocessValidationError("do not co-sweep model and preprocessing in ordinary baseline comparison")
    if contract.scaling_scope in {"datewise_cross_sectional", "groupwise", "categorywise"}:
        raise PreprocessValidationError("current runtime slice does not support non-train global scaling scopes")
    if contract.additional_preprocessing not in {"none", "hp_filter"}:
        raise PreprocessValidationError("current runtime slice does not support additional_preprocessing beyond none / hp_filter")
    if contract.x_lag_creation not in {"no_x_lags", "fixed_x_lags"}:
        raise PreprocessValidationError("current runtime slice does not support x_lag_creation beyond no_x_lags / fixed_x_lags")
    if contract.feature_grouping != "none":
        raise PreprocessValidationError("current runtime slice does not support feature_grouping beyond none")


def preprocess_to_dict(contract: PreprocessContract) -> dict[str, str]:
    return {
        "target_transform_policy": contract.target_transform_policy,
        "x_transform_policy": contract.x_transform_policy,
        "tcode_policy": contract.tcode_policy,
        "target_missing_policy": contract.target_missing_policy,
        "x_missing_policy": contract.x_missing_policy,
        "target_outlier_policy": contract.target_outlier_policy,
        "x_outlier_policy": contract.x_outlier_policy,
        "scaling_policy": contract.scaling_policy,
        "dimensionality_reduction_policy": contract.dimensionality_reduction_policy,
        "feature_selection_policy": contract.feature_selection_policy,
        "preprocess_order": contract.preprocess_order,
        "preprocess_fit_scope": contract.preprocess_fit_scope,
        "inverse_transform_policy": contract.inverse_transform_policy,
        "evaluation_scale": contract.evaluation_scale,
        "representation_policy": contract.representation_policy,
        "tcode_application_scope": contract.tcode_application_scope,
        "target_transform": contract.target_transform,
        "target_normalization": contract.target_normalization,
        "target_domain": contract.target_domain,
        "scaling_scope": contract.scaling_scope,
        "additional_preprocessing": contract.additional_preprocessing,
        "x_lag_creation": contract.x_lag_creation,
        "feature_grouping": contract.feature_grouping,
    }


def preprocess_summary(contract: PreprocessContract) -> str:
    return "; ".join(f"{key}={value}" for key, value in preprocess_to_dict(contract).items())
