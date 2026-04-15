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
_MISSING = {
    "none",
    "drop",
    "em_impute",
    "custom",
}
_OUTLIER = {
    "none",
    "clip",
    "outlier_to_nan",
    "custom",
}
_SCALING = {
    "none",
    "standard",
    "robust",
    "minmax",
    "custom",
}
_DIMRED = {
    "none",
    "pca",
    "ipca",
    "custom",
}
_FEATURE_SELECTION = {
    "none",
    "correlation_filter",
    "lasso_select",
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
    "transformed_scale",
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
        )
    )


def is_operational_preprocess_contract(contract: PreprocessContract) -> bool:
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
    train_only_raw_panel = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_without_tcode",
        target_missing_policy="none",
        x_missing_policy="em_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    train_only_raw_panel_robust = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_without_tcode",
        target_missing_policy="none",
        x_missing_policy="em_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="robust",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    return contract in {raw_only, train_only_raw_panel, train_only_raw_panel_robust}


def check_preprocess_governance(
    contract: PreprocessContract,
    *,
    preprocessing_sweep: bool = False,
    model_sweep: bool = False,
) -> None:
    extra_present = _has_extra_preprocessing(contract)

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
            raise PreprocessValidationError(
                "tcode_only requires x_transform_policy='dataset_tcode_transformed'"
            )
        if contract.preprocess_order != "tcode_only":
            raise PreprocessValidationError("tcode_only requires preprocess_order='tcode_only'")
        if extra_present:
            raise PreprocessValidationError("tcode_only cannot carry extra preprocessing")

    if contract.tcode_policy == "tcode_then_extra_preprocess":
        if contract.preprocess_order != "tcode_then_extra":
            raise PreprocessValidationError(
                "tcode_then_extra_preprocess requires preprocess_order='tcode_then_extra'"
            )
        if contract.target_transform_policy != "tcode_transformed" or contract.x_transform_policy != "dataset_tcode_transformed":
            raise PreprocessValidationError(
                "tcode_then_extra_preprocess requires tcode-transformed target and x representation"
            )
        if not extra_present:
            raise PreprocessValidationError(
                "tcode_then_extra_preprocess requires at least one extra preprocessing policy"
            )

    if contract.tcode_policy == "extra_preprocess_without_tcode":
        if contract.preprocess_order != "extra_only":
            raise PreprocessValidationError(
                "extra_preprocess_without_tcode requires preprocess_order='extra_only'"
            )
        if contract.target_transform_policy == "tcode_transformed" or contract.x_transform_policy == "dataset_tcode_transformed":
            raise PreprocessValidationError(
                "extra_preprocess_without_tcode cannot use tcode-transformed representation"
            )
        if not extra_present:
            raise PreprocessValidationError(
                "extra_preprocess_without_tcode requires at least one extra preprocessing policy"
            )

    if contract.tcode_policy == "extra_then_tcode":
        if contract.preprocess_order != "extra_then_tcode":
            raise PreprocessValidationError("extra_then_tcode requires preprocess_order='extra_then_tcode'")
        if contract.target_transform_policy != "tcode_transformed" or contract.x_transform_policy != "dataset_tcode_transformed":
            raise PreprocessValidationError(
                "extra_then_tcode requires tcode-transformed target and x representation"
            )
        if not extra_present:
            raise PreprocessValidationError("extra_then_tcode requires at least one extra preprocessing policy")

    if not extra_present and contract.preprocess_fit_scope != "not_applicable":
        raise PreprocessValidationError(
            "preprocess_fit_scope can be non-trivial only when extra preprocessing is explicit"
        )
    if extra_present and contract.preprocess_fit_scope == "not_applicable":
        raise PreprocessValidationError(
            "extra preprocessing requires explicit preprocess_fit_scope"
        )
    if contract.scaling_policy != "none" and contract.preprocess_fit_scope not in {"train_only", "expanding_train_only", "rolling_train_only"}:
        raise PreprocessValidationError("scaling requires explicit train-only fit scope")
    if contract.x_missing_policy == "em_impute" and contract.preprocess_fit_scope not in {"train_only", "expanding_train_only", "rolling_train_only"}:
        raise PreprocessValidationError("em_impute requires explicit train-only fit scope")
    if preprocessing_sweep and model_sweep:
        raise PreprocessValidationError(
            "do not co-sweep model and preprocessing in ordinary baseline comparison"
        )


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
    }


def preprocess_summary(contract: PreprocessContract) -> str:
    return "; ".join(f"{key}={value}" for key, value in preprocess_to_dict(contract).items())
