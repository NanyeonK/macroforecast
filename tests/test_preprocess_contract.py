from __future__ import annotations

import pytest

from macrocast import (
    PreprocessContract,
    PreprocessValidationError,
    build_preprocess_contract,
    check_preprocess_governance,
    is_operational_preprocess_contract,
    preprocess_summary,
    preprocess_to_dict,
)
from macrocast.preprocessing import (
    CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION,
    CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION,
    CUSTOM_FINAL_Z_SELECTION_CONTRACT_VERSION,
    FeatureBlockCallableResult,
    FeatureCombinerCallableResult,
    build_target_scale_contract,
    custom_feature_block_contract_metadata,
    custom_feature_combiner_contract_metadata,
    custom_final_z_selection_contract_metadata,
    validate_feature_block_callable_result,
    validate_feature_combiner_callable_result,
)


def _raw_only_contract() -> PreprocessContract:
    return build_preprocess_contract(
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


def _train_only_raw_panel_contract() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
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


def _train_only_raw_panel_robust_contract() -> PreprocessContract:
    return build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
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


def test_build_preprocess_contract_raw_only_operational() -> None:
    contract = _raw_only_contract()

    assert isinstance(contract, PreprocessContract)
    assert contract.tcode_policy == "raw_only"
    assert is_operational_preprocess_contract(contract) is True
    check_preprocess_governance(contract, preprocessing_sweep=False)


def test_build_preprocess_contract_train_only_raw_panel_is_operational() -> None:
    contract = _train_only_raw_panel_contract()

    assert contract.tcode_policy == "extra_preprocess_only"
    assert contract.x_missing_policy == "em_impute"
    assert contract.scaling_policy == "standard"
    assert is_operational_preprocess_contract(contract) is True
    check_preprocess_governance(contract, preprocessing_sweep=False)


def test_build_preprocess_contract_train_only_raw_panel_robust_is_operational() -> None:
    contract = _train_only_raw_panel_robust_contract()

    assert contract.tcode_policy == "extra_preprocess_only"
    assert contract.x_missing_policy == "em_impute"
    assert contract.scaling_policy == "robust"
    assert is_operational_preprocess_contract(contract) is True
    check_preprocess_governance(contract, preprocessing_sweep=False)


def test_build_preprocess_contract_tcode_then_extra_is_not_supported() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="official_tcode_transformed",
        x_transform_policy="official_tcode_transformed",
        tcode_policy="official_tcode_then_extra_preprocess",
        target_missing_policy="none",
        x_missing_policy="em_impute",
        target_outlier_policy="none",
        x_outlier_policy="outlier_to_nan",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="official_tcode_then_extra",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="target_only",
        evaluation_scale="raw_level",
        representation_policy="official_tcode_only",
        tcode_application_scope="target_and_predictors",
    )

    assert contract.preprocess_order == "official_tcode_then_extra"
    assert is_operational_preprocess_contract(contract) is False
    check_preprocess_governance(contract, preprocessing_sweep=True)


def test_build_preprocess_contract_tcode_then_train_only_extra_is_operational() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="official_tcode_transformed",
        x_transform_policy="official_tcode_transformed",
        tcode_policy="official_tcode_then_extra_preprocess",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="winsorize",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="official_tcode_then_extra",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
        representation_policy="official_tcode_only",
        tcode_application_scope="target_and_predictors",
    )

    assert is_operational_preprocess_contract(contract) is True
    check_preprocess_governance(contract, preprocessing_sweep=False)


def test_target_normalization_is_operational_with_window_fit() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
        target_normalization="zscore_train_only",
    )

    assert is_operational_preprocess_contract(contract) is True
    check_preprocess_governance(contract)


def test_target_scale_contract_records_operational_normalization() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="standard",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="target_only",
        evaluation_scale="both",
        target_transform="log",
        target_normalization="zscore_train_only",
    )

    scale = build_target_scale_contract(contract)

    assert scale["schema_version"] == "target_scale_contract_v1"
    assert scale["runtime_status"] == "operational"
    assert scale["model_target_scale"] == "transformed_target_scale"
    assert scale["forecast_scale"] == "original_target_scale"
    assert scale["normalization_fit_scope"] == "train_only"
    assert scale["blockers"] == []


def test_custom_feature_block_contract_validation() -> None:
    metadata = custom_feature_block_contract_metadata(block_kind="temporal")
    assert metadata["schema_version"] == CUSTOM_FEATURE_BLOCK_CONTRACT_VERSION
    assert metadata["block_kind"] == "temporal"
    assert "feature_names" in metadata["required_fields"]

    result = FeatureBlockCallableResult(
        train_features=object(),
        pred_features=object(),
        feature_names=("custom_ma",),
        runtime_feature_names=("custom__ma",),
        fit_state={"window": 3},
        leakage_metadata={"lookahead": "forbidden"},
        provenance={"source": "test"},
    )
    validate_feature_block_callable_result(result)


def test_custom_feature_combiner_contract_validation() -> None:
    metadata = custom_feature_combiner_contract_metadata()
    assert metadata["schema_version"] == CUSTOM_FEATURE_COMBINER_CONTRACT_VERSION
    assert "Z_train" in metadata["required_fields"]

    result = FeatureCombinerCallableResult(
        Z_train=[[1.0], [2.0]],
        Z_pred=[[3.0]],
        feature_names=("custom_combo",),
        block_roles={"custom_combo": "custom"},
        fit_state={"alpha": 1.0},
        leakage_metadata={"lookahead": "forbidden"},
        provenance={"test": "combiner"},
    )
    validate_feature_combiner_callable_result(result)


def test_custom_final_z_selection_contract_metadata() -> None:
    metadata = custom_final_z_selection_contract_metadata()
    assert metadata["schema_version"] == CUSTOM_FINAL_Z_SELECTION_CONTRACT_VERSION
    assert "selected_feature_names" in metadata["required_fields"]


def test_custom_feature_block_contract_requires_leakage_metadata() -> None:
    result = FeatureBlockCallableResult(
        train_features=object(),
        pred_features=object(),
        feature_names=("custom_ma",),
    )
    with pytest.raises(PreprocessValidationError, match="lookahead"):
        validate_feature_block_callable_result(result)


def test_preprocess_governance_rejects_dual_axis_sweep() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
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

    # fixed preprocessing + model sweep is the normal fair-comparison case — must pass
    check_preprocess_governance(contract, preprocessing_sweep=False, model_sweep=True)

    # co-sweeping both model and preprocessing must be rejected
    with pytest.raises(PreprocessValidationError):
        check_preprocess_governance(contract, preprocessing_sweep=True, model_sweep=True)


def test_preprocess_governance_rejects_raw_only_with_hidden_transform() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="official_tcode_transformed",
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

    with pytest.raises(PreprocessValidationError):
        check_preprocess_governance(contract)


def test_preprocess_summary_and_dict_expose_execution_semantics() -> None:
    contract = _train_only_raw_panel_contract()

    summary = preprocess_summary(contract)
    payload = preprocess_to_dict(contract)

    assert "x_missing_policy=em_impute" in summary
    assert "scaling_policy=standard" in summary
    assert payload["preprocess_fit_scope"] == "train_only"
    assert payload["evaluation_scale"] == "raw_level"



def test_build_preprocess_contract_supports_stage2_governance_defaults() -> None:
    contract = _raw_only_contract()
    payload = preprocess_to_dict(contract)
    assert payload["representation_policy"] == "raw_only"
    assert payload["tcode_application_scope"] == "none"


def test_build_preprocess_contract_mean_impute_minmax_is_operational() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="winsorize",
        scaling_policy="minmax",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    assert is_operational_preprocess_contract(contract) is True
    check_preprocess_governance(contract)


def test_build_preprocess_contract_pca_path_is_operational() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="median_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="standard",
        dimensionality_reduction_policy="pca",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    assert is_operational_preprocess_contract(contract) is True
    check_preprocess_governance(contract)


def test_build_preprocess_contract_accepts_combined_dimred_and_feature_selection() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="standard",
        dimensionality_reduction_policy="pca",
        feature_selection_policy="correlation_filter",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    assert is_operational_preprocess_contract(contract) is True


def test_build_preprocess_contract_accepts_select_after_factor_semantics() -> None:
    contract = build_preprocess_contract(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_only",
        target_missing_policy="none",
        x_missing_policy="mean_impute",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="standard",
        dimensionality_reduction_policy="pca",
        feature_selection_policy="lasso_select",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
        feature_selection_semantics="select_after_factor",
    )
    assert contract.feature_selection_semantics == "select_after_factor"
    assert is_operational_preprocess_contract(contract) is True
