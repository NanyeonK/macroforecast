"""macrocast.

Horse race research benchmarking package for macro forecasting.

Public surface (v0.1):
- ``macrocast.run(recipe)`` -- execute a recipe end-to-end (L1->L8) and
  return a :class:`ManifestExecutionResult`. Iterates every sweep cell.
- ``macrocast.replicate(manifest_path)`` -- re-execute a stored manifest
  and verify per-cell sink hashes match bit-for-bit.

Importable submodule surface:
- ``macrocast.custom``        - user-defined model / preprocessor / feature registration
- ``macrocast.defaults``      - default profile dict template
- ``macrocast.preprocessing`` - preprocessing contract helpers
- ``macrocast.raw``           - FRED-MD/QD/SD adapters and custom CSV/Parquet loaders
- ``macrocast.core``          - 12-layer DAG runtime (foundation, layers, ops, runtime, execution)
- ``macrocast.tuning``        - hyperparameter search engines
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.2.0"

_LAZY_EXPORTS = {
    # public top-level API
    "run": ".api",
    "run_file": ".api",
    "replicate": ".api",
    "ManifestExecutionResult": ".api",
    "ReplicationResult": ".api",
    # defaults
    "DEFAULT_PROFILE": ".defaults",
    "DEFAULT_PROFILE_NAME": ".defaults",
    "build_default_recipe_dict": ".defaults",
    # custom
    "custom_feature_block": ".custom",
    "custom_feature_combiner": ".custom",
    "custom_method_extension_contracts": ".custom",
    "custom_model_contract_metadata": ".custom",
    "custom_preprocessor": ".custom",
    "custom_preprocessor_contract_metadata": ".custom",
    "target_transformer": ".custom",
    "target_transformer_contract_metadata": ".custom",
    "CUSTOM_MODEL_CONTRACT_VERSION": ".custom",
    "CUSTOM_PREPROCESSOR_CONTRACT_VERSION": ".custom",
    "TARGET_TRANSFORMER_CONTRACT_VERSION": ".custom",
    "custom_model": ".custom",
    "register_feature_block": ".custom",
    "register_feature_combiner": ".custom",
    "register_preprocessor": ".custom",
    "register_target_transformer": ".custom",
    "register_model": ".custom",
    "get_custom_feature_block": ".custom",
    "get_custom_feature_combiner": ".custom",
    "get_custom_preprocessor": ".custom",
    "get_custom_target_transformer": ".custom",
    "get_custom_model": ".custom",
    "is_custom_feature_block": ".custom",
    "is_custom_feature_combiner": ".custom",
    "is_custom_preprocessor": ".custom",
    "is_custom_target_transformer": ".custom",
    "is_custom_model": ".custom",
    "list_custom_feature_blocks": ".custom",
    "list_custom_feature_combiners": ".custom",
    "list_custom_preprocessors": ".custom",
    "list_custom_target_transformers": ".custom",
    "list_custom_models": ".custom",
    "clear_custom_feature_blocks": ".custom",
    "clear_custom_feature_combiners": ".custom",
    "clear_custom_preprocessors": ".custom",
    "clear_custom_target_transformers": ".custom",
    "clear_custom_models": ".custom",
    "clear_custom_extensions": ".custom",
    # raw adapters
    "normalize_version_request": ".raw",
    "list_vintages": ".raw",
    "get_raw_cache_root": ".raw",
    "get_manifest_path": ".raw",
    "get_raw_file_path": ".raw",
    "build_raw_artifact_record": ".raw",
    "append_raw_manifest_entry": ".raw",
    "read_raw_manifest": ".raw",
    "parse_fred_csv": ".raw",
    "load_custom_csv": ".raw",
    "load_custom_parquet": ".raw",
    "load_fred_md": ".raw",
    "load_fred_qd": ".raw",
    "load_fred_sd": ".raw",
    "RawVersionRequest": ".raw",
    "RawDatasetMetadata": ".raw",
    "RawArtifactRecord": ".raw",
    "RawLoadResult": ".raw",
    # preprocessing
    "build_preprocess_contract": ".preprocessing",
    "check_preprocess_governance": ".preprocessing",
    "is_operational_preprocess_contract": ".preprocessing",
    "preprocess_summary": ".preprocessing",
    "preprocess_to_dict": ".preprocessing",
    "PreprocessContractError": ".preprocessing",
    "PreprocessValidationError": ".preprocessing",
    "PreprocessContract": ".preprocessing",
    "TargetTransformPolicy": ".preprocessing",
    "XTransformPolicy": ".preprocessing",
    "TcodePolicy": ".preprocessing",
    "MissingPolicy": ".preprocessing",
    "OutlierPolicy": ".preprocessing",
    "ScalingPolicy": ".preprocessing",
    "DimensionalityReductionPolicy": ".preprocessing",
    "FeatureSelectionPolicy": ".preprocessing",
    "PreprocessOrder": ".preprocessing",
    "PreprocessFitScope": ".preprocessing",
    "InverseTransformPolicy": ".preprocessing",
    "EvaluationScale": ".preprocessing",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
