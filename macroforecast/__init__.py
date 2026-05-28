"""macroforecast.

Horse race research benchmarking package for macro forecasting.

**Public surface**

- ``macroforecast.forecast(...)`` -- one-shot helper (v0.8.0); assembles a
  default recipe and runs it.
- ``macroforecast.Experiment(...)`` -- builder class (v0.8.0) with
  ``compare_models`` / ``compare`` / ``sweep`` / ``run`` /
  ``replicate`` / ``to_yaml`` / ``validate``.
- ``macroforecast.ForecastResult`` -- thin façade over
  :class:`ManifestExecutionResult` returned by both.
- ``macroforecast.run(recipe)`` -- execute a recipe end-to-end (L1->L8) and
  return a :class:`ManifestExecutionResult`. Iterates every sweep cell.
- ``macroforecast.replicate(manifest_path)`` -- re-execute a stored manifest
  and verify per-cell sink hashes match bit-for-bit.

**Importable submodule surface**

- ``macroforecast.recipes`` -- recipe orchestration namespace; ``mf.run`` /
  ``mf.replicate`` / ``mf.Experiment`` / ``mf.forecast`` are aliases here.
- ``macroforecast.meta`` -- package-wide execution settings.
- ``macroforecast.data`` -- canonical panels, metadata, FRED/custom loaders, and data specs.
- ``macroforecast.preprocessing`` -- preprocessing schemas and contract helpers.
- ``macroforecast.features`` -- feature engineering ops, transforms, selectors.
- ``macroforecast.models`` -- public model classes, model ops, paper helpers, tuning.
- ``macroforecast.evaluation`` -- forecast metric schema and ops.
- ``macroforecast.stat_tests`` -- forecast-comparison statistical tests.
- ``macroforecast.interpretation`` -- interpretation schemas, ops, and methods.
- ``macroforecast.output`` -- output/provenance schema and export ops.
- ``macroforecast.diagnostics`` -- data/preprocessing/features/generator diagnostics.
- ``macroforecast.core`` -- cross-layer runtime, registry, manifest, cache, and execution.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.9.5a0"

_LAZY_EXPORTS = {
    # public top-level API -- routed through macroforecast.recipes (v0.9.5a)
    # mf.run / mf.replicate / etc. are silent aliases for mf.recipes.run / etc.
    "run": ".recipes",
    "run_file": ".recipes",
    "replicate": ".recipes",
    "ManifestExecutionResult": ".recipes",
    "ReplicationResult": ".recipes",
    # high-level façade -- also routed through macroforecast.recipes (v0.9.5a)
    "forecast": ".recipes",
    "Experiment": ".recipes",
    "ForecastResult": ".recipes",
    # defaults (now at macroforecast.api.defaults)
    "DEFAULT_PROFILE": ".api.defaults",
    "DEFAULT_PROFILE_NAME": ".api.defaults",
    "build_default_recipe_dict": ".api.defaults",
    # custom (now at macroforecast.api.custom)
    "custom_feature_block": ".api.custom",
    "custom_feature_combiner": ".api.custom",
    "custom_method_extension_contracts": ".api.custom",
    "custom_model_contract_metadata": ".api.custom",
    "custom_preprocessor": ".api.custom",
    "custom_preprocessor_contract_metadata": ".api.custom",
    "target_transformer": ".api.custom",
    "target_transformer_contract_metadata": ".api.custom",
    "CUSTOM_MODEL_CONTRACT_VERSION": ".api.custom",
    "CUSTOM_PREPROCESSOR_CONTRACT_VERSION": ".api.custom",
    "TARGET_TRANSFORMER_CONTRACT_VERSION": ".api.custom",
    "custom_model": ".api.custom",
    "register_feature_block": ".api.custom",
    "register_feature_combiner": ".api.custom",
    "register_preprocessor": ".api.custom",
    "register_target_transformer": ".api.custom",
    "register_model": ".api.custom",
    "get_custom_feature_block": ".api.custom",
    "get_custom_feature_combiner": ".api.custom",
    "get_custom_preprocessor": ".api.custom",
    "get_custom_target_transformer": ".api.custom",
    "get_custom_model": ".api.custom",
    "is_custom_feature_block": ".api.custom",
    "is_custom_feature_combiner": ".api.custom",
    "is_custom_preprocessor": ".api.custom",
    "is_custom_target_transformer": ".api.custom",
    "is_custom_model": ".api.custom",
    "list_custom_feature_blocks": ".api.custom",
    "list_custom_feature_combiners": ".api.custom",
    "list_custom_preprocessors": ".api.custom",
    "list_custom_target_transformers": ".api.custom",
    "list_custom_models": ".api.custom",
    "clear_custom_feature_blocks": ".api.custom",
    "clear_custom_feature_combiners": ".api.custom",
    "clear_custom_preprocessors": ".api.custom",
    "clear_custom_target_transformers": ".api.custom",
    "clear_custom_models": ".api.custom",
    "clear_custom_extensions": ".api.custom",
    # data
    "list_vintages": ".data",
    "load_custom_csv": ".data",
    "load_custom_parquet": ".data",
    "load_fred_md": ".data",
    "load_fred_qd": ".data",
    "load_fred_sd": ".data",
    "as_panel": ".data",
    "metadata": ".data",
    "spec": ".data",
    "validate_panel": ".data",
    "panel_info": ".data",
    "DataBundle": ".data",
    "DataSpec": ".data",
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

_LAZY_MODULES: tuple[str, ...] = (
    "recipes",
    "api",
    "meta",
    "data",
    "preprocessing",
    "features",
    "models",
    "evaluation",
    "stat_tests",
    "interpretation",
    "output",
    "diagnostics",
    # Promoted compatibility namespaces
    "feature_selection",
    "transforms",
)
"""Submodules exposed as ``macroforecast.<name>`` via lazy import.

v0.1.0: ``functions`` added for the per-op standalone callable namespace
(``mf.functions.ridge_fit``, ``mf.functions.theil_u1``, etc.).
v0.9.5: semantic module namespaces expose the former numbered layer bodies;
``feature_selection`` and ``transforms`` remain promoted compatibility namespaces.
"""

__all__ = sorted(set(_LAZY_EXPORTS) | set(_LAZY_MODULES))


# Backward-compat aliases: top-level names that moved into macroforecast.api/
_SUBMODULE_ALIASES: dict[str, str] = {
    "functions": ".api.functions",
    "custom": ".api.custom",
    "defaults": ".api.defaults",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    # backward-compat: old top-level submodule names now live under api/
    alias_path = _SUBMODULE_ALIASES.get(name)
    if alias_path is not None:
        module = import_module(alias_path, __name__)
        globals()[name] = module
        return module
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | set(_LAZY_MODULES) | set(_SUBMODULE_ALIASES))
